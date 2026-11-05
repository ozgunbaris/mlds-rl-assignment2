"""Validation experiments for Double DQN on MsPacman-v5 (writes validation.md / validation.json + figures).

    python -m dqn.validate_mspacman --run results/mspacman/main_seed0 --out results/mspacman/validation

P1  preprocessing: raw frame, preprocessed frame and the 4-frame state; which pixels the
    Ms. Pac-Man colour (sum 448) removes over a whole game.
P2  random policy: score and length of 100 games with uniformly random actions.
P3  learning progress: 20 greedy games for every saved snapshot of the network.
P4  calibration: predicted Q(s_t, a_t) against the realised discounted clipped return of the same
    greedy policy, cut at every lost life exactly as in the training target.
P5  event-triggered average of max_a Q around lost lives, power pellets and eaten ghosts.
P6  the first 10 steps of a greedy game.
P7  determinism (sticky actions off: the same game every time) and the frozen start of a game
    (steps where no action changes the emulator), with the agent's action gap there.
P8  where Ms. Pac-Man goes: position heat maps early and late in training.
P9  composition of the score (pellets, power pellets, ghosts and fruit).
"""
import argparse
import glob
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

import viz
from .common import discounted_returns
from .mspacman_dqn import ENV_ID, N_ACTIONS, FrameStack, frame, load_net, mspacman_color, preprocess_observation

ACTIONS = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
PAC_RGB = (210, 164, 74)


def pac_position(obs):
    m = (obs[..., 0] == PAC_RGB[0]) & (obs[..., 1] == PAC_RGB[1]) & (obs[..., 2] == PAC_RGB[2])
    if not m.any():
        return (np.nan, np.nan)
    r, c = np.nonzero(m)
    return (float(r.mean()), float(c.mean()))


def play_record(net, env, eps, rng, max_steps=30_000, keep_frames=0):
    """One game with full per-step records."""
    obs, info = env.reset()
    st = FrameStack()
    st.reset(frame(obs))
    rec = {"q": [], "a": [], "r": [], "lives": [], "lives_before": [], "pos": [], "frames": [], "raw": []}
    lives = info["lives"]
    done = False
    t = 0
    while not done and t < max_steps:
        with torch.no_grad():
            q = net(torch.from_numpy(st.get())[None])[0].numpy() if net is not None else np.zeros(N_ACTIONS)
        a = int(rng.integers(N_ACTIONS)) if (net is None or rng.random() < eps) else int(q.argmax())
        rec["q"].append(q)
        rec["a"].append(a)
        rec["pos"].append(pac_position(obs))
        rec["lives_before"].append(lives)
        if t < keep_frames:
            rec["frames"].append(st.get().copy())
            rec["raw"].append(obs.copy())
        obs, r, term, trunc, info = env.step(a)
        st.push(frame(obs))
        rec["r"].append(float(r))
        rec["lives"].append(info["lives"])
        lives = info["lives"]
        done = term or trunc
        t += 1
    out = {k: np.array(v) for k, v in rec.items()}
    out["life_lost"] = out["lives"] < out["lives_before"]
    return out


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- P1
def p1_preprocessing(out):
    import matplotlib.pyplot as plt
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=3)
    st = FrameStack()
    st.reset(frame(obs))
    counts, sums_present, raw_keep = [], set(), None
    for t in range(600):
        obs, r, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
        st.push(frame(obs))
        crop = obs[1:176:2, ::2].astype(np.int64).sum(2)
        counts.append(int((crop == mspacman_color).sum()))
        full_mask = (obs[..., 0] == PAC_RGB[0]) & (obs[..., 1] == PAC_RGB[1]) & (obs[..., 2] == PAC_RGB[2])
        # pixels whose colour sum is 448 but whose colour is NOT Ms. Pac-Man's would be removed by mistake
        other = (obs.astype(np.int64).sum(2) == mspacman_color) & ~full_mask
        sums_present.add(int(other.sum()))
        if t == 150:
            raw_keep = (obs.copy(), preprocess_observation(obs.copy())[:, :, 0].copy(), st.get().copy())
        if term or trunc:
            break
    env.close()
    raw, pre, stack = raw_keep
    viz.setup()
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1, 1]})
    axes[0].imshow(raw)
    axes[0].set_title("raw frame 210 x 160 x 3", fontsize=9)
    axes[1].imshow(pre, cmap="gray", vmin=-128, vmax=127)
    axes[1].set_title("preprocessed 88 x 80 (int8)", fontsize=9)
    for k in range(4):
        axes[2 + k].imshow(stack[k], cmap="gray", vmin=-128, vmax=127)
        axes[2 + k].set_title(f"state frame t-{3 - k}" if k < 3 else "state frame t", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Preprocessing: crop rows 1-175, keep every 2nd row and column, sum the colour channels, "
                 "Ms. Pac-Man (colour sum 448) set to the darkest value", fontsize=10)
    fig.savefig(os.path.join(out, "fig_val_preprocess.png"))
    plt.close(fig)
    c = np.array(counts)
    return {"frames": len(c), "pixels_with_sum_448_in_crop": {"min": int(c.min()), "max": int(c.max()),
            "mean": float(c.mean()), "frames_with_none": int((c == 0).sum())},
            "pixels_with_sum_448_that_are_not_mspacman_colour_max": max(sums_present),
            "value_range_preprocessed": [int(pre.min()), int(pre.max())]}


# ----------------------------------------------------------------------------- P2 / P3
def evaluate_games(net, games, seed, eps=0.0, sticky=0.25):
    env = gym.make(ENV_ID, repeat_action_probability=sticky)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    scores, lengths = [], []
    for _ in range(games):
        rec = play_record(net, env, eps, rng)
        scores.append(rec["r"].sum())
        lengths.append(len(rec["r"]))
    env.close()
    return np.array(scores), np.array(lengths)


def p3_snapshots(run, out, games=20):
    import matplotlib.pyplot as plt
    snaps = sorted(glob.glob(os.path.join(run, "snapshots", "ep*.pt")))
    rows = []
    for p in snaps:
        ep = int(os.path.basename(p)[2:7])
        sc, ln = evaluate_games(load_net(p), games, seed=7_000 + ep)
        rows.append({"episode": ep, "mean": float(sc.mean()), "std": float(sc.std()), "se": float(sc.std() / np.sqrt(games)),
                     "length": float(ln.mean())})
        print(f"  snapshot ep {ep}: {sc.mean():.1f} +- {sc.std() / np.sqrt(games):.1f}", flush=True)
    fw = os.path.join(run, "final_weights.pt")
    if os.path.exists(fw):
        ep_final = int(np.loadtxt(os.path.join(run, "episodes.csv"), delimiter=",", skiprows=1, usecols=0)[-1])
        if not rows or ep_final != rows[-1]["episode"]:
            sc, ln = evaluate_games(load_net(fw), games, seed=7_000 + ep_final)
            rows.append({"episode": ep_final, "mean": float(sc.mean()), "std": float(sc.std()),
                         "se": float(sc.std() / np.sqrt(games)), "length": float(ln.mean())})
    viz.setup()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = [r["episode"] for r in rows]
    ax.errorbar(x, [r["mean"] for r in rows], yerr=[r["se"] for r in rows], color=viz.BLUE, marker="o", capsize=3,
                lw=2, label=f"greedy policy of the snapshot, {games} games (+- s.e.)")
    ax.set_xlabel("training episode of the snapshot")
    ax.set_ylabel("game score")
    ax.set_title("MsPacman-v5: score of saved networks along training")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(out, "fig_val_snapshots.png"))
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------------- P4 / P5 / P9 from recorded games
def recorded_games(net, n, seed):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    games = [play_record(net, env, 0.0, rng) for _ in range(n)]
    env.close()
    return games


def p4_calibration(games, gamma, out):
    import matplotlib.pyplot as plt
    pred, true, true_raw = [], [], []
    for g in games:
        rc = np.clip(g["r"], -1, 1)
        G = discounted_returns(rc, gamma, cut=g["life_lost"])
        pred.append(g["q"][np.arange(len(g["a"])), g["a"]])
        true.append(G)
    pred, true = np.concatenate(pred), np.concatenate(true)
    bins = np.quantile(pred, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(pred, bins) - 1, 0, 19)
    bp = np.array([pred[idx == k].mean() for k in range(20)])
    bt = np.array([true[idx == k].mean() for k in range(20)])
    viz.setup()
    fig, ax = plt.subplots(figsize=(6, 5))
    sub = np.random.default_rng(0).choice(len(pred), size=min(4000, len(pred)), replace=False)
    ax.scatter(pred[sub], true[sub], s=3, alpha=0.15, color=viz.BLUE, label="individual steps (4000 shown)")
    ax.plot(bp, bt, "o-", color=viz.ORANGE, lw=2, label="mean per predicted-value bin")
    lim = [min(bp.min(), bt.min(), 0), max(pred.max(), true.max())]
    ax.plot(lim, lim, "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted Q(s_t, a_t)")
    ax.set_ylabel("realised discounted clipped return")
    ax.set_title("MsPacman-v5: predicted vs realised value")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(out, "fig_val_calibration.png"))
    plt.close(fig)
    return {"steps": int(len(pred)), "mean_pred": float(pred.mean()), "mean_true": float(true.mean()),
            "mean_bias": float((pred - true).mean()), "corr": float(np.corrcoef(pred, true)[0, 1]),
            "bins_pred": bp.round(3).tolist(), "bins_true": bt.round(3).tolist()}


def event_steps(g):
    r = g["r"]
    ev = {"life lost": np.flatnonzero(g["life_lost"]),
          "power pellet eaten": np.flatnonzero(np.isin(r, [50, 60])),
          "ghost eaten": np.flatnonzero(np.isin(r % 1000, [200, 210, 400, 410, 800, 810]) | np.isin(r, [1600, 1610]))}
    return ev


def p5_events(games, out, before=60, after=30):
    import matplotlib.pyplot as plt
    traces = {k: [] for k in ["life lost", "power pellet eaten", "ghost eaten"]}
    for g in games:
        vmax = g["q"].max(1)
        for k, steps in event_steps(g).items():
            for t in steps:
                if t - before >= 0 and t + after < len(vmax):
                    traces[k].append(vmax[t - before:t + after + 1])
    viz.setup()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    lags = np.arange(-before, after + 1)
    summary = {}
    for k, (name, tr) in enumerate(traces.items()):
        if not tr:
            continue
        tr = np.array(tr)
        m, se = tr.mean(0), tr.std(0) / np.sqrt(len(tr))
        ax.plot(lags, m, color=viz.SERIES[k], lw=2, label=f"{name} (n = {len(tr)})")
        ax.fill_between(lags, m - se, m + se, color=viz.SERIES[k], alpha=0.2, lw=0)
        summary[name] = {"n": int(len(tr)), "q_at_-60": float(m[0]), "q_at_-10": float(m[before - 10]),
                         "q_at_0": float(m[before]), "q_at_+10": float(m[before + 10])}
    ax.axvline(0, color=viz.TEXT2, lw=1, ls="--")
    ax.set_xlabel("agent steps relative to the event")
    ax.set_ylabel("max_a Q(s, a), mean over events (+- s.e.)")
    ax.set_title("MsPacman-v5: how the agent's value estimate moves around game events")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(os.path.join(out, "fig_val_events.png"))
    plt.close(fig)
    return summary


def p9_composition(games):
    allr = np.concatenate([g["r"] for g in games])
    vals, cnt = np.unique(allr[allr > 0], return_counts=True)
    total = allr.sum()
    big = allr[allr >= 100].sum()
    return {"games": len(games), "mean_score": float(total / len(games)),
            "reward_values": {str(int(v)): int(c) for v, c in zip(vals, cnt)},
            "share_of_score_from_rewards_>=100": float(big / total) if total else 0.0,
            "ghosts_per_game": float(sum(len(event_steps(g)["ghost eaten"]) for g in games) / len(games)),
            "power_pellets_per_game": float(sum(len(event_steps(g)["power pellet eaten"]) for g in games) / len(games))}


# ----------------------------------------------------------------------------- P6
def p6_trace(net, steps=10):
    env = gym.make(ENV_ID)
    env.reset(seed=11)
    g = play_record(net, env, 0.0, np.random.default_rng(11), max_steps=steps + 1)
    env.close()
    rows = []
    for t in range(steps):
        q = g["q"][t]
        nxt = g["q"][t + 1].max() if t + 1 < len(g["q"]) else float("nan")
        y = min(g["r"][t], 1.0) + 0.99 * nxt
        rows.append([t, int(g["lives_before"][t]), f"({g['pos'][t][0]:.0f}, {g['pos'][t][1]:.0f})", f"{q.max():.3f}",
                     ACTIONS[int(q.argmax())], f"{q.max() - q.min():.4f}", ACTIONS[g["a"][t]], int(g["r"][t]),
                     f"{y:.3f}"])
    return rows


# ----------------------------------------------------------------------------- P7
def p7_determinism(net, games=5, seed=21):
    """Greedy games with a different reset seed each time. Without sticky actions the emulator and the
    greedy policy are both deterministic, so every game must be identical whatever the seed; with sticky
    actions (probability 0.25, drawn from the seeded generator) the games must differ."""
    out = {}
    for sticky in [0.0, 0.25]:
        sc, ln = [], []
        for k in range(games):
            env = gym.make(ENV_ID, repeat_action_probability=sticky)
            env.reset(seed=seed + k)
            g = play_record(net, env, 0.0, np.random.default_rng(seed + k))
            sc.append(float(g["r"].sum()))
            ln.append(int(len(g["r"])))
            env.close()
        out[f"sticky_{sticky}"] = {"scores": sc, "lengths": ln}
    return out


def p7_frozen_start(net, steps=120):
    """At each of the first `steps` steps of a game (sticky actions off), clone the emulator, apply each of
    the 9 actions for one agent step (4 frames) and compare the resulting screens and RAM. A step is frozen
    when every action leads to the same emulator state."""
    env = gym.make(ENV_ID, repeat_action_probability=0.0)
    obs, _ = env.reset(seed=5)
    ale = env.unwrapped.ale
    aset = env.unwrapped._action_set
    st = FrameStack()
    st.reset(frame(obs))
    frozen, gaps, vmax = [], [], []
    for t in range(steps):
        s0 = ale.cloneState()
        outs = []
        for a in range(N_ACTIONS):
            ale.restoreState(s0)
            for _ in range(4):
                ale.act(aset[a])
            outs.append(np.concatenate([ale.getRAM().astype(np.int64), ale.getScreenRGB().ravel().astype(np.int64)]))
        ale.restoreState(s0)
        same = all(np.array_equal(outs[0], o) for o in outs[1:])
        with torch.no_grad():
            q = net(torch.from_numpy(st.get())[None])[0].numpy()
        frozen.append(same)
        gaps.append(float(q.max() - q.min()))
        vmax.append(float(q.max()))
        obs, r, term, trunc, _ = env.step(int(q.argmax()))
        st.push(frame(obs))
    env.close()
    frozen, gaps, vmax = np.array(frozen), np.array(gaps), np.array(vmax)
    first_live = int(np.argmin(frozen)) if (~frozen).any() else steps
    # nothing can happen during the frozen intro, so each step only brings the first reward one step closer:
    # max_a Q should grow by a factor 1/gamma per step, i.e. log(max Q) should rise with slope -log(0.99) = 0.01005
    k = first_live
    growth = None
    if k >= 10 and np.all(vmax[:k] > 0):
        growth = float(np.polyfit(np.arange(k), np.log(vmax[:k]), 1)[0])
    return {"steps_checked": steps, "frozen_steps": int(frozen.sum()), "first_step_where_actions_matter": first_live,
            "action_gap_frozen_mean": float(gaps[frozen].mean()) if frozen.any() else None,
            "action_gap_after_mean": float(gaps[~frozen].mean()) if (~frozen).any() else None,
            "log_maxq_growth_per_frozen_step": growth, "predicted_growth": float(-np.log(0.99)),
            "maxq_first_and_last_frozen_step": [float(vmax[0]), float(vmax[k - 1])] if k > 0 else None}


# ----------------------------------------------------------------------------- P8
def p8_positions(run, out, early_ep=250, games=10):
    import matplotlib.pyplot as plt
    paths = [(f"episode {early_ep}", os.path.join(run, "snapshots", f"ep{early_ep:05d}.pt")),
             ("final network", os.path.join(run, "final_weights.pt"))]
    env = gym.make(ENV_ID)
    bg, _ = env.reset(seed=0)
    env.close()
    viz.setup()
    fig, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 5.6))
    res = {}
    for ax, (name, p) in zip(np.atleast_1d(axes), paths):
        if not os.path.exists(p):
            continue
        env = gym.make(ENV_ID)
        env.reset(seed=33)
        rng = np.random.default_rng(33)
        pos = []
        for _ in range(games):
            g = play_record(load_net(p), env, 0.0, rng)
            pos.append(g["pos"])
        env.close()
        pos = np.concatenate(pos)
        pos = pos[~np.isnan(pos[:, 0])]
        H, ye, xe = np.histogram2d(pos[:, 0], pos[:, 1], bins=[np.arange(0, 176, 4), np.arange(0, 161, 4)])
        ax.imshow(bg[:176].mean(2), cmap="gray", alpha=0.35, extent=[0, 160, 176, 0])
        im = ax.imshow(np.log1p(H), cmap="magma", alpha=0.75, extent=[0, 160, 176, 0])
        ax.set_title(f"{name}: {games} greedy games", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        cover = float((H > 0).mean())
        res[name] = {"positions": int(len(pos)), "fraction_of_4x4_cells_visited": cover}
    fig.suptitle("Where Ms. Pac-Man goes (log visit counts per 4 x 4-pixel cell over the maze)", fontsize=11)
    fig.savefig(os.path.join(out, "fig_val_positions.png"))
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/mspacman/main_seed0")
    ap.add_argument("--out", default="results/mspacman/validation")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    jpath = os.path.join(args.out, "validation.json")
    res = json.load(open(jpath)) if os.path.exists(jpath) else {}
    todo = set(args.only) if args.only else {"p1", "p2", "p3", "p4", "p6", "p7", "p8"}
    net = load_net(os.path.join(args.run, "final_weights.pt")) if os.path.exists(
        os.path.join(args.run, "final_weights.pt")) else None
    t0 = time.time()
    if "p1" in todo:
        res["P1_preprocessing"] = p1_preprocessing(args.out)
        print("P1", res["P1_preprocessing"], flush=True)
    if "p2" in todo:
        sc, ln = evaluate_games(None, 100, seed=99)
        res["P2_random"] = {"games": 100, "mean": float(sc.mean()), "std": float(sc.std()), "min": float(sc.min()),
                            "max": float(sc.max()), "length_mean": float(ln.mean())}
        print("P2", res["P2_random"], flush=True)
    if "p3" in todo:
        res["P3_snapshots"] = p3_snapshots(args.run, args.out, args.games)
    if "p4" in todo and net is not None:
        games = recorded_games(net, args.games, seed=4242)
        res["P4_calibration"] = p4_calibration(games, 0.99, args.out)
        res["P5_events"] = p5_events(games, args.out)
        res["P9_composition_final"] = p9_composition(games)
        env = gym.make(ENV_ID)
        env.reset(seed=4243)
        rnd = [play_record(None, env, 1.0, np.random.default_rng(4243 + k)) for k in range(args.games)]
        env.close()
        res["P9_composition_random"] = p9_composition(rnd)
        print("P4", json.dumps(res["P4_calibration"])[:300], flush=True)
        print("P5", res["P5_events"], flush=True)
    if "p6" in todo and net is not None:
        res["P6_trace"] = p6_trace(net)
    if "p7" in todo and net is not None:
        res["P7_determinism"] = p7_determinism(net)
        res["P7_frozen_start"] = p7_frozen_start(net)
        print("P7", res["P7_determinism"], res["P7_frozen_start"], flush=True)
    if "p8" in todo and net is not None:
        res["P8_positions"] = p8_positions(args.run, args.out)
    res["run"] = args.run
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2)
    lines = [f"# MsPacman validation results ({args.run})", "", f"Wall time of this invocation: {time.time() - t0:.0f} s", ""]
    for k, v in res.items():
        if k == "P6_trace":
            lines += ["## P6 first 10 steps", "", md_table(["t", "lives", "Ms. Pac-Man (row, col)", "max Q", "argmax",
                                                              "max Q - min Q", "action", "reward", "y = clip(r) + 0.99 max Q(s')"], v), ""]
        else:
            lines += [f"## {k}", "", "```", json.dumps(v, indent=1), "```", ""]
    open(os.path.join(args.out, "validation.md"), "w").write("\n".join(lines))
    print("wrote", os.path.join(args.out, "validation.md"), f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
