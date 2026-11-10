"""Validation experiments for DQN on CartPole-v1 (writes validation.md / validation.json + figures).

    python -m dqn.validate_cartpole --results results/cartpole --out results/cartpole/validation

C1  chain MDP with known Q*: the same DQN code path must recover Q* (goal chain) and the
    bootstrapped value 1/(1-gamma) under a time limit (survive chain); treating the time limit as
    terminal must give the predicted biased value 1/(1-gamma(1-1/T)).
C2  scale of Q: on CartPole, with a policy that never fails, Q* = 1/(1-gamma) exactly; checked for
    gamma 0.9 / 0.95 / 0.99 and for rewards scaled by 2 and by -1.
C3  true return vs predicted Q (overestimation), with roll-outs continued to 2000 steps so the
    500-step truncation does not hide failures.
C4  first 10 steps of a greedy episode: state, Q-values, action, reward and Bellman target.
C5  mirror symmetry of the learned Q (the dynamics are symmetric under s -> -s, left <-> right).
C6  value and policy maps over (theta, theta_dot) during training and over (x, x_dot) at the end.
C7  component ablations and the truncation edge case, summarised from the runs in --results.
C8  why the training reward stays low (run with --only exploration_noise): the kept networks rolled out
    greedily and with epsilon = 0.1, and the greedy checks made while epsilon was at its final value.
"""
import argparse
import glob
import json
import os

import gymnasium as gym
import numpy as np
import torch

from .cartpole_dqn import ENV_ID, evaluate, failure_reason, load_net
from .common import read_csv
from .toy_chain import exact_q, run_dqn

import viz  # noqa: E402  (repository root on sys.path when run with -m)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- C1 chain MDP
def chain_tests(out, seeds=(0, 1, 2)):
    res = {}
    n, gamma = 8, 0.95
    qstar = exact_q(n, gamma, "goal")
    goal = {}
    for name, kw in [("double_dqn", dict(double=True)), ("dqn", dict(double=False)),
                     ("no_target", dict(use_target=False))]:
        errs, pol_ok, qs = [], [], []
        for s in seeds:
            q = run_dqn("goal", n, gamma, steps=20_000, seed=s, **kw)
            errs.append(float(np.abs(q[:n - 1] - qstar[:n - 1]).max()))
            pol_ok.append(bool(np.all(q[:n - 1].argmax(1) == 1)))
            qs.append(q)
        goal[name] = {"max_abs_error_per_seed": errs, "greedy_policy_optimal_per_seed": pol_ok,
                      "q_seed0": qs[0].tolist()}
    res["goal"] = {"n": n, "gamma": gamma, "q_star": qstar.tolist(), **goal}
    surv = {}
    for name, tt, loss in [("bootstrap_on_truncation, Huber", False, "huber"),
                           ("truncation_as_terminal, Huber", True, "huber"),
                           ("bootstrap_on_truncation, squared", False, "mse"),
                           ("truncation_as_terminal, squared", True, "mse")]:
        means = []
        for s in seeds:
            q = run_dqn("survive", n, gamma, steps=30_000, seed=s, truncation_terminal=tt, time_limit=50, loss=loss)
            means.append(float(q.mean()))
        surv[name] = {"mean_q_per_seed": means,
                      "predicted": float(exact_q(n, gamma, "survive", 50, tt, loss=loss)[0, 0])}
    res["survive"] = {"time_limit": 50, **surv}
    # figure: learned vs exact Q on the goal chain
    viz.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = np.arange(n - 1)
    ax.plot(xs, qstar[:n - 1, 1], "k-", lw=2, label="exact Q*(s, right)")
    ax.plot(xs, qstar[:n - 1, 0], "k--", lw=2, label="exact Q*(s, left)")
    q0 = np.array(goal["double_dqn"]["q_seed0"])
    ax.plot(xs, q0[:n - 1, 1], "o", color=viz.BLUE, ms=8, label="Double DQN, right")
    ax.plot(xs, q0[:n - 1, 0], "s", color=viz.ORANGE, ms=7, label="Double DQN, left")
    ax.set_xlabel("state s (goal is state 7)")
    ax.set_ylabel("action value")
    ax.set_title("Chain MDP: learned vs exact action values (gamma = 0.95)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(out, "fig_val_chain.png"))
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- roll-out helpers
def rollout(net, seed, max_steps=500, eps=0.0):
    env = gym.make(ENV_ID, max_episode_steps=max_steps)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    S, Q, A, R = [], [], [], []
    done = False
    while not done:
        with torch.no_grad():
            q = net(torch.as_tensor(obs)[None])[0].numpy()
        a = int(rng.integers(2)) if rng.random() < eps else int(q.argmax())
        S.append(obs.copy())
        Q.append(q)
        A.append(a)
        obs, r, term, trunc, _ = env.step(a)
        R.append(r)
        done = term or trunc
    env.close()
    return np.array(S), np.array(Q), np.array(A), np.array(R), failure_reason(obs, term, trunc)


def discounted(R, gamma):
    G = np.zeros(len(R))
    run = 0.0
    for t in range(len(R) - 1, -1, -1):
        run = R[t] + gamma * run
        G[t] = run
    return G


# ----------------------------------------------------------------------------- C3 overestimation
def overestimation(net, gamma, seeds=range(20), horizon=2000):
    """Predicted Q(s_t, a_t) for t < 500 against the realised discounted return of the same greedy
    policy, with the episode continued to `horizon` steps (gamma^1500 ~ 1e-34, so the tail is exact)."""
    pred, true, lengths, ends = [], [], [], []
    for s in seeds:
        S, Q, A, R, end = rollout(net, 50_000 + s, max_steps=horizon)
        G = discounted(R, gamma)
        k = min(500, len(R))
        pred.append(Q[np.arange(k), A[:k]])
        true.append(G[:k])
        lengths.append(len(R))
        ends.append(end)
    pred, true = np.concatenate(pred), np.concatenate(true)
    return {"episodes": len(lengths), "lengths_to_2000": lengths, "ends": ends,
            "mean_pred": float(pred.mean()), "mean_true": float(true.mean()),
            "mean_bias": float((pred - true).mean()), "max_abs_bias": float(np.abs(pred - true).max()),
            "bound": 1.0 / (1.0 - gamma)}


# ----------------------------------------------------------------------------- C4 trace
def trace(net, gamma, seed=7, steps=10):
    S, Q, A, R, _ = rollout(net, seed)
    rows = []
    for t in range(steps):
        y = R[t] + gamma * Q[t + 1].max()
        rows.append([t, f"{S[t][0]:+.3f}", f"{S[t][1]:+.3f}", f"{S[t][2]:+.3f}", f"{S[t][3]:+.3f}",
                     f"{Q[t][0]:.3f}", f"{Q[t][1]:.3f}", ["left", "right"][A[t]], int(R[t]), f"{y:.3f}",
                     f"{y - Q[t][A[t]]:+.4f}"])
    return rows


# ----------------------------------------------------------------------------- C5 mirror
def mirror(net, seeds=range(10)):
    S = np.concatenate([rollout(net, 60_000 + s)[0] for s in seeds])
    with torch.no_grad():
        q = net(torch.as_tensor(S)).numpy()
        qm = net(torch.as_tensor(-S)).numpy()
    d = np.abs(q[:, 0] - qm[:, 1])  # Q(s, left) vs Q(-s, right)
    adv = q[:, 1] - q[:, 0]
    adv_m = qm[:, 0] - qm[:, 1]
    by_gap = {}
    for th in [0.1, 0.5, 1.0]:
        m = np.abs(adv) > th
        by_gap[f"|gap|>{th}"] = {"share_of_states": float(m.mean()),
                                 "greedy_action_agreement": float(np.mean((adv[m] > 0) == (adv_m[m] > 0)))}
    return {"states": int(len(S)), "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
            "corr_action_gap": float(np.corrcoef(adv, adv_m)[0, 1]),
            "greedy_action_agreement": float(np.mean((adv > 0) == (adv_m > 0))),
            "median_abs_action_gap": float(np.median(np.abs(adv))), "agreement_by_gap": by_gap}


# ----------------------------------------------------------------------------- C6 maps
def value_maps(run_dir, out, gamma, tag):
    import matplotlib.pyplot as plt
    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "ep*.pt")))
    want = [0, 100, 300, 600, 1000]
    pick = []
    for w in want:
        cand = [p for p in snaps if int(os.path.basename(p)[2:6]) == w]
        if cand:
            pick.append((w, cand[0]))
    sel = json.load(open(os.path.join(run_dir, "best_selection.json")))
    pick = [(w, p) for w, p in pick if w <= sel["episode"]]
    pick.append((f"selected (episode {sel['episode']})", os.path.join(run_dir, "best_weights.pt")))
    pick.append(("final", os.path.join(run_dir, "final_weights.pt")))
    th = np.linspace(-0.21, 0.21, 121)
    thd = np.linspace(-2.0, 2.0, 121)
    T, TD = np.meshgrid(th, thd)
    grid = np.stack([np.zeros_like(T), np.zeros_like(T), T, TD], -1).reshape(-1, 4).astype(np.float32)
    viz.setup()
    fig, axes = plt.subplots(2, len(pick), figsize=(3.1 * len(pick), 6.2), sharex=True, sharey=True)
    for k, (w, path) in enumerate(pick):
        net = load_net(path)
        with torch.no_grad():
            q = net(torch.as_tensor(grid)).numpy()
        V = q.max(1).reshape(T.shape)
        gap = (q[:, 1] - q[:, 0]).reshape(T.shape)
        im = axes[0, k].pcolormesh(T, TD, V, cmap="viridis", vmin=0, vmax=1 / (1 - gamma) * 1.05, shading="auto")
        axes[0, k].set_title(f"episode {w}" if isinstance(w, int) else
                             ("final network (episode 1000)" if w == "final" else w), fontsize=9)
        axes[1, k].pcolormesh(T, TD, np.sign(gap), cmap="coolwarm", vmin=-1.5, vmax=1.5, shading="auto")
        axes[1, k].contour(T, TD, gap, levels=[0], colors="k", linewidths=0.8)
        axes[1, k].set_xlabel("pole angle theta (rad)")
    axes[0, 0].set_ylabel("theta_dot (rad/s)\nmax_a Q(s, a)")
    axes[1, 0].set_ylabel("theta_dot (rad/s)\ngreedy action")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.85, label="max_a Q (x = x_dot = 0)")
    fig.suptitle("Learned value (top) and greedy action (bottom; red = push right, blue = push left) "
                 "over pole angle and angular velocity", fontsize=11)
    fig.savefig(os.path.join(out, f"fig_val_policy_maps_{tag}.png"))
    plt.close(fig)
    # cart position map of the selected network
    x = np.linspace(-2.4, 2.4, 121)
    xd = np.linspace(-2.0, 2.0, 121)
    X, XD = np.meshgrid(x, xd)
    g2 = np.stack([X, XD, np.zeros_like(X), np.zeros_like(X)], -1).reshape(-1, 4).astype(np.float32)
    net = load_net(os.path.join(run_dir, "best_weights.pt"))
    with torch.no_grad():
        V2 = net(torch.as_tensor(g2)).numpy().max(1).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.pcolormesh(X, XD, V2, cmap="viridis", shading="auto")
    ax.contour(X, XD, V2, levels=[15, 18, 19, 19.5], colors="w", linewidths=0.7)
    ax.set_xlabel("cart position x (track edges at +-2.4)")
    ax.set_ylabel("cart velocity x_dot")
    ax.set_title("Selected network: max_a Q over cart position and velocity (pole upright)")
    fig.colorbar(im, ax=ax, label="max_a Q")
    fig.savefig(os.path.join(out, f"fig_val_position_map_{tag}.png"))
    plt.close(fig)
    return {"position_map_min": float(V2.min()), "position_map_center": float(V2[60, 60])}


# ----------------------------------------------------------------------------- C2 / C7 summaries
def group_summary(results, prefix):
    rows = []
    for d in sorted(glob.glob(os.path.join(results, prefix + "_seed*"))):
        p = os.path.join(d, "eval_summary_greedy.json")
        if not os.path.exists(p):
            continue
        ev = json.load(open(p))
        ep = read_csv(os.path.join(d, "episodes.csv"))
        cfg = json.load(open(os.path.join(d, "config.json")))
        last = slice(-100, None)
        sel = os.path.join(d, "eval_summary_greedy_best.json")
        evb = json.load(open(sel)) if os.path.exists(sel) else None
        rows.append({"run": os.path.basename(d), "gamma": cfg["gamma"], "eval_mean": ev["mean"], "eval_std": ev["std"],
                     "solved_fraction": ev["solved_fraction"], "ends": ev["ends"],
                     "best_eval_mean": evb["mean"] if evb else None,
                     "best_solved_fraction": evb["solved_fraction"] if evb else None,
                     "final_ma100": float(np.mean(ep["reward"][last])),
                     "qmax_last100": float(np.mean(ep["qmax_mean"][last])),
                     "heldout_last100": float(np.mean(ep["heldout_qmax"][last])),
                     "episodes_to_ma475": next((int(i + 1) for i, m in enumerate(
                         np.convolve(ep["reward"], np.ones(100) / 100, mode="full")[:len(ep["reward"])])
                         if i >= 99 and m >= 475), None)})
    return rows


# ----------------------------------------------------------------------------- C8 exploration noise
def exploration_noise(results, prefix="main", episodes=200):
    """Two possible reasons for a low epsilon-greedy training reward: the random actions themselves, and a
    greedy policy that is poor at that moment. (1) Each kept network plays `episodes` episodes greedily and
    with epsilon = 0.1 (same seeds). (2) The greedy checks logged while epsilon was at its final value."""
    out = {}
    for d in sorted(glob.glob(os.path.join(results, prefix + "_seed*"))):
        net = load_net(os.path.join(d, "best_weights.pt"))
        seed = int(d[-1])
        row = {}
        for eps in (0.0, 0.1):
            _, sm = evaluate(net, episodes, seed=7000 + seed, eps=eps)
            row[f"kept_network_eps_{eps}"] = {"mean": sm["mean"], "share_at_500": sm["solved_fraction"], "ends": sm["ends"]}
        ep = read_csv(os.path.join(d, "episodes.csv"))
        cfg = json.load(open(os.path.join(d, "config.json")))
        at_final = ep["epsilon"] <= cfg["eps_final"] + 1e-4
        checks = ep["eval_greedy_mean"][at_final & ~np.isnan(ep["eval_greedy_mean"])]
        row["greedy_checks_at_final_epsilon"] = {"n": int(len(checks)), "median": float(np.median(checks)),
                                                 "min": float(checks.min()), "max": float(checks.max()),
                                                 "number_at_500": int((checks >= 500).sum())}
        row["training_reward_mean_at_final_epsilon"] = float(ep["reward"][at_final].mean())
        out[os.path.basename(d)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cartpole")
    ap.add_argument("--out", default="results/cartpole/validation")
    ap.add_argument("--main-prefix", default="main")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    jpath = os.path.join(args.out, "validation.json")
    res = json.load(open(jpath)) if os.path.exists(jpath) else {}
    todo = set(args.only) if args.only else {"chain", "network", "groups"}
    if "overest_groups" in todo:
        # C3 for every kept network of the Double DQN and the DQN-target runs: overestimation per variant
        out = {}
        for pfx in ["main", "dqn"]:
            for d in sorted(glob.glob(os.path.join(args.results, pfx + "_seed*"))):
                w = os.path.join(d, "best_weights.pt")
                if os.path.exists(w):
                    o = overestimation(load_net(w), json.load(open(os.path.join(d, "config.json")))["gamma"], seeds=range(10))
                    out[os.path.basename(d)] = {k: o[k] for k in ["mean_pred", "mean_true", "mean_bias", "max_abs_bias"]}
                    out[os.path.basename(d)]["survived_2000"] = int(sum(l == 2000 for l in o["lengths_to_2000"]))
        res["C3_overestimation_by_variant"] = out
        print("C3 by variant", json.dumps(out), flush=True)
    if "exploration_noise" in todo:
        res["C8_exploration_noise"] = exploration_noise(args.results, args.main_prefix)
        print("C8", json.dumps(res["C8_exploration_noise"]), flush=True)
    if "chain_noclip" in todo:
        # the squared-loss variant without gradient-norm clipping: clipping scales down exactly the minibatches
        # that contain the large errors of the wrong targets, which also down-weights them
        means = [float(run_dqn("survive", 8, 0.95, steps=30_000, seed=s, truncation_terminal=True, time_limit=50,
                               loss="mse", grad_clip=0.0).mean()) for s in (0, 1, 2)]
        res.setdefault("C1_chain", {}).setdefault("survive", {})["truncation_as_terminal, squared, no gradient clipping"] = {
            "mean_q_per_seed": means, "predicted": float(exact_q(8, 0.95, "survive", 50, True, loss="mse")[0, 0])}
        print("C1 no clipping", means, flush=True)
    if "chain" in todo:
        res["C1_chain"] = chain_tests(args.out)
        print("C1", json.dumps(res["C1_chain"]["survive"]), flush=True)
    groups = {}
    if "groups" in todo or "network" in todo:
        for pfx in [args.main_prefix, "dqn", "notarget", "noreplay", "truncterm", "gamma0.9", "gamma0.99",
                    "rscale2", "rscale-1", "lr0"]:
            g = group_summary(args.results, pfx)
            if g:
                groups[pfx] = g
        res["C7_groups"] = groups
    if "network" in todo and groups.get(args.main_prefix):
        main_runs = groups[args.main_prefix]
        best = max(main_runs, key=lambda r: (r["best_solved_fraction"] or 0, r["best_eval_mean"] or 0, -int(r["run"][-1])))
        run_dir = os.path.join(args.results, best["run"])
        net = load_net(os.path.join(run_dir, "best_weights.pt"))
        gamma = best["gamma"]
        res["network_run"] = best["run"]
        res["C3_overestimation"] = overestimation(net, gamma)
        res["C4_trace"] = trace(net, gamma)
        res["C5_mirror"] = mirror(net)
        res["C6_maps"] = value_maps(run_dir, args.out, gamma, "main")
        print("C3", json.dumps(res["C3_overestimation"])[:400], flush=True)
        print("C5", json.dumps(res["C5_mirror"]), flush=True)
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2)
    # markdown digest
    lines = ["# CartPole validation results", ""]
    if "C1_chain" in res:
        c = res["C1_chain"]
        lines += ["## C1 chain MDP", "", md_table(["variant", "max |Q - Q*| per seed", "greedy policy optimal"], [
            [k, ", ".join(f"{e:.4f}" for e in c["goal"][k]["max_abs_error_per_seed"]),
             ", ".join(str(b) for b in c["goal"][k]["greedy_policy_optimal_per_seed"])]
            for k in ["double_dqn", "dqn", "no_target"]]), "",
            md_table(["survive chain (T = 50)", "mean Q per seed", "predicted"], [
                [k, ", ".join(f"{m:.3f}" for m in c["survive"][k]["mean_q_per_seed"]), f"{c['survive'][k]['predicted']:.3f}"]
                for k in c["survive"] if k != "time_limit"]), ""]
    if "C7_groups" in res:
        lines += ["## C2 / C7 run groups", ""]
        rows = []
        for g, runs in res["C7_groups"].items():
            for r in runs:
                rows.append([r["run"], r["gamma"], f"{r['eval_mean']:.1f}", f"{r['eval_std']:.1f}",
                             f"{r['solved_fraction']:.3f}", r["ends"], r["best_eval_mean"], f"{r['final_ma100']:.1f}",
                             f"{r['qmax_last100']:.2f}", f"{r['heldout_last100']:.2f}", r["episodes_to_ma475"]])
        lines += [md_table(["run", "gamma", "greedy mean", "std", "solved", "ends", "selected net mean",
                            "final MA100", "visited max Q", "held-out max Q", "episodes to MA100 >= 475"], rows), ""]
    if "C3_overestimation" in res:
        lines += ["## C3 overestimation", "", "```", json.dumps(res["C3_overestimation"], indent=1), "```", "",
                  "## C4 first 10 steps", "",
                  md_table(["t", "x", "x_dot", "theta", "theta_dot", "Q(left)", "Q(right)", "action", "r",
                            "y = r + g max Q(s')", "y - Q(s,a)"], res["C4_trace"]), "",
                  "## C5 mirror", "", "```", json.dumps(res["C5_mirror"], indent=1), "```", ""]
    if "C8_exploration_noise" in res:
        rows = [[k, f"{v['kept_network_eps_0.0']['mean']:.1f}", f"{v['kept_network_eps_0.1']['mean']:.1f}",
                 v["kept_network_eps_0.1"]["ends"], f"{v['greedy_checks_at_final_epsilon']['median']:.0f}",
                 f"{v['greedy_checks_at_final_epsilon']['min']:.0f} to {v['greedy_checks_at_final_epsilon']['max']:.0f}",
                 f"{v['training_reward_mean_at_final_epsilon']:.0f}"] for k, v in res["C8_exploration_noise"].items()]
        lines += ["## C8 exploration noise", "",
                  md_table(["run", "kept network, greedy (200 episodes)", "kept network, epsilon 0.1", "ends with epsilon 0.1",
                            "greedy checks at final epsilon: median", "range", "training reward at final epsilon"], rows), ""]
    open(os.path.join(args.out, "validation.md"), "w").write("\n".join(lines))
    print("wrote", os.path.join(args.out, "validation.md"))


if __name__ == "__main__":
    main()
