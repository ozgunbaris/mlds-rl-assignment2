"""Figures for the MsPacman-v5 runs.

    python -m dqn.make_plots_mspacman --runs results/mspacman/main_seed0 results/mspacman/main_seed1 \
        --out results/mspacman/figures

Per run: (i) <run>_maxq.png, (ii) <run>_training.png, (iii) <run>_hist.png (500 greedy roll-out games).
Comparisons: compare_seeds.png (moving average of both seeds), eval_periodic.png (greedy check games
every 100 training episodes), diagnostics.png (loss, TD error, episode length, epsilon), actions.png.
"""
import argparse
import json
import os

import numpy as np

import viz
from dqn.common import moving_average, read_csv

ACTIONS = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']


def maxq_plot(ep, held, path, title):
    import matplotlib.pyplot as plt
    viz.setup()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = ep["episode"]
    ax.plot(x, ep["qmax_mean"], color=viz.BLUE, lw=0.5, alpha=0.3, label="mean over the episode's states of max_a Q(s, a)")
    ax.plot(x, moving_average(ep["qmax_mean"], 100), color=viz.BLUE, lw=2, label="its moving average (100 episodes)")
    ax.plot(x, moving_average(ep["qmax_max"], 100), color=viz.AQUA, lw=1.2,
            label="largest max_a Q in the episode (moving average)")
    if held is not None:
        ax.plot(held["episode"], held["qmax_mean"], color=viz.ORANGE, lw=1.6, marker="o", ms=2,
                label="average over 1000 fixed held-out states")
    ax.set_xlabel("training episode")
    ax.set_ylabel("max_a Q(s, a)  (clipped-reward units)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["results/mspacman/main_seed0", "results/mspacman/main_seed1"])
    ap.add_argument("--out", default="results/mspacman/figures")
    ap.add_argument("--eval-tag", default="eps0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib.pyplot as plt
    data = {}
    for d in args.runs:
        name = os.path.basename(d.rstrip("/"))
        ep = read_csv(os.path.join(d, "episodes.csv"))
        hp = os.path.join(d, "heldout_q.csv")
        held = read_csv(hp) if os.path.exists(hp) else None
        evp = os.path.join(d, "eval_periodic.csv")
        ev = read_csv(evp) if os.path.exists(evp) else None
        data[name] = (ep, held, ev)
        viz.training_curve(ep["reward"], os.path.join(args.out, f"{name}_training.png"),
                           f"MsPacman-v5, Double DQN, {name}: score per training episode", ylabel="game score")
        maxq_plot(ep, held, os.path.join(args.out, f"{name}_maxq.png"),
                  f"MsPacman-v5, Double DQN, {name}: maximum Q-value per training episode")
        rp = os.path.join(d, f"eval_rewards_{args.eval_tag}.npy")
        if os.path.exists(rp):
            r = np.load(rp)
            viz.reward_histogram(r, os.path.join(args.out, f"{name}_hist.png"),
                                 f"MsPacman-v5, {name}: 500 roll-out games of the final network (greedy)",
                                 bins=30, xlabel="game score")
    if len(data) > 1:
        viz.compare_curves({k: v[0]["reward"] for k, v in data.items()}, os.path.join(args.out, "compare_seeds.png"),
                           "MsPacman-v5: 100-episode moving-average score of the two seeds", ylabel="game score (moving average)")
    # periodic greedy evaluation
    viz.setup()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (name, (ep, held, ev)) in enumerate(data.items()):
        if ev is None or "eval_mean" not in ev:
            continue
        ax.errorbar(ev["episode"], ev["eval_mean"], yerr=ev["eval_std"] / np.sqrt(ev["n"]), color=viz.SERIES[k],
                    lw=1.5, marker="o", ms=3, capsize=2, label=f"{name}: mean of 10 greedy games (+- s.e.)")
        ax.plot(ep["episode"], moving_average(ep["reward"], 100), color=viz.SERIES[k], lw=1, alpha=0.5, ls="--",
                label=f"{name}: training score, moving average (epsilon-greedy)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("game score")
    ax.set_title("MsPacman-v5: greedy evaluation games during training")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(args.out, "eval_periodic.png"))
    plt.close(fig)
    # diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for k, (name, (ep, held, ev)) in enumerate(data.items()):
        c = viz.SERIES[k]
        x = ep["episode"]
        axes[0, 0].plot(x, moving_average(np.nan_to_num(ep["loss_mean"]), 100), color=c, lw=1.5, label=name)
        axes[0, 1].plot(x, moving_average(np.nan_to_num(ep["td_abs_mean"]), 100), color=c, lw=1.5, label=name)
        axes[1, 0].plot(x, moving_average(ep["length"], 100), color=c, lw=1.5, label=name)
        axes[1, 1].plot(x, ep["epsilon"], color=c, lw=1.5, label=f"{name}: epsilon")
        ax2 = axes[1, 1]
        ax2.plot(x, ep["lr"] / ep["lr"][0], color=c, lw=1.0, ls="--", label=f"{name}: lr / initial lr")
    axes[0, 0].set_title("Huber loss (moving average)")
    axes[0, 1].set_title("|Q(s, a) - y| (moving average)")
    axes[1, 0].set_title("episode length in agent steps (moving average)")
    axes[1, 1].set_title("exploration rate and learning-rate schedule")
    for ax in axes.ravel():
        ax.set_xlabel("training episode")
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(args.out, "diagnostics.png"))
    plt.close(fig)
    # action distribution over training (first run)
    name, (ep, _, _) = next(iter(data.items()))
    A = np.stack([ep[f"a{i}"] for i in range(9)], 1)
    w = 100
    k = len(A) // w
    frac = np.array([A[i * w:(i + 1) * w].sum(0) / A[i * w:(i + 1) * w].sum() for i in range(k)])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.stackplot((np.arange(k) + 1) * w, frac.T, labels=ACTIONS, colors=viz.SERIES + ["#999999"], alpha=0.9)
    ax.set_xlabel(f"training episode (blocks of {w})")
    ax.set_ylabel("fraction of actions")
    ax.set_ylim(0, 1)
    ax.set_title(f"MsPacman-v5, {name}: actions chosen during training")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.savefig(os.path.join(args.out, "actions.png"))
    plt.close(fig)
    summ = {}
    for name, (ep, held, ev) in data.items():
        summ[name] = {"episodes": int(len(ep["episode"])), "steps": int(ep["step"][-1]),
                      "first100_mean": float(np.mean(ep["reward"][:100])), "last100_mean": float(np.mean(ep["reward"][-100:])),
                      "best_ma100": float(moving_average(ep["reward"], 100).max()),
                      "hours": float(ep["wall_time_s"][-1] / 3600)}
        rp = os.path.join(args.runs[list(data).index(name)], f"eval_summary_{args.eval_tag}.json")
        if os.path.exists(rp):
            summ[name]["rollout"] = json.load(open(rp))
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
