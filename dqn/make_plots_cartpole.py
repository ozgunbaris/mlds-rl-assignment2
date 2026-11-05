"""Figures and summary table for the CartPole-v1 runs.

    python -m dqn.make_plots_cartpole --results results/cartpole --out results/cartpole/figures

Required plots for every run of the main configuration (figures/per_run/):
  <run>_maxq.png      (i)   max_a Q(s, a) per training episode (mean over the states visited in the
                            episode, its 100-episode moving average, the average over a fixed set of
                            500 held-out states, and the bound 1 / (1 - gamma))
  <run>_training.png  (ii)  episode reward with the 100-episode moving average
  <run>_hist.png      (iii) histogram of the 500 greedy roll-out episodes with mean and std
Comparisons: seeds of the main configuration, ablations, discount factors, failure reasons.
"""
import argparse
import glob
import json
import os

import numpy as np

import viz
from dqn.common import moving_average, read_csv

GROUPS = [("main", "Double DQN (main)"), ("dqn", "DQN target (no Double)"), ("notarget", "no target network"),
          ("noreplay", "no replay (buffer = 64)"), ("truncterm", "time limit treated as terminal")]


def runs(results, prefix):
    out = []
    for d in sorted(glob.glob(os.path.join(results, f"{prefix}_seed*"))):
        if os.path.exists(os.path.join(d, "episodes.csv")):
            out.append(d)
    return out


def maxq_plot(ep, gamma, path, title):
    import matplotlib.pyplot as plt
    viz.setup()
    x = ep["episode"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, ep["qmax_mean"], color=viz.BLUE, lw=0.6, alpha=0.35, label="mean over the episode's states of max_a Q(s, a)")
    ax.plot(x, moving_average(ep["qmax_mean"], 100), color=viz.BLUE, lw=2, label="its moving average (100 episodes)")
    ax.plot(x, ep["heldout_qmax"], color=viz.ORANGE, lw=1.5, label="average over 500 fixed held-out states")
    ax.plot(x, ep["qmax_max"], color=viz.AQUA, lw=0.6, alpha=0.5, label="largest max_a Q in the episode")
    ax.axhline(1 / (1 - gamma), color=viz.TEXT2, ls="--", lw=1, label=f"1 / (1 - gamma) = {1 / (1 - gamma):.0f}")
    ax.set_xlabel("training episode")
    ax.set_ylabel("max_a Q(s, a)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def training_with_checks(ep, path, title):
    """Required plot (ii) plus the greedy evaluation checks made during training."""
    import matplotlib.pyplot as plt
    viz.setup()
    x = ep["episode"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, ep["reward"], color=viz.BLUE, lw=0.6, alpha=0.35, label="episode reward (epsilon-greedy, epsilon = 0.1 after step 21,000)")
    ax.plot(x, moving_average(ep["reward"], 100), color=viz.ORANGE, lw=2.0, label="moving average (100 episodes)")
    if "eval_greedy_mean" in ep:
        m = ~np.isnan(ep["eval_greedy_mean"])
        ax.plot(x[m], ep["eval_greedy_mean"][m], "o", color=viz.GREEN, ms=3.5,
                label="greedy check: mean of 20 separate evaluation episodes")
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode reward")
    ax.set_title(title)
    ax.set_ylim(0, 520)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def end_fractions(ep, path, title, window=100):
    import matplotlib.pyplot as plt
    viz.setup()
    ends = ep["end"]
    n = len(ends)
    k = n // window
    xs = (np.arange(k) + 1) * window
    fr = {e: [np.mean(ends[i * window:(i + 1) * window] == e) for i in range(k)] for e in ["angle", "position", "time limit"]}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.stackplot(xs, fr["time limit"], fr["position"], fr["angle"], colors=[viz.AQUA, viz.ORANGE, viz.BLUE],
                 labels=["reached the 500-step limit", "cart left the track (|x| > 2.4)", "pole fell (|theta| > 12 deg)"],
                 alpha=0.85)
    ax.set_xlabel(f"training episode (blocks of {window})")
    ax.set_ylabel("fraction of episodes")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cartpole")
    ap.add_argument("--out", default="results/cartpole/figures")
    args = ap.parse_args()
    per_run = os.path.join(args.out, "per_run")
    os.makedirs(per_run, exist_ok=True)
    table = []
    for prefix, label in GROUPS + [("gamma0.9", "gamma 0.9"), ("gamma0.99", "gamma 0.99"), ("rscale2", "reward x 2"),
                                   ("rscale-1", "reward x (-1)"), ("lr0", "learning rate 0")]:
        for d in runs(args.results, prefix):
            name = os.path.basename(d)
            ep = read_csv(os.path.join(d, "episodes.csv"))
            cfg = json.load(open(os.path.join(d, "config.json")))
            ev_p = os.path.join(d, "eval_summary_greedy.json")
            ev = json.load(open(ev_p)) if os.path.exists(ev_p) else None
            evb_p = os.path.join(d, "eval_summary_greedy_best.json")
            evb = json.load(open(evb_p)) if os.path.exists(evb_p) else None
            table.append({"run": name, "group": prefix, "episodes": int(len(ep["episode"])),
                          "steps": int(ep["step"][-1]), "final_ma100": float(moving_average(ep["reward"])[-1]),
                          "eval": ev, "eval_selected": evb})
            if prefix == "main":
                g = cfg["gamma"]
                training_with_checks(ep, os.path.join(per_run, f"cartpole_{name}_training.png"),
                                     f"CartPole-v1, Double DQN, {name}: episode reward")
                maxq_plot(ep, g, os.path.join(per_run, f"cartpole_{name}_maxq.png"),
                          f"CartPole-v1, Double DQN, {name}: maximum Q-value per training episode")
                end_fractions(ep, os.path.join(per_run, f"cartpole_{name}_ends.png"), f"CartPole-v1, {name}: how episodes ended")
                if ev is not None:
                    r = np.load(os.path.join(d, "eval_rewards_greedy.npy"))
                    viz.reward_histogram(r, os.path.join(per_run, f"cartpole_{name}_hist.png"),
                                         f"CartPole-v1, {name}: 500 greedy roll-outs, network after the last episode",
                                         bins=np.arange(0, 525, 25))
                if evb is not None:
                    r = np.load(os.path.join(d, "eval_rewards_greedy_best.npy"))
                    viz.reward_histogram(r, os.path.join(per_run, f"cartpole_{name}_hist_selected.png"),
                                         f"CartPole-v1, {name}: 500 greedy roll-outs, kept network",
                                         bins=np.arange(0, 525, 25))
    # seeds and ablations
    curves = {}
    for prefix, label in GROUPS:
        rs = [read_csv(os.path.join(d, "episodes.csv"))["reward"] for d in runs(args.results, prefix)]
        if rs:
            curves[label] = rs
    if curves:
        viz.compare_curves(curves, os.path.join(args.out, "compare_ablations.png"),
                           "CartPole-v1: moving-average reward, mean and std over seeds")
    # value scale of every ablation (log axis: without a target network the values diverge)
    import matplotlib.pyplot as plt
    viz.setup()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (prefix, label) in enumerate(GROUPS):
        eps_ = [read_csv(os.path.join(d, "episodes.csv")) for d in runs(args.results, prefix)]
        if not eps_:
            continue
        n = min(len(e["episode"]) for e in eps_)
        m = np.mean([moving_average(e["qmax_mean"], 100)[:n] for e in eps_], 0)
        ax.plot(np.arange(1, n + 1), m, color=viz.SERIES[k], lw=2, label=f"{label} ({len(eps_)} seeds)")
    ax.axhline(20, color=viz.TEXT2, ls="--", lw=1, label="1 / (1 - gamma) = 20")
    ax.set_yscale("log")
    ax.set_xlabel("training episode")
    ax.set_ylabel("max_a Q, 100-episode moving average (log scale)")
    ax.set_title("CartPole-v1: value of the visited states for every ablation")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(args.out, "compare_ablations_maxq.png"))
    plt.close(fig)
    main_runs = runs(args.results, "main")
    if main_runs:
        viz.compare_curves({os.path.basename(d): read_csv(os.path.join(d, "episodes.csv"))["reward"] for d in main_runs},
                           os.path.join(args.out, "seeds_main.png"),
                           "CartPole-v1, Double DQN: 100-episode moving-average reward of every seed")
        import matplotlib.pyplot as plt
        viz.setup()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k, d in enumerate(main_runs):
            ep = read_csv(os.path.join(d, "episodes.csv"))
            ax.plot(ep["episode"], moving_average(ep["qmax_mean"], 100), color=viz.SERIES[k], lw=1.6,
                    label=os.path.basename(d))
        ax.axhline(20, color=viz.TEXT2, ls="--", lw=1, label="1 / (1 - 0.95) = 20")
        ax.set_xlabel("training episode")
        ax.set_ylabel("max_a Q, 100-episode moving average")
        ax.set_title("CartPole-v1, Double DQN: maximum Q-value of every seed")
        ax.legend(loc="lower right", fontsize=8)
        fig.savefig(os.path.join(args.out, "seeds_main_maxq.png"))
        plt.close(fig)
    # discount factors: Q scale
    gam = {}
    for prefix, g in [("gamma0.9", 0.9), ("main", 0.95), ("gamma0.99", 0.99)]:
        rs = runs(args.results, prefix)
        if rs:
            gam[g] = [read_csv(os.path.join(d, "episodes.csv")) for d in rs]
    if len(gam) > 1:
        import matplotlib.pyplot as plt
        viz.setup()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k, (g, eps) in enumerate(sorted(gam.items())):
            n = min(len(e["episode"]) for e in eps)
            m = np.mean([moving_average(e["qmax_mean"], 100)[:n] for e in eps], 0)
            ax.plot(np.arange(1, n + 1), m, color=viz.SERIES[k], lw=2, label=f"gamma = {g} ({len(eps)} seeds)")
            ax.axhline(1 / (1 - g), color=viz.SERIES[k], ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("training episode")
        ax.set_ylabel("max_a Q (moving average, log scale)")
        ax.set_title("CartPole-v1: the learned Q-value approaches 1 / (1 - gamma) (dashed)")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(args.out, "compare_gamma_maxq.png"))
        plt.close(fig)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote figures for {len(table)} runs to {args.out}")


if __name__ == "__main__":
    main()
