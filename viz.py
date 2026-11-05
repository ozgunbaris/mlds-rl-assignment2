"""Shared matplotlib style and plotting helpers for the report figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dqn.common import moving_average  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#dddcd7"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": TEXT2,
        "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "savefig.dpi": 160, "figure.constrained_layout.use": True,
    })


def training_curve(rewards, path, title, window=100, ylabel="episode reward", xlabel="training episode"):
    setup()
    rewards = np.asarray(rewards, dtype=float)
    ep = np.arange(1, len(rewards) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ep, rewards, color=BLUE, lw=0.6, alpha=0.35, label="episode reward")
    ax.plot(ep, moving_average(rewards, window), color=ORANGE, lw=2.0, label=f"moving average ({window} episodes)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def reward_histogram(rewards, path, title, bins=30, xlabel="episode reward"):
    setup()
    rewards = np.asarray(rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(rewards, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    m, s = rewards.mean(), rewards.std()
    ax.axvline(m, color=ORANGE, lw=2, label=f"mean = {m:.2f}, std = {s:.2f}, n = {len(rewards)}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of episodes")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def compare_curves(curves, path, title, window=100, ylabel="episode reward (moving average)",
                   xlabel="training episode", band=None):
    """curves: ordered dict name -> 1-D rewards (or list of per-seed arrays when band=True).
    With per-seed lists, the mean of the moving averages is drawn with a +-1 std band."""
    setup()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (name, data) in enumerate(curves.items()):
        color = SERIES[k % len(SERIES)]
        if isinstance(data, (list, tuple)):
            mas = [moving_average(np.asarray(r, dtype=float), window) for r in data]
            n = min(len(m) for m in mas)
            mas = np.stack([m[:n] for m in mas])
            ep = np.arange(1, n + 1)
            mean, std = mas.mean(0), mas.std(0)
            ax.plot(ep, mean, color=color, lw=2, label=f"{name} (mean of {len(mas)} seeds)")
            ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        else:
            r = np.asarray(data, dtype=float)
            ax.plot(np.arange(1, len(r) + 1), moving_average(r, window), color=color, lw=2, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def line_plot(x, ys, path, title, xlabel, ylabel, labels=None, step=False, markers=False, logy=False, annotate=None):
    setup()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ys = [ys] if np.ndim(ys[0]) == 0 else ys
    for k, y in enumerate(ys):
        color = SERIES[k % len(SERIES)]
        lab = labels[k] if labels else None
        if step:
            ax.step(x, y, where="post", color=color, lw=2, label=lab)
        else:
            ax.plot(x, y, color=color, lw=2, label=lab, marker="o" if markers else None, ms=4)
    if logy:
        ax.set_yscale("log")
    if annotate:
        for text, (xa, ya) in annotate:
            ax.annotate(text, xy=(xa, ya), xytext=(8, 8), textcoords="offset points", color=TEXT2, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if labels:
        ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)
