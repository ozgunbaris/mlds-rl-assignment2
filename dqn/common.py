"""Shared helpers for the DQN experiments (no plotting here)."""
import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x, window: int = 100) -> np.ndarray:
    """Trailing moving average; the first window-1 entries average what is available."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(x))
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def discounted_returns(rewards, gamma: float, cut=None, bootstrap: float = 0.0) -> np.ndarray:
    """G_t = r_t + gamma * G_{t+1} computed backwards.

    cut: optional boolean array; cut[t] = True means the return is not carried
    across step t (G_t = r_t), e.g. at a life loss when life loss is treated as
    terminal. bootstrap: value added after the last step (0 for a terminal end).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros(len(rewards))
    running = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        if cut is not None and cut[t]:
            running = 0.0
        running = rewards[t] + gamma * running
        out[t] = running
    return out


class PiecewiseLinear:
    """Piecewise-linear schedule through (x, y) knots, constant outside them."""

    def __init__(self, knots):
        self.x = np.array([k[0] for k in knots], dtype=np.float64)
        self.y = np.array([k[1] for k in knots], dtype=np.float64)
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("knots must have increasing x")

    def __call__(self, x) -> float:
        return float(np.interp(x, self.x, self.y))


class CsvLogger:
    """Append-only CSV writer that flushes after every row (safe to plot mid-run)."""

    def __init__(self, path: str, fieldnames, resume: bool = False):
        self.path = path
        self.fieldnames = list(fieldnames)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if (resume and exists) else "w"
        self.f = open(path, mode, newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if mode == "w":
            self.w.writeheader()
            self.f.flush()

    def write(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.f.flush()

    def close(self):
        self.f.close()


def read_csv(path: str):
    """Read a CSV written by CsvLogger into a dict of numpy arrays (numeric where possible)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    if not rows:
        return out
    for k in rows[0].keys():
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col)
    return out


def atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
