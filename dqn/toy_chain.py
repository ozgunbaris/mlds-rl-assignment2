"""A deterministic chain MDP with known optimal action values, used to test the DQN code path.

States 0..n-1 on a line, observed as one-hot vectors; action 0 = left, 1 = right (deterministic).
mode "goal":     moving right from state n-2 into state n-1 gives reward 1 and ends the episode;
                 every other step gives 0. Q*(s, right) = gamma^(n-2-s), Q*(s, left) = gamma V*(max(s-1, 0)).
mode "survive":  reward 1 on every step, the episode never terminates and is truncated after
                 `time_limit` steps. The state carries no clock, so the correct (bootstrapped) values are
                 Q* = 1 / (1 - gamma) for every state and action. If the truncation is wrongly treated as
                 terminal, a fraction p = 1/time_limit of the targets is 1 instead of 1 + gamma Q. With a squared
                 loss the fixed point is the mean target: Q = 1 / (1 - gamma (1 - p)) (= 14.49 for gamma 0.95,
                 limit 50). With the Huber loss (delta = 1) the fixed point solves
                 (1 - p) psi(1 + gamma Q - Q) + p psi(1 - Q) = 0 with psi clipped at +-1; the wrong targets are
                 far below Q, so psi = -1 for them and Q = (1 - p / (1 - p)) / (1 - gamma) (= 19.59): the Huber
                 loss treats the wrong targets as outliers, like a median.

The same MLPQNet, VectorReplay, dqn_update (Double DQN, Huber loss) and target-network code as for
CartPole are trained on it with a uniformly random behaviour policy (off-policy learning).
"""
import copy

import numpy as np
import torch

from .core import dqn_update, sync_target
from .networks import MLPQNet
from .replay import VectorReplay


class ChainEnv:
    def __init__(self, n=8, mode="goal", time_limit=50, seed=0):
        self.n, self.mode, self.time_limit = n, mode, time_limit
        self.rng = np.random.default_rng(seed)
        self.s, self.t = 0, 0

    def obs(self):
        o = np.zeros(self.n, dtype=np.float32)
        o[self.s] = 1.0
        return o

    def reset(self):
        # exploring starts: every non-goal state is a possible start, so every state is visited
        self.s = int(self.rng.integers(0, self.n - 1 if self.mode == "goal" else self.n))
        self.t = 0
        return self.obs()

    def step(self, a):
        self.t += 1
        if a == 1:
            self.s = min(self.s + 1, self.n - 1)
        else:
            self.s = max(self.s - 1, 0)
        if self.mode == "goal":
            term = self.s == self.n - 1
            r = 1.0 if term else 0.0
        else:
            term, r = False, 1.0
        trunc = (not term) and self.t >= self.time_limit
        return self.obs(), r, term, trunc


def exact_q(n, gamma, mode, time_limit=50, truncation_terminal=False, loss="mse"):
    """Optimal action values by value iteration (goal) or in closed form (survive; for the biased
    truncation-as-terminal variant, the fixed point of the given loss)."""
    if mode == "survive":
        if truncation_terminal:
            p = 1.0 / time_limit
            if loss == "huber":
                return np.full((n, 2), (1.0 - p / (1.0 - p)) / (1.0 - gamma))
            return np.full((n, 2), 1.0 / (1.0 - gamma * (1.0 - p)))
        return np.full((n, 2), 1.0 / (1.0 - gamma))
    V = np.zeros(n)
    for _ in range(10_000):
        Q = np.zeros((n, 2))
        for s in range(n - 1):
            for a in (0, 1):
                s2 = min(s + 1, n - 1) if a == 1 else max(s - 1, 0)
                Q[s, a] = 1.0 if s2 == n - 1 else gamma * V[s2]
        V_new = Q.max(1)
        V_new[n - 1] = 0.0
        if np.max(np.abs(V_new - V)) < 1e-12:
            break
        V = V_new
    return Q


def run_dqn(mode="goal", n=8, gamma=0.95, steps=20_000, seed=0, double=True, use_target=True,
            truncation_terminal=False, time_limit=50, lr=1e-3, target_update=100, batch=64, loss="huber",
            grad_clip=10.0):
    """Train with a uniformly random behaviour policy; return the learned Q table (n x 2)."""
    torch.manual_seed(seed)
    env = ChainEnv(n, mode, time_limit, seed)
    rng = np.random.default_rng(seed + 1)
    online = MLPQNet(n, 2, (64, 64))
    target = copy.deepcopy(online) if use_target else online
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = VectorReplay(10_000, n, seed=seed + 2)
    o = env.reset()
    updates = 0
    for t in range(steps):
        a = int(rng.integers(2))
        o2, r, term, trunc = env.step(a)
        buf.add(o, a, r, o2, term or (truncation_terminal and trunc))
        o = env.reset() if (term or trunc) else o2
        if len(buf) >= batch:
            s, a_b, r_b, d_b, s2, _ = buf.sample(batch)
            dqn_update(online, target, opt, (torch.as_tensor(s), torch.as_tensor(a_b), torch.as_tensor(r_b),
                                             torch.as_tensor(d_b), torch.as_tensor(s2)),
                       gamma, double=double, loss=loss, grad_clip=grad_clip)
            updates += 1
            if use_target and updates % target_update == 0:
                sync_target(online, target)
    with torch.no_grad():
        q = online(torch.eye(n)).numpy()
    if mode == "goal":
        q[n - 1] = 0.0  # the goal is terminal; its value is never used
    return q


def aliased_corridor_fixed_point(n_steps, gamma, exit_reward, loss):
    """Value that the update converges to when `n_steps` consecutive states share one observation.

    n_steps - 1 transitions have the target gamma * q (reward 0, same observation), and the last one has
    the target exit_reward (terminal). Squared loss: the mean target, q = R / ((n - 1)(1 - gamma) + 1).
    Huber loss (delta = 1) with R - q > 1: the exit transition contributes a gradient of size 1, so
    (n - 1)(1 - gamma) q = 1 and q = 1 / ((n - 1)(1 - gamma)), independent of R."""
    if loss == "huber":
        return 1.0 / ((n_steps - 1) * (1.0 - gamma))
    return exit_reward / ((n_steps - 1) * (1.0 - gamma) + 1.0)


def run_aliased_corridor(n_steps=66, gamma=0.99, exit_reward=25.0, loss="huber", updates=40_000, seed=0,
                         lr=1e-3, target_update=50, batch=64):
    """A corridor of `n_steps` steps whose states all look the same (one constant observation), with reward
    0 on every step except the last, which pays `exit_reward` and ends the episode. No action has any effect.
    This copies the frozen start of a MsPacman game: many steps with (almost) the same frames, then a state
    of high value. The true values are exit_reward * gamma^k for k = 0 .. n_steps - 1; the network cannot
    represent them because it sees one observation. Same MLPQNet, VectorReplay and dqn_update as CartPole.
    Returns the learned value (mean over both actions), averaged over the last 10 % of the updates."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    online = MLPQNet(1, 2, (64, 64))
    target = copy.deepcopy(online)
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = VectorReplay(100 * n_steps, 1, seed=seed + 2)
    o = np.ones(1, dtype=np.float32)
    for _ in range(100):  # 100 episodes fill the buffer: every transition of the corridor equally often
        for k in range(n_steps):
            last = k == n_steps - 1
            buf.add(o, int(rng.integers(2)), exit_reward if last else 0.0, o, last)
    tail = []
    for u in range(1, updates + 1):
        s, a_b, r_b, d_b, s2, _ = buf.sample(batch)
        dqn_update(online, target, opt, (torch.as_tensor(s), torch.as_tensor(a_b), torch.as_tensor(r_b),
                                         torch.as_tensor(d_b), torch.as_tensor(s2)), gamma, double=True, loss=loss)
        if u % target_update == 0:
            sync_target(online, target)
        if u > 0.9 * updates and u % 50 == 0:
            with torch.no_grad():
                tail.append(float(online(torch.ones(1, 1)).mean()))
    return float(np.mean(tail))
