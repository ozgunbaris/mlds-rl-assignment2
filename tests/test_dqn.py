"""Unit checks of the DQN mechanics (replay buffers, Bellman targets, preprocessing, schedules).

These are the mechanical checks; the experiments that test whether the algorithm behaves as
reinforcement learning predicts are in dqn/validate_*.py and the report.
"""
import copy

import numpy as np
import pytest
import torch

from dqn.common import PiecewiseLinear, discounted_returns, moving_average
from dqn.core import dqn_update, sync_target, td_targets
from dqn.networks import AtariQNet, MLPQNet
from dqn.replay import FrameStackReplay, VectorReplay


def _frame(i):
    """A distinct 88 x 80 int8 frame that encodes the integer i."""
    f = np.zeros((88, 80), dtype=np.int8)
    f[0, 0] = i % 100
    f[0, 1] = (i // 100) % 100
    return f


def _frame_id(f):
    return int(f[0, 0]) + 100 * int(f[0, 1])


class LiveStack:
    def __init__(self):
        self.s = None

    def reset(self, f):
        self.s = np.stack([f] * 4)

    def push(self, f):
        self.s = np.concatenate([self.s[1:], f[None]])


@pytest.mark.parametrize("capacity", [37, 50, 1000])
def test_framestack_replay_matches_live_stack(capacity):
    """Every transition the buffer can return equals (s, a, r, terminal, s') as seen by the acting
    loop, across episode boundaries (states never mix two episodes) and after wrap-around."""
    rng = np.random.default_rng(0)
    buf = FrameStackReplay(capacity, (88, 80), 4, seed=1)
    live = LiveStack()
    expected = {}
    fid = 0
    for _ in range(60):
        n = int(rng.integers(1, 15))
        f = _frame(fid); fid += 1
        live.reset(f)
        buf.add_first(f)
        for t in range(n):
            s = live.s.copy()
            a, r = int(rng.integers(9)), float(rng.integers(0, 3))
            f = _frame(fid); fid += 1
            live.push(f)
            term = t == n - 1 and rng.random() < 0.5
            slot = buf.ptr
            buf.add(f, a, r, term)
            expected[slot] = (s, a, r, term, live.s.copy())
    valid = np.flatnonzero(buf.valid(np.arange(buf.size)))
    assert len(valid) > 0
    for j in valid:
        s, a, r, term, s2 = expected[j]
        got_s = buf.stack_at(np.array([(j - 1) % capacity]))[0]
        got_s2 = buf.stack_at(np.array([j]))[0]
        assert [_frame_id(x) for x in got_s] == [_frame_id(x) for x in s]
        assert [_frame_id(x) for x in got_s2] == [_frame_id(x) for x in s2]
        assert buf.action[j] == a and buf.reward[j] == r and bool(buf.terminal[j]) == term
    # sampled batches only contain valid slots
    s, a, r, d, s2, j = buf.sample(256)
    assert buf.valid(j).all()
    assert s.shape == (256, 4, 88, 80) and s.dtype == np.int8


def test_framestack_excludes_slots_that_straddle_the_write_pointer():
    buf = FrameStackReplay(10, (88, 80), 4, seed=0)
    buf.add_first(_frame(0))
    for i in range(1, 25):
        buf.add(_frame(i), 0, 0.0, False)
    # full buffer, oldest slot at ptr: the 4 oldest slots cannot build a 4-frame state for s
    ages = (np.arange(10) - buf.ptr) % 10
    assert not buf.valid(np.arange(10))[ages < 4].any()
    assert buf.valid(np.arange(10))[ages >= 4].all()


def test_vector_replay_roundtrip():
    buf = VectorReplay(5, 4, seed=0)
    for i in range(8):
        buf.add(np.full(4, i), i % 2, float(i), np.full(4, i + 1), i == 7)
    assert len(buf) == 5
    s, a, r, d, s2, idx = buf.sample(100)
    assert np.all(s2[:, 0] == s[:, 0] + 1)  # next state belongs to the same transition
    assert set(r.tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}  # only the 5 newest survive
    assert np.all((d == 1.0) == (r == 7.0))


class FixedQ(torch.nn.Module):
    """Q(s, .) = table[s] for integer states."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(table, dtype=torch.float32))

    def forward(self, s):
        return self.table[s.long()]


def test_td_targets_terminal_and_double():
    online = FixedQ([[0.0, 0.0], [5.0, 1.0]])   # online prefers action 0 in state 1
    target = FixedQ([[0.0, 0.0], [2.0, 9.0]])   # target's own max is action 1 (9)
    s2 = torch.tensor([1, 1])
    r = torch.tensor([1.0, 1.0])
    term = torch.tensor([0.0, 1.0])
    y_dqn = td_targets(online, target, s2, r, term, 0.9, double=False)
    y_ddqn = td_targets(online, target, s2, r, term, 0.9, double=True)
    assert torch.allclose(y_dqn, torch.tensor([1 + 0.9 * 9.0, 1.0]))      # max of target
    assert torch.allclose(y_ddqn, torch.tensor([1 + 0.9 * 2.0, 1.0]))     # target evaluates online argmax
    # no target network: the online network evaluates its own argmax
    y_self = td_targets(online, online, s2, r, term, 0.9, double=True)
    assert torch.allclose(y_self, torch.tensor([1 + 0.9 * 5.0, 1.0]))


def test_dqn_update_moves_q_toward_target_and_target_net_is_frozen():
    torch.manual_seed(0)
    online = MLPQNet(4, 2, (32,))
    target = copy.deepcopy(online)
    before = [p.clone() for p in target.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=0.05)
    s = torch.randn(64, 4)
    a = torch.randint(0, 2, (64,))
    r = torch.ones(64)
    d = torch.ones(64)  # terminal: y = r = 1 exactly
    s2 = torch.randn(64, 4)
    err0 = (online(s).gather(1, a[:, None]).squeeze(1) - 1).abs().mean().item()
    for _ in range(200):
        dqn_update(online, target, opt, (s, a, r, d, s2), 0.95, double=True, loss="mse", grad_clip=0)
    err1 = (online(s).gather(1, a[:, None]).squeeze(1) - 1).abs().mean().item()
    assert err1 < 0.1 * err0
    assert all(torch.equal(p, q) for p, q in zip(before, target.parameters()))
    sync_target(online, target)
    assert all(torch.equal(p, q) for p, q in zip(online.parameters(), target.parameters()))


def test_reward_clipping_only_inside_target():
    online = FixedQ([[0.0], [0.0]])
    batch = (torch.tensor([0]), torch.tensor([0]), torch.tensor([200.0]), torch.tensor([1.0]), torch.tensor([1]))
    opt = torch.optim.SGD(online.parameters(), lr=0.0)
    out = dqn_update(online, online, opt, batch, 0.99, reward_clip=1.0, loss="mse", grad_clip=0)
    assert out["y_mean"] == pytest.approx(1.0)


def test_preprocess_matches_signed_arithmetic():
    from dqn.mspacman_dqn import mspacman_color, preprocess_observation
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    obs[5, 10] = (210, 164, 74)  # a Ms. Pac-Man pixel (row 5 is kept by the crop, column 10 too)
    out = preprocess_observation(obs.copy())
    ref = obs[1:176:2, ::2].astype(np.int64).sum(axis=2)
    ref[ref == mspacman_color] = 0
    ref = ref // 3 - 128
    assert out.shape == (88, 80, 1) and out.dtype == np.int8
    assert np.array_equal(out[:, :, 0].astype(np.int64), ref)
    assert out[2, 5, 0] == -128  # the Ms. Pac-Man colour becomes the darkest value


def test_atari_net_shapes():
    net = AtariQNet(9)
    x = torch.zeros(3, 4, 88, 80, dtype=torch.int8)
    assert net(x).shape == (3, 9)
    xf = torch.ones(2, 4, 88, 80) * 64
    xf_copy = xf.clone()
    net(xf)
    assert torch.equal(xf, xf_copy)  # forward does not modify its input


def test_schedules_and_returns():
    sch = PiecewiseLinear([(0, 1.0), (100, 1.0), (200, 0.1), (1000, 0.01)])
    assert sch(0) == 1.0 and sch(100) == 1.0 and sch(150) == pytest.approx(0.55)
    assert sch(200) == pytest.approx(0.1) and sch(5000) == pytest.approx(0.01)
    g = discounted_returns([1, 1, 1], 0.5)
    assert np.allclose(g, [1.75, 1.5, 1.0])
    g = discounted_returns([1, 1, 1], 0.5, cut=np.array([False, True, False]))
    assert np.allclose(g, [1.5, 1.0, 1.0])  # nothing is carried across the cut after step 1
    assert np.allclose(moving_average([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
