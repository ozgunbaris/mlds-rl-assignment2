"""Experience replay buffers.

VectorReplay      small vector observations (CartPole): stores (s, a, r, s', terminal).
FrameStackReplay  image observations (MsPacman): stores every preprocessed frame once and
                  rebuilds the 4-frame stacks at sampling time, so 500k transitions need
                  500k frames (3.5 GB as int8) instead of 8 x 500k.

`terminal` is the bootstrap mask of the Bellman target y = r + gamma (1 - terminal) max Q(s', .).
It is True only when the episode really ended (pole fell, game over, and optionally a lost
life). A time-limit truncation is NOT terminal: s' is a valid state and the target bootstraps.
"""
import numpy as np


class VectorReplay:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminal = np.zeros(self.capacity, dtype=np.float32)
        self.added = np.zeros(self.capacity, dtype=np.int64)  # global step at which the slot was written
        self.ptr = 0
        self.size = 0
        self.n_added = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminal: bool):
        k = self.ptr
        self.obs[k] = s
        self.action[k] = a
        self.reward[k] = r
        self.next_obs[k] = s2
        self.terminal[k] = float(terminal)
        self.added[k] = self.n_added
        self.n_added += 1
        self.ptr = (k + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.action[idx], self.reward[idx], self.terminal[idx], self.next_obs[idx], idx)

    def __len__(self):
        return self.size


class FrameStackReplay:
    """Circular buffer of single frames with episode-start flags.

    Slot k holds frame f_k. Either f_k is the first frame of an episode (start[k] = True,
    no transition leads into it), or it is the frame reached by the transition stored in
    the same slot (action[k], reward[k], terminal[k]) from the frame in slot k - 1.

    The state at slot k is the stack of the last `stack` frames ending at k; frames that
    belong to the previous episode are replaced by the first frame of the current episode
    (the same rule the acting loop uses after env.reset()). Transition j (start[j] False) is
        s = state(j - 1), a = action[j], r = reward[j], s' = state(j), terminal[j].
    """

    def __init__(self, capacity: int, frame_shape=(88, 80), stack: int = 4, seed: int = 0):
        self.capacity = int(capacity)
        self.stack = int(stack)
        self.frames = np.zeros((self.capacity, *frame_shape), dtype=np.int8)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminal = np.zeros(self.capacity, dtype=np.bool_)
        self.start = np.zeros(self.capacity, dtype=np.bool_)
        self.added = np.zeros(self.capacity, dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self.n_added = 0
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- writing
    def _write(self, frame, start, action, reward, terminal):
        k = self.ptr
        self.frames[k] = frame
        self.start[k] = start
        self.action[k] = action
        self.reward[k] = reward
        self.terminal[k] = terminal
        self.added[k] = self.n_added
        self.n_added += 1
        self.ptr = (k + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_first(self, frame):
        """First frame of an episode (after env.reset())."""
        self._write(frame, True, 0, 0.0, False)

    def add(self, frame, action: int, reward: float, terminal: bool):
        """Frame reached by taking `action` in the previous slot's state."""
        self._write(frame, False, action, reward, terminal)

    # ---------------------------------------------------------------- reading
    def valid(self, j):
        """Slots that hold a usable transition (vectorised)."""
        j = np.asarray(j)
        ok = ~self.start[j] & (j < self.size)
        if self.size == self.capacity:
            # the window j - stack .. j must not reach past the oldest slot (which sits at ptr)
            age = (j - self.ptr) % self.capacity
            ok &= age >= self.stack
        else:
            ok &= j >= 1
        return ok

    def stack_at(self, ends):
        """Stacked states ending at slots `ends` -> int8 array (B, stack, H, W)."""
        ends = np.asarray(ends)
        offs = np.arange(-(self.stack - 1), 1)
        pos = (ends[:, None] + offs[None, :]) % self.capacity
        st = self.start[pos]
        src = pos.copy()
        replaced = np.zeros(len(ends), dtype=bool)
        for i in range(self.stack - 2, -1, -1):
            # position i belongs to the previous episode if a later position is a start
            replaced = replaced | st[:, i + 1]
            src[:, i] = np.where(replaced, src[:, i + 1], pos[:, i])
        return self.frames[src]

    def sample_indices(self, batch_size: int):
        out = np.empty(0, dtype=np.int64)
        while len(out) < batch_size:
            cand = self.rng.integers(0, self.size, size=2 * batch_size)
            out = np.concatenate([out, cand[self.valid(cand)]])
        return out[:batch_size]

    def sample(self, batch_size: int):
        j = self.sample_indices(batch_size)
        s = self.stack_at((j - 1) % self.capacity)
        s2 = self.stack_at(j)
        return s, self.action[j], self.reward[j], self.terminal[j].astype(np.float32), s2, j

    def num_transitions(self):
        return int(self.valid(np.arange(self.size)).sum()) if self.size else 0

    def __len__(self):
        return self.size
