"""Q-networks: an MLP for CartPole and the Nature-DQN convolutional network for MsPacman."""
import torch
import torch.nn as nn


class MLPQNet(nn.Module):
    """obs (B, obs_dim) -> Q(s, .) (B, n_actions).

    in_scale divides each input feature by a fixed constant (stored with the weights). For CartPole
    the default scales (2.4, 3, 0.21, 3.5) are the cart-position and pole-angle failure thresholds
    and typical velocity magnitudes, so all four inputs are of order 1."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden=(128, 128), in_scale=None):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)
        scale = torch.ones(obs_dim) if in_scale is None else torch.as_tensor(in_scale, dtype=torch.float32)
        self.register_buffer("in_scale", scale)

    def forward(self, x):
        return self.net(x / self.in_scale)


class AtariQNet(nn.Module):
    """Mnih et al. (2015) architecture on a stack of 4 preprocessed 88 x 80 frames.

    Input: int8 tensor (B, 4, 88, 80) with values in [-128, 127] from the assignment's
    preprocess_observation(); it is scaled to [-1, 1) inside the network.
    conv 8x8/4 (32) -> conv 4x4/2 (64) -> conv 3x3/1 (64) -> fc 512 -> fc n_actions.
    Spatial sizes: 88x80 -> 21x19 -> 9x8 -> 7x6, so the flattened size is 64*7*6 = 2688.
    """

    def __init__(self, n_actions: int = 9, in_frames: int = 4, height: int = 88, width: int = 80, arch: str = "nature"):
        super().__init__()
        if arch == "nature":
            self.conv = nn.Sequential(
                nn.Conv2d(in_frames, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            )
            hidden = 512
        elif arch == "nips":
            # Mnih et al. (2013): conv 8x8/4 (16) -> conv 4x4/2 (32) -> fc 256; 88x80 -> 21x19 -> 9x8
            self.conv = nn.Sequential(
                nn.Conv2d(in_frames, 16, 8, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            )
            hidden = 256
        else:
            raise ValueError(arch)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_frames, height, width)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.to(memory_format=torch.channels_last)  # 1.3-2x faster convolutions on CPU (oneDNN)

    def forward(self, x):
        x = (x.float() / 128.0).contiguous(memory_format=torch.channels_last)
        return self.head(self.conv(x))
