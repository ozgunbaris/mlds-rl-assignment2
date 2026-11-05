"""The DQN update shared by CartPole and MsPacman.

For a minibatch (s, a, r, terminal, s') from the replay buffer the target is
    y = r + gamma * (1 - terminal) * Q_target(s', a*)
with a* = argmax_a Q_target(s', a) (DQN, Mnih et al. 2015) or
     a* = argmax_a Q_online(s', a) (Double DQN, van Hasselt et al. 2016),
and the online network takes one gradient step on the Huber (or squared) loss
between Q_online(s, a) and y. y is computed without gradient. The target network
is a copy of the online network that is refreshed every `target_update` updates
(or never different from it, in the no-target-network ablation).
"""
import torch
import torch.nn.functional as F


def td_targets(online, target, s2, r, terminal, gamma: float, double: bool = True):
    """Bellman targets (no gradient). `target` may be the online network itself."""
    with torch.no_grad():
        q2_target = target(s2)
        if double:
            a2 = online(s2).argmax(1, keepdim=True) if target is not online else q2_target.argmax(1, keepdim=True)
            q2 = q2_target.gather(1, a2).squeeze(1)
        else:
            q2 = q2_target.max(1).values
        return r + gamma * (1.0 - terminal) * q2


def dqn_update(online, target, optimizer, batch, gamma: float, double: bool = True, loss: str = "huber",
               grad_clip: float = 10.0, reward_clip: float = 0.0):
    """One gradient step. batch = (s, a, r, terminal, s2) as torch tensors.
    reward_clip > 0 clips rewards to [-reward_clip, reward_clip] inside the target."""
    s, a, r, terminal, s2 = batch
    if reward_clip > 0:
        r = r.clamp(-reward_clip, reward_clip)
    y = td_targets(online, target, s2, r, terminal, gamma, double)
    q = online(s).gather(1, a[:, None]).squeeze(1)
    if loss == "huber":
        l = F.smooth_l1_loss(q, y)
    elif loss == "mse":
        l = F.mse_loss(q, y)
    else:
        raise ValueError(loss)
    optimizer.zero_grad(set_to_none=True)
    l.backward()
    gn = torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip if grad_clip > 0 else float("inf"))
    optimizer.step()
    td = (q.detach() - y)
    return {"loss": float(l.item()), "q_mean": float(q.detach().mean()), "y_mean": float(y.mean()),
            "td_abs": float(td.abs().mean()), "grad_norm": float(gn)}


def sync_target(online, target, tau: float = 1.0):
    """Hard copy (tau = 1) or Polyak averaging target <- tau * online + (1 - tau) * target."""
    with torch.no_grad():
        if tau >= 1.0:
            target.load_state_dict(online.state_dict())
        else:
            for p_t, p in zip(target.parameters(), online.parameters()):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)
