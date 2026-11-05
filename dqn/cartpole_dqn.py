"""Double DQN on CartPole-v1 (discount factor 0.95).

Train + roll out 500 greedy episodes:
    python -m dqn.cartpole_dqn --out results/cartpole/ddqn_seed0 --seed 0
Ablations used in the report (same script):
    --no-double                vanilla DQN target (max over the target network)
    --no-target                no target network (targets from the online network itself)
    --buffer-size 64           no experience replay (each update uses the 64 most recent transitions)
    --truncation-terminal      treat the 500-step time limit as a terminal state (the edge-case bug)
    --gamma 0.9 / 0.99         changes the predicted scale of Q, 1 / (1 - gamma)

Every step: act epsilon-greedily on Q_online, store (s, a, r, s', terminal) in the replay buffer,
take one Adam step on a minibatch of 64 (Huber loss, Double DQN target), and copy the online
network into the target network every 500 updates. terminal is True only when the pole falls or
the cart leaves the track; a time-limit truncation at 500 steps bootstraps from s'.
"""
import argparse
import copy
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from .common import CsvLogger, PiecewiseLinear, set_seed
from .core import dqn_update, sync_target
from .networks import MLPQNet
from .replay import VectorReplay

ENV_ID = "CartPole-v1"


def failure_reason(obs, term, trunc):
    """Why an episode ended: 'angle' (|theta| > 12 deg), 'position' (|x| > 2.4) or 'time limit'."""
    if term:
        return "angle" if abs(obs[2]) > 0.2095 else "position"
    return "time limit" if trunc else ""


IN_SCALE = (2.4, 3.0, 0.21, 3.5)


def greedy_episode(env, net):
    obs, _ = env.reset()
    done, R = False, 0.0
    while not done:
        with torch.no_grad():
            a = int(net(torch.as_tensor(obs)[None])[0].argmax())
        obs, r, term, trunc, _ = env.step(a)
        R += r
        done = term or trunc
    return R


def collect_heldout_states(n, seed):
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = []
    while len(out) < n:
        out.append(obs.copy())
        obs, r, term, trunc, _ = env.step(int(rng.integers(2)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(out, dtype=np.float32)


def train(args):
    torch.set_num_threads(1)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    snap_dir = os.path.join(args.out, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    env = gym.make(ENV_ID)
    env.action_space.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    online = MLPQNet(4, 2, tuple(args.hidden), in_scale=IN_SCALE if args.scale_inputs else None)
    target = online if args.no_target else copy.deepcopy(online)
    eval_env = gym.make(ENV_ID)
    eval_env.reset(seed=args.seed + 500)
    best = {"mean": -1.0, "episode": 0}
    opt = torch.optim.Adam(online.parameters(), lr=args.lr)
    lr_at = PiecewiseLinear([(0, args.lr), (args.episodes, args.lr_final)])
    eps_at = PiecewiseLinear([(0, 1.0), (args.learning_starts, 1.0),
                              (args.learning_starts + args.eps_decay_steps, args.eps_final)])
    buf = VectorReplay(args.buffer_size, 4, seed=args.seed + 1)
    heldout = torch.as_tensor(collect_heldout_states(500, seed=10_000 + args.seed))
    log = CsvLogger(os.path.join(args.out, "episodes.csv"),
                    ["episode", "step", "reward", "length", "end", "epsilon", "lr", "qmax_mean", "qmax_max", "q0",
                      "heldout_qmax", "loss_mean", "td_abs_mean", "grad_norm_mean", "y_mean", "n_updates",
                     "eval_greedy_mean", "wall_time_s"])
    torch.save(online.state_dict(), os.path.join(snap_dir, "ep0000.pt"))
    step, updates, t0 = 0, 0, time.time()
    stop_info = {"solved_episode": None, "step": None}
    for ep in range(1, args.episodes + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep - 1)
        obs, _ = env.reset(seed=args.seed if ep == 1 else None)
        qmax, stats, q0 = [], [], None
        done, R, L = False, 0.0, 0
        while not done:
            with torch.no_grad():
                q = online(torch.as_tensor(obs)[None])[0].numpy()
            qmax.append(float(q.max()))
            q0 = float(q.max()) if q0 is None else q0
            a = int(rng.integers(2)) if rng.random() < eps_at(step) else int(q.argmax())
            obs2, r, term, trunc, _ = env.step(a)
            terminal = bool(term or (args.truncation_terminal and trunc))
            buf.add(obs, a, args.reward_scale * r, obs2, terminal)
            obs = obs2
            R += r
            L += 1
            step += 1
            done = term or trunc
            if step >= args.learning_starts and len(buf) >= args.batch_size and step % args.train_freq == 0:
                s, a_b, r_b, d_b, s2, _ = buf.sample(args.batch_size)
                batch = (torch.as_tensor(s), torch.as_tensor(a_b), torch.as_tensor(r_b), torch.as_tensor(d_b),
                         torch.as_tensor(s2))
                stats.append(dqn_update(online, target, opt, batch, args.gamma, double=args.double, loss=args.loss,
                                        grad_clip=args.grad_clip))
                updates += 1
                if not args.no_target:
                    if args.tau < 1.0:
                        sync_target(online, target, args.tau)
                    elif updates % args.target_update == 0:
                        sync_target(online, target)
        with torch.no_grad():
            hq = float(online(heldout).max(1).values.mean())
        mean = lambda k: float(np.mean([x[k] for x in stats])) if stats else ""  # noqa: E731
        ev, solved = "", False
        if args.eval_every > 0 and ep % args.eval_every == 0:
            scores = [greedy_episode(eval_env, online) for _ in range(args.eval_periodic_episodes)]
            # the rule is armed only once exploration has reached its final rate, so training does not stop
            # while the behaviour policy is still mostly random and Q is far from its fixed point
            armed = step >= args.learning_starts + args.eps_decay_steps
            if args.stop_when_solved and armed and min(scores) >= 500:
                # confirmation: the stopping rule needs `solve_episodes` consecutive 500-step greedy episodes
                more = [greedy_episode(eval_env, online) for _ in range(args.solve_episodes - len(scores))]
                solved = min(more, default=500) >= 500
                scores += more
            ev = float(np.mean(scores))
            if ev >= best["mean"]:  # ties go to the later network
                best = {"mean": ev, "episode": ep, "n": len(scores)}
                torch.save(online.state_dict(), os.path.join(args.out, "best_weights.pt"))
        log.write(episode=ep, step=step, reward=R, length=L, end=failure_reason(obs, term, trunc),
                  epsilon=round(eps_at(step), 4), lr=opt.param_groups[0]["lr"], qmax_mean=float(np.mean(qmax)),
                  qmax_max=float(np.max(qmax)), q0=q0, heldout_qmax=hq, loss_mean=mean("loss"),
                  td_abs_mean=mean("td_abs"), grad_norm_mean=mean("grad_norm"), y_mean=mean("y_mean"),
                  n_updates=len(stats), eval_greedy_mean=ev, wall_time_s=round(time.time() - t0, 2))
        if ep % args.snapshot_every == 0:
            torch.save(online.state_dict(), os.path.join(snap_dir, f"ep{ep:04d}.pt"))
        if solved:
            print(f"solved at episode {ep}: {args.solve_episodes} consecutive greedy evaluation episodes "
                  f"of 500 steps; training stops", flush=True)
            stop_info = {"solved_episode": ep, "step": step}
            break
        if ep % args.print_every == 0:
            print(f"ep {ep:4d} | step {step:7d} | reward {R:5.0f} | eps {eps_at(step):.3f} | qmax {np.mean(qmax):6.2f} "
                  f"| held-out {hq:6.2f} | {time.time() - t0:6.1f}s", flush=True)
    log.close()
    torch.save(online.state_dict(), os.path.join(args.out, "final_weights.pt"))
    with open(os.path.join(args.out, "best_selection.json"), "w") as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(args.out, "stopping.json"), "w") as f:
        json.dump({**stop_info, "episodes_trained": ep, "steps_trained": step}, f, indent=2)
    env.close()
    eval_env.close()
    return online


def load_net(path, hidden=(128, 128)):
    sd = torch.load(path, weights_only=True)
    net = MLPQNet(4, 2, tuple(hidden))
    net.load_state_dict(sd)  # in_scale is a buffer stored with the weights
    net.eval()
    return net


def evaluate(net, episodes, seed, eps=0.0, out=None, tag=None):
    """Roll out `episodes` episodes with the epsilon-greedy policy (eps = 0: greedy)."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    rewards, ends = [], []
    obs, _ = env.reset(seed=seed)
    for i in range(episodes):
        if i > 0:
            obs, _ = env.reset()
        done, R = False, 0.0
        while not done:
            with torch.no_grad():
                q = net(torch.as_tensor(obs)[None])[0].numpy()
            a = int(rng.integers(2)) if rng.random() < eps else int(q.argmax())
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        rewards.append(R)
        ends.append(failure_reason(obs, term, trunc))
    env.close()
    rewards = np.array(rewards)
    summary = {"episodes": int(len(rewards)), "epsilon": eps, "mean": float(rewards.mean()),
               "std": float(rewards.std()), "min": float(rewards.min()), "max": float(rewards.max()),
               "solved_fraction": float(np.mean(rewards >= 500)),
               "ends": {k: int(sum(e == k for e in ends)) for k in ["angle", "position", "time limit"]},
               "seed": seed}
    if out is not None:
        tag = tag or ("greedy" if eps == 0 else f"eps{eps:g}")
        np.save(os.path.join(out, f"eval_rewards_{tag}.npy"), rewards)
        with open(os.path.join(out, f"eval_summary_{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return rewards, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/cartpole/ddqn_seed0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1500, help="maximum number of training episodes")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-final", type=float, default=5e-5, help="linear decay over the training episodes")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--target-update", type=int, default=500, help="in gradient updates (hard copy)")
    p.add_argument("--tau", type=float, default=1.0, help="< 1: Polyak update of the target after every update")
    p.add_argument("--scale-inputs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=25, help="greedy check every N episodes (0: off)")
    p.add_argument("--eval-periodic-episodes", type=int, default=10)
    p.add_argument("--stop-when-solved", action=argparse.BooleanOptionalAction, default=True,
                   help="stop training once the greedy policy lasts 500 steps in --solve-episodes consecutive "
                        "evaluation episodes (separate environment instance and seeds)")
    p.add_argument("--solve-episodes", type=int, default=50)
    p.add_argument("--eps-decay-steps", type=int, default=20_000)
    p.add_argument("--eps-final", type=float, default=0.02)
    p.add_argument("--double", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-target", action="store_true")
    p.add_argument("--truncation-terminal", action="store_true")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="multiplies the reward stored for learning (control experiments); logs keep the raw reward")
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--eval-only", default=None, help="weights file to roll out instead of training")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.eval_only:
        net = load_net(args.eval_only, args.hidden)
    else:
        net = train(args)
    _, summary = evaluate(net, args.eval_episodes, seed=args.seed + 1000, eps=0.0, out=args.out)
    print("eval final network (greedy):", json.dumps(summary), flush=True)
    best_path = os.path.join(args.out, "best_weights.pt")
    if not args.eval_only and os.path.exists(best_path):
        _, sb = evaluate(load_net(best_path, args.hidden), args.eval_episodes, seed=args.seed + 1000, eps=0.0,
                         out=args.out, tag="greedy_best")
        print("eval selected network (greedy):", json.dumps(sb), flush=True)


if __name__ == "__main__":
    main()
