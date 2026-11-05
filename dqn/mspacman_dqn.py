"""Double DQN on ALE/MsPacman-v5 from pixels.

Train (resumable; Ctrl-C or the time limit saves a checkpoint):
    python -m dqn.mspacman_dqn --out results/mspacman/main_seed0 --seed 0 --max-episodes 5000
Roll out a trained network:
    python -m dqn.mspacman_dqn --eval results/mspacman/main_seed0/checkpoint.pt --eval-episodes 500 \
        --out results/mspacman/main_seed0

Environment: ALE/MsPacman-v5 with its defaults (frame skip 4, sticky actions with probability 0.25,
9 actions, at most 108,000 frames per game). Every frame goes through the assignment's
preprocess_observation() (88 x 80 int8); the state is the stack of the last 4 preprocessed frames.
Discount factor 0.99. Rewards are clipped to [-1, 1] inside the Bellman target only; every
reported score is the raw game score of a full game (3 lives).
"""
import argparse
import copy
import csv
import json
import os
import random
import signal
import time

import gymnasium as gym
import ale_py
import numpy as np
import torch

from .common import CsvLogger, PiecewiseLinear, atomic_torch_save, set_seed
from .core import dqn_update, sync_target
from .networks import AtariQNet
from .replay import FrameStackReplay

gym.register_envs(ale_py)
ENV_ID = "ALE/MsPacman-v5"
N_ACTIONS = 9
STACK = 4

# ----------------------------------------------------------------------------- preprocessing
# Verbatim from the assignment. Note: obs is uint8, so img.sum(axis=2) is uint64 and "img // 3 - 128"
# wraps around for values below 128; the final astype(np.int8) keeps the low 8 bits, which equal the
# signed result, so the output is still correct (checked against a signed computation in the tests).
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


def frame(obs):
    """(88, 80) int8 frame for the replay buffer and the stack."""
    return preprocess_observation(obs)[:, :, 0]


def make_env():
    return gym.make(ENV_ID)


class FrameStack:
    """Live stack of the last 4 frames; after reset it holds 4 copies of the first frame
    (the same rule FrameStackReplay.stack_at uses at episode starts)."""

    def __init__(self):
        self.s = np.zeros((STACK, 88, 80), dtype=np.int8)

    def reset(self, f):
        self.s[:] = f

    def push(self, f):
        self.s[:-1] = self.s[1:]
        self.s[-1] = f

    def get(self):
        return self.s


def net_device(net):
    return next(net.parameters()).device


def q_values(net, state):
    with torch.no_grad():
        return net(torch.from_numpy(state)[None].to(net_device(net)))[0].cpu().numpy()


def play_episode(env, net, eps, rng, record=False):
    """One full game with an epsilon-greedy policy. Returns (score, length, extra)."""
    obs, info = env.reset()
    st = FrameStack()
    st.reset(frame(obs))
    done, score, length = False, 0.0, 0
    rec = {"q": [], "a": [], "r": [], "lives": [], "obs": []} if record else None
    while not done:
        q = q_values(net, st.get())
        a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(q.argmax())
        if record:
            rec["q"].append(q)
            rec["a"].append(a)
        obs, r, term, trunc, info = env.step(a)
        st.push(frame(obs))
        score += r
        length += 1
        done = term or trunc
        if record:
            rec["r"].append(r)
            rec["lives"].append(info["lives"])
    return score, length, rec


def collect_heldout_states(n, seed, every=4):
    """Fixed set of states from a uniformly random policy (Mnih et al. 2015, Fig. 2): the average of
    max_a Q(s, a) over this set is tracked during training."""
    env = make_env()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    st = FrameStack()
    st.reset(frame(obs))
    out, t = [], 0
    while len(out) < n:
        obs, r, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
        st.push(frame(obs))
        t += 1
        if t % every == 0:
            out.append(st.get().copy())
        if term or trunc:
            obs, _ = env.reset()
            st.reset(frame(obs))
    env.close()
    return np.stack(out)


def heldout_q(net, states, batch=250):
    vals = []
    dev = net_device(net)
    with torch.no_grad():
        for i in range(0, len(states), batch):
            vals.append(net(torch.from_numpy(states[i:i + batch]).to(dev)).max(1).values.cpu().numpy())
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std())


def truncate_csv(path, key, max_value):
    """Drop rows written after the checkpoint we resume from."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else None
    if fields is None:
        return
    keep = [r for r in rows if float(r[key]) <= max_value]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(keep)


class Stop:
    flag = False


def _handle(sig, frm):
    Stop.flag = True


def train(args):
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    snap_dir = os.path.join(args.out, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.pt")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}", flush=True)
    online = AtariQNet(N_ACTIONS, arch=args.arch).to(device)
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=args.lr, eps=args.adam_eps)
    lr_decay_updates = max(1, args.lr_decay_steps // args.train_freq)
    lr_factor = PiecewiseLinear([(0, 1.0), (lr_decay_updates, args.lr_final / args.lr)])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: lr_factor(u))
    eps_sched = PiecewiseLinear([(0, 1.0), (args.learning_starts, 1.0),
                                 (args.learning_starts + args.eps_decay_steps, args.eps_mid),
                                 (args.eps_final_step, args.eps_final)])

    episode, step, updates = 0, 0, 0
    wall_before = 0.0
    refill_until = args.learning_starts
    resumed = False
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False, map_location=device)
        online.load_state_dict(ck["online"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        episode, step, updates = ck["episode"], ck["step"], ck["updates"]
        wall_before = ck.get("wall_time_s", 0.0)
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        # the replay buffer is not checkpointed (3.5 GB): refill it before updating again
        refill_until = step + args.resume_refill
        resumed = True
        for name, key in [("episodes.csv", "episode"), ("heldout_q.csv", "episode"), ("eval_periodic.csv", "episode")]:
            truncate_csv(os.path.join(args.out, name), key, episode)
        print(f"resumed from episode {episode}, step {step}, updates {updates}; refilling the buffer "
              f"until step {refill_until}", flush=True)
    else:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(args.seed + 7 * episode)
    env = make_env()
    eval_env = make_env()
    eval_env.reset(seed=args.seed + 10_000 + episode)
    eval_rng = np.random.default_rng(args.seed + 20_000 + episode)
    buf = FrameStackReplay(args.buffer_size, (88, 80), STACK, seed=args.seed + 30_000 + episode)
    heldout = collect_heldout_states(args.heldout_states, seed=10_000 + args.seed)

    ep_fields = ["episode", "step", "reward", "clipped_reward", "length", "epsilon", "lr", "qmax_mean",
                 "qmax_max", "q0", "loss_mean", "td_abs_mean", "grad_norm_mean", "y_mean", "n_updates",
                 "truncated", "lives_lost"] + [f"a{i}" for i in range(N_ACTIONS)] + ["wall_time_s", "steps_per_s"]
    log = CsvLogger(os.path.join(args.out, "episodes.csv"), ep_fields, resume=resumed)
    hlog = CsvLogger(os.path.join(args.out, "heldout_q.csv"), ["episode", "step", "qmax_mean", "qmax_std"],
                     resume=resumed)
    elog = CsvLogger(os.path.join(args.out, "eval_periodic.csv"),
                     ["episode", "step", "eval_mean", "eval_std", "eval_min", "eval_max", "eval_len_mean", "n",
                      "epsilon"], resume=resumed)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    t_start = time.time()

    def save_checkpoint():
        atomic_torch_save({"online": online.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
                           "sched": sched.state_dict(), "episode": episode, "step": step, "updates": updates,
                           "wall_time_s": wall_before + time.time() - t_start, "py_rng": random.getstate(),
                           "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                           "config": vars(args)}, ckpt_path)

    if not resumed:
        h = heldout_q(online, heldout)
        hlog.write(episode=0, step=0, qmax_mean=h[0], qmax_std=h[1])
        atomic_torch_save(online.state_dict(), os.path.join(snap_dir, "ep00000.pt"))

    snapshot_set = set(args.snapshot_episodes)
    st = FrameStack()
    while True:
        elapsed_h = (wall_before + time.time() - t_start) / 3600
        if episode >= args.max_episodes or step >= args.max_steps or elapsed_h >= args.time_limit_hours or Stop.flag:
            break
        obs, info = env.reset(seed=args.seed + episode if episode == 0 or resumed else None)
        resumed = False
        f = frame(obs)
        st.reset(f)
        buf.add_first(f)
        lives = info["lives"]
        score, clipped, length, lives_lost = 0.0, 0, 0, 0
        qmax, stats = [], []
        acounts = np.zeros(N_ACTIONS, dtype=np.int64)
        t_ep = time.time()
        q0 = None
        while True:
            q = q_values(online, st.get())
            qmax.append(float(q.max()))
            if q0 is None:
                q0 = float(q.max())
            eps = eps_sched(step)
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(q.argmax())
            acounts[a] += 1
            obs, r, term, trunc, info = env.step(a)
            f = frame(obs)
            st.push(f)
            life_lost = info["lives"] < lives
            lives = info["lives"]
            lives_lost += int(life_lost)
            terminal = bool(term or (args.life_loss_terminal and life_lost))
            buf.add(f, a, float(r), terminal)
            score += r
            clipped += int(r != 0)
            length += 1
            step += 1
            if step > refill_until and step % args.train_freq == 0:
                s, a_b, r_b, d_b, s2, _ = buf.sample(args.batch_size)
                batch = tuple(torch.from_numpy(x).to(device) for x in (s, a_b, r_b, d_b, s2))
                stats.append(dqn_update(online, target, opt, batch, args.gamma, double=args.double,
                                        loss=args.loss, grad_clip=args.grad_clip, reward_clip=args.reward_clip))
                sched.step()
                updates += 1
                if updates % args.target_update == 0:
                    sync_target(online, target)
            if term or trunc:
                break
        episode += 1
        dt = time.time() - t_ep
        mean = lambda k: float(np.mean([x[k] for x in stats])) if stats else ""  # noqa: E731
        row = dict(episode=episode, step=step, reward=score, clipped_reward=clipped, length=length,
                   epsilon=round(eps_sched(step), 5), lr=opt.param_groups[0]["lr"], qmax_mean=float(np.mean(qmax)),
                   qmax_max=float(np.max(qmax)), q0=q0, loss_mean=mean("loss"), td_abs_mean=mean("td_abs"),
                   grad_norm_mean=mean("grad_norm"), y_mean=mean("y_mean"), n_updates=len(stats),
                   truncated=int(bool(trunc)), lives_lost=lives_lost,
                   wall_time_s=round(wall_before + time.time() - t_start, 1), steps_per_s=round(length / dt, 1))
        row.update({f"a{i}": int(acounts[i]) for i in range(N_ACTIONS)})
        log.write(**row)
        if episode % args.heldout_every == 0:
            h = heldout_q(online, heldout)
            hlog.write(episode=episode, step=step, qmax_mean=h[0], qmax_std=h[1])
        if args.eval_every > 0 and episode % args.eval_every == 0:
            sc, ln = [], []
            for _ in range(args.eval_periodic_episodes):
                s_, l_, _ = play_episode(eval_env, online, args.eval_periodic_eps, eval_rng)
                sc.append(s_)
                ln.append(l_)
            elog.write(episode=episode, step=step, eval_mean=float(np.mean(sc)), eval_std=float(np.std(sc)),
                       eval_min=float(np.min(sc)), eval_max=float(np.max(sc)), eval_len_mean=float(np.mean(ln)),
                       n=len(sc), epsilon=args.eval_periodic_eps)
        if episode in snapshot_set:
            atomic_torch_save(online.state_dict(), os.path.join(snap_dir, f"ep{episode:05d}.pt"))
        if episode % args.print_every == 0:
            print(f"ep {episode:5d} | step {step:8d} | score {score:6.0f} | len {length:5d} | eps {eps_sched(step):.3f} | "
                  f"qmax {np.mean(qmax):6.2f} | upd {updates:7d} | {length / dt:5.0f} sps | "
                  f"{(wall_before + time.time() - t_start) / 3600:5.2f} h", flush=True)
        if episode % args.ckpt_every == 0:
            save_checkpoint()
    save_checkpoint()
    atomic_torch_save(online.state_dict(), os.path.join(args.out, "final_weights.pt"))
    log.close()
    hlog.close()
    elog.close()
    env.close()
    eval_env.close()
    print(f"stopped at episode {episode}, step {step}, updates {updates}, "
          f"{(wall_before + time.time() - t_start) / 3600:.2f} h", flush=True)
    return online


def load_net(path):
    sd = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(sd, dict) and "online" in sd:
        sd = sd["online"]
    net = AtariQNet(N_ACTIONS, arch="nature" if "conv.4.weight" in sd else "nips")
    net.load_state_dict(sd)
    net.eval()
    return net


def evaluate(net, episodes, seed, eps=0.0, out=None, tag=None):
    env = make_env()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    scores, lengths = [], []
    t0 = time.time()
    for i in range(episodes):
        s, l, _ = play_episode(env, net, eps, rng)
        scores.append(s)
        lengths.append(l)
        if (i + 1) % 50 == 0:
            print(f"  eval {i + 1}/{episodes}: mean so far {np.mean(scores):.1f} ({time.time() - t0:.0f}s)", flush=True)
    env.close()
    scores = np.array(scores)
    lengths = np.array(lengths)
    summary = {"episodes": int(len(scores)), "epsilon": eps, "mean": float(scores.mean()), "std": float(scores.std()),
               "min": float(scores.min()), "max": float(scores.max()), "median": float(np.median(scores)),
               "length_mean": float(lengths.mean()), "seed": seed}
    if out is not None:
        tag = tag or f"eps{eps:g}"
        np.save(os.path.join(out, f"eval_rewards_{tag}.npy"), scores)
        np.save(os.path.join(out, f"eval_lengths_{tag}.npy"), lengths)
        with open(os.path.join(out, f"eval_summary_{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return scores, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/mspacman/main_seed0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cpu, cuda, cuda:0, ...")
    p.add_argument("--arch", choices=["nature", "nips"], default="nips",
                   help="nips: Mnih et al. 2013 network (2.3x cheaper on CPU); nature: Mnih et al. 2015")
    p.add_argument("--max-episodes", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=10_000_000)
    p.add_argument("--time-limit-hours", type=float, default=1e9)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=5e-5, help="learning rate after --lr-decay-steps (linear)")
    p.add_argument("--lr-decay-steps", type=int, default=5_000_000)
    p.add_argument("--adam-eps", type=float, default=1.5e-4)
    p.add_argument("--learning-starts", type=int, default=50_000)
    p.add_argument("--eps-decay-steps", type=int, default=250_000, help="1.0 -> eps-mid over this many steps")
    p.add_argument("--eps-mid", type=float, default=0.1)
    p.add_argument("--eps-final", type=float, default=0.01)
    p.add_argument("--eps-final-step", type=int, default=2_500_000, help="eps-mid -> eps-final until this step")
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update", type=int, default=2000, help="in gradient updates (x train-freq steps)")
    p.add_argument("--double", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--reward-clip", type=float, default=1.0)
    p.add_argument("--life-loss-terminal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--heldout-states", type=int, default=1000)
    p.add_argument("--heldout-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-periodic-episodes", type=int, default=10)
    p.add_argument("--eval-periodic-eps", type=float, default=0.0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--snapshot-episodes", type=int, nargs="*",
                   default=[100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-refill", type=int, default=50_000)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--eval", default=None, help="checkpoint/weights to roll out instead of training")
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--eval-eps", type=float, default=0.0)
    p.add_argument("--eval-seed", type=int, default=2026)
    p.add_argument("--eval-tag", default=None)
    args = p.parse_args()
    if args.eval:
        torch.set_num_threads(args.threads)
        net = load_net(args.eval)
        os.makedirs(args.out, exist_ok=True)
        _, summary = evaluate(net, args.eval_episodes, args.eval_seed, args.eval_eps, args.out, args.eval_tag)
        print(json.dumps(summary))
        return
    train(args)


if __name__ == "__main__":
    main()
