---
title: "IEMS 469 Reinforcement Learning, Assignment 2: Deep Q-Network"
author: "Özgün Barış Kır, student number 3676663, GitHub username ozgunbaris"
date: "November 5, 2026"
geometry: margin=0.8in
fontsize: 11pt
colorlinks: true
---

# Repository and how to run the code

Özgün Barış Kır (3676663). GitHub username: **ozgunbaris**. Repository: **mlds-rl-assignment2**, <https://github.com/ozgunbaris/mlds-rl-assignment2>.

All code is in the repository and `README.md` lists every command. Short version (Python 3.12, run from the repository root):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q                                       # 11 mechanical checks
python -m dqn.cartpole_dqn --out results/cartpole/main_seed0 --seed 0 \
    --episodes 1000 --eps-final 0.1 --no-stop-when-solved \
    --eval-periodic-episodes 20                                 # (a), one seed
scripts/run_cartpole.sh                                         # (a), all 24 runs
python -m dqn.make_plots_cartpole                               # (a) figures
python -m dqn.validate_cartpole                                 # (a) validation
scripts/run_mspacman.sh                                         # (b) training + 500 games
python -m dqn.make_plots_mspacman --runs results/mspacman/main_seed0
python -m dqn.validate_mspacman --run results/mspacman/main_seed0
```

Every `results/*` folder holds the logs (`episodes.csv`, `heldout_q.csv`, `eval_periodic.csv`), the network weights, the roll-out rewards (`eval_rewards_*.npy`, `eval_summary_*.json`), the figures, and `validation.md` with the numbers quoted below.

**Compute.** All runs were made on a cloud machine with 4 CPU cores and no GPU, inside a fixed time window of about three hours. MsPacman therefore trained for a fixed wall-clock budget of 1.7 hours on 3 CPU threads (one run), while the 4th core ran the 24 CartPole runs one after another. The consequences for MsPacman are discussed in its section.

# The DQN implementation (shared by both games)

`dqn/core.py`, `dqn/replay.py` and `dqn/networks.py` are shared. One training step does the following.

1. Act $\epsilon$-greedily on the online network $Q_\theta$: a uniformly random action with probability $\epsilon$, otherwise $\arg\max_a Q_\theta(s, a)$. The exploration rate follows a piecewise-linear schedule (an $\epsilon$ scheduler).
2. Store the transition $(s, a, r, s', d)$ in the **replay buffer**. $d$ is the bootstrap mask: it is 1 only when the episode really ended (pole fell, cart left the track, game over, or a lost life in MsPacman). A time-limit truncation is not terminal: $s'$ is a real state and its value is bootstrapped (edge cases below).
3. Sample a uniform minibatch from the buffer and form the **Double DQN** target with the **target network** $Q_{\bar\theta}$:
$$y = r + \gamma\,(1 - d)\,Q_{\bar\theta}\big(s', \arg\max_{a'} Q_\theta(s', a')\big).$$
The online network chooses the next action and the target network evaluates it, which removes most of the upward bias of $\max_{a'} Q_{\bar\theta}(s', a')$ (van Hasselt et al. 2016). `--no-double` gives the original DQN target.
4. Take one Adam step on the **Huber loss** between $Q_\theta(s, a)$ and $y$ ($y$ carries no gradient), with the gradient norm clipped at 10. The learning rate decays linearly (a learning-rate scheduler).
5. Copy $\theta$ into $\bar\theta$ every fixed number of updates.

The three required components are therefore a neural network ($Q_\theta$), a replay buffer (`VectorReplay`, `FrameStackReplay`) and a target network ($Q_{\bar\theta}$). Ablations that remove each of them are reported in part (a).

**Maximum Q-value plots.** For plot (i) every training episode logs, from the forward passes the agent already makes to act, the mean over the states it visited of $\max_a Q_\theta(s, a)$, the largest such value in the episode and the value of the first state. Each run also tracks the average of $\max_a Q_\theta$ over a fixed set of held-out states collected with a random policy before training (Mnih et al. 2015, Fig. 2), which does not move with the agent's own state distribution.

# Part (a): CartPole-v1

## Setup

The Q-network is a 4-128-128-2 MLP with ReLU. Its inputs are divided by $(2.4, 3, 0.21, 3.5)$, the cart-position and pole-angle failure limits and typical velocities, so that all four are of order 1. Discount factor $\gamma = 0.95$ as prescribed. Adam with learning rate $5 \times 10^{-4}$ decaying linearly to $5 \times 10^{-5}$ over the 1000 training episodes; minibatch 64; replay buffer of 50,000 transitions; updates start after 1000 steps; one update per environment step; the target network is refreshed every 500 updates. $\epsilon$ decays linearly from 1 to 0.1 between steps 1,000 and 21,000 and stays at 0.1.

Every 25 training episodes the greedy policy plays 20 evaluation episodes on a separate environment instance. The network with the best such score (ties go to the later network) is kept as the trained model; the reasons for this choice and the behaviour of the network after the last episode are part of the edge cases. After training, 500 new greedy episodes (new seeds) are rolled out with the kept network and with the final network. Five seeds were run.

## Results

«CARTPOLE_RESULTS»

![(a)(i) CartPole-v1, seed «CP_SEED»: maximum Q-value per training episode. The dashed line is $1/(1-\gamma) = 20$, the exact value of a policy that never fails.](../results/cartpole/figures/per_run/cartpole_main_seed«CP_SEED»_maxq.png){ width=78% }

![(a)(ii) CartPole-v1, seed «CP_SEED»: episode reward (with $\epsilon$-greedy exploration) and its 100-episode moving average.](../results/cartpole/figures/per_run/cartpole_main_seed«CP_SEED»_training.png){ width=78% }

![(a)(iii) CartPole-v1, seed «CP_SEED»: 500 greedy roll-out episodes of the trained (kept) network.](../results/cartpole/figures/per_run/cartpole_main_seed«CP_SEED»_hist_selected.png){ width=62% }

![CartPole-v1: 100-episode moving-average reward of all five seeds.](../results/cartpole/figures/seeds_main.png){ width=78% }

## Edge cases (part a)

«CARTPOLE_EDGE»

## Tests that validate correctness (part a)

«CARTPOLE_TESTS»

## With three more months (part a)

- Repeat every configuration with 20 or more seeds and report confidence intervals for the probability that the kept network solves the task, for the episode at which it is found, and for the size of the collapse after it.
- Measure the action gap $Q(s, a_1) - Q(s, a_2)$ along training and test directly whether the collapses start when the gap falls below the noise of the updates, as argued above. Then test the remedies that act on it: prioritised replay of rare failure transitions, a replay buffer that keeps a fixed share of old episodes, a larger final $\epsilon$, a smaller final learning rate, and the advantage-learning operator that widens the gap.
- Compare the stopping or selection rule with the alternatives (fixed budget, stopping when the moving average reaches 475, selection on 100 instead of 20 evaluation episodes) by their false-positive rate on 500 fresh episodes.
- Test the learned policy beyond its training distribution: random initial states far from upright, pushes during the episode, changed pole length and mass, and the time limit raised to 5000 steps.
- Check $\gamma = 0.95$ against $\gamma = 0.99$ at equal compute: the longer horizon should remove the cart-position failures described in the edge cases.

# Part (b): MsPacman-v5

## Setup

`ALE/MsPacman-v5` with its defaults: frame skip 4, sticky actions with probability 0.25 (the previous action is repeated regardless of the agent's choice), 9 actions, at most 108,000 frames per game, 3 lives. Every frame goes through the assignment's `preprocess_observation()` unchanged (88 x 80, int8). The state is the stack of the last 4 preprocessed frames, which shows the direction in which Ms. Pac-Man and the ghosts move; after a reset the stack holds four copies of the first frame. The network divides the input by 128.

The network is the original DQN network of Mnih et al. (2013): convolution 8x8 stride 4 (16 filters), convolution 4x4 stride 2 (32), a 256-unit layer and 9 outputs. The larger Mnih et al. (2015) network is implemented as well (`--arch nature`), but on this CPU machine it trained at 140 steps per second against about 300 for the small one, so the small network gets about twice as many updates in the same time.

$\gamma = 0.99$ as prescribed. Adam (learning rate $10^{-4}$ decaying linearly towards $5 \times 10^{-5}$ over 3 million steps, $\epsilon_{\text{Adam}} = 1.5 \times 10^{-4}$), Huber loss, gradient-norm clipping at 10, minibatch 32, one update every 4 agent steps, target network copied every 1000 updates (4000 steps), replay buffer of 500,000 frames, updates start after 20,000 random steps. $\epsilon$ decays from 1 to 0.1 over the next 250,000 steps and then to 0.01 at step 1.5 million. Rewards are clipped to $[-1, 1]$ inside the Bellman target only, and a lost life ends the bootstrap (but not the game); both are edge cases below. All reported scores are raw game scores of complete games.

## Results

«MSPACMAN_RESULTS»

![(b)(i) MsPacman-v5: maximum Q-value per training episode (clipped-reward units).](../results/mspacman/figures/mspacman_main_seed0_maxq.png){ width=78% }

![(b)(ii) MsPacman-v5: score per training episode and its 100-episode moving average.](../results/mspacman/figures/mspacman_main_seed0_training.png){ width=78% }

![(b)(iii) MsPacman-v5: 500 greedy roll-out games of the final network.](../results/mspacman/figures/mspacman_main_seed0_hist.png){ width=62% }

![MsPacman-v5: greedy evaluation games (10 every 100 training episodes) next to the training score.](../results/mspacman/figures/mspacman_eval_periodic.png){ width=78% }

## Edge cases (part b)

«MSPACMAN_EDGE»

## Tests that validate correctness (part b)

«MSPACMAN_TESTS»

## With three more months (part b)

- Train for the full 4000 to 6000 episodes (10 to 50 million frames) with several seeds on a GPU, and report the learning curves with confidence bands; the run here is one seed and a short budget.
- Ablate each design choice on MsPacman at equal compute: reward clipping against a log or Pop-Art normalised reward (clipping makes a ghost worth as much as a pellet), life loss as terminal or not, Double against plain DQN, the 2013 against the 2015 network, the frame stack against a max-pooled pair of frames.
- Add the standard improvements one at a time (prioritised replay, dueling network, n-step returns, noisy networks) and measure their effect on the same seeds.
- Extend the calibration test: compare $Q$ with Monte Carlo returns state by state with many roll-outs from cloned emulator states, so that the overestimation can be measured without the noise of a single trajectory.
- Probe what the network has learned with saliency maps and with controlled states (ghost placed next to Ms. Pac-Man, power pellet just eaten), and test robustness to sticky-action probability, frame skip and the other mazes of the game.
