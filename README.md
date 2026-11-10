# IEMS 469 Reinforcement Learning, Assignment 2: Deep Q-Network

Double DQN (neural network + replay buffer + target network) for

| Part | Game | Code | Results |
|---|---|---|---|
| (a) | CartPole-v1, discount factor 0.95 | `dqn/cartpole_dqn.py` | `results/cartpole/` |
| (b) | MsPacman-v5 from pixels, discount factor 0.99 | `dqn/mspacman_dqn.py` | `results/mspacman/` |

The written report is `report/report.pdf` (source `report/report.md`; rebuild with `report/build.sh`, which needs pandoc and xelatex).

## Setup

Python 3.12 was used. All dependencies are in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q        # 11 checks of the replay buffers, Bellman targets, preprocessing and schedules
```

`gymnasium[atari]` + `ale-py` provide `ALE/MsPacman-v5` (the ROMs ship with `ale-py >= 0.9`). Every command below is run from the repository root.

## Part (a): CartPole-v1

One run trains for 1000 episodes, checks the greedy policy on 20 separate evaluation episodes every 25 training episodes, keeps the network with the best check (`best_weights.pt`) as well as the final one (`final_weights.pt`), and rolls out 500 greedy episodes with each:

```bash
python -m dqn.cartpole_dqn --out results/cartpole/main_seed0 --seed 0 --episodes 1000 --eps-final 0.1 \
    --no-stop-when-solved --eval-periodic-episodes 20
scripts/run_cartpole.sh                                                          # all 23 runs of the report
python -m dqn.make_plots_cartpole --results results/cartpole --out results/cartpole/figures
python -m dqn.validate_cartpole --results results/cartpole --out results/cartpole/validation
```

Ablation switches of `dqn.cartpole_dqn`: `--no-double` (DQN target), `--no-target` (no target network), `--buffer-size 64` (no replay), `--truncation-terminal` (the time-limit edge case), `--gamma`, `--reward-scale`, `--lr`. Other options: `--tau` (soft target updates), `--stop-when-solved` (stop after 50 consecutive 500-step greedy evaluation episodes, armed once epsilon reached its final value).

Validation experiments (`dqn/validate_cartpole.py`, results in `results/cartpole/validation/validation.md`):

| test | what it does | how it validates |
|---|---|---|
| C1 chain MDP (`dqn/toy_chain.py`) | trains the same DQN code on a deterministic 8-state chain | learned Q must equal the exact Q* from value iteration; with a 50-step time limit, bootstrapping must give Q = 1/(1-gamma) = 20, and treating the limit as terminal must give the predicted biased value (14.49 with a squared loss, 19.59 with the Huber loss) |
| C2 Q scale | runs with gamma 0.9 / 0.99 and with rewards x2 and x(-1) | a policy that never fails has Q = r/(1-gamma) exactly: 10, 20, 100, 40; with negative rewards the agent must learn to end episodes as fast as possible |
| C3 overestimation | greedy roll-outs continued to 2000 steps | predicted Q(s, a) against the realised discounted return |
| C4 first 10 steps | table of state, Q-values, action, reward, Bellman target | the step-by-step mechanics |
| C5 mirror symmetry | Q(s, left) against Q(-s, right) on visited states | the dynamics are symmetric, so a correct Q-function must be too |
| C6 value and policy maps | max_a Q and the greedy action over (theta, theta_dot) during training, and over (x, x_dot) | the non-standard plot; shows how the policy forms and where the value drops |
| C7 ablations | no target network, no replay, no Double DQN, time limit as terminal | each required component is removed and the effect is measured over seeds |
| C8 exploration noise (`--only exploration_noise`) | kept networks rolled out greedily and with epsilon 0.1; greedy checks made at the final epsilon | separates the two causes of the low training reward: random actions and a changing greedy policy |

## Part (b): MsPacman-v5

The reported MsPacman results are three seeds of 6,000 games each, trained on one GPU of the lab node mlds-deepdish5 (`results/mspacman/cluster_seed{0,1,2}`, figures in `results/mspacman/figures_cluster/`, validation in `results/mspacman/validation_cluster_seed*/`). Their final networks average 2,314.3, 2,207.0 and 1,503.0 points over 500 greedy test games.

```bash
python -m dqn.mspacman_dqn --out results/mspacman/cluster_seed0 --seed 0 --arch nature --threads 1 --max-episodes 6000
python -m dqn.mspacman_dqn --out results/mspacman/cluster_seed0 --resume      # continue after an interruption
python -m dqn.mspacman_dqn --eval results/mspacman/cluster_seed0/final_weights.pt --eval-episodes 500 --eval-eps 0 --out results/mspacman/cluster_seed0
python -m dqn.make_plots_mspacman --runs results/mspacman/cluster_seed0 results/mspacman/cluster_seed1 results/mspacman/cluster_seed2 --out results/mspacman/figures_cluster
python -m dqn.validate_mspacman --run results/mspacman/cluster_seed0 --out results/mspacman/validation_cluster_seed0
```

Earlier runs on a 4-core cloud machine without a GPU used five shortened settings (`results/mspacman/main_seed*`, `cont_seed*`; scripts `scripts/run_mspacman.sh`, `scripts/run_mspacman_seed1.sh`, `scripts/run_mspacman_continue.sh`). The report describes them in one paragraph. `results/mspacman/gpu_run1_partial/` holds the logs of a first GPU run on mlds-deepdish4 whose results were deleted by the cluster before they were copied.

Main options: `--arch {nips,nature}` (Mnih et al. 2013 / 2015 network), `--device {auto,cpu,cuda}`, `--threads`, `--max-episodes`, `--time-limit-hours`, `--buffer-size`, `--lr`, `--lr-final`, `--eps-*`, `--target-update`, `--no-double`, `--loss {huber,mse}`, `--reward-clip`, `--no-life-loss-terminal`. Training writes a checkpoint every 50 episodes and on Ctrl-C; `--resume` continues from it (the replay buffer is refilled for 50,000 steps first).

Validation experiments (`dqn/validate_mspacman.py`, results in `results/mspacman/validation_cluster_seed*/validation.md`):

| test | what it does | how it validates |
|---|---|---|
| P1 preprocessing | raw frame, preprocessed frame, 4-frame state; counts the pixels removed as Ms. Pac-Man | the given function is used unchanged; its unsigned-integer arithmetic is shown to give the signed result |
| P2 random policy | 100 games with uniformly random actions | the score every learning curve is compared with |
| P3 snapshots | 20 greedy games for every saved network | learning shows as rising scores of the saved networks, free of exploration noise |
| P4 calibration | predicted Q(s_t, a_t) against the realised discounted clipped return, cut at lost lives as in training | the network must predict the quantity it is trained on |
| P5 events | average max_a Q around lost lives, power pellets and eaten ghosts | the non-standard plot; the value must fall before a death and rise after a power pellet |
| P6 first 10 steps | table of Q-values, actions, rewards, targets | the step-by-step mechanics |
| P7 determinism and frozen start | sticky actions off: the same game every time; the first steps where no action changes the emulator | with no stochasticity the outcome must repeat; where actions do not matter the action gap should be small |
| P8 positions | heat map of Ms. Pac-Man's position early and late in training | a second non-standard plot: the trained agent covers the maze |
| P9 score composition | pellets, power pellets, ghosts and fruit per game | effect of reward clipping |
| P10 input coverage | finds the input rows and columns with zero gradient, overwrites them with noise, redraws the lives indicator | the convolution sizes drop preprocessed rows 84-87 and columns 76-79, so Q must not change; the lives indicator (rows 86-87) is therefore not read |
| P11 frozen phases | measures the game start (66 steps), death animation and restart (29 steps) with the emulator-clone test; trains a corridor of identical observations (`run_aliased_corridor` in `dqn/toy_chain.py`) | the Huber fixed point for N steps that look the same, 1 / ((N - 1)(1 - gamma)), predicts the value the network gives these phases (1.54 and 3.57) |

### Running on the lab cluster (Nomad)

`scripts/nomad/make_job.sh` writes a Nomad batch job that embeds the code as a base64 tarball, installs the dependencies in a `python:3.12-slim` container, runs the unit tests, trains several seeds in parallel, rolls out 500 games per run and packs the results; `scripts/fetch_cluster_results.sh` copies them back. See `HANDOFF.md` for the exact procedure.

```bash
GPU=1 HOURS=16 SEEDS="0 1 2" ARCH=nature THREADS=1 CORES=6 MEMORY=20480 EPISODES=6000 PREFIX=cluster \
    HOLD_HOURS=12 JOB_NAME=mspacman-dqn-gpu scripts/nomad/make_job.sh    # the job of the reported runs
scripts/fetch_cluster_results.sh <alloc-id> mspacman-dqn-gpu           # copy the packed results back
```

Nomad on this cluster deletes a finished job and its files within hours. The job therefore keeps its task alive after the test games until `/alloc/data/FETCHED` exists (or `HOLD_HOURS` pass); release it with `nomad alloc exec <alloc-id> touch /alloc/data/FETCHED` after fetching.

## Repository layout

```
dqn/core.py                  Bellman targets (DQN / Double DQN), the update step, target-network sync
dqn/replay.py                VectorReplay (CartPole) and FrameStackReplay (MsPacman, frames stored once)
dqn/networks.py              MLP and convolutional Q-networks
dqn/cartpole_dqn.py          part (a): training, model selection, roll-outs
dqn/mspacman_dqn.py          part (b): preprocessing, training, checkpoints, roll-outs
dqn/toy_chain.py             chain MDP with known Q* (validation C1)
dqn/make_plots_*.py          figures and summaries
dqn/validate_*.py            validation experiments
viz.py                       plotting style
tests/test_dqn.py            pytest checks
scripts/                     experiment scripts, Nomad job generator, cluster fetch
results/                     logs, weights, roll-outs, figures, validation outputs
report/                      report.md, build.py (figure numbering + pandoc), report.pdf
```
