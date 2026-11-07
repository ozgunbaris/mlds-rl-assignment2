# Handoff: IEMS 469 Assignment 2 (Deep Q-Network), continue the MsPacman training on the lab cluster

This file is written for the next Claude Code session (running on Özgün's Mac, which can reach the lab
cluster). The previous session ran in a cloud container with 4 CPU cores and no GPU, and its time ran out on
2026-11-05 at 07:59 UTC. Everything it produced is in this repository.

## 1. The assignment (summary of the handout `mlds_assignment2.pdf`)

- Due **November 11, 2026, 11:59 pm**. Submit on Canvas a PDF report with: GitHub username, repository name and
  link to the **public** repository; instructions to run the code; labeled plots for each part; answers to all
  discussion questions. All code must be in the repository.
- (a) **CartPole-v1**, discount factor 0.95; the model should solve the environment (pole upright for 500 steps).
- (b) **MsPacman-v5** from images, discount factor 0.99, 9 actions, the given `preprocess_observation()`
  (88 x 80 int8). Graded on convergence and performance: "whether your model shows learning/improvement over the
  training process"; "the model may take a long time to train (4000-6000 episodes)". Suggested: Huber loss,
  learning-rate scheduler, epsilon scheduler.
- For both: (i) plot the maximum Q-values versus training episodes; (ii) plot episode rewards versus training
  episodes with the moving average of the last 100 episodes; (iii) after training, roll out 500 episodes with the
  trained network, plot the histogram of episode rewards, report mean and standard deviation.
- For each part also: edge cases and how they were handled; tests that validate correctness (not unit tests;
  quantitative experiments such as changing parameters, making the process deterministic, tracing the first 10
  steps); at least one non-standard plot; directions for running the validation; and "Given an additional 3 months,
  what other validation tests and experiments would you perform?"
- The DQN must use a neural network, a replay buffer and a target network. "You should use a single GPU card to
  run your program on deepdish."
- Grading: implementation 20, plots and discussion 20, performance 20, edge cases and tests 30, report formatting
  and readability 10.

## 2. State at handoff

Everything below is pushed to `main`.

- **Code** (complete, `python -m pytest tests -q` passes 11 checks): `dqn/` (shared Double DQN core, replay buffers,
  networks, CartPole and MsPacman trainers, chain-MDP test problem, plotting and validation scripts), `scripts/`
  (experiment loops, Nomad job generator `scripts/nomad/make_job.sh` + `scripts/nomad/run.sh`, fetch script).
- **Part (a) CartPole: done.** 23 runs in `results/cartpole/` (5 main seeds; ablations: no target network x3,
  no replay x3, DQN target x3, time limit as terminal x2, gamma 0.9 x2, gamma 0.99 x2, reward x2, reward x(-1),
  learning rate 0), validation in `results/cartpole/validation/`, figures in `results/cartpole/figures/`,
  preliminary tuning runs in `results/cartpole/preliminary/`. The report section is written. Kept networks:
  500.0, 495.4, 455.8, 500.0, 500.0 (3 of 5 seeds perfect over 500 roll-outs).
- **Part (b) MsPacman: preliminary.** Two runs of the Mnih et al. (2013) network on 4 CPU cores. A restart of the cloud machine stopped them; the reported networks are the last checkpoints: seed 0 after 1,100 games (0.92 h, 670,642 agent steps), 500 greedy games 968.8 +- 362.0 (random policy 245.1 +- 50.4); seed 1 after 300 games (0.25 h), 141.2 (its greedy policy is still stuck: 494 of 500 games score exactly 140). Validation P1-P9 for seed 0 is in `results/mspacman/validation/`, figures in `results/mspacman/figures/`, and the report text was generated from these files by `report/fill_mspacman.py`.
  Continued training (several more machine restarts, each resumed from the last checkpoint): seed 0 reached game 1,500 (1.34 h, 945,654 steps; `results/mspacman/cont_seed0/weights_ep01500.pt`), 500 greedy games 1074.9 +- 385.8, which the report now quotes as "Continued training"; seed 1 reached game 700 (`weights_ep00700.pt`, not evaluated). Later: seed 0 game 2,150 (1.97 h) scores 997.7 +- 366.5 (worse than game 1,500; quoted in the report), seed 1 game 1,250 (`weights_ep01250.pt`, not evaluated).
  Continuation runs first resumed from the two checkpoints at 13:13 UTC on 2026-11-05 in `results/mspacman/cont_seed0` (2 threads, 4 h total budget) and `results/mspacman/cont_seed1` (1 thread, 3 h), via `scripts/run_mspacman_continue.sh`; each evaluates 500 games when it ends. Whatever they reached is pushed; if a later session finds them complete (an `eval_summary_eps0.json` in the folder), it can rerun `validate_mspacman` / `make_plots_mspacman` on them and swap them into the report (the text describes main_seed0/main_seed1).
- **Report:** `report/report.md` and `report/report.pdf` (built with `report/build.sh`) describe all of the above.
  Superseded on 2026-11-07: part (b) now reports the deepdish5 GPU runs (section 6f below).

## 3. What remains

Status on 2026-11-07 17:30 UTC (observed): the long MsPacman training, the 500-game roll-outs, validation P1 to P9,
the figures, the part (b) rewrite and the PDF rebuild are done (section 6f). What remains is to review the report,
commit and push (ask the user first), and submit the PDF by 2026-11-11 23:59.

## 4. Cluster procedure

Follow the user's rules file `kranj_nomad_rules.md` (Özgün has it; the previous session received it as an
upload). The essentials:

- Never run compute on kranj; only `nomad` commands and file edits there. Ask the user before any other command.
- Never read or print `~/.ssh/id_ed25519` or its passphrase. Always pass `-o BatchMode=yes` to ssh and scp.
  The user must have the Northwestern GlobalProtect VPN on and the key loaded:
  `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` (check with `ssh-add -l`).
- If ssh asks for a passphrase or fails with "Permission denied", stop and tell the user. Do not retry failed
  ssh or nomad commands in a loop.
- **Ask the user before every `nomad job run`, `nomad job stop` and `-purge`.** Touch only jobs the user names.
- Use node `mlds-deepdish5` unless the user names another. Request only the cores and memory the job needs.
- Observed on 2026-10-07: mlds-deepdish5 has 48 logical CPUs at 5.36 GHz and 94 GiB memory, and **Nomad
  detects no GPU on it** (node meta says `gpu = nvidia`). A GPU run therefore needs a node the user names and
  whose GPU Nomad can see; `GPU=1 scripts/nomad/make_job.sh` adds `device "nvidia/gpu" { count = 1 }` and
  `--device cuda`, but this path has **not been tested** (the cloud machine had no GPU). Start it with a short
  `HOURS=0.2` job first.

Commands (from the repository root on the Mac):

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests -q                                   # must print 11 passed
# CPU job, 3 seeds x 8 threads on mlds-deepdish5 (about 5 GB of memory per seed for the replay buffer)
HOURS=10 SEEDS="0 1 2" ARCH=nature THREADS=8 CORES=24 MEMORY=16384 JOB_NAME=mspacman-dqn scripts/nomad/make_job.sh
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "test -e ~/mspacman-dqn.nomad && echo EXISTS || echo FREE"
scp -o ConnectTimeout=10 -o BatchMode=yes scripts/nomad/mspacman-dqn.nomad obkir@kranj.iems.private:~/mspacman-dqn.nomad
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad job validate mspacman-dqn.nomad && nomad job plan mspacman-dqn.nomad"
# ask the user, then:
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad job run -detach mspacman-dqn.nomad"
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad job status mspacman-dqn"
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad alloc logs <alloc-id> run | tail -20"
scripts/fetch_cluster_results.sh <alloc-id> mspacman-dqn      # any time: results are packed every 10 minutes
```

Speed reference (measured in the cloud container, Intel CPU, per run): Nature network 204 agent steps/s with
4 threads alone, about 140 with 2 threads when two runs share 4 cores; Mnih 2013 network about 270 with 3
threads. A trained agent's game lasts roughly 600 to 1200 agent steps, so 5000 games are about 4 to 5 million
steps. On mlds-deepdish5 (faster cores, 8 threads per run) expect a few hundred steps/s per run; 10 hours should
reach several million steps per seed, but measure it from the first `ep ... sps` lines in `train.log` and adjust
`HOURS` (the run stops at `--max-episodes 6000` or the time limit, whichever comes first, and still evaluates).

## 5. After the job: post-processing and report

```bash
for S in 0 1 2; do python -m dqn.validate_mspacman --run results/mspacman/cluster_seed$S --out results/mspacman/validation_cluster_seed$S; done
python -m dqn.make_plots_mspacman --runs results/mspacman/cluster_seed0 results/mspacman/cluster_seed1 results/mspacman/cluster_seed2 --out results/mspacman/figures_cluster
report/build.sh
git add -A && git commit -m "MsPacman cluster runs" && git push
```

In `report/report.md`: update the MsPacman setup paragraph (network, schedules, compute), the results paragraph,
the three required figures (point them to `figures_cluster/mspacman_cluster_seedK_*.png`; figure file stems must
be unique because `report/build.py` numbers figures by file stem), the edge-case and test numbers that come from
`validation.md`, and the "Compute" paragraph near the top. Keep every statement tied to a number in a results file.

## 6. Lessons from the cloud runs (use them when choosing the cluster settings)

- The Mnih 2013 network (`--arch nips`) learned steadily on CPU: the 100-game average score of seed 0 rose from 277
  (random play: 245) to about 900 by game 950 (0.8 h, 560k agent steps), with the Q-values still rising, i.e. far
  from converged. On the cluster the Nature network (`--arch nature`, default of `make_job.sh`) with the default
  (long-run) schedules is the better choice if the run gets at least ~4M steps; otherwise keep `--arch nips`.
- Early greedy policies get stuck (identical scores in all 10 periodic evaluation games at episodes 100 and 200);
  sticky actions help a deterministic greedy policy out of loops. This is in the edge cases of the report.
- The first ~66 agent steps of every game are a frozen intro (no action changes the emulator); test P7 measures it.
- `dqn/validate_mspacman.py` was dry-run on an intermediate network and works end to end (about 1 minute with
  `--games 3`, several minutes with the default `--games 20`).
- Never edit a shell script while a launcher is executing it (bash reads scripts incrementally); copy it first.

## 6b. Mac session, 2026-11-05 17:15 UTC (observed)

- The Mac reaches kranj (VPN on, key in the agent). `python -m pytest tests -q` passes 11 checks in the Mac venv.
- Nomad now sees GPUs on mlds-deepdish5: 1 RTX PRO 5000 Blackwell (48 GB) and 2 RTX 5060 Ti (16 GB), driver 615.71.09.
  The "no GPU" note in section 4 is out of date.
- All five nodes had their cores reserved at 17:15 UTC. The user said mlds-deepdish5 frees up on the evening of 2026-11-06.
- Prepared job: `scripts/nomad/mspacman-dqn-gpu.nomad` (also copied to `~/mspacman-dqn-gpu.nomad` on kranj), generated with
  `GPU=1 HOURS=16 SEEDS="0 1 2" ARCH=nature THREADS=2 CORES=9 MEMORY=20480 EPISODES=6000 PREFIX=cluster JOB_NAME=mspacman-dqn-gpu`.
  `nomad job validate` passes; `nomad job plan` fails only on the "cores" dimension of mlds-deepdish5 (the GPU request fits).
- Mac speed for the Nature network (20-second averages after updates began): 237 agent steps/s on MPS (`--device mps`),
  35 on the CPU with 4 threads.

## 6c. Cluster run submitted on 2026-11-06 (observed)

- At 18:55 UTC mlds-deepdish5 was still full. mlds-deepdish4 had about 6 free cores (21,850 MHz) and 41 GiB.
- With the user's approval the job `mspacman-dqn-gpu-dd4` was submitted at about 19:00 UTC: allocation `abec26f3` on
  mlds-deepdish4, 1 RTX A6000 (shared with another process that held 24 of its 48 GB), 6 cores, 20 GB.
- Generated with `NODE=mlds-deepdish4 GPU=1 HOURS=16 SEEDS="0 1 2" ARCH=nature THREADS=2 CORES=6 MEMORY=20480 EPISODES=6000
  PREFIX=cluster JOB_NAME=mspacman-dqn-gpu-dd4 scripts/nomad/make_job.sh` (code revision ea6247a).
- Results go to `results/mspacman/cluster_seed{0,1,2}`. Fetch with `scripts/fetch_cluster_results.sh abec26f3 mspacman-dqn-gpu-dd4`.
  The job packs the results every 10 minutes, so fetching works while it runs.
- The deepdish5 job file `~/mspacman-dqn-gpu.nomad` was not submitted.
- Job log: PyTorch 2.15.0+cu132, `cuda True`, 11 tests passed, seeds started 19:00 UTC. Speed per seed: 202 to 230 agent
  steps/s before updates, 96 to 113 steps/s with updates (19:05 to 19:10 UTC). At about 105 steps/s the 16-hour limit
  allows about 6.0M steps per seed; the 6,000-game limit is expected to end the run first (guessed: games of 800 to 900
  steps on average give 4.8M to 5.4M steps, 13 to 15 hours).
- Report decision (user, 2026-11-06): part (b) reports the three GPU seeds. The cloud CPU MsPacman runs (`main_seed*`,
  `cont_seed*`) stay in the report as one short paragraph that names their five differences from the GPU job:
  network Mnih 2013 (GPU: 2015), updates start at 20,000 steps (GPU: 50,000), epsilon reaches 0.01 at 1.5M steps
  (GPU: 2.5M), learning-rate decay over 3M steps (GPU: 5M), target copy every 1,000 updates (GPU: 2,000).
  The CartPole runs stay as they are: CartPole is CPU-only and uses the same `core.py` and `replay.py` as the GPU job.

## 6d. The deepdish4 run was lost (observed 2026-11-07 13:55 UTC)

- At 04:54 UTC the run was healthy: about 5,100 games per seed, 3.79M steps, 100-game averages 1,801 to 1,918.
- At 13:55 UTC Nomad had no job `mspacman-dqn-gpu-dd4`, no allocation `abec26f3` and no evaluations for it. Nomad had
  removed the finished job and its files (the results tarball lived in the allocation directory). The results were
  never fetched, so the final weights and the 500 test games are lost.
- Kept: the logs downloaded at 04:54 UTC (`episodes.csv`, `eval_periodic.csv`, `heldout_q.csv`, up to games 5,105 to
  5,168) and the first 2 hours of `train.log`, in `results/mspacman/gpu_run1_partial/seed{0,1,2}`.
- Fix: `scripts/nomad/run.sh` now keeps the task alive after the final pack until `/alloc/data/FETCHED` exists or
  `HOLD_HOURS` (default 12, set in `make_job.sh`) have passed. After fetching, release it with
  `nomad alloc exec <alloc-id> touch /alloc/data/FETCHED` (ask the user first). Also fetch snapshots during the run.
- Rerun prepared: `scripts/nomad/mspacman-dqn-gpu.nomad` (mlds-deepdish5, same settings as the deepdish4 run but
  9 cores, hold 12 h), copied to `~/mspacman-dqn-gpu.nomad` on kranj; `nomad job plan` places it (deepdish5 was idle).

## 6e. Rerun on mlds-deepdish5 (submitted 2026-11-07 14:10 UTC with the user's approval)

- Job `mspacman-dqn-gpu`, allocation `534e2b63` on mlds-deepdish5. Generated with `GPU=1 HOURS=16 SEEDS="0 1 2"
  ARCH=nature THREADS=1 CORES=6 MEMORY=20480 EPISODES=6000 PREFIX=cluster HOLD_HOURS=12 JOB_NAME=mspacman-dqn-gpu
  scripts/nomad/make_job.sh` (code revision ea6247a; the training code is the same as the lost deepdish4 run).
- Why 1 thread and 2 cores per seed: on the Mac GPU one seed ran at 267 / 248 / 224 steps/s with 1 / 2 / 4 PyTorch
  threads; a profile put 61 % of the time in `dqn_update` (15 ms per update) and 24 % in the action forward pass
  (1.3 ms), 6.6 % in the emulator and 1.4 % in buffer sampling. The loop is one sequential Python process per seed
  that waits on many small GPU operations, so more CPU cores do not make a seed faster.
- Job log: PyTorch 2.15.0+cu132, `cuda True`, 11 tests passed, seeds started 14:11 UTC. Speed per seed with updates:
  563 to 573 agent steps/s (14:12 to 14:13 UTC), 5.3 times the deepdish4 run (107). At this rate 6,000 games
  (about 4.6M steps, from the deepdish4 logs) take about 2.3 hours.
- A fetch loop on the Mac (`caffeinate -i`, scratchpad `fetch_loop.sh`) runs `scripts/fetch_cluster_results.sh
  534e2b63 mspacman-dqn-gpu` every 2 hours and once more when the job log shows `HOLD up to`. After the final fetch,
  release the hold with `nomad alloc exec 534e2b63 touch /alloc/data/FETCHED` (the user approved this release).

## 6f. Results of the deepdish5 run (observed 2026-11-07)

- Training ended 16:30 to 16:45 UTC. The 500 test games per seed took about 12 minutes. The final fetch ran at
  17:12 UTC, and the hold was released at 17:13 UTC (`RELEASED after 7 min` in the job log).
- Another job reserved 40 of deepdish5's 48 cores and 68 GiB from about 16:00 UTC. Our seeds then slowed from about
  578 to 363 to 463 steps/s (our 6 reserved cores stayed ours). Training took 2.32 to 2.56 hours per seed.
- Results in `results/mspacman/cluster_seed{0,1,2}` (6,000 games, 4.34M to 4.71M steps, final weights, 12 snapshots,
  500 test games): test means 2,314.3 (std 381.6), 2,207.0 (630.5), 1,503.0 (319.9). Figures in
  `results/mspacman/figures_cluster/`, validation in `results/mspacman/validation_cluster_seed*/`.
- Seed 2 is lower because it eats few ghosts (0.4 per game against 2.2 and 2.6; P9). On deepdish4 the same seed
  had a 100-game average of 1,754 at game 5,000, against 1,468 here. The two runs were identical until updates
  began (games 100 to 121), so the seed gap is within run-to-run variation.
- `dqn/make_plots_mspacman.py`: the seed-comparison title now uses the number of seeds, and the histogram title was
  shortened so it fits the figure.
- Report part (b) rewritten for the GPU runs, with one paragraph on the cloud CPU runs and their 5 setting
  differences (user decision). The compute paragraph, run commands, date and README were updated. The PDF has
  21 pages and 20 figures.
- Size note: the three `cluster_seed*/snapshots/` folders hold 67 MB each (Nature network, 5.8 MB per file). Decide
  with the user whether to commit all snapshots or only `final_weights.pt` plus a few.

## 7. Caveats

- The replay buffer is not checkpointed (3.5 GB). `--resume` reloads the networks, optimizer and counters and
  refills the buffer for 50,000 steps before updating again.
- `results/mspacman/*/checkpoint.pt` is git-ignored (optimizer state, 20+ MB); `final_weights.pt` and the
  snapshots are kept. The live logs of the continuation runs (`cont_seed*/train.log`, `episodes.csv`, `heldout_q.csv`,
  `eval_periodic.csv`) are git-ignored while the runs go; copies are committed in `cont_seed*/log_snapshot/`.
- The CartPole "trained network" is the one with the best periodic greedy check (`best_weights.pt`); the network
  after the last episode collapses in most seeds. This is documented in the report as an edge case.
