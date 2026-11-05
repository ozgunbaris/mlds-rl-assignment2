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
- **Part (b) MsPacman: preliminary.** «HANDOFF_MSP»
- **Report:** `report/report.md` and `report/report.pdf` (built with `report/build.sh`) describe all of the above.
  The MsPacman sections describe the short cloud runs and must be updated after the cluster runs.

## 3. What remains

1. **Long MsPacman training on the cluster** (the main gap): 3 seeds, 4000-6000 episodes each, with the
   Mnih et al. (2015) network (`--arch nature`, the default in `scripts/nomad/make_job.sh`) and the default
   schedules of `dqn/mspacman_dqn.py` (learning starts after 50k steps, epsilon 1 -> 0.1 over 250k steps then
   -> 0.01 at 2.5M steps, learning rate 1e-4 -> 5e-5 over 5M steps, target network every 2000 updates, buffer
   500k frames). These defaults are meant for a run of about 5 million agent steps.
2. Roll out 500 games per run (the job does this), run the validation and the plots on the Mac, and replace the
   MsPacman sections of `report/report.md` with the cluster results (the current text describes the short
   1.7-hour cloud run; keep it as the "short run" comparison or remove it).
3. Rebuild `report/report.pdf` (`report/build.sh`, needs pandoc and xelatex) and push.

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

## 6. Caveats

- The replay buffer is not checkpointed (3.5 GB). `--resume` reloads the networks, optimizer and counters and
  refills the buffer for 50,000 steps before updating again.
- `results/mspacman/*/checkpoint.pt` is git-ignored (optimizer state, 20+ MB); `final_weights.pt` and the
  snapshots are kept.
- The CartPole "trained network" is the one with the best periodic greedy check (`best_weights.pt`); the network
  after the last episode collapses in most seeds. This is documented in the report as an edge case.
