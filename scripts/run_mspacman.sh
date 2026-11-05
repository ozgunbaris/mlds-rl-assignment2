#!/usr/bin/env bash
# The two MsPacman-v5 training runs in the report (Double DQN, seeds 0 and 1), run side by side on a
# 4-core CPU machine with 2 PyTorch threads each, then the 500-game roll-outs of the final networks.
# Training is resumable: re-running this script with RESUME=1 continues from checkpoint.pt.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
EPISODES=${EPISODES:-6000}
HOURS=${HOURS:-16}
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi
for S in 0 1; do
  D=results/mspacman/main_seed$S
  mkdir -p $D
  nohup $PY -m dqn.mspacman_dqn --out $D --seed $S --threads 2 --max-episodes $EPISODES \
      --time-limit-hours $HOURS $RESUME_FLAG >> $D/train.log 2>&1 &
  echo "started $D (pid $!)"
done
wait
for S in 0 1; do
  D=results/mspacman/main_seed$S
  $PY -m dqn.mspacman_dqn --eval $D/final_weights.pt --eval-episodes 500 --eval-eps 0 --out $D --threads 2 \
      >> $D/eval.log 2>&1 &
done
wait
echo "done"
