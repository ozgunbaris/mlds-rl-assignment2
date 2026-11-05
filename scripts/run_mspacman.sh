#!/usr/bin/env bash
# The MsPacman-v5 run in the report: Double DQN with the Mnih et al. (2013) network, trained for a fixed
# wall-clock budget on a 4-core CPU machine (3 PyTorch threads; the 4th core ran the CartPole experiments),
# then 500 greedy roll-out games of the final network.
# Training is resumable: re-running this script with RESUME=1 continues from checkpoint.pt.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
HOURS=${HOURS:-1.7}
SEED=${SEED:-0}
D=results/mspacman/main_seed$SEED
mkdir -p $D
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi
$PY -m dqn.mspacman_dqn --out $D --seed $SEED --arch nips --threads 3 --max-episodes 6000 \
    --time-limit-hours $HOURS --learning-starts 20000 --eps-decay-steps 250000 --eps-final-step 1500000 \
    --lr-decay-steps 3000000 --target-update 1000 $RESUME_FLAG >> $D/train.log 2>&1
$PY -m dqn.mspacman_dqn --eval $D/final_weights.pt --eval-episodes 500 --eval-eps 0 --out $D --threads 4 \
    >> $D/eval.log 2>&1
echo "done"
