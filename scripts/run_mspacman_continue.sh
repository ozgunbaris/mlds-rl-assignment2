#!/usr/bin/env bash
# Continue the two MsPacman runs from their last checkpoints after the cloud machine restarted (it killed the
# training at episodes 1130 and 320; the last checkpoints are at episodes 1100 and 300). Each continuation lives
# in results/mspacman/cont_seed<S>, which starts as a copy of main_seed<S>; the replay buffer is refilled for
# 50,000 steps before the updates resume. HOURS is the total wall-clock budget including the first 0.92 / 0.25 h.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
S=${SEED:-0}
THREADS=${THREADS:-2}
HOURS=${HOURS:-4.0}
D=results/mspacman/cont_seed$S
$PY -m dqn.mspacman_dqn --out $D --seed $S --arch nips --threads $THREADS --max-episodes 6000 \
    --time-limit-hours $HOURS --learning-starts 20000 --eps-decay-steps 250000 --eps-final-step 1500000 \
    --lr-decay-steps 3000000 --target-update 1000 --resume >> $D/train.log 2>&1
$PY -m dqn.mspacman_dqn --eval $D/final_weights.pt --eval-episodes 500 --eval-eps 0 --out $D --threads $THREADS \
    >> $D/eval.log 2>&1
echo "done"
