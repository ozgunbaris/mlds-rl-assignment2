#!/usr/bin/env bash
# Second MsPacman-v5 seed, started once the CartPole runs had freed a core: same settings as
# scripts/run_mspacman.sh but 1 PyTorch thread and a shorter wall-clock budget, then 500 greedy games.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
HOURS=${HOURS:-0.95}
D=results/mspacman/main_seed1
mkdir -p $D
$PY -m dqn.mspacman_dqn --out $D --seed 1 --arch nips --threads 1 --max-episodes 6000 \
    --time-limit-hours $HOURS --learning-starts 20000 --eps-decay-steps 250000 --eps-final-step 1500000 \
    --lr-decay-steps 3000000 --target-update 1000 >> $D/train.log 2>&1
$PY -m dqn.mspacman_dqn --eval $D/final_weights.pt --eval-episodes 500 --eval-eps 0 --out $D --threads 2 \
    >> $D/eval.log 2>&1
echo "done"
