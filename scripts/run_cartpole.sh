#!/usr/bin/env bash
# Every CartPole-v1 run in the report, one after another on one core (in order of priority).
# Main configuration: Double DQN, gamma 0.95, epsilon 1 -> 0.1, 1000 training episodes; the network kept is the
# one with the best greedy score in the periodic checks (20 separate evaluation episodes every 25 episodes).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
COMMON="--episodes 1000 --eps-final 0.1 --no-stop-when-solved --eval-periodic-episodes 20 --print-every 250"
run() {  # run <name> <seed> [extra args]
  local name=$1 seed=$2; shift 2
  local d=results/cartpole/${name}_seed$seed
  if [ -f $d/eval_summary_greedy.json ]; then echo "skip $d"; return; fi
  mkdir -p $d
  $PY -m dqn.cartpole_dqn --out $d --seed $seed $COMMON "$@" > $d/train.log 2>&1
  echo "$(date -u +%H:%M:%S) done $d: $(grep -o '"mean": [0-9.]*' $d/eval_summary_greedy*.json | tr '\n' ' ')"
}
for s in 0 1 2 3 4; do run main $s; done
for s in 0 1; do run truncterm $s --truncation-terminal; done   # seed 2 was stopped: identical to main_seed2 (no truncation reached)
for s in 0 1 2; do run notarget $s --no-target; done
for s in 0 1 2; do run noreplay $s --buffer-size 64; done
for s in 0 1 2; do run dqn $s --no-double; done
for s in 0 1; do run gamma0.99 $s --gamma 0.99; done
for s in 0 1; do run gamma0.9 $s --gamma 0.9; done
run rscale2 0 --reward-scale 2
run rscale-1 0 --reward-scale -1 --episodes 300
run lr0 0 --lr 0 --lr-final 0 --episodes 300
echo "ALL DONE $(date -u +%H:%M:%S)"
