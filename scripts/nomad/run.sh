#!/bin/sh
# Runs inside the python:3.12-slim container on the cluster node. Plain POSIX sh.
# No dollar-brace, percent-brace or double-curly sequences (HCL heredoc and Nomad template rules).
set -u
BASE=/alloc/data
WORK=$BASE/work
LOGS=$BASE/logs
OUT=$BASE/mspacman_results.tar.gz
mkdir -p $WORK $LOGS

pack() {
  cd $BASE
  if [ -d $WORK/results ]; then
    tar -czf $OUT.tmp --exclude=checkpoint.pt --exclude=checkpoint.pt.tmp -C $WORK results -C $BASE logs && mv $OUT.tmp $OUT
  else
    tar -czf $OUT.tmp -C $BASE logs && mv $OUT.tmp $OUT
  fi
  echo "PACKED $(wc -c < $OUT) bytes $(date -u +%H:%M:%S)"
  cd $WORK
}
trap pack EXIT

CORES=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))")
echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) node=$NODE_NAME cores=$CORES hours=$HOURS seeds=$SEEDS arch=$ARCH threads=$THREADS device=$DEVICE episodes=$EPISODES prefix=$PREFIX extra=$EXTRA memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
base64 -d /local/code.b64 > $BASE/code.tar.gz
tar -xzf $BASE/code.tar.gz -C $WORK
cd $WORK
if [ "$DEVICE" = "cpu" ]; then
  TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX=""
fi
echo "pip torch ($DEVICE) $(date -u +%H:%M:%S)"
pip install --no-cache-dir --disable-pip-version-check torch $TORCH_INDEX > $LOGS/pip_torch.log 2>&1 || { tail -n 20 $LOGS/pip_torch.log; exit 1; }
echo "pip gymnasium/ale $(date -u +%H:%M:%S)"
pip install --no-cache-dir --disable-pip-version-check "numpy>=2,<3" "gymnasium[atari]==1.4.0" "ale-py==0.12.1" pytest > $LOGS/pip_other.log 2>&1 || { tail -n 20 $LOGS/pip_other.log; exit 1; }
python3 -c "import torch, gymnasium, ale_py; print('versions', torch.__version__, gymnasium.__version__, ale_py.__version__, 'cuda', torch.cuda.is_available())"
python3 -m pytest -q tests/test_dqn.py > $LOGS/pytest.log 2>&1
echo "pytest rc=$? $(tail -n 1 $LOGS/pytest.log)"

PIDS=""
for S in $SEEDS; do
  D=results/mspacman/"$PREFIX"_seed"$S"
  mkdir -p $D
  python3 -m dqn.mspacman_dqn --out $D --seed $S --arch $ARCH --threads $THREADS --device $DEVICE --max-episodes $EPISODES --time-limit-hours $HOURS $EXTRA > $D/train.log 2>&1 &
  PIDS="$PIDS $!"
  echo "started $D pid $!"
done

while true; do
  alive=0
  for p in $PIDS; do
    if kill -0 $p 2>/dev/null; then alive=1; fi
  done
  if [ $alive -eq 0 ]; then break; fi
  sleep 600
  for S in $SEEDS; do
    D=results/mspacman/"$PREFIX"_seed"$S"
    echo "$D: $(grep '^ep' $D/train.log | tail -n 1)"
  done
  pack
done
echo "TRAINING_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for S in $SEEDS; do
  D=results/mspacman/"$PREFIX"_seed"$S"
  python3 -m dqn.mspacman_dqn --eval $D/final_weights.pt --eval-episodes 500 --eval-eps 0 --out $D --threads $THREADS > $D/eval.log 2>&1 &
done
wait
echo "EVAL done $(date -u +%H:%M:%S)"
pack
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Keep the allocation running so its files stay readable with nomad alloc fs. On this cluster Nomad removed a
# finished job, its allocation and its files within hours (job mspacman-dqn-gpu-dd4, 2026-11-07). The task exits
# when /alloc/data/FETCHED exists (nomad alloc exec <alloc> touch /alloc/data/FETCHED) or after HOLD_HOURS.
HOLD_MIN=$(( HOLD_HOURS * 60 ))
echo "HOLD up to $HOLD_HOURS h until /alloc/data/FETCHED exists"
i=0
while [ ! -e $BASE/FETCHED ] && [ $i -lt $HOLD_MIN ]; do
  sleep 60
  i=$(( i + 1 ))
done
echo "RELEASED after $i min $(date -u +%Y-%m-%dT%H:%M:%SZ)"
