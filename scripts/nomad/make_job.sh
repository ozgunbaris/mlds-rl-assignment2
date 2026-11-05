#!/usr/bin/env bash
# Generate scripts/nomad/$JOB_NAME.nomad: a Nomad batch job that trains MsPacman-v5 Double DQN for several seeds
# in parallel on one node (docker, python:3.12-slim), rolls out 500 greedy games per run and packs the results
# into /alloc/data/mspacman_results.tar.gz (fetch with scripts/fetch_cluster_results.sh).
# The code is embedded as a base64 tarball in a template block (the cluster has no shared storage).
#
# Defaults: 3 seeds of the Mnih et al. (2015) network, 8 CPU threads each, 10 hours, at most 6000 episodes,
# on mlds-deepdish5 (CPU only: Nomad detects no GPU there). Memory: each run's replay buffer of 500,000
# 88x80 int8 frames takes 3.5 GB, so request about 5 GB per seed.
# GPU=1 adds a request for one NVIDIA GPU and DEVICE=cuda; it needs a node whose GPU Nomad can see (untested).
set -e
cd "$(dirname "$0")/../.."
REV=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
HOURS=${HOURS:-10}
SEEDS=${SEEDS:-"0 1 2"}
ARCH=${ARCH:-nature}
THREADS=${THREADS:-8}
DEVICE=${DEVICE:-cpu}
EPISODES=${EPISODES:-6000}
CORES=${CORES:-24}
MEMORY=${MEMORY:-16384}
JOB_NAME=${JOB_NAME:-mspacman-dqn}
NODE=${NODE:-mlds-deepdish5}
PREFIX=${PREFIX:-cluster}
EXTRA=${EXTRA:-""}
GPU=${GPU:-0}
if [ "$GPU" = "1" ]; then DEVICE=cuda; fi
OUTFILE=scripts/nomad/$JOB_NAME.nomad
TMP=$(mktemp -d)
tar -czf $TMP/code.tar.gz dqn/*.py tests/*.py viz.py requirements.txt
B64=$(base64 < $TMP/code.tar.gz)
if [ "$GPU" = "1" ]; then
  GPU_BLOCK='        device "nvidia/gpu" {
          count = 1
        }'
else
  GPU_BLOCK=""
fi
cat > $OUTFILE <<EOF
job "$JOB_NAME" {
  datacenters = ["dc1"]
  type        = "batch"

  meta {
    revision = "$REV"
    hours    = "$HOURS"
    seeds    = "$SEEDS"
    arch     = "$ARCH"
    prefix   = "$PREFIX"
  }

  constraint {
    attribute = "\${attr.unique.hostname}"
    value     = "$NODE"
  }

  group "main" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    ephemeral_disk {
      sticky  = true
      migrate = false
      size    = 10000
    }

    task "run" {
      driver = "docker"

      config {
        image   = "python:3.12-slim"
        command = "/bin/sh"
        args    = ["/local/run.sh"]
      }

      env {
        NODE_NAME        = "\${node.unique.name}"
        PYTHONUNBUFFERED = "1"
        OMP_NUM_THREADS  = "$THREADS"
        MKL_NUM_THREADS  = "$THREADS"
        HOURS            = "$HOURS"
        SEEDS            = "$SEEDS"
        ARCH             = "$ARCH"
        THREADS          = "$THREADS"
        DEVICE           = "$DEVICE"
        EPISODES         = "$EPISODES"
        PREFIX           = "$PREFIX"
        EXTRA            = "$EXTRA"
      }

      template {
        destination = "local/run.sh"
        data        = <<PAYLOAD_EOF
$(cat scripts/nomad/run.sh)
PAYLOAD_EOF
      }

      template {
        destination = "local/code.b64"
        data        = <<PAYLOAD_EOF
$B64
PAYLOAD_EOF
      }

      resources {
        cores  = $CORES
        memory = $MEMORY
$GPU_BLOCK
      }
    }
  }
}
EOF
rm -rf $TMP
echo "wrote $OUTFILE (job $JOB_NAME on $NODE, revision $REV, hours $HOURS, seeds '$SEEDS', arch $ARCH, threads $THREADS, device $DEVICE, cores $CORES, memory $MEMORY MB, gpu $GPU)"
