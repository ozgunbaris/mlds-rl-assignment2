#!/usr/bin/env bash
# Fetch the packed results of a Nomad MsPacman job and unpack them into results/mspacman/.
# Usage (from the Mac, with the VPN on and the SSH key loaded in ssh-agent):
#   scripts/fetch_cluster_results.sh <alloc-id> <job-name>
set -e
cd "$(dirname "$0")/.."
ALLOC=${1:?alloc id}
JOB=${2:?job name}
TMP=$(mktemp -d)
ssh -o ConnectTimeout=10 -o BatchMode=yes obkir@kranj.iems.private "nomad alloc fs $ALLOC alloc/data/mspacman_results.tar.gz" > $TMP/mspacman_results.tar.gz
ls -la $TMP/mspacman_results.tar.gz
tar -xzf $TMP/mspacman_results.tar.gz -C $TMP
mkdir -p results/mspacman/cluster_logs/$JOB
cp -R $TMP/results/mspacman/. results/mspacman/
cp -R $TMP/logs/. results/mspacman/cluster_logs/$JOB/
ls -d results/mspacman/*_seed*
echo FETCH_DONE
