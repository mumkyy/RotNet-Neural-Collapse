#!/bin/bash
# Submit downstream (ConvClassifier probe) jobs from the canonical downstream/ trees.
# Each downstream config references its backbone as './experiments/<backbone>/model_net_epoch200';
# --pretrained_path reroots that to SCRATCH, where the trained backbones live.
#
# Usage:
#   ./run_downstreams.sh                      # ALL tasks, all backbones  (1292 jobs - careful)
#   ./run_downstreams.sh cifar                # one task
#   ./run_downstreams.sh jig9 NC1_inv_log_LW  # one task, backbones matching a substring
#   DRYRUN=1 ./run_downstreams.sh cifar       # print what would be submitted
set -euo pipefail
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/

declare -A TREE=(
  [cifar]="CIFAR10/RotNet/MSE/downstream"
  [jig4]="STL10/jigsaw4/MSE/downstream"
  [jig9]="STL10/jigsaw9/MSE/downstream"
  [rotnet]="STL10/RotNet/MSE/downstream"
)

tasks="${1:-cifar jig4 jig9 rotnet}"
filter="${2:-}"
n=0
for t in $tasks; do
  tree="${TREE[$t]:-}"
  if [ -z "$tree" ]; then echo "unknown task '$t' (use: cifar jig4 jig9 rotnet)" >&2; exit 1; fi
  for cfg in $(find "config/$tree" -name '*.py' | sort); do
    exp="${cfg#config/}"; exp="${exp%.py}"
    if [ -n "$filter" ] && [[ "$exp" != *"$filter"* ]]; then continue; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "would submit: $exp"; n=$((n+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$(basename "$exp")" \
          submit_one.sh --exp "$exp" \
                        --output_model_path "$SCRATCH" \
                        --pretrained_path "$SCRATCH" \
                        --num_workers 8 --cuda)
    echo "$jid  $exp"
    n=$((n+1))
  done
done
echo "total: $n"
