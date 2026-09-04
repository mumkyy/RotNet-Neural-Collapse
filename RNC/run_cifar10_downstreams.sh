#!/bin/bash
# Submit ONLY the downstreams for this group (each as its own job). No dependencies.
# Backbone checkpoints are read from SCRATCH via --pretrained_path.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/
for f in $(find config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/nc3Layerwise_WD config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/nc3Layerwise_noWD -name '*.py' | sort); do
  exp=${f#config/}; exp=${exp%.py}
  jid=$(sbatch --parsable --job-name="$(basename "$exp")"         submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --pretrained_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
