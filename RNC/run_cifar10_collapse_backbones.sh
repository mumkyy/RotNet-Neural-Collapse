#!/bin/bash
# Submit the 4 force-collapse (no-WD) CIFAR NIN rotnet backbones, each its own job. No dependencies.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/
for v in NC1Layerwise NC1Final NC3Layerwise NC3Final; do
  exp="CIFAR10/RotNet/MSE/Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_${v}_noWD"
  jid=$(sbatch --parsable --job-name="$(basename "$exp")" \
        submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
