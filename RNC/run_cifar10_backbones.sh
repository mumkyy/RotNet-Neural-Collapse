#!/bin/bash
# Submit ONLY the backbones for this group (each as its own job). No dependencies.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/
for exp in   "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_nc3Layerwise_withWD"   "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_nc3Layerwise_withoutWD"; do
  jid=$(sbatch --parsable --job-name="$(basename "$exp")"         submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
