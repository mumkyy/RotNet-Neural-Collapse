#!/bin/bash
# Run ON THE CLUSTER. Deletes scratch experiment dirs that became obsolete after the config
# cleanup (their config was DELETED and not renamed). Backbones only -- their downstream probe
# dirs (same families) are part of the separate downstream cleanup.
set -u
SCRATCH=/scratch/amr239/ai342/RNC_experiments/experiments
OBSOLETE=(
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_Reg
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_Reg_Inverse
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_LamNC3_1e2
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_LamNC3_1e2_WD
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_nc3Layerwise_withWD
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_wreg
  CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_wreg_all
  CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_NC3Layerwise_noWD
  CIFAR10_RotNet_NIN4blocks_NC1pushAway_NC3force_MSE
  STL10_Jigsaw4_resnet_Not_Collapsed_MSE_LamNC3_1e2_jitter_colordist
  STL10_Jigsaw4_resnet_Not_Collapsed_MSE_nc3Layerwise_withWD
  STL10_Jigsaw9_resnet_Collapsed_MSE_NC3Layerwise_noWD
  STL10_Jigsaw9_resnet_Not_Collapsed_MSE_LamNC3_1e2_jitter_colordist
  STL10_Jigsaw9_resnet_Not_Collapsed_MSE_nc3Layerwise_withWD
  STL10_RotNet_Resnet_Not_Collapsed_MSE_LamNC3_1e2
  STL10_RotNet_Resnet_Not_Collapsed_MSE_nc3Layerwise_withWD
)
for d in "${OBSOLETE[@]}"; do
  p="$SCRATCH/$d"
  if [ -d "$p" ]; then echo "rm -rf $d"; rm -rf "$p"; else echo "absent (ok): $d"; fi
done
