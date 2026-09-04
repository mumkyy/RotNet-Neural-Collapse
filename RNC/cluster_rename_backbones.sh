#!/bin/bash
# Run ON THE CLUSTER. Renames trained backbone scratch dirs (old name -> canonical name) so the
# content-UNCHANGED backbones are reused instead of retrained.
# PAIR THIS with rewriting downstream feat_pretrained_file refs in the repo configs -- otherwise the
# downstreams that point at the old names will fail to load their backbone.
# Content-CHANGED / new backbones are NOT here (they are retrained by run_required_backbones.sh, and
# their stale old scratch dirs are removed by cluster_cleanup_scratch.sh).
set -u
S=/scratch/amr239/ai342/RNC_experiments/experiments

RENAMES=(
  # CIFAR NIN
  "CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_NC1Layerwise_noWD|CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_inv_log_LW"
  "CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_NC1Final_noWD|CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_inv_log_Final"
  "CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_NC3Final_noWD|CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC3_inv_log"
  # STL jigsaw4
  "STL10_Jigsaw4_resnet_Collapsed_MSE_Four_classes_jitter_colordist|STL10_Jigsaw4_resnet_Collapsed_MSE"
  "STL10_Jigsaw4_resnet_Not_Collapsed_MSE_nc3Layerwise_withoutWD|STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC3lw_reg"
  # STL jigsaw9
  "STL10_Jigsaw9_resnet_Collapsed_MSE_Ten_classes_jitter_colordist|STL10_Jigsaw9_resnet_Collapsed_MSE"
  "STL10_Jigsaw9_resnet_Collapsed_MSE_NC1Layerwise_noWD|STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_inv_log_LW"
  "STL10_Jigsaw9_resnet_Collapsed_MSE_NC1Final_noWD|STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_inv_log_Final"
  "STL10_Jigsaw9_resnet_Collapsed_MSE_NC3Final_noWD|STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC3_inv_log"
  "STL10_Jigsaw9_resnet_Not_Collapsed_MSE_nc3Layerwise_withoutWD|STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC3lw_reg"
  # STL rotnet: Resnet -> resnet (content-unchanged reused ones) + nc3Layerwise -> NC3lw_reg
  "STL10_RotNet_Resnet_Collapsed_MSE|STL10_RotNet_resnet_Collapsed_MSE"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_log_LW|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_inv_log_LW"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_log_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_inv_log_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_nonlog_LW|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_LW"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_nonlog_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_log_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_nonlog_LW|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_LW"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_nonlog_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_reg_log|STL10_RotNet_resnet_Not_Collapsed_MSE_NC3_reg_log"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_inv_log|STL10_RotNet_resnet_Not_Collapsed_MSE_NC3_inv_log"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_inv_nonlog|STL10_RotNet_resnet_Not_Collapsed_MSE_NC3_inv_nonlog"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_log_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final|STL10_RotNet_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final"
  "STL10_RotNet_Resnet_Not_Collapsed_MSE_nc3Layerwise_withoutWD|STL10_RotNet_resnet_Not_Collapsed_MSE_NC3lw_reg"
)
for pair in "${RENAMES[@]}"; do
  old="${pair%%|*}"; new="${pair##*|}"
  if [ -d "$S/$old" ]; then
    if [ -e "$S/$new" ]; then echo "SKIP (target exists): $new"; else mv "$S/$old" "$S/$new"; echo "$old -> $new"; fi
  else echo "absent (ok / already renamed): $old"; fi
done
