#!/bin/bash
# Submit the canonical backbones that need a FRESH training run after the config cleanup.
# A backbone is here if it is NEW or its penalty CONTENT changed (so any old scratch model is stale).
# Backbones that were only RENAMED (same content) are NOT here -- reuse their trained model by
# renaming the scratch dir instead (see the reuse-rename list I gave you).
# Each is its own job, written to SCRATCH.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/

EXPS=(
  # --- CIFAR NIN rotnet (7) ---
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE"                 # plain baseline (verify not already on scratch)
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_reg_log_LW"  # layers standardized
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC3_reg_log"     # lambda 0.1 (was 100)
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC3lw_reg"       # NIN nc3lw was broken; rebuilt
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC3lw_inv"       # new
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_log_LW"    # penalty changed (nc3lw)
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW" # penalty changed (nc3lw)
  # --- STL jigsaw9 (5) ---
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_reg_log_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC3lw_inv"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  # --- STL rotnet (5) ---
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_reg_log_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE_NC3lw_inv"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  # --- STL jigsaw4 (17: unstarted except reused Collapsed_MSE + NC3lw_reg) ---
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_reg_log_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_Final"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_inv_log_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_inv_log_Final"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_Final"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC3_reg_log"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC3_inv_log"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC3_inv_nonlog"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC3lw_inv"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_Final"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  "STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final"
)
for exp in "${EXPS[@]}"; do
  jid=$(sbatch --parsable --job-name="$(basename "$exp")" \
        submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
