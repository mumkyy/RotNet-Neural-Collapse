#!/bin/bash
# Submit the NC1/NC3/NC1+NC3 penalty-matrix backbones (reg/inverse x log/non-log,
# layerwise/final) for CIFAR NIN rotnet + STL jigsaw9 + STL rotnet.
# Log duplicates of existing configs are intentionally omitted. Each is its own job.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/
EXPS=(
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_log_Final"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_inv_nonlog_Final"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_inv_nonlog_LW"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_reg_log_Final"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_reg_nonlog_Final"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC1_reg_nonlog_LW"
  "CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_NC3_inv_nonlog"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_log_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_log_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_log_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_nonlog_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_inv_nonlog_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_nonlog_Final"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC1_reg_nonlog_LW"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_inv_log"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_inv_nonlog"
  "STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_Resnet_Not_Collapsed_MSE_NC3_reg_log"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_Final"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_log_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_Final"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1NC3_inv_nonlog_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_Final"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_inv_nonlog_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_Final"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_reg_nonlog_LW"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC3_inv_nonlog"
  "STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC3_reg_log"
)
for exp in "${EXPS[@]}"; do
  jid=$(sbatch --parsable --job-name="$(basename "$exp")" \
        submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
