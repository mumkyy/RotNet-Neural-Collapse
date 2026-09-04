#!/bin/bash
# Submit ONLY the downstreams for this group (each as its own job). No dependencies.
# Backbone checkpoints are read from SCRATCH via --pretrained_path.
# Already-completed probes (classifier_net_epoch100 present) are listed in SKIP and not resubmitted.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/

# completed as of last scan -- skip these
SKIP="
STL10_ConvClassifier_on_Jigsaw4_resnet_conv1_feats_Not_Collapsed_MSE_nc3Layerwise_noWD
STL10_ConvClassifier_on_Jigsaw4_resnet_conv2_block0_feats_Not_Collapsed_MSE_nc3Layerwise_noWD
STL10_ConvClassifier_on_Jigsaw4_resnet_conv2_block0_feats_Not_Collapsed_MSE_nc3Layerwise_WD
STL10_ConvClassifier_on_Jigsaw4_resnet_conv2_block2_feats_Not_Collapsed_MSE_nc3Layerwise_noWD
STL10_ConvClassifier_on_Jigsaw4_resnet_conv5_block0_feats_Not_Collapsed_MSE_nc3Layerwise_noWD
STL10_ConvClassifier_on_Jigsaw4_resnet_conv5_block2_feats_Not_Collapsed_MSE_nc3Layerwise_noWD
"

for f in $(find config/STL10/jigsaw4/MSE/Not_Collapsed/nc3Layerwise_WD config/STL10/jigsaw4/MSE/Not_Collapsed/nc3Layerwise_noWD -name '*.py' | sort); do
  exp=${f#config/}; exp=${exp%.py}
  name=$(basename "$exp")
  case " $SKIP " in *" $name "*) echo "skip (done)  $name"; continue;; esac
  jid=$(sbatch --parsable --job-name="$name" \
        submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --pretrained_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done
