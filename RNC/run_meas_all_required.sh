#!/bin/bash
# Measure EVERY required backbone that exists on scratch (one job each), correctly per task.
# Measurements only need the model architecture from --exp, so we use one representative
# config per task (num_stages/num_classes/arch) and point --exp-dir at each backbone's own
# checkpoint dir. This lets us measure backbones whose training config isn't in this repo
# (the _Four/Ten_classes, _wreg, _LamNC3 aliases from the other repo).
# Non-required families (nc3Layerwise / NC3Layerwise / Sigmoid / NC1pushAway) and all
# downstream dirs are skipped.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/experiments

NIN_LAYERS="conv1.Block1_ConvB1,conv1.Block1_ConvB2,conv1.Block1_ConvB3,conv2.Block2_ConvB1,conv2.Block2_ConvB2,conv2.Block2_ConvB3,conv3.Block3_ConvB1,conv3.Block3_ConvB2,conv3.Block3_ConvB3,conv4.Block4_ConvB1,conv4.Block4_ConvB2,conv4.Block4_ConvB3,classifier"
RES_LAYERS="conv1,conv2.block0,conv2.block1,conv2.block2,conv3.block0,conv3.block1,conv3.block2,conv3.block3,conv4.block0,conv4.block1,conv4.block2,conv4.block3,conv4.block4,conv4.block5,conv5.block0,conv5.block1,conv5.block2,lin1,lin2,classifier"

# Representative config per task (architecture only). Must exist in ./config.
REP_CIFAR="CIFAR10/RotNet/MSE/Not_Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE"
REP_JIG4="STL10/jigsaw4/MSE/Not_Collapsed/Backbone/STL10_Jigsaw4_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
REP_JIG9="STL10/jigsaw9/MSE/Not_Collapsed/Backbone/STL10_Jigsaw9_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"
REP_ROT="STL10/RotNet/MSE/Not_Collapsed/Backbone/STL10_RotNet_resnet_Not_Collapsed_MSE_NC1_reg_log_Final"

for d in "$SCRATCH"/*/; do
  name="$(basename "$d")"
  # skip downstream dirs
  case "$name" in *ConvClassifier*|*LinearClassifier*|*_feats_*) continue;; esac
  # skip not-required families
  case "$name" in *nc3Layerwise*|*NC3Layerwise*|*Sigmoid*|*pushAway*) continue;; esac
  # route by task prefix
  case "$name" in
    CIFAR10_RotNet_NIN4blocks*) exp="$REP_CIFAR"; ds=cifar10; pt=rotation; nc=4;  arch=NetworkInNetwork;  layers="$NIN_LAYERS"; sub="cifar10/rotnet";;
    STL10_Jigsaw4_resnet*)      exp="$REP_JIG4";  ds=stl10;   pt=jigsaw;   nc=4;  arch=ResNet34_NIN_Style; layers="$RES_LAYERS"; sub="stl10/jigsaw4";;
    STL10_Jigsaw9_resnet*)      exp="$REP_JIG9";  ds=stl10;   pt=jigsaw_9; nc=10; arch=ResNet34_NIN_Style; layers="$RES_LAYERS"; sub="stl10/jigsaw9";;
    STL10_RotNet_[Rr]esnet*)    exp="$REP_ROT";   ds=stl10;   pt=rotation; nc=4;  arch=ResNet34_NIN_Style; layers="$RES_LAYERS"; sub="stl10/rotnet";;
    *) continue;;
  esac
  jid=$(sbatch --parsable --job-name="meas_${name}" submit_meas_one.sh \
        --exp "$exp" --config-root config --dataset_name_arg "$ds" \
        --exp-dir "$SCRATCH/$name" --workers 8 \
        --pretext-mode "$pt" --num-classes "$nc" --split test \
        --arch-class "$arch" --out-root "results_nc/$sub/$name" \
        --layers "$layers" --nc4 --nc4-layerwise --pabs --nc2)
  echo "$jid  meas $name"
done
