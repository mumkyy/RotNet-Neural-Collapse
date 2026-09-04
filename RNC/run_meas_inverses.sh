#!/bin/bash
# Submit one measurement job per force-collapse ("inverse") backbone.
# Each job runs measurements.measurementsFix (NC1 layerwise via --nc4-layerwise,
# NC2, NC3/pabs, NC4) over the pretext test split, reading checkpoints from SCRATCH.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/experiments

NIN_LAYERS="conv1.Block1_ConvB1,conv1.Block1_ConvB2,conv1.Block1_ConvB3,conv2.Block2_ConvB1,conv2.Block2_ConvB2,conv2.Block2_ConvB3,conv3.Block3_ConvB1,conv3.Block3_ConvB2,conv3.Block3_ConvB3,conv4.Block4_ConvB1,conv4.Block4_ConvB2,conv4.Block4_ConvB3,classifier"

RES_LAYERS="conv1,conv2.block0,conv2.block1,conv2.block2,conv3.block0,conv3.block1,conv3.block2,conv3.block3,conv4.block0,conv4.block1,conv4.block2,conv4.block3,conv4.block4,conv4.block5,conv5.block0,conv5.block1,conv5.block2,lin1,lin2,classifier"

# ---- CIFAR NIN rotnet collapse backbones ----
for V in NC1Layerwise NC1Final NC3Layerwise NC3Final; do
  name="CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_${V}_noWD"
  jid=$(sbatch --parsable --job-name="meas_${name}" submit_meas_one.sh \
        --exp "CIFAR10/RotNet/MSE/Collapsed/backbone/${name}" \
        --config-root config \
        --dataset_name_arg cifar10 \
        --exp-dir "${SCRATCH}/${name}" \
        --workers 8 \
        --pretext-mode rotation --num-classes 4 --split test \
        --arch-class NetworkInNetwork \
        --out-root "results_nc/cifar10/rotnet/collapse_${V}" \
        --layers "$NIN_LAYERS" \
        --nc4 --nc4-layerwise --pabs --nc2)
  echo "$jid  meas ${name}"
done

# ---- STL10 jigsaw9 resnet34 collapse backbones ----
for V in NC1Layerwise NC1Final NC3Layerwise NC3Final; do
  name="STL10_Jigsaw9_resnet_Collapsed_MSE_${V}_noWD"
  jid=$(sbatch --parsable --job-name="meas_${name}" submit_meas_one.sh \
        --exp "STL10/jigsaw9/MSE/Collapsed/Backbone/${name}" \
        --config-root config \
        --dataset_name_arg stl10 \
        --exp-dir "${SCRATCH}/${name}" \
        --workers 8 \
        --pretext-mode jigsaw_9 --num-classes 10 --split test \
        --arch-class ResNet34_NIN_Style \
        --out-root "results_nc/stl10/jigsaw9/collapse_${V}" \
        --layers "$RES_LAYERS" \
        --nc4 --nc4-layerwise --pabs --nc2)
  echo "$jid  meas ${name}"
done
