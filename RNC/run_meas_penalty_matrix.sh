#!/bin/bash
# Submit one measurement job per penalty-matrix backbone (NC1/NC3/NC1+NC3 x reg/inverse
# x log/non-log x LW/Final) for CIFAR NIN rotnet + STL jigsaw9 + STL rotnet.
# Selects only the matrix configs by the "_<reg|inv>_<log|nonlog>" descriptor,
# so it ignores the other backbones sharing the same folder.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/experiments

NIN_LAYERS="conv1.Block1_ConvB1,conv1.Block1_ConvB2,conv1.Block1_ConvB3,conv2.Block2_ConvB1,conv2.Block2_ConvB2,conv2.Block2_ConvB3,conv3.Block3_ConvB1,conv3.Block3_ConvB2,conv3.Block3_ConvB3,conv4.Block4_ConvB1,conv4.Block4_ConvB2,conv4.Block4_ConvB3,classifier"
RES_LAYERS="conv1,conv2.block0,conv2.block1,conv2.block2,conv3.block0,conv3.block1,conv3.block2,conv3.block3,conv4.block0,conv4.block1,conv4.block2,conv4.block3,conv4.block4,conv4.block5,conv5.block0,conv5.block1,conv5.block2,lin1,lin2,classifier"

submit_dir () { # $1 config-subdir  $2 dataset  $3 pretext  $4 nclasses  $5 arch  $6 layers  $7 out-sub
  for f in $(find "config/$1" -maxdepth 1 -name '*.py' | grep -E '_(reg|inv)_(log|nonlog)' | sort); do
    exp="${f#config/}"; exp="${exp%.py}"; name="$(basename "$exp")"
    jid=$(sbatch --parsable --job-name="meas_${name}" submit_meas_one.sh \
          --exp "$exp" --config-root config --dataset_name_arg "$2" \
          --exp-dir "${SCRATCH}/${name}" --workers 8 \
          --pretext-mode "$3" --num-classes "$4" --split test \
          --arch-class "$5" --out-root "results_nc/$7/${name}" \
          --layers "$6" --nc4 --nc4-layerwise --pabs --nc2)
    echo "$jid  meas ${name}"
  done
}

submit_dir "CIFAR10/RotNet/MSE/Not_Collapsed/backbone" cifar10 rotation 4  NetworkInNetwork  "$NIN_LAYERS" "cifar10/rotnet"
submit_dir "STL10/jigsaw9/MSE/Not_Collapsed/Backbone"  stl10  jigsaw_9 10 ResNet34_NIN_Style "$RES_LAYERS" "stl10/jigsaw9"
submit_dir "STL10/RotNet/MSE/Not_Collapsed/Backbone"   stl10  rotation 4  ResNet34_NIN_Style "$RES_LAYERS" "stl10/rotnet"
