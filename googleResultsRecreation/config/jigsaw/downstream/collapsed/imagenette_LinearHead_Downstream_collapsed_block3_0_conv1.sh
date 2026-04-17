#!/bin/sh

python train_and_eval.py \
  --workdir /project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/DOWNSTREAM_COLLAPSED/LinearHead_Jig9_10Perms_40Epoch/block2_0_conv1 \
  --task downstream \
  --architecture resnet50 \
  --dataset imagenette \
  --dataset_dir /project/amr239/gma35/RotNet-Neural-Collapse/RNC/datasets/Imagenette/imagenette2-160 \
  --permutations_path /project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/permutations_100_max.bin \
  --train_split train \
  --val_split val \
  --batch_size 32 \
  --eval_batch_size 4 \
  --preprocessing resize,to_gray,crop,crop_patches,standardization \
  --downstream_preprocessing resize,crop,standardization \
  --resize_size 292,292 \
  --lr 0.1 \
  --lr_scale_batch_size 128 \
  --decay_epochs 60,90,120 \
  --epochs 100 \
  --warmup_epochs 5 \
  --serving_input_shape None,64,64,3 \
  --load_model /project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/LinearHead_Jig9_10Perms_40Epoch/checkpoint_last.pt \
  --layer_extractor block3.0.conv1 \
  "$@"