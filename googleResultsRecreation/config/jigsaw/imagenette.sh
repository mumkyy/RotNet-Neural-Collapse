#!/bin/sh

python train_and_eval.py \
  --workdir /project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/jigsaw_imagenette_resnet50_NC10 \
  --task jigsaw \
  --dataset imagenette \
  --dataset_dir /project/amr239/gma35/RotNet-Neural-Collapse/RNC/datasets/Imagenette/imagenette2-160 \
  --permutations_path /project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/permutations_100_max.bin \
  --train_split train \
  --val_split val \
  --batch_size 32 \
  --eval_batch_size 4 \
  --architecture resnet50 \
  --filters_factor 8 \
  --last_relu True \
  --mode v1 \
  --preprocessing resize,to_gray,crop,crop_patches,standardization \
  --resize_size 292,292 \
  --crop_size 255 \
  --grayscale_probability 0.66 \
  --splits_per_side 3 \
  --patch_jitter 21 \
  --embed_dim 10 \
  --perm_subset_size 10 \
  --lr 0.1 \
  --lr_scale_batch_size 128 \
  --decay_epochs 15,25 \
  --epochs 25 \
  --warmup_epochs 5 \
  --serving_input_shape None,64,64,3 \
  "$@"
