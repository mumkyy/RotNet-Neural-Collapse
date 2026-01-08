#!/bin/bash
set -euo pipefail

# Work from project root
NOT="config/CIFAR10/RotNet/CE/Not_Collapsed/linearclassifier/conv2/CIFAR10_LinearClassifier_on_RotNet_NIN4blocks_Conv2_feats_Not_Collapsed_CE_NC1_Reg.py"

convs=(1 2 3 4)

blocks=(
  "ConvB1"
  "ConvB2"
  "ConvB3"
)

special=(
  "MaxPool"
  "AvgPool"
  ""
  ""
)

hooks=()
count=0

for conv in "${convs[@]}"; do
  for block in "${blocks[@]}"; do
    hooks+=("conv${conv}.Block${conv}_${block}")
  done

  sp="${special[$count]}"
  if [[ -n "$sp" ]]; then
    hooks+=("conv${conv}.Block${conv}_${sp}")
  fi

  ((count+=1))
done



for i in "${!hooks[@]}"; do
  hook="${hooks[$i]}"

  subdir="${hook%%.*}"

  # not collapsed config 
  NOT_OUT="config/CIFAR10/RotNet/CE/Not_Collapsed/linearclassifier/${subdir}/"
  NOT_EXP="${NOT_OUT}CIFAR10_LinearClassifier_on_RotNet_NIN4blocks_${hook}_feats_Not_Collapsed_CE_NC1_Reg.py"


  cp -p "$NOT" "$NOT_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$NOT_EXP"

  echo "Wrote $NOT_EXP"
done
