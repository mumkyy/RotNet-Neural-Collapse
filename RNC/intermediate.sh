#!/bin/bash
set -euo pipefail

# Work from project root
COLL="config/CIFAR10/RotNet/MSE/Collapsed/convclassifier/conv2/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Collapsed_MSE.py"
NOT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/conv2/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Not_Collapsed_MSE.py"

convs=(1 2 3 4)

blocks=(
  "ConvB1"
  "ConvB2"
  "ConvB3"
)

special=(
  "MaxPool"
  "AvgPool"
  "AvgPool"
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

hooks+=("classifier")



for i in "${!hooks[@]}"; do
  hook="${hooks[$i]}"

  # collapsed config 
  COLL_OUT="config/CIFAR10/RotNet/MSE/Collapsed/convclassifier/${hook}/"
  mkdir -p "$COLL_OUT"
  COLL_EXP="${COLL_OUT}CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_${hook}_feats_Collapsed_MSE.py"

  cp -p "$COLL" "$COLL_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$COLL_EXP"

  echo "Wrote $COLL_EXP"

  # not collapsed config 
  NOT_OUT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/${hook}/"
  mkdir -p "$NOT_OUT"
  NOT_EXP="${NOT_OUT}CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_${hook}_feats_Not_Collapsed_MSE.py"

  cp -p "$NOT" "$NOT_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$NOT_EXP"

  echo "Wrote $NOT_EXP"

done
