#!/bin/bash
set -euo pipefail

# Work from project root
NOT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/conv2/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Not_Collapsed_MSE_NC1_Reg_Detach.py"

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
  NOT_OUT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/${subdir}/"
  NOT_EXP="${NOT_OUT}CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_${hook}_feats_Not_Collapsed_MSE_NC1_Reg_Detach.py"


  cp -p "$NOT" "$NOT_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$NOT_EXP"

  if [[ "$hook" == conv1.Block1_ConvB2* ]]; then
    sed -E -i "s|(\['nChannels'\][[:space:]]*=[[:space:]]*)[0-9]+|\1160|" "$NOT_EXP"
  elif [[ "$hook" == conv1.Block1_ConvB3* || "$hook" == conv1.Block1_MaxPool* ]]; then
    sed -E -i "s|(\['nChannels'\][[:space:]]*=[[:space:]]*)[0-9]+|\196|" "$NOT_EXP"
  else
    sed -E -i "s|(\['nChannels'\][[:space:]]*=[[:space:]]*)[0-9]+|\1192|" "$NOT_EXP"
  fi

  echo "Wrote $NOT_EXP"
done
