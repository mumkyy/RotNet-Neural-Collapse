#!/bin/bash
set -euo pipefail

# Work from project root
COLL="config/CIFAR10/RotNet/MSE/Collapsed/convclassifier/conv2/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Collapsed_MSE.py"
NOT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/conv2/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Not_Collapsed_MSE_wreg.py"

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

  if [[ "$hook" == "classifier" ]]; then
    subdir="classifier"
  else
    subdir="${hook%%.*}"
  fi

  # collapsed config 
  COLL_OUT="config/CIFAR10/RotNet/MSE/Collapsed/convclassifier/${subdir}/"
  COLL_EXP="${COLL_OUT}CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_${hook}_feats_Collapsed_MSE.py"

  # not collapsed config 
  NOT_OUT="config/CIFAR10/RotNet/MSE/Not_Collapsed/convclassifier/${subdir}/"
  NOT_EXP="${NOT_OUT}CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_${hook}_feats_Not_Collapsed_MSE.py"

  mkdir -p "$COLL_OUT" "$NOT_OUT"

  cp -p "$COLL" "$COLL_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$COLL_EXP"

  echo "Wrote $COLL_EXP"

  cp -p "$NOT" "$NOT_EXP"

  sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$NOT_EXP"

  echo "Wrote $NOT_EXP"

  job_safe="${hook//./_}"

  # collapsed
  jid_coll=$(sbatch --parsable --job-name="gpu_collapsed_${job_safe}" --export=ALL,EXP_NAME="$COLL_EXP" ../scripts/submit.sh)
  echo "Submitted collapsed: $jid_coll"

  # not collapsed
  jid_not=$(sbatch --parsable --job-name="gpu_not_collapsed_${job_safe}" --export=ALL,EXP_NAME="$NOT_EXP" ../scripts/submit.sh)
  echo "Submitted not collapsed: $jid_not"

done
