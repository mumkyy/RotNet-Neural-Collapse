#!/bin/bash
set -euo pipefail

# Work from project root
cd config
mkdir -p testkernels
cd ..

BASE="config/CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE.py"
DOWN4="config/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv4_feats_Collapsed.py"
DOWN2="config/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv2_feats_Collapsed.py"

odds=(1 3 5 7 9 11 15)
KERNEL_SETS=()
for ((a=0; a<${#odds[@]}-3; a++)); do
  for ((b=a+1; b<${#odds[@]}-2; b++)); do
    for ((c=b+1; c<${#odds[@]}-1; c++)); do
      for ((d=c+1; d<${#odds[@]}; d++)); do
        KERNEL_SETS+=("[${odds[a]},${odds[b]},${odds[c]},${odds[d]}]")
      done
    done
  done
done

for i in "${!KERNEL_SETS[@]}"; do
  ks="${KERNEL_SETS[$i]}"
  KS_TAG=$(echo "$ks" | tr -d '[] ' | tr ',' '_')

  # ---------- Upstream config ----------
  BASE_OUT="config/testkernels/CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE_${KS_TAG}.py"
  BASE_EXP="testkernels/CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE_${KS_TAG}"

  cp -p "$BASE" "$BASE_OUT"

  sed -E -i \
    "s|(^[[:space:]]*data_test_opt\['kernel_sizes'\][[:space:]]*=[[:space:]])\[[^]]*\]|\1${ks}|;" \
    "$BASE_OUT"
  sed -E -i \
    "s|(^[[:space:]]*data_train_opt\['kernel_sizes'\][[:space:]]*=[[:space:]])\[[^]]*\]|\1${ks}|;" \
    "$BASE_OUT"

  echo "Wrote $BASE_OUT with kernel_sizes = ${ks}"

  # ---------- Downstream conv4 config ----------
  DOWN4_OUT="config/testkernels/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv4_feats_Collapsed_${KS_TAG}.py"
  DOWN4_EXP="testkernels/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv4_feats_Collapsed_${KS_TAG}"

  cp -p "$DOWN4" "$DOWN4_OUT"

  sed -E -i \
    "s|(^[[:space:]]*config\['out_feat_keys'\][[:space:]]*=[[:space:]])\[[^]]*\]|\1\['conv4'\]|;" \
    "$DOWN4_OUT"

  sed -E -i \
    "s|(^[[:space:]]*feat_pretrained_file[[:space:]]*=[[:space:]])['\"][^'\"]*['\"]|\1'./experiments/${BASE_EXP}/model_net_epoch200'|;" \
    "$DOWN4_OUT"

  echo "Wrote $DOWN4_OUT with out_feat_keys=['conv4'] and feat_pretrained_file=./experiments/${BASE_EXP}/model_net_epoch200"

  # ---------- Downstream conv2 config ----------
  DOWN2_OUT="config/testkernels/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv2_feats_Collapsed_${KS_TAG}.py"
  DOWN2_EXP="testkernels/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv2_feats_Collapsed_${KS_TAG}"

  cp -p "$DOWN2" "$DOWN2_OUT"

  sed -E -i \
    "s|(^[[:space:]]*config\['out_feat_keys'\][[:space:]]*=[[:space:]])\[[^]]*\]|\1\['conv2'\]|;" \
    "$DOWN2_OUT"

  sed -E -i \
    "s|(^[[:space:]]*feat_pretrained_file[[:space:]]*=[[:space:]])['\"][^'\"]*['\"]|\1'./experiments/${BASE_EXP}/model_net_epoch200'|;" \
    "$DOWN2_OUT"

  echo "Wrote $DOWN2_OUT with out_feat_keys=['conv2'] and feat_pretrained_file=./experiments/${BASE_EXP}/model_net_epoch200"

  # upstream
  jid_up=$(sbatch --parsable --job-name="gpu_collapsed_gb_kernel_up_${KS_TAG}" --export=ALL,EXP_NAME="$BASE_EXP",KS_TAG="$KS_TAG" ./submit.sh)
  echo "Submitted upstream: $jid_up"

  # conv2 waits for upstream to succeed
  jid_c2=$(sbatch --parsable --dependency=afterok:${jid_up} --job-name="gpu_collapsed_gb_kernel_conv2_${KS_TAG}" --export=ALL,EXP_NAME="$DOWN2_EXP",KS_TAG="$KS_TAG" ./submit.sh)
  echo "Submitted conv2: $jid_c2 (after $jid_up)"

  # conv4 waits for upstream to succeed
  jid_c4=$(sbatch --parsable --dependency=afterok:${jid_up} --job-name="gpu_collapsed_gb_kernel_conv4_${KS_TAG}" --export=ALL,EXP_NAME="$DOWN4_EXP",KS_TAG="$KS_TAG" ./submit.sh)
  echo "Submitted conv4: $jid_c4 (after $jid_up)"

done
