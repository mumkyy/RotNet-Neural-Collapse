#!/bin/bash
set -euo pipefail

cd config

mkdir -p testkernels


BASE="CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE.py"


KERNEL_SETS=(
  "[1,3,5,7]"
  "[1,3,5,9]"
  "[1,3,5,11]"
  "[1,3,7,9]"
  "[1,3,7,11]"
  "[1,3,9,11]"
  "[1,5,7,9]"
  "[1,5,7,11]"
  "[1,5,9,11]"
  "[1,7,9,11]"
  "[3,5,7,9]"
  "[3,5,7,11]"
  "[3,5,9,11]"
  "[3,7,9,11]"
  "[5,7,9,11]"
)

# Loop through and create 5 config copies
for i in "${!KERNEL_SETS[@]}"; do
  ks="${KERNEL_SETS[$i]}"
  OUT="testkernels/CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE_${i}.py"


  cp -p "$BASE" "$OUT"


  sed -E -i \
    "s|(^[[:space:]]*data_test_opt\['kernel_sizes'\][[:space:]]*=[[:space:]])\[[^]]*\]|\1${ks}|;" \
    "$OUT"

  echo "Wrote $OUT with data_test_opt['kernel_sizes'] = $ks"

  
  KS_TAG=$(echo "$ks" | tr -d '[] ' | tr ',' '_')


  sbatch --export=ALL,CFG_PATH="$PWD/$OUT",KS_TAG="$KS_TAG" ../submit.sh
done
