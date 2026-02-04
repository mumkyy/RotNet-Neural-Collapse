#!/bin/bash
set -euo pipefail

# Work from project root
variants=(
  "Not_Collapsed|config/Imagenette/jigsaw/MSE/Not_Collapsed/conv2/Imagenette_ConvClassifier_on_Jigsaw_four_classes_jitter_colordist_resnet_conv2_feats_Not_Collapsed_MSE.py"
  "Collapsed|config/Imagenette/jigsaw/MSE/Collapsed/conv2/Imagenette_ConvClassifier_on_Jigsaw_four_classes_jitter_colordist_resnet_conv2_feats_Collapsed_MSE.py"
)

# ResNet34_NIN_Style feature keys (see architectures/Resnet.py)
hooks=(
  "conv1.Stem_Conv"
  "conv1.Stem_BN"
  "conv1.Stem_ReLU"
  "conv1.Stem_MaxPool"
)

for i in 0 1 2; do
  hooks+=("conv2.block${i}")
done

for i in 0 1 2 3; do
  hooks+=("conv3.block${i}")
done

for i in 0 1 2 3 4 5; do
  hooks+=("conv4.block${i}")
done

for i in 0 1 2; do
  hooks+=("conv5.block${i}")
done

declare -A channels=(
  [conv1]=64
  [conv2]=64
  [conv3]=128
  [conv4]=256
  [conv5]=512
)

for variant in "${variants[@]}"; do
  IFS="|" read -r label base <<< "$variant"

  for hook in "${hooks[@]}"; do
    subdir="${hook%%.*}"
    ch="${channels[$subdir]:-}"

    if [[ -z "$ch" ]]; then
      echo "Unknown stage for hook: $hook" >&2
      exit 1
    fi

    OUT_DIR="config/Imagenette/jigsaw/MSE/${label}/${subdir}/"
    OUT_FILE="${OUT_DIR}Imagenette_ConvClassifier_on_Jigsaw_four_classes_jitter_colordist_resnet_${hook}_feats_${label}_MSE.py"

    mkdir -p "$OUT_DIR"
    cp -p "$base" "$OUT_FILE"

    sed -E -i "s|^config\\['out_feat_keys'\\][[:space:]]*=.*|config['out_feat_keys'] = ['${hook}']|" "$OUT_FILE"
    sed -E -i "s|([\"']nChannels[\"'][[:space:]]*:[[:space:]]*)[0-9]+|\\1${ch}|" "$OUT_FILE"

    echo "Wrote $OUT_FILE"
  done
done
