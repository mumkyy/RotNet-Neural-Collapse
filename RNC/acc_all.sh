#!/bin/bash
# Dump last-epoch accuracy for ALL NC-experiment backbones + downstreams in one go.
#   bash acc_all.sh > acc_dump.txt 2>&1
# Override the experiments root with: ROOT=/some/experiments bash acc_all.sh
# NOTE: no `set -e` -- a failure/missing dir must not abort the whole dump.
set -uo pipefail
cd "$(dirname "$0")"
root="${ROOT:-/scratch/amr239/ai342/RNC_experiments/experiments}"

print_section () { echo -e "\n ======== $1 ======== \n"; }

run_acc () {
  local exp="$1"
  local logs="${root}/${exp}/logs/"
  echo -e "\n .... ${exp} .... \n"
  if [[ -d "$logs" ]]; then
    python accuracy.py "$logs" --last-epoch --recursive
  else
    echo "MISSING: $logs"
  fi
}

# Run accuracy on every dir under $root matching a glob pattern.
run_glob () {  # $1 = section title, $2 = glob pattern (relative to $root)
  print_section "$1"
  shopt -s nullglob
  local matched=0 d
  for d in "$root"/$2/; do
    run_acc "$(basename "$d")"
    matched=1
  done
  shopt -u nullglob
  [[ $matched -eq 1 ]] || echo "(no dirs matched: $2)"
}

# ============ INVERSE / force-collapse ============
run_glob "CIFAR NIN collapse BACKBONES"   "CIFAR10_RotNet_NIN4blocks_Collapsed_MSE_*_noWD"
run_glob "CIFAR NIN collapse DOWNSTREAMS" "CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_*_feats_Collapsed_MSE_NC*"
run_glob "STL jig9 collapse BACKBONES"    "STL10_Jigsaw9_resnet_Collapsed_MSE_*_noWD"
run_glob "STL jig9 collapse DOWNSTREAMS"  "STL10_ConvClassifier_on_Jigsaw9_resnet_*_feats_Collapsed_MSE_NC*"

# ============ nc3 layerwise (not collapsed) ============
run_glob "CIFAR NIN nc3LW BACKBONES"      "CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "CIFAR NIN nc3LW DOWNSTREAMS"    "CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_*_feats_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL jig4 nc3LW BACKBONES"       "STL10_Jigsaw4_resnet_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL jig4 nc3LW DOWNSTREAMS"     "STL10_ConvClassifier_on_Jigsaw4_resnet_*_feats_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL jig9 nc3LW BACKBONES"       "STL10_Jigsaw9_resnet_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL jig9 nc3LW DOWNSTREAMS"     "STL10_ConvClassifier_on_Jigsaw9_resnet_*_feats_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL rotnet nc3LW BACKBONES"     "STL10_RotNet_Resnet_Not_Collapsed_MSE_nc3Layerwise_*"
run_glob "STL rotnet nc3LW DOWNSTREAMS"   "STL10_ConvClassifier_on_RotNet_resnet_*_feats_Not_Collapsed_MSE_nc3Layerwise_*"

# ============ penalty matrix (NC1/NC3/NC1+NC3 x reg/inverse x log/non-log) BACKBONES ============
run_glob "CIFAR NIN penmat reg"    "CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_*_reg_*"
run_glob "CIFAR NIN penmat inv"    "CIFAR10_RotNet_NIN4blocks_Not_Collapsed_MSE_*_inv_*"
run_glob "STL jig9 penmat reg"     "STL10_Jigsaw9_resnet_Not_Collapsed_MSE_*_reg_*"
run_glob "STL jig9 penmat inv"     "STL10_Jigsaw9_resnet_Not_Collapsed_MSE_*_inv_*"
run_glob "STL rotnet penmat reg"   "STL10_RotNet_Resnet_Not_Collapsed_MSE_*_reg_*"
run_glob "STL rotnet penmat inv"   "STL10_RotNet_Resnet_Not_Collapsed_MSE_*_inv_*"
