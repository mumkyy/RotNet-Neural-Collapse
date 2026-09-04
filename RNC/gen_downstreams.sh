#!/bin/bash
# Deterministically regenerate ALL downstream (ConvClassifier probe) configs for the 4 owned tasks.
# Target: per task, 19 canonical backbones x 17 probe layers = 323 configs.
# Layout: config/<task>/MSE/downstream/<backbone_basename>/<stage>/<name>.py
# Every feat_pretrained_file points at a CANONICAL backbone name.
set -euo pipefail
cd "$(dirname "$0")"

# ---- probe specs: "layer:nChannels" ----
# NOTE: the model registers a bare 'convN' key that is assigned the SAME tensor as that block's
# last child (see NetworkInNetwork.forward). So bare conv2/conv3/conv4 are exact aliases of
# Block2_AvgPool / Block3_ConvB3 / Block4_ConvB3 and are deliberately EXCLUDED - we keep the
# explicit sub-block name instead (matching how conv1 is handled, and the ResNet convention).
# 14 distinct NIN probe layers.
NIN_LAYERS=(
  "conv1.Block1_ConvB1:192" "conv1.Block1_ConvB2:160" "conv1.Block1_ConvB3:96" "conv1.Block1_MaxPool:96"
  "conv2.Block2_ConvB1:192" "conv2.Block2_ConvB2:192" "conv2.Block2_ConvB3:192" "conv2.Block2_AvgPool:192"
  "conv3.Block3_ConvB1:192" "conv3.Block3_ConvB2:192" "conv3.Block3_ConvB3:192"
  "conv4.Block4_ConvB1:192" "conv4.Block4_ConvB2:192" "conv4.Block4_ConvB3:192"
)
# ResNet: bare 'conv1' is the stem output (alias of its last child), and it is the ONLY stem probe
# here - no conv1.Stem_* keys - so there is no aliasing within this set. Stages conv2..conv5 use
# explicit block keys only, never the bare 'convN' alias. 17 distinct ResNet probe layers.
RES_LAYERS=(
  "conv1:64" "conv2.block0:64" "conv2.block1:64" "conv2.block2:64"
  "conv3.block0:128" "conv3.block1:128" "conv3.block2:128" "conv3.block3:128"
  "conv4.block0:256" "conv4.block1:256" "conv4.block2:256" "conv4.block3:256" "conv4.block4:256" "conv4.block5:256"
  "conv5.block0:512" "conv5.block1:512" "conv5.block2:512"
)

# task -> tree, backbone dirs, arch, feat num_classes, dataset, ds-name-prefix, on-token
gen_task () {
  local task="$1" tree="$2" nbdir="$3" cbdir="$4" arch="$5" featnc="$6" ds="$7" dspfx="$8" ontok="$9"
  local -n LAYERS=${10}
  local netfile out_root count=0
  if [ "$arch" = nin ]; then netfile="architectures/NetworkInNetwork.py"; else netfile="architectures/Resnet.py"; fi
  out_root="config/${tree}/downstream"
  rm -rf "$out_root"; mkdir -p "$out_root"

  for bcfg in "$nbdir"/*.py "$cbdir"/*.py; do
    [ -e "$bcfg" ] || continue
    local bname tail
    bname="$(basename "$bcfg" .py)"                 # canonical backbone name
    # ds filename tail = everything from Collapsed/Not_Collapsed onward
    tail="${bname##*resnet_}"; tail="${tail##*NIN4blocks_}"
    for spec in "${LAYERS[@]}"; do
      local layer="${spec%%:*}" nch="${spec##*:}"
      local stage="${layer%%.*}"
      local ftok="${layer//./_}"                    # filename token: dots -> underscores
      local d="$out_root/$bname/$stage"; mkdir -p "$d"
      local out="$d/${dspfx}_ConvClassifier_on_${ontok}_${ftok}_feats_${tail}.py"
      cat > "$out" <<EOF
batch_size = 128

config = {}

data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = '${ds}'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = '${ds}'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 100

networks = {}
feat_net_opt = {'num_classes': ${featnc}, 'num_stages': 4, 'use_avg_on_conv3': False}
feat_pretrained_file = './experiments/${bname}/model_net_epoch200'
networks['feat_extractor'] = {'def_file': '${netfile}', 'pretrained': feat_pretrained_file, 'opt': feat_net_opt, 'optim_params': None}

cls_net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr': [(35, 0.1), (70, 0.02), (85, 0.004), (100, 0.0008)]}
cls_net_opt = {'num_classes': 10, 'nChannels': ${nch}, 'cls_type': 'NIN_ConvBlock3'}
networks['classifier'] = {'def_file': 'architectures/NonLinearClassifier.py', 'pretrained': None, 'opt': cls_net_opt, 'optim_params': cls_net_optim_params}
config['out_feat_keys'] = ['${layer}']

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype': 'MSELoss', 'opt': None}
config['criterions'] = criterions
config['algorithm_type'] = 'FeatureClassificationModel'
config['best_metric'] = 'prec1'
EOF
      count=$((count+1))
    done
  done
  echo "$task: wrote $count downstream configs -> $out_root"
}

gen_task cifar  "CIFAR10/RotNet/MSE" \
  "config/CIFAR10/RotNet/MSE/Not_Collapsed/backbone" "config/CIFAR10/RotNet/MSE/Collapsed/backbone" \
  nin 4  cifar10 CIFAR10 "RotNet_NIN4blocks" NIN_LAYERS

gen_task jig4   "STL10/jigsaw4/MSE" \
  "config/STL10/jigsaw4/MSE/Not_Collapsed/Backbone" "config/STL10/jigsaw4/MSE/Collapsed/Backbone" \
  res 4  stl10 STL10 "Jigsaw4_resnet" RES_LAYERS

gen_task jig9   "STL10/jigsaw9/MSE" \
  "config/STL10/jigsaw9/MSE/Not_Collapsed/Backbone" "config/STL10/jigsaw9/MSE/Collapsed/Backbone" \
  res 10 stl10 STL10 "Jigsaw9_resnet" RES_LAYERS

gen_task rotnet "STL10/RotNet/MSE" \
  "config/STL10/RotNet/MSE/Not_Collapsed/Backbone" "config/STL10/RotNet/MSE/Collapsed/Backbone" \
  res 4  stl10 STL10 "RotNet_resnet" RES_LAYERS
