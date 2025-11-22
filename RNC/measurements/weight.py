#!/usr/bin/env python 
#coding: utf-8
"""
RUN THE FILE
python -m measurements.weight --exp CIFAR10/RotNet/MSE/Collapsed/backbone/CIFAR10_RotNet_NIN4blocks_Collapsed_MSE --exp_dir ../experiments/CIFAR10_RotNet_NIN4blocks_Collapsed_MSE --checkpoint 200
"""
import argparse, importlib.util, os, pickle, random
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# set the seed
def set_seed(seed: int): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#weight function
def flatten_weight_matrix(weight: torch.Tensor) -> torch.Tensor: 
    #reshape the weight tensor automatically to a 2d array 
    return weight.view(weight.shape[0], -1)

@torch.no_grad()
def compute_weight_stats(name: str, module: nn.Module):
    #weight tensor
    W=module.weight 
    #flatten W using cpu for rank calculation
    W2d = flatten_weight_matrix(W).detach().cpu()
    #torch.linalg gets the rank of the flattened matrix
    rank = torch.linalg.matrix_rank(W2d).item()

    stats = {
        "name":         name, 
        "shape":        tuple(W.shape),
        "flat_shape":   tuple(W2d.shape),
        "rank":         rank, 
    }
    return stats

#CLI
def parse_args():
    p = argparse.ArgumentParser("Measure NC on a RotNet experiment")
    p.add_argument('--exp',        required=True, help='experiment name, used to find config/config_<exp>.py enter config name without config/ prefix this is automatically appended')
    p.add_argument('--exp_dir',    default=None, help='(optional) full path to the experiment directory; overrides experiments/<exp> enter experiment')
    p.add_argument('--checkpoint', type=int, default=0, help='epoch id of model_net_epochXX to load')
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--no_cuda',    action='store_true')
    p.add_argument('--metricEval',  type=str,default=None,help='Name of evaluation metric (for non-rotation / non-gaussian-blur experiments)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:  # you passed --no_cuda or no GPU
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, map_location=torch.device('cpu'))

    set_seed(42)

    # 1) load config and experimemnt (trained model)
    cfg_file = Path('config') / f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = Path('experiments') / args.exp
    config['exp_dir'] = str(exp_dir)

    # 2) build loader
    from dataloader import GenericDataset, DataLoader as RotLoader
    
    dt = config['data_train_opt']
    batch_size=dt['batch_size']
    dc = config['networks']['model']['opt']
    C = dc.get('num_classes', 4) # RotNet 4 rotations default , otheriwse get from netopt
    feat_layer = config.get('feature_layer', 'encoder')
    # 3) instantiate alg + load checkpoint
    import algorithms as alg
    algo = getattr(alg, config['algorithm_type'])(config)
    if use_cuda:
        algo.load_to_gpu()
    algo.load_checkpoint(args.checkpoint, train=False)

    # backbone / feature extractor
    feat_extractor = algo.networks.get(
        'feat_extractor',
        algo.networks.get('model', list(algo.networks.values())[0])
    )
    model = feat_extractor

    print("Available feature layers:", model.all_feat_names)
    #weights extraction (onece per checkpoint)
    conv_layers = [] 
    linear_layers = []
    #grab the model layers convolutional and linear
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append((name, m))
        elif isinstance(m, nn.Linear):
            linear_layers.append((name, m))

    #print the layers with their shape
    print("\n[Weights] Found convolutional layers:")
    for i, (name, m) in enumerate(conv_layers):
        print(f"  Conv[{i}]: {name}, weight shape = {tuple(m.weight.shape)}")

    print("\n[Weights] Found linear (fully-connected) layers:")
    for i, (name, m) in enumerate(linear_layers):
        print(f"  Linear[{i}]: {name}, weight shape = {tuple(m.weight.shape)}")

    # Focus: conv2 + output layer (same index across collapsed / non-collapsed 4-block models)
    focus_stats = []
    #second convolutional block conv_layers = [ conv1 , conv2, conv3, conv4]
    if len(conv_layers) >= 2:
        conv2_name, conv2_mod = conv_layers[1]
        #this will store a dict[str] with rank, compressed and noncompressed shape and name
        stats_conv2 = compute_weight_stats(conv2_name, conv2_mod)
        focus_stats.append(stats_conv2)
        print("\n[Weights] Conv2 stats:", stats_conv2)
    else:
        print("\n[Weights] WARNING: < 2 conv layers; cannot define conv2.")
    #output layer weight computation 
    if len(linear_layers) >= 1:
        out_name, out_mod = linear_layers[-1]
        stats_out = compute_weight_stats(out_name, out_mod)
        focus_stats.append(stats_out)
        print("\n[Weights] Output layer stats:", stats_out)
    else:
        print("\n[Weights] WARNING: no linear layers; cannot define output layer.")

    # All blocks (for full comparison)
    all_block_stats = []
    for name, m in conv_layers + linear_layers:
        s = compute_weight_stats(name, m)
        all_block_stats.append(s)

    #path = results/configFile->args._modelTYPE(CNN).__main__/bs(from config)
    save_dir = Path('results') / f"{args.exp}_{type(model).__name__}" / f"bs{batch_size}"
    (save_dir / 'plots').mkdir(parents=True, exist_ok=True)
    #weights file will be saved at results/configFile_CNN.__main__/bs(from_config)/weight_stats_checkpoint
    weights_file = save_dir / f"weight_stats_checkpoint{args.checkpoint}.txt"
    
    with open(weights_file, 'w') as fp:
        fp.write(f"Weight statistics for checkpoint {args.checkpoint}\n\n")

        fp.write("=== Focus: conv2 + output layer ===\n")
        for s in focus_stats:
            fp.write(f"{s['name']}\n")
            fp.write(f"  shape:        {s['shape']}\n")
            fp.write(f"  flat_shape:   {s['flat_shape']}\n")
            fp.write(f"  rank:         {s['rank']}\n")

        fp.write("\n=== All conv + linear blocks ===\n")
        for s in all_block_stats:
            fp.write(f"{s['name']}\n")
            fp.write(f"  shape:        {s['shape']}\n")
            fp.write(f"  flat_shape:   {s['flat_shape']}\n")
            fp.write(f"  rank:         {s['rank']}\n")


    print(f"\n✓ Weight stats written to {weights_file}\n")