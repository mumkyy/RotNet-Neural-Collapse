#!/usr/bin/env python 
#coding: utf-8

"""
# collapsed run
python NC1_with_weights.py --exp CIFAR10_RotNet_NIN4blocks_collapsed --checkpoint 200 --batch-size 128 --workers 0

# non-collapsed run
python NC1_with_weights.py --exp CIFAR10_RotNet_NIN4blocks_notcollapsed --checkpoint 200 --batch-size 128 --workers 0


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

#NC metrics Collector
class Measurements: 
    def __init__(self):
        self.accuracy = [] 
        self.loss = [] # MSE 
        self.trSwtrSb = [] #nc1 metric

@torch.no_grad() 
def compute_metrics(M: Measurements, model: nn.Module, loader: DataLoaderataloader, C: int, feat_layer: str): 
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval().to(device)
    #this is the index of a specific layer within the models structure 
    idx = model.all_feat_names.index(feat_layer)

    #grapb a block 
    feat_module = model.feature_blocks[idx]

    #hook to that block 
    feats: Dict[str, torch.Tensor] = {}
    def feat_hook(module, inp, out):
        feats['h'] = out.detach().view(out.size(0), -1)
    handle = feat_module.register_forward_hook(feat_hook)

    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[torch.Tensor] = []
    total_ss = 0.0
    loss_fn = nn.MSELoss()
    
    #Pass 1 : Means , loss, acc i.e. forward pass

    print(f"[Measurement] PASS 1 ({feat_layer}): computing class means, accuracy, and loss")
    #iterate through the data loader object and start a smart progress bar
    for x, y in tqdm(loader, desc=f"PASS 1 ({feat_layer})", unit="batch"): 
        #makes the current input data to x and output labels as y 
        x, y = x.to(device), y.to(device)
        #execute a forward pass with input x
        out = model(x)
        #grab the features 
        h = feats['h'].view(len(x), -1)

        #calculate loss 
        total_loss += loss_fn(out, y).item() * len(x)
        #number of correctly classified samples : argmax being dimension wise max index
        net_correct += (out.argmax(1).cpu() == y.cpu()).sum().item()

        #calculate the sum of features
        for c in range(C):
            idx = (y.cpu() == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                hc = h[idx]
                if len(sum_per_class) < C:
                    sum_per_class.append(hc.sum(0))
                else:
                    sum_per_class[c] += hc.sum(0)
                N_per_class += hc.size(0)
    means = [s / max(n,1) for s , n in zip(sum_per_class, N_per_class)]
    means = [m.to(device) for m in means]
    Mmat = torch.stack(means).T
    muG = Mmat.mean(1, keepdim=True)

# Pass 2 : within class cov

    print(f"[Measurement] PASS 2 ({feat_layer}): computing within-class covariance...")
    for x, y in tqdm(loader, desc=f"PASS 2 ({feat_layer})", unit="batch"):
        x,y = x.to(device), y.to(device)
        out = model(x)
        h = feats['h'].view(len(x), -1)

        for c in range(C): 
            idx = (y.cpu() == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                hc = h[idx]
                #current instance - mean for that class?
                z = hc - means[c].to(hc.device)
                #calculate the square of the difference between curr and avg
                total_ss += (z.pow(2).sum(dim=1)).sum().item()

    print(f"[Measurements] Finalizing metrics for {feat_layer}...")


    #calculate all the useful values 
    N = N_per_class.sum().item()
    #trace of within class matrix is this total over N samples 
    trace_Sw = total_ss / N 
    loss = total_loss / N 
    acc = net_correct / N

    #NC1 CALCULATION 

    """
    Sb = 1 / C (SUM[from c = 1 -> C](avg(c) - avg(Global))(avg(c) - avg(Global))^T)
    """
    Sb = (Mmat - mug) @ (Mmat - muG).T / C
    trace_Sb = Sb.trace() 
    #error value 
    eps = 1e-12
    nc1_ratio = trace_Sw / (trace_Sb + eps)
    trSwtrSb = nc1_ratio.item()

    M.accuracy.append(trSwtrSb)

    handle.remove()
    feats.clear()


#THE BELOW IS CHAT GPT GENERATED NEED TO MANUALLY PARSE 
# -------------------------------------------------------------------------
# CLI & Main script
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Measure NC on a RotNet experiment")
    p.add_argument('--exp',        required=True, help='experiment name, used to find config/config_<exp>.py')
    p.add_argument('--exp_dir',    default=None, help='(optional) full path to the experiment directory; overrides experiments/<exp>')
    p.add_argument('--checkpoint', type=int, default=0, help='epoch id of model_net_epochXX to load')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--no_cuda',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:  # you passed --no_cuda or no GPU
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, map_location=torch.device('cpu'))

    set_seed(42)

    # 1) load config
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
    ds_train = GenericDataset(
        dataset_name=dt['dataset_name'],
        split=dt['split'],
        random_sized_crop=dt['random_sized_crop'],
        num_imgs_per_cat=dt.get('num_imgs_per_cat')
    )
    loader = RotLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        unsupervised=dt['unsupervised'],
        epoch_size=dt['epoch_size'],
        num_workers=args.workers,
        shuffle=False
    )(0)

    C = 4  # RotNet 4 rotations

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

    # ------------------------------------------------------------------
    # Weight analysis (once per checkpoint)
    # ------------------------------------------------------------------
    conv_layers = []
    linear_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append((name, m))
        elif isinstance(m, nn.Linear):
            linear_layers.append((name, m))

    print("\n[Weights] Found convolutional layers:")
    for i, (name, m) in enumerate(conv_layers):
        print(f"  Conv[{i}]: {name}, weight shape = {tuple(m.weight.shape)}")

    print("\n[Weights] Found linear (fully-connected) layers:")
    for i, (name, m) in enumerate(linear_layers):
        print(f"  Linear[{i}]: {name}, weight shape = {tuple(m.weight.shape)}")

    # Focus: conv2 + output layer (same index across collapsed / non-collapsed 4-block models)
    focus_stats = []
    if len(conv_layers) >= 2:
        conv2_name, conv2_mod = conv_layers[1]
        stats_conv2 = compute_weight_stats(conv2_name, conv2_mod)
        focus_stats.append(stats_conv2)
        print("\n[Weights] Conv2 stats:", stats_conv2)
    else:
        print("\n[Weights] WARNING: < 2 conv layers; cannot define conv2.")

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

    save_dir = Path('results') / f"{args.exp}_{type(model).__name__}" / f"bs{args.batch_size}"
    (save_dir / 'plots').mkdir(parents=True, exist_ok=True)

    weights_file = save_dir / f"weight_stats_checkpoint{args.checkpoint}.txt"
    with open(weights_file, 'w') as fp:
        fp.write(f"Weight statistics for checkpoint {args.checkpoint}\n\n")

        fp.write("=== Focus: conv2 + output layer ===\n")
        for s in focus_stats:
            fp.write(f"{s['name']}\n")
            fp.write(f"  shape:        {s['shape']}\n")
            fp.write(f"  flat_shape:   {s['flat_shape']}\n")
            fp.write(f"  rank:         {s['rank']}\n")
            fp.write(f"  fro_norm:     {s['fro_norm']:.6f}\n")
            fp.write(f"  spec_norm:    {s['spec_norm']:.6f}\n")
            fp.write(f"  stable_rank:  {s['stable_rank']:.6f}\n\n")

        fp.write("\n=== All conv + linear blocks ===\n")
        for s in all_block_stats:
            fp.write(f"{s['name']}\n")
            fp.write(f"  shape:        {s['shape']}\n")
            fp.write(f"  flat_shape:   {s['flat_shape']}\n")
            fp.write(f"  rank:         {s['rank']}\n")
            fp.write(f"  fro_norm:     {s['fro_norm']:.6f}\n")
            fp.write(f"  spec_norm:    {s['spec_norm']:.6f}\n")
            fp.write(f"  stable_rank:  {s['stable_rank']:.6f}\n\n")

    print(f"\n✓ Weight stats written to {weights_file}\n")

    # ------------------------------------------------------------------
    # NC1 across layers
    # ------------------------------------------------------------------
    layers = model.all_feat_names
    nc1_per_layer = {}

    for layer in tqdm(layers, desc="Measuring NC1 per layer"):
        metrics = Measurements()
        compute_metrics(metrics, model, loader, C, layer)
        nc1_per_layer[layer] = metrics.trSwtrSb[-1]

    out_file = save_dir / 'NC1WrittenVals.txt'
    with open(out_file, 'w') as fp:
        fp.write(f"NC1 metrics for checkpoint {args.checkpoint}\n")
        for layer, val in nc1_per_layer.items():
            fp.write(f"{layer}: {val:.6f}\n")
    print(f"✓ NC1 values written to {out_file}")

    x = list(nc1_per_layer.keys())
    y = [nc1_per_layer[l] for l in x]

    plt.figure()
    plt.plot(range(len(x)), y, 'bx-')
    plt.xticks(range(len(x)), x, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('NC1')
    plt.title(f'NC1 across layers at checkpoint {args.checkpoint}')
    plt.tight_layout()
    plt.savefig(save_dir / 'plots' / f"NC1_layers_checkpoint.pdf")
    plt.close()

    print("✓ Done – results in", save_dir)