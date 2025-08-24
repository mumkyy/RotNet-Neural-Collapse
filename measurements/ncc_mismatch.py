#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
NCC Mismatch per Layer for RotNet Checkpoints
================================================================================

This script loads a single RotNet checkpoint and computes the NCC mismatch
rate for each backbone feature layer. "NCC mismatch" is the fraction of samples
for which the classifier head's prediction disagrees with a nearest-class-center
(NCC) classifier built in the chosen feature space.

Typical one-liners
------------------
# compute NCC mismatch across ALL backbone layers at checkpoint 200
python NCC_mismatch.py \
    --exp CIFAR10_RotNet_NIN4blocks \
    --checkpoint 200 \
    --batch-size 128 \
    --workers 0

# compute NCC mismatch only at 'conv3' layer
python NCC_mismatch.py \
    --exp CIFAR10_RotNet_NIN4blocks \
    --checkpoint 200 \
    --batch-size 128 \
    --feature-layer conv3

Notes
-----
- Assumes your backbone exposes `all_feat_names` and `_feature_blocks`.
- RotNet has 4 classes (rotations), so we set C = 4.
- Features are flattened (global-average-pooled if 4D) before computing class means.
"""
import argparse, importlib.util, random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def _flatten_feats(feat: torch.Tensor) -> torch.Tensor:
    # NCHW -> global avg pool to N,C
    if feat.ndim == 4:
        return feat.mean(dim=(2, 3))
    # otherwise flatten to N,D
    return feat.view(feat.size(0), -1)

@torch.no_grad()
def ncc_mismatch_for_layer(model: nn.Module,
                           loader: DataLoader,
                           C: int,
                           layer_name: str,
                           device: str) -> float:
    """
    Computes NCC mismatch at a single feature layer:
      1) First pass: compute class means μ_c in that feature space
      2) Second pass: classify by nearest μ_c and compare to head's argmax
    Returns mismatch rate in [0,1].
    """
    # locate layer
    try:
        idx = model.all_feat_names.index(layer_name)
        feat_module = model._feature_blocks[idx]
    except Exception as e:
        raise RuntimeError(f"Could not resolve feature layer '{layer_name}': {e}")

    # hook
    feats: Dict[str, torch.Tensor] = {}
    def hook(_m, _in, out):
        feats['h'] = out.detach().cpu()
    hndl = feat_module.register_forward_hook(hook)

    model.eval().to(device)

    # ----- PASS 1: accumulate sums per class to build means -----
    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[Optional[torch.Tensor]] = [None for _ in range(C)]

    for x, y in tqdm(loader, desc=f"[{layer_name}] PASS 1", leave=False):
        x = x.to(device)
        logits = model(x)                      # triggers hook
        h = _flatten_feats(feats['h'])         # on CPU
        y_cpu = y.cpu()

        for c in range(C):
            idxs = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idxs.numel():
                vecs = h[idxs]                 # CPU
                if sum_per_class[c] is None:
                    sum_per_class[c] = vecs.sum(0)
                else:
                    sum_per_class[c] += vecs.sum(0)
                N_per_class[c] += vecs.size(0)

    means: List[torch.Tensor] = []
    for c in range(C):
        if N_per_class[c] > 0 and sum_per_class[c] is not None:
            means.append(sum_per_class[c] / N_per_class[c].item())
        else:
            # In case a class is missing, create a zero mean (rare on full train split)
            means.append(torch.zeros_like(sum_per_class[0]))
    Mmat = torch.stack(means).T   # D×C, on CPU

    # ----- PASS 2: compute NCC predictions and compare to head -----
    n_match = 0
    n_total = 0
    MmatT = Mmat.T.unsqueeze(0)   # 1×C×D for broadcast

    for x, y in tqdm(loader, desc=f"[{layer_name}] PASS 2", leave=False):
        x = x.to(device)
        logits = model(x)                      # triggers hook
        h = _flatten_feats(feats['h'])         # CPU, N×D
        y_cpu = y.cpu()

        # pairwise L2 to class means
        # d(n, c) = ||h_n - μ_c||_2
        diffs = h.unsqueeze(1) - MmatT         # N×C×D
        dists = torch.norm(diffs, dim=2)       # N×C
        ncc_pred = dists.argmin(dim=1)         # N

        head_pred = logits.argmax(1).cpu()     # N
        n_match += (ncc_pred == head_pred).sum().item()
        n_total += h.size(0)

    mismatch = 1.0 - (n_match / max(1, n_total))
    hndl.remove()
    feats.clear()
    return float(mismatch)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("NCC mismatch per layer on a RotNet checkpoint")
    p.add_argument('--exp',        required=True, help='experiment name (config/<exp>.py & experiments/<exp>/)')
    p.add_argument('--exp_dir',    default=None, help='override experiment dir (default: experiments/<exp>)')
    p.add_argument('--checkpoint', type=int, required=True, help='epoch id of model_net_epochXX to load')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--no_cuda',    action='store_true')
    p.add_argument('--feature-layer', type=str, default=None,
                   help='optional: compute only this layer (e.g., conv3). If omitted, compute all layers.')
    return p.parse_args()

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    if not use_cuda:
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, map_location=torch.device('cpu'))
    device = 'cuda' if use_cuda else 'cpu'
    set_seed(42)

    # 1) load config
    cfg_file = Path('config') / f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config

    # exp dir
    exp_dir = Path(args.exp_dir) if args.exp_dir else Path('experiments') / args.exp
    config['exp_dir'] = str(exp_dir)

    # 2) dataloader
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

    C = 4  # RotNet rotations

    # 3) model + checkpoint
    import algorithms as alg
    algo = getattr(alg, config['algorithm_type'])(config)
    if use_cuda:
        algo.load_to_gpu()
    algo.load_checkpoint(args.checkpoint, train=False)

    # pick the backbone module that has feature blocks
    feat_extractor = algo.networks.get(
        'feat_extractor',
        algo.networks.get('model', list(algo.networks.values())[0])
    )

    # pick layers
    all_layers = list(getattr(feat_extractor, 'all_feat_names', []))
    if not all_layers:
        raise RuntimeError("Backbone does not expose all_feat_names.")
    if args.feature_layer:
        if args.feature_layer not in all_layers:
            raise RuntimeError(f"--feature-layer '{args.feature_layer}' not in available layers: {all_layers}")
        layers = [args.feature_layer]
    else:
        layers = all_layers

    print("Available feature layers:", all_layers)
    print("Measuring layers:", layers)

    # 4) measure per-layer NCC mismatch
    ncc_per_layer = {}
    for layer in tqdm(layers, desc="Measuring NCC mismatch per layer"):
        mismatch = ncc_mismatch_for_layer(
            model=feat_extractor,
            loader=loader,
            C=C,
            layer_name=layer,
            device=device
        )
        ncc_per_layer[layer] = mismatch

    # 5) save in results/<exp>_<arch>/bs<B>/
    arch_name = type(feat_extractor).__name__
    save_dir = Path('results') / f"{args.exp}_{arch_name}" / f"bs{args.batch_size}"
    (save_dir / 'plots').mkdir(parents=True, exist_ok=True)

    # text dump
    out_txt = save_dir / f"NCC_mismatch_checkpoint{args.checkpoint}.txt"
    with open(out_txt, 'w') as fp:
        fp.write(f"NCC mismatch per layer for checkpoint {args.checkpoint}\n")
        for layer, val in ncc_per_layer.items():
            fp.write(f"{layer}: {val:.6f}\n")
    print(f"✓ NCC mismatch values written to {out_txt}")

    # plot
    xs = list(ncc_per_layer.keys())
    ys = [ncc_per_layer[k] for k in xs]
    plt.figure()
    plt.plot(range(len(xs)), ys, 'bx-')
    plt.xticks(range(len(xs)), xs, rotation=45, ha='right')
    plt.ylabel('NCC mismatch (lower is better)')
    plt.xlabel('Layer')
    plt.title(f'NCC mismatch across layers @ checkpoint {args.checkpoint}')
    plt.tight_layout()
    plt.savefig(save_dir / 'plots' / f"NCC_mismatch_layers_checkpoint{args.checkpoint}.pdf")
    plt.close()
    print("✓  Done – results in", save_dir)
