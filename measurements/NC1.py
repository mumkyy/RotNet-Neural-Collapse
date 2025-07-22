#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
Neural-Collapse NC1-per-Layer Measurement for RotNet Checkpoints
================================================================================

This script loads a single RotNet checkpoint and computes the Neural Collapse 1 metric
(NC‑1 = trace(S_w Σ_b^{-1})) for each feature layer in the backbone.  The per-layer
NC‑1 scores are saved to a results directory and a PDF plot showing NC‑1 vs. layer is
written.

Typical one-liner
-----------------
    # compute NC1 across all backbone layers at checkpoint 200
    python NC1.py \
        --exp       CIFAR10_RotNet_NIN4blocks \
        --checkpoint 200 \
        --batch-size 128 \
        --workers    0

Arguments
---------
`--exp` *NAME*        Experiment name (must match config/<exp>.py and experiments/<exp>/)
`--checkpoint` *N*    Epoch ID to load (e.g. 200)
`--batch-size` *B*    Batch size for feature extraction
`--workers` *W*       Number of dataloader workers (0 for single-process on Windows)
`--no_cuda`           Force CPU even if GPU is available

What the script does
---------------------
1. Loads the training config (config/<exp>.py) to locate checkpoints and data options.
2. Builds a RotNet train loader (unsupervised, 4-rotation task) with the specified batch size.
3. Instantiates the RotNet backbone network and loads the specified checkpoint.
4. Iterates over each feature layer name in `backbone.all_feat_names`:
   • Hooks the layer's output, flattens spatial dimensions.
   • Runs a forward pass over the train split to collect features and labels.
   • Computes within-class scatter S_w and between-class scatter S_b.
   • Calculates NC‑1 = trace(S_w Σ_b^{-1}) for that layer.
5. Saves a PDF plot `NC1_layers_checkpoint.pdf` under `results/<exp>_NetworkInNetwork/bs<B>/plots/`.

Results
-------
- A dict of {layer_name: NC1_value} is printed.
- The PDF plot shows NC1 vs. layer in order of `all_feat_names`.

The script is read-only with respect to your experiment folder; it never overwrites
existing checkpoints.
"""
import argparse, importlib.util, os, pickle, random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm  # added for progress bars
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds, ArpackError
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Simple seed helper (same as utils.set_seed)
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# NC metrics collector
# -----------------------------------------------------------------------------
class Measurements:
    def __init__(self):
        self.accuracy = []
        self.loss     = []
        self.Sw_invSb = [] #nc1 
        
# -----------------------------------------------------------------------------
# Compute NC metrics in one full pass
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(M: Measurements, model: nn.Module, loader: DataLoader, C: int, feat_layer: str):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval().to(device)

    idx = model.all_feat_names.index(feat_layer)

    # 2) grab that block directly
    feat_module = model._feature_blocks[idx]

    # 3) hook it
    feats: Dict[str,torch.Tensor] = {}
    def feat_hook(module, inp, out):
        feats['h'] = out.detach().view(out.size(0),-1)
    handle = feat_module.register_forward_hook(feat_hook)

    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[torch.Tensor] = []
    Sw = torch.zeros(0)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = net_correct = 0

    # PASS 1: means, loss, accuracy
    print("[Measurement] PASS 1: computing class means, accuracy, and loss...")
    for x, y in tqdm(loader, desc="PASS 1", unit="batch"):  # progress bar

        x, y = x.to(device), y.to(device)
        out = model(x)
        h   = feats['h'].view(len(x), -1)

        if Sw.numel()==0:
            Sw = torch.zeros(h.size(1), h.size(1),device=device)

        total_loss += loss_fn(out, y).item() * len(x)
        net_correct += (out.argmax(1).cpu()==y.cpu()).sum().item()

        for c in range(C):
            idx = (y.cpu()==c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                hc = h[idx]
                if len(sum_per_class)<C:
                    sum_per_class.append(hc.sum(0))
                else:
                    sum_per_class[c] += hc.sum(0)
                N_per_class[c] += hc.size(0)

    means = [s / max(n,1) for s,n in zip(sum_per_class, N_per_class)]
    means = [m.to(device) for m in means]
    Mmat  = torch.stack(means).T
    muG   = Mmat.mean(1, keepdim=True)

    # PASS 2: within-class cov, NCC match
    print("[Measurement] PASS 2: computing within-class covariance and NCC mismatch...")
    for x, y in tqdm(loader, desc="PASS 2", unit="batch"):  # progress bar
        x, y = x.to(device), y.to(device)
        out = model(x)
        h   = feats['h'].view(len(x), -1)

        for c in range(C):
            idx = (y.cpu()==c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                hc = h[idx]
                z  = hc - means[c].to(hc.device)
                Sw += z.T @ z


    print("[Measurement] Finalizing metrics...")

    # finalize
    N = N_per_class.sum().item()
    Sw /= N
    loss = total_loss / N
    acc  = net_correct / N

    # ------------------------------------------------------------------
    # NC-1  -- trace(S_w Σ_b-1)  (GPU-aware)
    # ------------------------------------------------------------------
    if device == 'cuda':
        # --- 1. compute Σ_b  (= between-class scatter) on GPU ----------
        Sb = (Mmat - muG) @ (Mmat - muG).T / C        # D×D torch tensor (GPU)
        # --- 2. eigendecomp.  We only need the non-zero eigen-pairs
        #     If D <= C you could use torch.linalg.inv, but in CNNs D>>C.
        evals, evecs = torch.linalg.eigh(Sb)          # *all* eigen-pairs
        # keep the C-1 largest (the smallest one is ≈0 by construction)
        keep = evals.argsort(descending=True)[:C-1]
        Λinv = torch.diag(1. / evals[keep])
        Σb_inv = evecs[:, keep] @ Λinv @ evecs[:, keep].T   # D×D
        Σb_inv = Σb_inv.to(device)  
        Sw_invSb = (Sw.to(device) * Σb_inv).sum().item()    # trace(A B)=sum(A*B)
    else:
        # ---------- CPU path: SciPy sparse SVD (identical to before) ---
        Sb_np  = ((Mmat - muG) @ (Mmat - muG).T / C).cpu().numpy()
        try:
            k      = min(C-1, Sb_np.shape[0] - 1)
            eigv, eigval, _ = svds(Sb_np, k=k)
            Σb_inv = eigv @ np.diag(eigval**-1) @ eigv.T
            Sw_invSb = np.trace(Sw.cpu().numpy() @ Σb_inv)
        except (ValueError, ArpackError):
            Sw_invSb = float('nan')


    # store
    M.accuracy.append(acc)
    M.loss.append(loss)
    M.Sw_invSb.append(Sw_invSb)
    handle.remove()
    feats.clear()


# -----------------------------------------------------------------------------
# CLI & Main script
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Measure NC on a RotNet experiment")
    p.add_argument('--exp',        required=True, help='experiment name, used to find config/config_<exp>.py')
    p.add_argument('--exp_dir',    default=None, help='(optional) full path to the experiment directory; overrides experiments/<exp>')
    p.add_argument('--checkpoint', type=int, default=0, help='epoch id of model_net_epochXX to load')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--no_cuda',    action='store_true')
    return p.parse_args()

if __name__=='__main__':
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:                       # i.e. you passed --no_cuda
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, map_location=torch.device('cpu'))
    set_seed(42)
    

    # 1) load config
    cfg_file = Path('config')/f"{args.exp}.py"
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


   # after loading your checkpoint…
    feat_extractor = algo.networks.get('feat_extractor',
                        algo.networks.get('model',
                                        list(algo.networks.values())[0]))

    # this is exactly the list of feature‐keys you can pass to forward(out_feat_keys=…)
    print("Available feature layers:", feat_extractor.all_feat_names)

    layers = feat_extractor.all_feat_names

    nc1_per_layer = {}
    for layer in tqdm(layers,desc="Measuring NC1 per layer"):
        # 4) measure over all layers
        metrics = Measurements()
        print(f"\n[Measurement] Loading checkpoint")
        # pull out the nn.Module exactly as before...
        # (your existing fallback logic that sets `model` from `algo`)
        model = None
        if hasattr(algo, 'networks'):
            for k,v in algo.networks.items():
                if isinstance(v, nn.Module):
                    model = v; break
        if model is None:
            for name in ('model','net'):
                cand = getattr(algo, name, None)
                if isinstance(cand, nn.Module):
                    model = cand; break
        if model is None:
            raise RuntimeError("Couldn't find a torch.nn.Module in algo")

        # now compute and append a single point
        compute_metrics(metrics, model, loader, C, layer)
        nc1_per_layer[layer] = metrics.Sw_invSb[-1]

    # 5) save plot curves vs epoch_list (instead of len=1)
    save_dir = Path('results')/f"{args.exp}_{type(model).__name__}"/f"bs{args.batch_size}"
    (save_dir / 'plots').mkdir(parents=True, exist_ok=True)

    x = list(nc1_per_layer.keys())
    y = [nc1_per_layer[l] for l in x]

    plt.figure()
    plt.plot(range(len(x)), y, 'bx-')
    plt.xticks(range(len(x)), x, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('NC1')
    plt.title(f'NC1 across layers at checkpoint')
    plt.tight_layout()
    plt.savefig(save_dir/'plots'/f"NC1_layers_checkpoint.pdf")
    plt.close()

    print("✓  Done – results in", save_dir)



