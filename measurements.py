#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
Neural-Collapse Metric Suite for RotNet Checkpoints
================================================================================

This script scans **one or many** saved checkpoints from a RotNet experiment
and produces the four canonical Neural-Collapse statistics (NC-1 … NC-4) plus
auxiliary curves (loss, accuracy, NCC mismatch).  Results are stored as a
`metrics.pkl` file *and* as tidy PDF plots vs. training epoch.

Typical one-liner
-----------------
    # analyse a single checkpoint on CPU
    python measurements.py \
        --exp       CIFAR10_RotNet_NIN4blocks \
        --checkpoint 44 \
        --batch-size 512 \
        --workers   0 \
        --no_cuda

Full feature set
----------------
Argument            | Meaning & examples
------------------- | ---------------------------------------------------------
`--exp` *NAME*      | Experiment name **without** path. Must have a matching
                    | `config/config_<NAME>.py` and (by default) checkpoints in
                    | `experiments/<NAME>/`.
`--exp_dir` *PATH*  | **Override** the automatic directory above, e.g.\
                    | `--exp_dir "D:/runs/RotNet/CIFAR10_RotNet_NIN4blocks"`.
`--checkpoint` *N*  | Epoch ID to load first (default 0 = start from earliest
                    | checkpoint found). \
                    | Use together with `--start-epoch / --end-epoch` to scan a
                    | **range** of epochs in one shot.
`--start-epoch` *N* | First epoch to analyse (inclusive, default 1).
`--end-epoch` *N*   | Last  epoch to analyse (inclusive, default = latest file).
`--batch-size`      | Forward-pass batch size for metric computation
                    | (does **not** affect training).
`--workers`         | Dataloader workers (same as in training config).
`--no_cuda`         | Force CPU evaluation even if a GPU is visible.

What the script actually does
-----------------------------
1. **Load config**   (`config/<exp>.py`) – identical to *main.py*.
2. **Recreate loaders** for the 4-rotation self-supervised task.
3. **Instantiate** the chosen `Algorithm` class, *but without* resuming
   optimisation (we only need the networks).
4. **Iterate over checkpoints**  
   (`<exp_dir>/*_net_epochXX`) and for each epoch:
   • feed the entire train split once  
   • compute NC-1 … NC-4, loss, accuracy, NCC-mismatch.
5. **Save outputs** to  
   `results/<exp>_<arch>/bs<batch>_epochs<first>-<last>/`:
   * `metrics.pkl`   – pickled dict with `epochs` and all curves  
   * `plots/*.pdf`   – one figure per metric.

The script is read-only with respect to your experiment folder; it never
overwrites checkpoints.
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
        self.Sw_invSb = []
        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M    = []
        self.cos_W    = []
        self.W_M_dist = []
        self.NCC_mismatch = []

# -----------------------------------------------------------------------------
# Compute NC metrics in one full pass
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(M: Measurements, model: nn.Module, loader: DataLoader, C: int, feat_layer: str):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval().to(device)

    # Automatically find the classification layer (last nn.Linear)
    cls_layer = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            cls_layer = m
    if cls_layer is None:
        raise RuntimeError("No nn.Linear layer found in the model to hook features.")

    # hook on classification layer to grab penultimate features
    feats: Dict[str, torch.Tensor] = {}
    def hook(module, inp, out):
        feats['h'] = inp[0].detach().cpu()
    handle = cls_layer.register_forward_hook(hook)

    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[torch.Tensor] = []
    Sw = torch.zeros(0)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = net_correct = NCC_match = 0

    # PASS 1: means, loss, accuracy
    print("[Measurement] PASS 1: computing class means, accuracy, and loss...")
    for x, y in tqdm(loader, desc="PASS 1", unit="batch"):  # progress bar

        x, y = x.to(device), y.to(device)
        out = model(x)
        h   = feats['h'].view(len(x), -1)

        if Sw.numel()==0:
            Sw = torch.zeros(h.size(1), h.size(1))

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
                z  = hc - means[c]
                Sw += (z.unsqueeze(2) @ z.unsqueeze(1)).sum(0)

                dists = torch.norm(hc.unsqueeze(1) - Mmat.T.unsqueeze(0), dim=2)
                NCCp  = dists.argmin(dim=1)
                netp  = out[idx].argmax(1).cpu()
                NCC_match += (NCCp==netp).sum().item()

        print("[Measurement] Finalizing metrics...")

    # finalize
    N = N_per_class.sum().item()
    Sw /= N
    loss = total_loss / N
    acc  = net_correct / N
    NCCm = 1 - NCC_match / N

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


    # NC-2
    W = cls_layer.weight.T
    M_c = (Mmat - muG).to(W.device)
    Mn, Wn = M_c.norm(p=2, dim=0), W.norm(p=2, dim=0)
    covM = (Mn.std(unbiased=False)/Mn.mean()).item()
    covW = (Wn.std(unbiased=False)/Wn.mean()).item()
    def coherence(V):
        G = V.T@V
        G = G/(G.norm(1,keepdim=True)+1e-9)
        G.fill_diagonal_(0)
        return (G.abs().sum()/(C*(C-1))).item()
    cosM = coherence(M_c/Mn)
    cosW = coherence(W/Wn)

    # NC-3
    W_M_dist = (W/W.norm() - M_c/M_c.norm()).norm().pow(2).item()

    # store
    M.accuracy.append(acc); M.loss.append(loss)
    M.Sw_invSb.append(Sw_invSb)
    M.norm_M_CoV.append(covM); M.norm_W_CoV.append(covW)
    M.cos_M.append(cosM); M.cos_W.append(cosW)
    M.W_M_dist.append(W_M_dist); M.NCC_mismatch.append(NCCm)

    handle.remove()

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
    p.add_argument('--start-epoch', type=int, default=1, help='first epoch to measure (inclusive)')
    p.add_argument('--end-epoch',   type=int, default=None, help='last epoch to measure (inclusive)')
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
        batch_size=dt['batch_size'],
        unsupervised=dt['unsupervised'],
        epoch_size=dt['epoch_size'],
        num_workers=args.workers,
        shuffle=False
    )(0)

    C = 4  # RotNet 4 rotations
    feat_layer = config.get('feature_layer', 'encoder')

    # 3) instantiate alg + load checkpoint
    import algorithms as alg
    algo = getattr(alg, config['algorithm_type'])(config)
    if use_cuda:
        algo.load_to_gpu()
    algo.load_checkpoint(args.checkpoint, train=False)

    # figure out which epochs to process
    exp_dir = Path(config['exp_dir'])
    # grab all "model_net_epochXX" files and extract XX
    all_files = list(exp_dir.glob('*_net_epoch*'))
    all_epochs = sorted(int(f.stem.split('epoch')[-1]) for f in all_files)
    start = args.start_epoch
    end   = args.end_epoch or max(all_epochs)
    epoch_list = [e for e in all_epochs if start <= e <= end]
    if not epoch_list:
        raise RuntimeError(f"No checkpoints found in {exp_dir} between epochs {start}–{end}")

    # 4) measure over all epochs
    metrics = Measurements()
    for e in epoch_list:
        print(f"\n[Measurement] Loading checkpoint epoch {e}")
        algo.load_checkpoint(e, train=False)
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
        compute_metrics(metrics, model, loader, C, feat_layer)

    # 5) save  plot curves vs epoch_list (instead of len=1)
    save_dir = Path('results')/f"{args.exp}_{type(model).__name__}"/f"bs{args.batch_size}_epochs{start}-{end}"
    (save_dir/'plots').mkdir(parents=True, exist_ok=True)
    with open(save_dir/'metrics.pkl','wb') as f:
        pickle.dump({'epochs': epoch_list, 'metrics': metrics}, f)

    # plotting
    for name in vars(metrics):
        plt.figure()
        plt.plot(epoch_list, getattr(metrics, name), 'bx-')
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.title(f"{name} vs epoch")
        plt.tight_layout()
        plt.savefig(save_dir/'plots'/f"{name}.pdf")
        plt.close()

    print("✓  Done – results in", save_dir)