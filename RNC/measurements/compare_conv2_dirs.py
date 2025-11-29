#!/usr/bin/env python
# coding: utf-8

"""
Compare conv2 weight *directions* across two experiments.

Example:

python compare_conv2_dirs.py \
  --exp-a CIFAR10_RotNet_NIN4blocks_Collapsed_MSE \
  --exp-b CIFAR10_RotNet_NIN4blocks_NotCollapsed_MSE \
  --exp-dir-a ../experiments/CIFAR10_RotNet_NIN4blocks_Collapsed_MSE \
  --exp-dir-b ../experiments/CIFAR10_RotNet_NIN4blocks_NotCollapsed_MSE \
  --checkpoint 200 \
  --arch-class AlexNetwork   # or NetworkInNetwork, etc.
"""

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_weight_matrix(W: torch.Tensor) -> torch.Tensor:
    """
    Flatten weight tensor to 2D: (out_channels, -1).
    Works for Conv2d and Linear.
    """
    return W.view(W.shape[0], -1)


def row_normalise(W: torch.Tensor) -> torch.Tensor:
    """
    Row-wise ℓ2 normalisation.
    """
    return W / (W.norm(dim=1, keepdim=True) + 1e-12)


def load_config(exp: str) -> Dict:
    """
    Load config/<exp>.py and return 'config' dict.
    """
    cfg_file = Path('config') / f"{exp}.py"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)  # type: ignore
    return cfg_mod.config


def build_fresh_model(
    config: dict,
    net_key: str,
    arch_class: Optional[str],
    use_cuda: bool,
) -> nn.Module:
    """
    Instantiate a fresh model according to config['networks'][net_key].

    Expected config:

        networks[net_key] = {
            'def_file': 'path/to/model_def.py',
            'opt':      { ... kwargs for constructor ... },
            'arch':     'ClassNameInFile',   # optional
            'optim_params': {...}            # ignored here
        }
    """
    net_cfg_all = config.get('networks', {})
    if net_key not in net_cfg_all:
        raise RuntimeError(
            f"net_key '{net_key}' not found in config['networks']. "
            f"Available: {list(net_cfg_all.keys())}"
        )

    net_cfg = net_cfg_all[net_key]

    def_file = net_cfg['def_file']
    opt_dict = net_cfg.get('opt', {})

    module_path = Path(def_file)
    if not module_path.is_file():
        raise FileNotFoundError(f"def_file not found: {module_path}")

    spec_model = importlib.util.spec_from_file_location(
        module_path.stem, module_path
    )
    mod_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(mod_model)  # type: ignore

    cls_name = arch_class or net_cfg.get('arch', module_path.stem)
    if not hasattr(mod_model, cls_name):
        raise RuntimeError(
            f"Could not find class '{cls_name}' in {def_file}. "
            "Either pass --arch-class, or add 'arch' to "
            f"config['networks']['{net_key}']."
        )

    ModelCls = getattr(mod_model, cls_name)
    model = ModelCls(**opt_dict)

    if use_cuda:
        model = model.cuda()

    return model


def discover_checkpoint(
    exp_dir: Path,
    net_key: str,
    checkpoint: int,
    ckpt_glob: Optional[str] = None,
) -> Path:
    """
    Given an experiment directory and an epoch id, find the corresponding checkpoint.

    By default, looks for files like '<net_key>_net_epoch*' and parses the
    integer after 'epoch' as the epoch id.
    """
    pattern = ckpt_glob or f"{net_key}_net_epoch*"
    ckpt_files = list(exp_dir.glob(pattern))
    if not ckpt_files:
        raise RuntimeError(
            f"No checkpoints found in {exp_dir} matching pattern '{pattern}'."
        )

    epoch_to_path: Dict[int, Path] = {}
    for f in ckpt_files:
        stem = f.stem
        try:
            ep_str = stem.split('epoch')[-1]
            ep = int(ep_str)
            epoch_to_path[ep] = f
        except ValueError:
            continue

    if not epoch_to_path:
        raise RuntimeError(
            f"Could not parse epochs from checkpoints in {exp_dir} "
            f"with pattern '{pattern}'."
        )

    if checkpoint not in epoch_to_path:
        raise RuntimeError(
            f"Requested checkpoint epoch {checkpoint} not found in {exp_dir}; "
            f"available: {sorted(epoch_to_path.keys())}"
        )

    return epoch_to_path[checkpoint]


def get_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """
    Return list of (name, Conv2d) in model.named_modules() order.
    """
    conv_layers: List[Tuple[str, nn.Conv2d]] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append((name, m))
    return conv_layers


def extract_conv2_weights(model: nn.Module, conv_index: int = 1) -> Tuple[str, torch.Tensor]:
    """
    Extract conv[layer_index] (default conv2 = index 1) weight as 2D (O, D).
    """
    conv_layers = get_conv_layers(model)
    if len(conv_layers) <= conv_index:
        raise RuntimeError(
            f"Model has only {len(conv_layers)} Conv2d layers; "
            f"cannot access index {conv_index}."
        )

    name, mod = conv_layers[conv_index]
    W = mod.weight.detach().cpu()
    W2d = flatten_weight_matrix(W)  # (out_channels, D)
    return name, W2d


def principal_angle_stats(WA: torch.Tensor, WB: torch.Tensor) -> Dict[str, float]:
    """
    Compute principal angles between the row-spaces of WA and WB.

    Steps:
      - WA: (Na, D), WB: (Nb, D)
      - SVD: WA = U_A S_A V_A^T; row-space is span of columns of V_A[:, :r_A]
      - Subspace overlap matrix: M = V_A_r^T V_B_r
      - Singular values of M are cos(theta_i).
    """
    # Center rows (optional; mainly to remove bias)
    WA_c = WA - WA.mean(dim=1, keepdim=True)
    WB_c = WB - WB.mean(dim=1, keepdim=True)

    # SVD
    Ua, Sa, Va = torch.linalg.svd(WA_c, full_matrices=False)
    Ub, Sb, Vb = torch.linalg.svd(WB_c, full_matrices=False)

    # effective rank
    tol = 1e-7
    ra = int((Sa > tol).sum())
    rb = int((Sb > tol).sum())
    if ra == 0 or rb == 0:
        raise RuntimeError("One of the conv2 matrices appears to be rank 0 after thresholding.")

    r = min(ra, rb)

    Va_r = Va[:, :r]  # (D, r)
    Vb_r = Vb[:, :r]

    # Subspace overlap
    M = Va_r.T @ Vb_r  # (r, r)
    _, S, _ = torch.linalg.svd(M, full_matrices=False)  # S = cos(theta_i)

    # Clip numerically
    S = torch.clamp(S, 0.0, 1.0)
    theta = torch.acos(S)  # radians

    return {
        "min_cos":    float(S.min().item()),
        "max_cos":    float(S.max().item()),
        "mean_cos":   float(S.mean().item()),
        "median_cos": float(S.median().item()),
        "min_angle_deg":  float(theta.max().item() * 180.0 / np.pi),  # note: max angle = acos(min_cos)
        "max_angle_deg":  float(theta.min().item() * 180.0 / np.pi),  # min angle = acos(max_cos)
        "mean_angle_deg": float(theta.mean().item() * 180.0 / np.pi),
    }


def filter_match_stats(WA: torch.Tensor, WB: torch.Tensor) -> Dict[str, float]:
    """
    For each filter in WA, find the most similar filter in WB (cosine similarity), and vice versa.

    WA: (Na, D)
    WB: (Nb, D)

    Returns mean / median / min / max of best-match cosine similarities
    in both directions.
    """
    WA_n = row_normalise(WA)
    WB_n = row_normalise(WB)

    # Cosine similarity matrix (Na, Nb)
    S = WA_n @ WB_n.T

    best_A = S.max(dim=1).values  # for each filter in A: best match in B
    best_B = S.max(dim=0).values  # for each filter in B: best match in A

    def stats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_mean":   float(x.mean().item()),
            f"{prefix}_median": float(x.median().item()),
            f"{prefix}_min":    float(x.min().item()),
            f"{prefix}_max":    float(x.max().item()),
        }

    out = {}
    out.update(stats(best_A, "A_to_B"))
    out.update(stats(best_B, "B_to_A"))
    return out


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Compare conv2 weight directions across two experiments")

    # Experiment A
    p.add_argument('--exp-a',        required=True,
                   help='Name of experiment A (used to find config/<exp-a>.py)')
    p.add_argument('--exp-dir-a',    default=None,
                   help='Optional explicit experiments directory for A; '
                        'if omitted, use experiments/<exp-a>')
    p.add_argument('--arch-class-a', type=str, default=None,
                   help='Optional architecture class for experiment A; overrides config["networks"][net_key]["arch"].')

    # Experiment B
    p.add_argument('--exp-b',        required=True,
                   help='Name of experiment B (used to find config/<exp-b>.py)')
    p.add_argument('--exp-dir-b',    default=None,
                   help='Optional explicit experiments directory for B; '
                        'if omitted, use experiments/<exp-b>')
    p.add_argument('--arch-class-b', type=str, default=None,
                   help='Optional architecture class for experiment B.')

    # Shared
    p.add_argument('--checkpoint',   type=int, default=200,
                   help='Epoch id to load (e.g. 200).')
    p.add_argument('--net-key',      type=str, default='model',
                   help="Key in config['networks'] for the primary model (default: 'model').")
    p.add_argument('--ckpt-glob',    type=str, default=None,
                   help="Glob pattern for checkpoints relative to exp_dir, e.g. 'model_net_epoch*'. "
                        "If omitted, defaults to f'{net_key}_net_epoch*'.")
    p.add_argument('--conv-index',   type=int, default=1,
                   help='Index of Conv2d layer to compare (0-based). Default 1 = conv2.')
    p.add_argument('--no_cuda',      action='store_true',
                   help='Force CPU evaluation even if a GPU is visible.')

    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    if not use_cuda:
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(
            f, map_location=torch.device('cpu'), **kw
        )

    set_seed(42)

    # ---------------- Experiment A ----------------
    config_a = load_config(args.exp_a)
    if args.exp_dir_a is not None:
        exp_dir_a = Path(args.exp_dir_a)
    else:
        exp_dir_a = Path('experiments') / args.exp_a
    if not exp_dir_a.is_dir():
        raise FileNotFoundError(f"Experiment directory A not found: {exp_dir_a}")

    ckpt_a = discover_checkpoint(exp_dir_a, args.net_key, args.checkpoint, args.ckpt_glob)
    print(f"[A] Using checkpoint: {ckpt_a}")

    model_a = build_fresh_model(
        config=config_a,
        net_key=args.net_key,
        arch_class=args.arch_class_a,
        use_cuda=use_cuda,
    )

    state_a = torch.load(ckpt_a, map_location='cpu')
    if isinstance(state_a, dict) and 'state_dict' in state_a:
        model_a.load_state_dict(state_a['state_dict'])
    else:
        model_a.load_state_dict(state_a)

    # ---------------- Experiment B ----------------
    config_b = load_config(args.exp_b)
    if args.exp_dir_b is not None:
        exp_dir_b = Path(args.exp_dir_b)
    else:
        exp_dir_b = Path('experiments') / args.exp_b
    if not exp_dir_b.is_dir():
        raise FileNotFoundError(f"Experiment directory B not found: {exp_dir_b}")

    ckpt_b = discover_checkpoint(exp_dir_b, args.net_key, args.checkpoint, args.ckpt_glob)
    print(f"[B] Using checkpoint: {ckpt_b}")

    model_b = build_fresh_model(
        config=config_b,
        net_key=args.net_key,
        arch_class=args.arch_class_b,
        use_cuda=use_cuda,
    )

    state_b = torch.load(ckpt_b, map_location='cpu')
    if isinstance(state_b, dict) and 'state_dict' in state_b:
        model_b.load_state_dict(state_b['state_dict'])
    else:
        model_b.load_state_dict(state_b)

    # ---------------- Extract conv2 weights ----------------
    name_a, W2d_a = extract_conv2_weights(model_a, conv_index=args.conv_index)
    name_b, W2d_b = extract_conv2_weights(model_b, conv_index=args.conv_index)

    print(f"\n[A] Conv layer index {args.conv_index}: {name_a}, weight shape {tuple(W2d_a.shape)}")
    print(f"[B] Conv layer index {args.conv_index}: {name_b}, weight shape {tuple(W2d_b.shape)}")

    if W2d_a.shape[1] != W2d_b.shape[1]:
        raise RuntimeError(
            f"Conv2 flattened dims differ between models: "
            f"D_A={W2d_a.shape[1]} vs D_B={W2d_b.shape[1]}"
        )

    # ---------------- Subspace comparison (principal angles) ----------------
    pa_stats = principal_angle_stats(W2d_a, W2d_b)
    print("\n=== Principal-angle statistics between conv2 row-spaces ===")
    for k, v in pa_stats.items():
        print(f"{k:>16}: {v:.6f}")

    # ---------------- Filter-wise best-match cosine similarities ------------ 
    match_stats = filter_match_stats(W2d_a, W2d_b)
    print("\n=== Filter-wise best-match cosine similarities ===")
    for k, v in match_stats.items():
        print(f"{k:>16}: {v:.6f}")

    print("\nDone.")
