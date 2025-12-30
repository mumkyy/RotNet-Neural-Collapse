#!/usr/bin/env python3
# coding: utf-8

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------- reproducibility ----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- simple container ----------------------

class Measurements:
    def __init__(self):
        self.epochs: List[int] = []
        self.accuracy: List[float] = []
        self.loss: List[float] = []
        self.nc1: List[float] = []
        self.nc3: List[float] = []


# ---------------------- config + model loading ----------------------

def load_config(config_root: str, exp: str) -> dict:
    cfg_file = Path(config_root) / f"{exp}.py"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.config


def discover_checkpoints(exp_dir: Path, ckpt_glob: str) -> Dict[int, Path]:
    files = list(exp_dir.glob(ckpt_glob))
    if not files:
        raise RuntimeError(f"No ckpts found in {exp_dir} matching '{ckpt_glob}'")

    out: Dict[int, Path] = {}
    for f in files:
        stem = f.stem
        if "epoch" not in stem:
            continue
        try:
            ep = int(stem.split("epoch")[-1])
            out[ep] = f
        except ValueError:
            continue

    if not out:
        raise RuntimeError("Found ckpt files but failed to parse epoch numbers.")
    return out


def load_state_dict(model: nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            sd = state["state_dict"]
        elif "network" in state:
            sd = state["network"]
        elif "model" in state:
            sd = state["model"]
        else:
            sd = state
    else:
        sd = state
    model.load_state_dict(sd, strict=True)


def find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear found (needed for NC3 with W).")
    return last


# ---------------------- dataloader builder (CIFAR10, 4-class pretext) ----------------------

def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None or s.strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def build_cifar10_pretext_loader(
    split: str,
    batch_size: int,
    workers: int,
    pretext_mode: str,
    sigmas: Optional[List[float]],
    kernel_sizes: Optional[List[int]],
    patch_jitter: int,
    color_distort: bool,
    color_dist_strength: float,
    shuffle: bool,
):
    """
    Returns an iterator that yields (x, y) where:
      x: Tensor (N, C, H, W)
      y: LongTensor (N,)
    and labels are in [0..3] for 4-class pretext.
    """
    from dataloader import GenericDataset, DataLoader as RotLoader

    ds = GenericDataset(
        dataset_name="cifar10",
        split=split,
        random_sized_crop=False,
        num_imgs_per_cat=None,
        pretext_mode=pretext_mode,
        sigmas=sigmas,
        kernel_sizes=kernel_sizes,
        patch_jitter=patch_jitter,
        color_distort=color_distort,
        color_dist_strength=color_dist_strength,
    )

    # Important: unsupervised=True so it emits pretext labels
    loader = RotLoader(
        dataset=ds,
        batch_size=batch_size,
        unsupervised=True,
        epoch_size=None,
        num_workers=workers,
        shuffle=shuffle,
    )(0)

    return loader


# ---------------------- NC1 + accuracy + loss (+ NC3 optional) ----------------------

@torch.no_grad()
def compute_nc1_acc_loss_nc3(
    M: Measurements,
    model: nn.Module,
    loader,
    C: int,
    feat_layer: str,
    device: torch.device,
):
    """
    Assumes your model exposes:
      - model.all_feat_names
      - model._feature_blocks
    and feat_layer is one of all_feat_names.

    Features used: output of the chosen feature block.
    NC1 uses class means + within scatter (trace form).
    NC3 uses last linear weights W aligned to centered means (standard NC3).
    """

    model.eval().to(device)

    # ---- hook chosen feature block ----
    if not hasattr(model, "all_feat_names") or not hasattr(model, "_feature_blocks"):
        raise RuntimeError("Model must have model.all_feat_names and model._feature_blocks for this simple script.")

    if feat_layer not in model.all_feat_names:
        raise RuntimeError(f"feat_layer='{feat_layer}' not in model.all_feat_names: {model.all_feat_names}")

    feat_idx = model.all_feat_names.index(feat_layer)
    feat_module = model._feature_blocks[feat_idx]

    feats: Dict[str, torch.Tensor] = {}

    def feat_hook(_m, _inp, out):
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("Feature hook output is not a tensor.")
        feats["h"] = out.detach()

    handle = feat_module.register_forward_hook(feat_hook)

    # ---- PASS 1: means + acc + loss ----
    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[Optional[torch.Tensor]] = [None for _ in range(C)]

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    net_correct = 0
    totalN = 0

    for x, y in tqdm(loader, desc="PASS1 (means/acc/loss)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        if "h" not in feats:
            raise RuntimeError("Feature hook did not fire.")

        h = feats["h"]
        if h.dim() > 2:
            h = h.flatten(1)  # (N, D)

        bs = y.size(0)
        totalN += bs

        total_loss += loss_fn(out, y).item() * bs
        net_correct += (out.argmax(1) == y).sum().item()

        # do class sums on CPU for safe indexing
        y_cpu = y.detach().cpu()
        h_cpu = h.detach().cpu()

        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            hc = h_cpu[idx]
            if sum_per_class[c] is None:
                sum_per_class[c] = hc.sum(0)
            else:
                sum_per_class[c] += hc.sum(0)
            N_per_class[c] += hc.size(0)

    if totalN == 0:
        raise RuntimeError("No samples seen.")

    means: List[torch.Tensor] = []
    for c in range(C):
        n = int(N_per_class[c].item())
        if n == 0:
            raise RuntimeError(f"Class {c} has 0 samples (cannot compute NC).")
        means.append(sum_per_class[c] / float(n))  # type: ignore

    # Mmat: (D, C) on CPU
    Mmat = torch.stack(means, dim=1)
    muG = Mmat.mean(dim=1, keepdim=True)

    # ---- PASS 2: within-class scatter ----
    total_ss = 0.0
    for x, y in tqdm(loader, desc="PASS2 (Sw)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        _ = model(x)

        h = feats["h"]
        if h.dim() > 2:
            h = h.flatten(1)

        y_cpu = y.detach().cpu()
        h_cpu = h.detach().cpu()

        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            hc = h_cpu[idx]
            z = hc - means[c]  # both CPU
            total_ss += z.pow(2).sum().item()

    # ---- finalize NC1 ----
    eps = 1e-12
    trace_Sw = total_ss / float(totalN)

    M_centered = Mmat - muG
    Sb = (M_centered @ M_centered.T) / float(C)
    trace_Sb = Sb.trace().item()

    nc1_ratio = trace_Sw / (trace_Sb + eps)

    # ---- NC3 (simple, standard) ----
    # W: (C, D) -> transpose to (D, C) on CPU
    cls_layer = find_last_linear(model)
    W = cls_layer.weight.detach().cpu().T  # (D, C)

    Wn = W / (W.norm() + eps)
    Mn = M_centered / (M_centered.norm() + eps)
    nc3 = (Wn - Mn).pow(2).sum().item()

    # ---- store ----
    M.accuracy.append(net_correct / float(totalN))
    M.loss.append(total_loss / float(totalN))
    M.nc1.append(float(nc1_ratio))
    M.nc3.append(float(nc3))

    handle.remove()
    feats.clear()


# ---------------------- plotting ----------------------

def plot_curves(save_dir: Path, M: Measurements) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    def _plot(name: str, y: List[float]):
        plt.figure()
        plt.plot(M.epochs, y, "bx-")
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(f"{name} vs epoch")
        plt.tight_layout()
        plt.savefig(save_dir / "plots" / f"{name}.pdf")
        plt.close()

    _plot("accuracy", M.accuracy)
    _plot("loss", M.loss)
    _plot("nc1", M.nc1)
    _plot("nc3", M.nc3)


# ---------------------- args ----------------------

def parse_args():
    p = argparse.ArgumentParser("Simple NC (CIFAR10, 4-class pretext)")

    # experiment / checkpoints
    p.add_argument("--exp", required=True, help="config/<exp>.py and experiments/<exp>/")
    p.add_argument("--config-root", default="config")
    p.add_argument("--exp-dir", default=None, help="Override experiments/<exp>")
    p.add_argument("--ckpt-glob", default=None, help="Default: model_net_epoch*")
    p.add_argument("--checkpoint", type=int, default=None, help="Evaluate only this epoch")
    p.add_argument("--start-epoch", type=int, default=None)
    p.add_argument("--end-epoch", type=int, default=None)
    p.add_argument("--stride", type=int, default=10, help="Subsample epochs if no range/checkpoint provided")

    # eval
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # pretext knobs (4-class assumption)
    p.add_argument("--pretext-mode", choices=["rotation", "gaussian_noise", "gaussian_blur", "jigsaw"],
                   required=True)
    p.add_argument("--num-classes", type=int, default=4)

    # gaussian_noise: provide 4 sigmas
    p.add_argument("--sigmas", type=str, default=None,
                   help="Comma list of 4 sigmas, e.g. 0.001,0.01,0.1,1.0")

    # gaussian_blur: provide 4 kernel sizes
    p.add_argument("--kernel-sizes", type=str, default=None,
                   help="Comma list of 4 kernel sizes, e.g. 3,5,7,9")

    # jigsaw knobs
    p.add_argument("--patch-jitter", type=int, default=0)
    p.add_argument("--color-distort", action="store_true")
    p.add_argument("--color-dist-strength", type=float, default=1.0)

    # which feature layer to measure on
    p.add_argument("--feat-layer", type=str, required=True,
                   help="Must be in model.all_feat_names (e.g. 'Block3_ConvB2' etc.)")

    # output
    p.add_argument("--out-root", default="results_simple_nc")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    config = load_config(args.config_root, args.exp)

    # exp_dir
    exp_dir = Path(args.exp_dir) if args.exp_dir else (Path("experiments") / args.exp)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    # checkpoints
    ckpt_glob = args.ckpt_glob or "model_net_epoch*"
    epoch_to_path = discover_checkpoints(exp_dir, ckpt_glob)
    all_epochs = sorted(epoch_to_path.keys())

    if args.checkpoint is not None:
        epochs = [args.checkpoint]
    elif args.start_epoch is not None or args.end_epoch is not None:
        s = args.start_epoch if args.start_epoch is not None else min(all_epochs)
        e = args.end_epoch if args.end_epoch is not None else max(all_epochs)
        epochs = [ep for ep in all_epochs if s <= ep <= e]
    else:
        # subsample
        mx = max(all_epochs)
        epochs = sorted({ep for ep in all_epochs if (ep % args.stride == 0) or (ep == mx)})

    # pretext param parsing
    sigmas = parse_float_list(args.sigmas)
    kernel_sizes = parse_int_list(args.kernel_sizes)

    # for gaussian modes, enforce 4 classes unless user overrides
    if args.pretext_mode == "gaussian_noise" and sigmas is None:
        sigmas = [1e-3, 1e-2, 1e-1, 1.0]
    if args.pretext_mode == "gaussian_blur" and kernel_sizes is None:
        kernel_sizes = [3, 5, 7, 9]

    # build loader (unsupervised pretext)
    loader = build_cifar10_pretext_loader(
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        pretext_mode=args.pretext_mode,
        sigmas=sigmas,
        kernel_sizes=kernel_sizes,
        patch_jitter=args.patch_jitter,
        color_distort=args.color_distort,
        color_dist_strength=args.color_dist_strength,
        shuffle=False,
    )

    # build model fresh from config
    # assumes config['networks']['model'] points at correct def_file + opts
    from measurements import build_fresh_model  # if you already have this helper
    model = build_fresh_model(config=config, net_key="model", arch_class=None, use_cuda=use_cuda)

    M = Measurements()

    # output dir
    out_dir = Path(args.out_root) / f"{args.exp}_{args.pretext_mode}_{args.split}_{args.feat_layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in epochs:
        ckpt = epoch_to_path[ep]
        print(f"\n[SimpleNC] epoch {ep} -> {ckpt.name}")

        load_state_dict(model, ckpt)
        model.to(device)

        compute_nc1_acc_loss_nc3(
            M=M,
            model=model,
            loader=loader,
            C=args.num_classes,
            feat_layer=args.feat_layer,
            device=device,
        )

        M.epochs.append(ep)
        print(f"[SimpleNC] acc={M.accuracy[-1]:.4f} loss={M.loss[-1]:.4f} nc1={M.nc1[-1]:.6f} nc3={M.nc3[-1]:.6f}")

    # save + plot
    with open(out_dir / "metrics.pkl", "wb") as f:
        pickle.dump({"epochs": M.epochs, "acc": M.accuracy, "loss": M.loss, "nc1": M.nc1, "nc3": M.nc3}, f)

    plot_curves(out_dir, M)
    print(f"\n✓ Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
