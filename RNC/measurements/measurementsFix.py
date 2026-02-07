#!/usr/bin/env python3
# coding: utf-8

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import permutations


# -------------------- utils --------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None or s.strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -------------------- config / model --------------------

def load_config(config_root: str, exp: str) -> dict:
    cfg_file = Path(config_root) / f"{exp}.py"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.config


def build_fresh_model(config: dict, net_key: str, arch_class: Optional[str], use_cuda: bool) -> nn.Module:
    net_cfg_all = config.get("networks", {})
    if net_key not in net_cfg_all:
        raise RuntimeError(f"net_key '{net_key}' not in config['networks']. Keys: {list(net_cfg_all.keys())}")

    net_cfg = net_cfg_all[net_key]
    def_file = Path(net_cfg["def_file"])
    if not def_file.is_file():
        raise FileNotFoundError(f"Architecture file not found: {def_file}")

    spec_model = importlib.util.spec_from_file_location(def_file.stem, def_file)
    mod_model = importlib.util.module_from_spec(spec_model)
    assert spec_model.loader is not None
    spec_model.loader.exec_module(mod_model)

    cls_name = arch_class or net_cfg.get("arch", def_file.stem)
    if not hasattr(mod_model, cls_name):
        raise RuntimeError(f"Class '{cls_name}' not found in {def_file}")

    ModelCls = getattr(mod_model, cls_name)
    opt_dict = net_cfg.get("opt", {}).copy()

    # Your NIN expects opt dict
    model = ModelCls(opt_dict)

    if use_cuda:
        model = model.cuda()
    return model


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
        raise RuntimeError("Found checkpoint files but failed to parse epoch numbers.")
    return out


# -------------------- CIFAR10 pretext loader --------------------

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
    fixed_perms: Optional[List[Tuple[int, ...]]] = None,
):
    """
    Returns the DataLoader OBJECT.
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
        fixed_perms=fixed_perms,
    )

    loader = RotLoader(
        dataset=ds,
        batch_size=batch_size,
        unsupervised=True,   # pretext labels
        epoch_size=None,
        num_workers=workers,
        shuffle=shuffle,
    )

    return loader


# -------------------- Imagenette pretext loader --------------------

def build_generic_pretext_loader(
    d_name: str,
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
    fixed_perms: Optional[List[Tuple[int, ...]]] = None,
    
):
    """
    Returns the DataLoader OBJECT.
    """
    from dataloader import GenericDataset, DataLoader as RotLoader

    ds = GenericDataset(
        dataset_name=d_name,
        split=split,
        random_sized_crop=False,
        num_imgs_per_cat=None,
        pretext_mode=pretext_mode,
        sigmas=sigmas,
        kernel_sizes=kernel_sizes,
        patch_jitter=patch_jitter,
        color_distort=color_distort,
        color_dist_strength=color_dist_strength,
        fixed_perms=fixed_perms,
    )

    loader = RotLoader(
        dataset=ds,
        batch_size=batch_size,
        unsupervised=True,   # pretext labels
        epoch_size=None,
        num_workers=workers,
        shuffle=shuffle,
    )

    return loader


# -------------------- NC math --------------------

def gapify(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert conv feature map -> vector per sample.
    If feat is (N, C, H, W): global average pool -> (N, C).
    If already (N, D): return as-is.
    """
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    if feat.dim() == 2:
        return feat
    return feat.flatten(1)


@torch.no_grad()
def nc4Fun(
    model: nn.Module,
    loader,  # DataLoader callable: loader(epoch) -> iterator
    means_penult: List[torch.Tensor],
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Returns: (nc4_match, nc4_mismatch, ncc_acc)
      nc4_match    = P(net_pred == ncc_pred)
      nc4_mismatch = 1 - nc4_match
      ncc_acc      = P(ncc_pred == y)
    """
    model.eval().to(device)

    # (C, D) on device
    Mu = torch.stack([m.to(device) for m in means_penult], dim=0)

    # final classifier prehook to grab penultimate h
    if not hasattr(model, "_feature_blocks"):
        raise RuntimeError("Expected model._feature_blocks (NetworkInNetwork).")
    cls_layer = model._feature_blocks[-1].Classifier

    penult: Dict[str, torch.Tensor] = {}

    def prehook(_m, inputs):
        penult["h"] = inputs[0].detach()

    handle = cls_layer.register_forward_pre_hook(prehook)

    total = 0
    match_ct = 0
    ncc_correct = 0

    it = loader(0)
    for x, y in tqdm(it, desc="NC4 (NCC match/acc)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)

        # Ensure we get logits in a consistent way
        out = model(x, out_feat_keys=["classifier"])
        logits = out[0] if isinstance(out, (list, tuple)) else out

        if "h" not in penult:
            raise RuntimeError("NC4 prehook didn't fire; penultimate not captured.")

        H = gapify(penult["h"])  # (B, D)

        # squared distances: (B, C)
        H2 = (H * H).sum(dim=1, keepdim=True)          # (B, 1)
        Mu2 = (Mu * Mu).sum(dim=1).view(1, -1)         # (1, C)
        dist2 = H2 - 2.0 * (H @ Mu.t()) + Mu2          # (B, C)

        ncc_pred = dist2.argmin(dim=1)                 # (B,)
        net_pred = logits.argmax(dim=1)                # (B,)

        total += y.numel()
        match_ct += (net_pred == ncc_pred).sum().item()
        ncc_correct += (ncc_pred == y).sum().item()

    handle.remove()

    if total == 0:
        raise RuntimeError("NC4 saw 0 samples.")

    nc4_match = match_ct / float(total)
    nc4_mismatch = 1.0 - nc4_match
    ncc_acc = ncc_correct / float(total)

    return nc4_match, nc4_mismatch, ncc_acc


# ==================== NEW: layerwise NC4 ====================

@torch.no_grad()
def nc4_layerwise(
    model: nn.Module,
    loader,  # callable: loader(epoch) -> iterator
    means_by_layer: Dict[str, List[torch.Tensor]],  # layer -> [mu_c] (CPU ok)
    layer_keys: List[str],
    num_classes: int,
    device: torch.device,
) -> Dict[str, Tuple[float, float, float]]:
    """
    For EACH layer ℓ, build NCC classifier using class means μ_c^(ℓ),
    then compare NCC predictions to the network's final-head predictions.

    Returns dict:
      layer_key -> (nc4_match, nc4_mismatch, ncc_acc)

    Definitions at layer ℓ:
      ncc_pred^(ℓ)(x) = argmin_c || h_ℓ(x) - μ_c^(ℓ) ||^2
      net_pred(x)     = argmax logits(x)  (final classifier output)

      nc4_match^(ℓ)   = P( net_pred == ncc_pred^(ℓ) )
      ncc_acc^(ℓ)     = P( ncc_pred^(ℓ) == y )
    """
    model.eval().to(device)

    # stack means on device: Mu[layer] is (C, Dℓ)
    Mu: Dict[str, torch.Tensor] = {}
    for k in layer_keys:
        if k not in means_by_layer:
            raise KeyError(f"means_by_layer missing key '{k}'.")
        Mu[k] = torch.stack([m.to(device) for m in means_by_layer[k]], dim=0)  # (C,D)

    total = 0
    match_ct: Dict[str, int] = {k: 0 for k in layer_keys}
    ncc_correct: Dict[str, int] = {k: 0 for k in layer_keys}

    out_keys = list(layer_keys)
    if "classifier" not in out_keys:
        out_keys.append("classifier")

    it = loader(0)
    for x, y in tqdm(it, desc="NC4 layerwise (NCC match/acc)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)

        outs = model(x, out_feat_keys=out_keys)
        if not isinstance(outs, (list, tuple)):
            raise RuntimeError("Expected model(..., out_feat_keys=...) to return list/tuple aligned with out_keys.")

        logits = outs[out_keys.index("classifier")]
        net_pred = logits.argmax(dim=1)

        total += y.numel()

        for k in layer_keys:
            feat = outs[out_keys.index(k)]
            H = gapify(feat)  # (B,D)

            Muk = Mu[k]  # (C,D)
            H2 = (H * H).sum(dim=1, keepdim=True)         # (B,1)
            Mu2 = (Muk * Muk).sum(dim=1).view(1, -1)      # (1,C)
            dist2 = H2 - 2.0 * (H @ Muk.t()) + Mu2        # (B,C)

            ncc_pred = dist2.argmin(dim=1)

            match_ct[k] += (net_pred == ncc_pred).sum().item()
            ncc_correct[k] += (ncc_pred == y).sum().item()

    if total == 0:
        raise RuntimeError("nc4_layerwise saw 0 samples.")

    out: Dict[str, Tuple[float, float, float]] = {}
    for k in layer_keys:
        m = match_ct[k] / float(total)
        out[k] = (m, 1.0 - m, ncc_correct[k] / float(total))

    return out


@torch.no_grad()
def compute_epoch_metrics_multilayer(
    model: nn.Module,
    loader,  # This is the DataLoader object (callable)
    num_classes: int,
    layer_keys: List[str],
    device: torch.device,
    return_means_penult: bool = False,
    return_means_by_layer: bool = False,   # NEW
) -> Union[
    Tuple[Dict[str, float], float, float, float],
    Tuple[Dict[str, float], float, float, float, List[torch.Tensor]],
    Tuple[Dict[str, float], float, float, float, Dict[str, List[torch.Tensor]]],
    Tuple[Dict[str, float], float, float, float, List[torch.Tensor], Dict[str, List[torch.Tensor]]],
]:
    """
    Returns:
      nc1_by_layer: dict layer_key -> NC1
      acc: accuracy
      loss: CE loss
      nc3: NC3 computed at classifier feature space (penultimate input to Linear)

    If return_means_penult=True:
      also returns means_penult: List[Tensor] of length C, each (D,)

    If return_means_by_layer=True:
      also returns means_by_layer: Dict[layer_key -> List[Tensor]] (each list length C)
    """
    model.eval().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # --- capture penultimate (input to final Linear) for NC3 ---
    if not hasattr(model, "_feature_blocks"):
        raise RuntimeError("Expected model._feature_blocks (NetworkInNetwork).")
    cls_layer = model._feature_blocks[-1].Classifier
    if not isinstance(cls_layer, nn.Linear):
        raise RuntimeError("Expected final classifier to be nn.Linear at model._feature_blocks[-1].Classifier")

    penult: Dict[str, torch.Tensor] = {}

    def prehook(_m, inputs):
        penult["h"] = inputs[0].detach()

    handle = cls_layer.register_forward_pre_hook(prehook)

    # ---- PASS 1: class means for each requested layer + penultimate, plus acc/loss ----
    C = num_classes
    sums_by_layer: Dict[str, List[Optional[torch.Tensor]]] = {
        k: [None for _ in range(C)] for k in layer_keys
    }
    N_per_class = torch.zeros(C, dtype=torch.long)

    sum_penult: List[Optional[torch.Tensor]] = [None for _ in range(C)]

    total_loss = 0.0
    correct = 0
    totalN = 0

    out_keys = list(layer_keys)
    if "classifier" not in out_keys:
        out_keys.append("classifier")

    iter_pass1 = loader(0)
    for x, y in tqdm(iter_pass1, desc="PASS1 (means/acc/loss)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)

        outs = model(x, out_feat_keys=out_keys)
        logits = outs[out_keys.index("classifier")]
        feats = outs if "classifier" in layer_keys else outs[:-1]

        if "h" not in penult:
            raise RuntimeError("Penultimate pre-hook didn't fire.")

        h_pen = gapify(penult["h"])

        bs = y.size(0)
        totalN += bs

        total_loss += loss_fn(logits, y).item() * bs
        correct += (logits.argmax(1) == y).sum().item()

        y_cpu = y.detach().cpu()

        hp_cpu = h_pen.detach().cpu()
        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            v = hp_cpu[idx]
            if sum_penult[c] is None:
                sum_penult[c] = v.sum(0)
            else:
                sum_penult[c] += v.sum(0)
            N_per_class[c] += v.size(0)

        for k, feat in zip(layer_keys, feats):
            fv = gapify(feat).detach().cpu()
            for c in range(C):
                idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                vc = fv[idx]
                if sums_by_layer[k][c] is None:
                    sums_by_layer[k][c] = vc.sum(0)
                else:
                    sums_by_layer[k][c] += vc.sum(0)

    if totalN == 0:
        raise RuntimeError("No samples seen in loader.")
    for c in range(C):
        if int(N_per_class[c].item()) == 0:
            raise RuntimeError(f"Class {c} has 0 samples in this split; cannot compute NC metrics.")

    means_by_layer: Dict[str, List[torch.Tensor]] = {}
    for k in layer_keys:
        means_by_layer[k] = [
            sums_by_layer[k][c] / float(int(N_per_class[c].item()))  # type: ignore
            for c in range(C)
        ]

    means_penult: List[torch.Tensor] = [
        sum_penult[c] / float(int(N_per_class[c].item()))  # type: ignore
        for c in range(C)
    ]

    # ---- PASS 2: within-class scatter for NC1 (per layer) ----
    total_ss_by_layer: Dict[str, float] = {k: 0.0 for k in layer_keys}

    iter_pass2 = loader(0)
    for x, y in tqdm(iter_pass2, desc="PASS2 (Sw per layer)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        outs = model(x, out_feat_keys=layer_keys)
        y_cpu = y.detach().cpu()

        for k, feat in zip(layer_keys, outs):
            fv = gapify(feat).detach().cpu()
            for c in range(C):
                idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                vc = fv[idx]
                z = vc - means_by_layer[k][c]
                total_ss_by_layer[k] += z.pow(2).sum().item()

    eps = 1e-12
    nc1_by_layer: Dict[str, float] = {}
    for k in layer_keys:
        Mmat = torch.stack(means_by_layer[k], dim=1)
        muG = Mmat.mean(dim=1, keepdim=True)
        M_centered = Mmat - muG

        Sb = (M_centered @ M_centered.T) / float(C)
        trSb = Sb.trace().item()
        trSw = total_ss_by_layer[k] / float(totalN)

        nc1_by_layer[k] = trSw / (trSb + eps)

    # ---- NC3 finalize (penultimate space only) ----
    Mmat_p = torch.stack(means_penult, dim=1)
    muG_p = Mmat_p.mean(dim=1, keepdim=True)
    Mc = Mmat_p - muG_p

    W = cls_layer.weight.detach().cpu().T
    Wn = W / (W.norm() + eps)
    Mn = Mc / (Mc.norm() + eps)
    nc3 = (Wn - Mn).pow(2).sum().item()

    acc = correct / float(totalN)
    loss = total_loss / float(totalN)

    handle.remove()
    penult.clear()

    # ---- NEW flexible returns ----
    if return_means_penult and return_means_by_layer:
        return nc1_by_layer, acc, loss, nc3, means_penult, means_by_layer
    if return_means_penult:
        return nc1_by_layer, acc, loss, nc3, means_penult
    if return_means_by_layer:
        return nc1_by_layer, acc, loss, nc3, means_by_layer
    return nc1_by_layer, acc, loss, nc3


# -------------------- plotting --------------------

def plot_and_save(save_dir: Path, epochs: List[int], series: Dict[str, List[float]], title_prefix: str) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)
    for name, vals in series.items():
        plt.figure()
        plt.plot(epochs, vals, "bx-")
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(f"{title_prefix} {name} vs epoch")
        plt.tight_layout()
        plt.savefig(save_dir / "plots" / f"{name}.pdf")
        plt.close()


def plot_nc1_layers(save_dir: Path, epochs: List[int], nc1_curves: Dict[str, List[float]], title: str) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    for k, vals in nc1_curves.items():
        plt.plot(epochs, vals, marker="o", label=k)
    plt.xlabel("epoch")
    plt.ylabel("NC1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "nc1_layers.pdf")
    plt.close()


def plot_nc1_by_layer(save_dir: Path, layer_keys: List[str], nc1_vals: List[float], title: str) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    x = list(range(len(layer_keys)))
    plt.plot(x, nc1_vals, "bx-")
    plt.xticks(x, layer_keys, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("NC1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "nc1_layers_final.pdf")
    plt.close()


def plot_nc4(save_dir: Path, epochs: List[int], match: List[float], mismatch: List[float], ncc_acc: List[float]) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, match, marker="o", label="nc4_match")
    plt.plot(epochs, mismatch, marker="o", label="nc4_mismatch")
    plt.plot(epochs, ncc_acc, marker="o", label="ncc_acc")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("NC4 / NCC metrics vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "nc4Plot.pdf")
    plt.close()


# ==================== NEW: plot layerwise NC4 at final epoch ====================

def plot_layerwise_nc4_final(
    save_dir: Path,
    layer_keys: List[str],
    lw_curves: Dict[str, Dict[str, List[float]]],
    epoch: int,
) -> None:
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    match = [lw_curves[k]["match"][-1] for k in layer_keys]
    mismatch = [lw_curves[k]["mismatch"][-1] for k in layer_keys]
    ncc_acc = [lw_curves[k]["ncc_acc"][-1] for k in layer_keys]

    plt.figure(figsize=(10, 5))
    x = list(range(len(layer_keys)))
    plt.plot(x, match, marker="o", label="match")
    plt.plot(x, mismatch, marker="o", label="mismatch")
    plt.plot(x, ncc_acc, marker="o", label="ncc_acc")
    plt.xticks(x, layer_keys, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("value")
    plt.title(f"Layerwise NC4 at epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "nc4_layerwise_final.pdf")
    plt.close()


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser("Simple NC1/NC3/Acc for NIN CIFAR10 4-class pretext")

    # experiment/config/ckpts
    p.add_argument("--exp", required=True, help="Config name (config-root/<exp>.py)")
    p.add_argument("--config-root", default="config")
    p.add_argument("--exp-dir", required=True, help="Directory where checkpoints live")
    p.add_argument("--ckpt-glob", default="model_net_epoch*", help='e.g. "model_net_epoch*"')
    p.add_argument("--net-key", default="model")
    p.add_argument("--arch-class", default="NetworkInNetwork")

    # epoch selection
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--start-epoch", type=int, default=None)
    p.add_argument("--end-epoch", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)

    # data/pretext
    p.add_argument("--split", choices=["train", "test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--pretext-mode", choices=["rotation", "gaussian_noise", "gaussian_blur", "jigsaw", "jigsaw_9"], required=True)
    p.add_argument("--num-classes", type=int, default=4)

    p.add_argument("--sigmas", type=str, default=None, help="Comma list (4 entries) for gaussian_noise")
    p.add_argument("--kernel-sizes", type=str, default=None, help="Comma list (4 entries) for gaussian_blur")

    p.add_argument("--patch-jitter", type=int, default=0)
    p.add_argument("--color-distort", action="store_true")
    p.add_argument("--color-dist-strength", type=float, default=1.0)

    p.add_argument(
        "--dataset_name_arg",
        type=str,
        default=None,
        help="if you want to use cifar10 do not use this flag, else type in the name of the dataset you want to build"
    )

    # layers to compute NC1 on (use exposed keys like conv1,conv2,conv3,...)
    p.add_argument(
        "--layers",
        type=str,
        default="conv1,conv2,conv3,conv4",
        help="Comma list of exposed feature keys for NC1 (e.g. conv1,conv2,conv3,conv4)",
    )

    # NC4 flags
    p.add_argument("--nc4", action="store_true", help="Compute NC4 (penultimate NCC Agreement / accuracy) and plot it")
    p.add_argument(
        "--nc4-layerwise",
        action="store_true",
        help="Compute layerwise NC4: NCC match/acc at each requested layer vs final-head predictions",
    )

    # misc
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=str, default="results_simple_nc")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    config = load_config(args.config_root, args.exp)

    # ---------------------------------------------------------
    # 1. SETUP & MODEL LOADING
    # ---------------------------------------------------------
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"exp-dir not found: {exp_dir}")

    model = build_fresh_model(
        config=config,
        net_key=args.net_key,
        arch_class=args.arch_class,
        use_cuda=use_cuda,
    )

    # Load the checkpoint we want to measure (or the last one found)
    epoch_to_path = discover_checkpoints(exp_dir, args.ckpt_glob)
    all_epochs = sorted(epoch_to_path.keys())

    # Determine epochs to process
    if args.checkpoint is not None:
        epochs = [args.checkpoint]
    elif args.start_epoch is not None or args.end_epoch is not None:
        s = args.start_epoch if args.start_epoch is not None else min(all_epochs)
        e = args.end_epoch if args.end_epoch is not None else max(all_epochs)
        epochs = [ep for ep in all_epochs if s <= ep <= e]
    else:
        mx = max(all_epochs)
        epochs = sorted({ep for ep in all_epochs if (ep % args.stride == 0) or (ep == mx)})

    # Load the FIRST epoch in the list just for calibration
    print(f"\n[Calibration] Loading epoch {epochs[0]} to find correct permutations...")
    load_state_dict(model, epoch_to_path[epochs[0]])
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 2. AUTO-DISCOVER PERMUTATIONS
    # ---------------------------------------------------------
    k = 4 #2x2 grid
    if args.pretext_mode == "jigsaw_9":
        K = 9  # 3x3 grid
    
    N = args.num_classes
    all_base_perms = list(permutations(range(1, K + 1)))  # 1-based logic from your file

    sigmas = parse_float_list(args.sigmas)
    kernel_sizes = parse_int_list(args.kernel_sizes)
    if args.pretext_mode == "gaussian_noise" and sigmas is None:
        sigmas = [1e-3, 1e-2, 1e-1, 1.0]
    if args.pretext_mode == "gaussian_blur" and kernel_sizes is None:
        kernel_sizes = [3, 5, 7, 9]

    best_perms = None
    best_acc = -1.0

    print(f"[Calibration] Testing 24 possible permutation sets...")

    def get_set_starting_at(start_idx: int) -> List[Tuple[int, ...]]:
        # EXACT logic from your maxHamming.py, but j is forced
        P_bar = all_base_perms.copy()
        P: List[Tuple[int, int, int, int]] = []
        j = start_idx  # Forced start

        i = 1
        while i <= N:
            P.append(P_bar[j])
            P_prime = P_bar[:j] + P_bar[j + 1 :]

            if i < N:
                n_p = len(P)
                n_pp = len(P_prime)
                D = np.zeros((n_p, n_pp), dtype=int)
                for r in range(n_p):
                    for c in range(n_pp):
                        D[r, c] = int(np.sum(np.array(P[r]) != np.array(P_prime[c])))

                D_bar = np.min(D, axis=0)
                j = int(np.argmax(D_bar))

            P_bar = P_prime
            i += 1

        return [tuple(x - 1 for x in p) for p in P]  # 0-based for loader

    for start_idx in range(len(all_base_perms)):
        candidate_perms = get_set_starting_at(start_idx)

        if args.dataset_name_arg is None: 
            calib_loader = build_cifar10_pretext_loader(
                split="test",
                batch_size=128,
                workers=0,  # fast, no MP overhead
                pretext_mode=args.pretext_mode,
                sigmas=sigmas,
                kernel_sizes=kernel_sizes,
                patch_jitter=args.patch_jitter,
                color_distort=False,
                color_dist_strength=0.0,
                shuffle=False,
                fixed_perms=candidate_perms,
            )
        else:
            calib_loader = build_generic_pretext_loader(
                d_name=args.dataset_name_arg,
                split="val",
                batch_size=128,
                workers=0,  # fast, no MP overhead
                pretext_mode=args.pretext_mode,
                sigmas=sigmas,
                kernel_sizes=kernel_sizes,
                patch_jitter=args.patch_jitter,
                color_distort=False,
                color_dist_strength=0.0,
                shuffle=False,
                fixed_perms=candidate_perms,

            )


        x, y = next(iter(calib_loader(0)))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out_raw = model(x)
            logits = out_raw[-1] if isinstance(out_raw, (list, tuple)) else out_raw
            acc = (logits.argmax(1) == y).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_perms = candidate_perms

        if best_acc > 0.90:
            print(f"[Calibration] Found match at index {start_idx}! Acc: {best_acc:.2%}")
            break

    if best_acc < 0.5:
        print(f"\n[WARNING] Best match was only {best_acc:.2%}. Something else might be wrong (e.g. Patch Jitter mismatch).")
    else:
        print(f"[Calibration] Locked in permutation set. Best Acc: {best_acc:.2%}")

    if best_perms is None:
        raise RuntimeError("Calibration failed to determine best_perms (unexpected).")

    # ---------------------------------------------------------
    # 3. MAIN METRICS LOOP
    # ---------------------------------------------------------
    if args.dataset_name_arg is None: 
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
            fixed_perms=best_perms,
        )
    else: 
        loader = build_generic_pretext_loader(
            d_name=args.dataset_name_arg,
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
            fixed_perms=best_perms,

        )

    # validate layer keys exist
    layer_keys = [x.strip() for x in args.layers.split(",") if x.strip()]
    if not hasattr(model, "all_feat_names"):
        raise RuntimeError("Model missing all_feat_names (expected NetworkInNetwork).")
    for k in layer_keys:
        if k not in model.all_feat_names:
            raise RuntimeError(f"Layer key '{k}' not in model.all_feat_names: {model.all_feat_names}")

    # output dir
    out_dir = Path(args.out_root) / f"{Path(args.exp).name}_{args.pretext_mode}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # storage
    epochs_logged: List[int] = []
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    nc3_curve: List[float] = []
    nc1_curves: Dict[str, List[float]] = {k: [] for k in layer_keys}

    # NC4 storage (penultimate)
    nc4_match_curve: List[float] = []
    nc4_mismatch_curve: List[float] = []
    ncc_acc_curve: List[float] = []

    # NEW: layerwise NC4 storage
    nc4_layerwise_curves: Dict[str, Dict[str, List[float]]] = {
        k: {"match": [], "mismatch": [], "ncc_acc": []} for k in layer_keys
    }

    for ep in epochs:
        ckpt = epoch_to_path[ep]
        print(f"\n[SimpleNIN] epoch {ep} -> {ckpt.name}")

        load_state_dict(model, ckpt)
        model.to(device)

        if args.nc4 or args.nc4_layerwise:
            want_pen = bool(args.nc4)
            want_layer_means = bool(args.nc4_layerwise)

            ret = compute_epoch_metrics_multilayer(
                model=model,
                loader=loader,
                num_classes=args.num_classes,
                layer_keys=layer_keys,
                device=device,
                return_means_penult=want_pen,
                return_means_by_layer=want_layer_means,
            )

            if want_pen and want_layer_means:
                nc1_by_layer, acc, loss, nc3, means_penult, means_by_layer = ret
            elif want_pen:
                nc1_by_layer, acc, loss, nc3, means_penult = ret
                means_by_layer = None
            else:
                nc1_by_layer, acc, loss, nc3, means_by_layer = ret
                means_penult = None

            # penultimate NC4 (original)
            if args.nc4:
                nc4_match, nc4_mismatch, ncc_acc = nc4Fun(
                    model=model,
                    loader=loader,
                    means_penult=means_penult,  # type: ignore[arg-type]
                    num_classes=args.num_classes,
                    device=device,
                )
                nc4_match_curve.append(nc4_match)
                nc4_mismatch_curve.append(nc4_mismatch)
                ncc_acc_curve.append(ncc_acc)
                print(f"[SimpleNIN] nc4_match={nc4_match:.4f} nc4_mismatch={nc4_mismatch:.4f} ncc_acc={ncc_acc:.4f}")

            # layerwise NC4 (NEW)
            if args.nc4_layerwise:
                if means_by_layer is None:
                    raise RuntimeError("Expected means_by_layer but got None (return_means_by_layer=True).")
                lw = nc4_layerwise(
                    model=model,
                    loader=loader,
                    means_by_layer=means_by_layer,
                    layer_keys=layer_keys,
                    num_classes=args.num_classes,
                    device=device,
                )
                for k in layer_keys:
                    m, mm, a = lw[k]
                    nc4_layerwise_curves[k]["match"].append(m)
                    nc4_layerwise_curves[k]["mismatch"].append(mm)
                    nc4_layerwise_curves[k]["ncc_acc"].append(a)

                deep = layer_keys[-1]
                m, mm, a = lw[deep]
                print(f"[SimpleNIN] layerwise-NC4 (deep={deep}) match={m:.4f} mismatch={mm:.4f} ncc_acc={a:.4f}")

        else:
            nc1_by_layer, acc, loss, nc3 = compute_epoch_metrics_multilayer(
                model=model,
                loader=loader,
                num_classes=args.num_classes,
                layer_keys=layer_keys,
                device=device,
            )

        epochs_logged.append(ep)
        acc_curve.append(acc)
        loss_curve.append(loss)
        nc3_curve.append(nc3)
        for k in layer_keys:
            nc1_curves[k].append(nc1_by_layer[k])

        print(f"[SimpleNIN] acc={acc:.4f} loss={loss:.4f} nc3={nc3:.6f}")
        print("[SimpleNIN] nc1:", " | ".join([f"{k}={nc1_by_layer[k]:.6f}" for k in layer_keys]))

        if use_cuda:
            torch.cuda.empty_cache()

    # save arrays
    import pickle
    payload = {
        "epochs": epochs_logged,
        "accuracy": acc_curve,
        "loss": loss_curve,
        "nc3": nc3_curve,
        "nc1_by_layer": nc1_curves,
        "layer_keys": layer_keys,
        "args": vars(args),
    }
    if args.nc4:
        payload["nc4_match"] = nc4_match_curve
        payload["nc4_mismatch"] = nc4_mismatch_curve
        payload["ncc_acc"] = ncc_acc_curve
    if args.nc4_layerwise:
        payload["nc4_layerwise"] = nc4_layerwise_curves
    #instead of a pkl json is easier to parse and will not require a full datastructure reparse as writing to csv would     
    import json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    
    # plots
    plot_and_save(out_dir, epochs_logged, {"accuracy": acc_curve, "loss": loss_curve, "nc3": nc3_curve}, "SimpleNIN")
    plot_and_save(out_dir, epochs_logged, {f"nc1_{k}": nc1_curves[k] for k in layer_keys}, "SimpleNIN")
    if epochs_logged:
        last_epoch = epochs_logged[-1]
        last_vals = [nc1_curves[k][-1] for k in layer_keys]
        plot_nc1_by_layer(out_dir, layer_keys, last_vals, f"NC1 across layers at epoch {last_epoch}")

    if args.nc4:
        plot_nc4(out_dir, epochs_logged, nc4_match_curve, nc4_mismatch_curve, ncc_acc_curve)

    if args.nc4_layerwise and epochs_logged:
        plot_layerwise_nc4_final(out_dir, layer_keys, nc4_layerwise_curves, epochs_logged[-1])

    print(f"\n✓ Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
