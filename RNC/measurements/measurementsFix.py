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
def compute_epoch_metrics_multilayer(
    model: nn.Module,
    loader, # This is the DataLoader object (callable)
    num_classes: int,
    layer_keys: List[str],
    device: torch.device,
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Returns:
      nc1_by_layer: dict layer_key -> NC1
      acc: accuracy
      loss: CE loss
      nc3: NC3 computed at classifier feature space (penultimate input to Linear)
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
        # inputs is a tuple; inputs[0] is (N, D)
        penult["h"] = inputs[0].detach()

    handle = cls_layer.register_forward_pre_hook(prehook)

    # ---- PASS 1: class means for each requested layer + penultimate, plus acc/loss ----
    C = num_classes
    sums_by_layer: Dict[str, List[Optional[torch.Tensor]]] = {
        k: [None for _ in range(C)] for k in layer_keys
    }
    N_per_class = torch.zeros(C, dtype=torch.long)

    # penultimate sums
    sum_penult: List[Optional[torch.Tensor]] = [None for _ in range(C)]

    total_loss = 0.0
    correct = 0
    totalN = 0

    out_keys = layer_keys + ["classifier"]  # logits at "classifier"

    # FIX: Call loader(0) to get the iterator
    iter_pass1 = loader(0)

    for x, y in tqdm(iter_pass1, desc="PASS1 (means/acc/loss)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)

        outs = model(x, out_feat_keys=out_keys)
        # outs is list aligned with out_keys
        feats = outs[:-1]
        logits = outs[-1]

        if "h" not in penult:
            raise RuntimeError("Penultimate pre-hook didn't fire.")

        h_pen = penult["h"]
        h_pen = gapify(h_pen)  # should already be (N, D)

        bs = y.size(0)
        totalN += bs

        total_loss += loss_fn(logits, y).item() * bs
        correct += (logits.argmax(1) == y).sum().item()

        # move to CPU for class aggregation
        y_cpu = y.detach().cpu()

        # penultimate class sums
        hp_cpu = h_pen.detach().cpu()
        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            v = hp_cpu[idx]  # (n_c, D)
            if sum_penult[c] is None:
                sum_penult[c] = v.sum(0)
            else:
                sum_penult[c] += v.sum(0)
            N_per_class[c] += v.size(0)

        # per-layer class sums
        for k, feat in zip(layer_keys, feats):
            fv = gapify(feat).detach().cpu()  # (N, Dk)
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

    # Build means per layer
    means_by_layer: Dict[str, List[torch.Tensor]] = {}
    for k in layer_keys:
        means_by_layer[k] = [
            sums_by_layer[k][c] / float(int(N_per_class[c].item()))  # type: ignore
            for c in range(C)
        ]

    # Penultimate means
    means_penult: List[torch.Tensor] = [
        sum_penult[c] / float(int(N_per_class[c].item()))  # type: ignore
        for c in range(C)
    ]

    # ---- PASS 2: within-class scatter for NC1 (per layer) ----
    total_ss_by_layer: Dict[str, float] = {k: 0.0 for k in layer_keys}

    # FIX: Call loader(0) to get the iterator
    iter_pass2 = loader(0)

    for x, y in tqdm(iter_pass2, desc="PASS2 (Sw per layer)", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        outs = model(x, out_feat_keys=layer_keys)  # list aligned to layer_keys
        y_cpu = y.detach().cpu()

        for k, feat in zip(layer_keys, outs):
            fv = gapify(feat).detach().cpu()  # (N, Dk)
            for c in range(C):
                idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                vc = fv[idx]
                z = vc - means_by_layer[k][c]
                total_ss_by_layer[k] += z.pow(2).sum().item()

    # ---- NC1 finalize per layer ----
    eps = 1e-12
    nc1_by_layer: Dict[str, float] = {}
    for k in layer_keys:
        # Mmat: (D, C)
        Mmat = torch.stack(means_by_layer[k], dim=1)
        muG = Mmat.mean(dim=1, keepdim=True)
        M_centered = Mmat - muG

        Sb = (M_centered @ M_centered.T) / float(C)
        trSb = Sb.trace().item()
        trSw = total_ss_by_layer[k] / float(totalN)

        nc1_by_layer[k] = trSw / (trSb + eps)

    # ---- NC3 finalize (penultimate space only) ----
    Mmat_p = torch.stack(means_penult, dim=1)            # (D, C)
    muG_p = Mmat_p.mean(dim=1, keepdim=True)
    Mc = Mmat_p - muG_p                                  # centered means

    W = cls_layer.weight.detach().cpu().T                # (D, C)
    Wn = W / (W.norm() + eps)
    Mn = Mc / (Mc.norm() + eps)
    nc3 = (Wn - Mn).pow(2).sum().item()

    acc = correct / float(totalN)
    loss = total_loss / float(totalN)

    handle.remove()
    penult.clear()

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
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--pretext-mode", choices=["rotation", "gaussian_noise", "gaussian_blur", "jigsaw"], required=True)
    p.add_argument("--num-classes", type=int, default=4)

    p.add_argument("--sigmas", type=str, default=None, help="Comma list (4 entries) for gaussian_noise")
    p.add_argument("--kernel-sizes", type=str, default=None, help="Comma list (4 entries) for gaussian_blur")

    p.add_argument("--patch-jitter", type=int, default=0)
    p.add_argument("--color-distort", action="store_true")
    p.add_argument("--color-dist-strength", type=float, default=1.0)

    # layers to compute NC1 on (use exposed keys like conv1,conv2,conv3,...)
    p.add_argument("--layers", type=str, default="conv1,conv2,conv3,conv4",
                   help="Comma list of exposed feature keys for NC1 (e.g. conv1,conv2,conv3,conv4)")

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
    # 1. SETUP & MODEL LOADING (Moved UP)
    # ---------------------------------------------------------
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"exp-dir not found: {exp_dir}")

    # Model
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
    
    # Re-implementing the core of maxHamming deterministically
    # There are 4! = 24 possible starting permutations.
    # We will try all 24 sets.
    
    K = 4 # 2x2 grid
    N = args.num_classes
    all_base_perms = list(permutations(range(1, K+1))) # 1-based logic from your file
    
    sigmas = parse_float_list(args.sigmas)
    kernel_sizes = parse_int_list(args.kernel_sizes)
    if args.pretext_mode == "gaussian_noise" and sigmas is None: sigmas = [1e-3, 1e-2, 1e-1, 1.0]
    if args.pretext_mode == "gaussian_blur" and kernel_sizes is None: kernel_sizes = [3, 5, 7, 9]

    best_perms = None
    best_acc = -1.0
    
    print(f"[Calibration] Testing 24 possible permutation sets...")
    
    # Function to generate set starting at index start_idx
    def get_set_starting_at(start_idx):
        # EXACT logic from your maxHamming.py, but j is forced
        P_bar = all_base_perms.copy()
        P = []
        j = start_idx # Forced start
        
        i = 1
        while i <= N:
            P.append(P_bar[j])
            P_prime = P_bar[:j] + P_bar[j+1:]
            
            if i < N:
                # Calculate Dist Matrix D
                # D[x,y] = hamming(P[x], P_prime[y])
                n_p = len(P)
                n_pp = len(P_prime)
                D = np.zeros((n_p, n_pp), dtype=int)
                for r in range(n_p):
                    for c in range(n_pp):
                        D[r,c] = np.sum(np.array(P[r]) != np.array(P_prime[c]))
                
                D_bar = np.min(D, axis=0)
                j = np.argmax(D_bar)
            
            P_bar = P_prime
            i += 1
        return [tuple(x-1 for x in p) for p in P] # Convert to 0-based for loader

    # Search loop
    for start_idx in range(len(all_base_perms)):
        candidate_perms = get_set_starting_at(start_idx)
        
        # Build quick loader (returns DataLoader callable object)
        calib_loader = build_cifar10_pretext_loader(
            split='test', batch_size=128, workers=0, # fast, no MP overhead
            pretext_mode=args.pretext_mode,
            sigmas=sigmas, kernel_sizes=kernel_sizes,
            patch_jitter=args.patch_jitter,
            color_distort=False, color_dist_strength=0.0,
            shuffle=False, fixed_perms=candidate_perms
        )
        
        # Run 1 batch (FIXED: calling loader(0) to get iterator)
        x, y = next(iter(calib_loader(0)))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)[-1] # Assume last output is logits
            acc = (logits.argmax(1) == y).float().mean().item()
            
        if acc > best_acc:
            best_acc = acc
            best_perms = candidate_perms
        
        # Optimization: If we hit >90%, we found it.
        if best_acc > 0.90:
            print(f"[Calibration] Found match at index {start_idx}! Acc: {best_acc:.2%}")
            break
            
    if best_acc < 0.5:
        print(f"\n[WARNING] Best match was only {best_acc:.2%}. Something else might be wrong (e.g. Patch Jitter mismatch).")
    else:
        print(f"[Calibration] Locked in permutation set. Best Acc: {best_acc:.2%}")

    # ---------------------------------------------------------
    # 3. MAIN METRICS LOOP
    # ---------------------------------------------------------
    
    # Create the REAL loader with the found perms
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
        fixed_perms=best_perms 
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

    for ep in epochs:
        ckpt = epoch_to_path[ep]
        print(f"\n[SimpleNIN] epoch {ep} -> {ckpt.name}")

        load_state_dict(model, ckpt)
        model.to(device)

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
    with open(out_dir / "metrics.pkl", "wb") as f:
        pickle.dump(payload, f)

    # plots
    plot_and_save(out_dir, epochs_logged, {"accuracy": acc_curve, "loss": loss_curve, "nc3": nc3_curve}, "SimpleNIN")
    plot_and_save(out_dir, epochs_logged, {f"nc1_{k}": nc1_curves[k] for k in layer_keys}, "SimpleNIN")

    print(f"\n✓ Done. Results in: {out_dir}")

if __name__ == "__main__":
    main()