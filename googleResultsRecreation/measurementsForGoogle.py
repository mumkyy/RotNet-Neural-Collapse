#!/usr/bin/env python3

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import train_and_eval as te


# -----------------------------
# config
# -----------------------------

WORKDIR = "/project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/Train_long_jigsaw_imagenette_resnet50_NC10" 
# results obtained needed to triain longer ^ : "/project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/jigsaw_imagenette_resnet50_NC10"

LAYER_KEYS = [
    "block1",
    "block2",
    "block3",
    "block4",
    "pre_logits",
]

USE_SPLIT = "val"
ONLY_LAST = False

# ridge for probe-NC3
PROBE_NC3_RIDGE = 1e-4


# -----------------------------
# utils
# -----------------------------

def load_args(workdir):
    with open(Path(workdir) / "args.json", "r") as f:
        return SimpleNamespace(**json.load(f))


def get_device():
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    return device


def gapify(x):
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 2:
        return x
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def build_loader(args, split_name):
    cfg = te.build_data_cfg(args)

    batch_size = args.eval_batch_size if split_name == args.val_split else args.batch_size

    loader = te.datasets.get_data(
        params={
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "pin_memory": not args.no_pin_memory,
        },
        split_name=split_name,
        is_training=False,
        shuffle=False,
        num_epochs=1,
        drop_remainder=False,
        cfg=cfg,
    )
    return loader


def list_checkpoints_sorted(workdir, only_last=False):
    workdir = Path(workdir)
    if only_last:
        ckpt = workdir / "checkpoint_last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        return [ckpt]

    ckpts = list(workdir.glob("checkpoint_step_*.pt"))
    last_ckpt = workdir / "checkpoint_last.pt"
    if last_ckpt.exists():
        ckpts.append(last_ckpt)

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {workdir}")

    def ckpt_key(path):
        if path.name == "checkpoint_last.pt":
            return (float("inf"), float("inf"))
        try:
            step = int(path.stem.split("_")[-1])
        except Exception:
            step = -1
        return (0, step)

    ckpts = sorted(ckpts, key=ckpt_key)
    return ckpts


def extract_layer_features_from_endpoints(endpoints, layer_key, B, P, selected_perms):
    """
    endpoints[layer_key] is patch-level backbone output:
      - block*: [B*P, C, H, W]
      - pre_logits: [B*P, C]

    Convert to permutation-example features:
      [B, P, ...] -> permute/concat -> [B*M, P*C, H, W] -> GAP -> [B*M, D]
    """
    feat = endpoints[layer_key]

    if feat.ndim == 4:
        _, C, H, W = feat.shape
        feat = feat.view(B, P, C, H, W)
    elif feat.ndim == 2:
        _, C = feat.shape
        feat = feat.view(B, P, C, 1, 1)
    else:
        raise ValueError(f"Unsupported endpoint shape for {layer_key}: {tuple(feat.shape)}")

    concat_feat = te.permute_and_concat_batch_patches(feat, selected_perms)
    concat_feat = gapify(concat_feat)
    return concat_feat


def init_running_stats(num_classes):
    stats = {}
    for layer in LAYER_KEYS:
        stats[layer] = {
            "N": 0,
            "class_counts": torch.zeros(num_classes, dtype=torch.long),
            "sum_total": None,
            "class_sums": None,
            "feature_dim": None,
        }
    return stats


def update_running_sums(stats, feat_cpu, y_cpu, num_classes):
    """
    feat_cpu: [N, D] float64 cpu
    y_cpu: [N] long cpu
    """
    N, D = feat_cpu.shape

    stats["N"] += N
    if stats["sum_total"] is None:
        stats["sum_total"] = torch.zeros(D, dtype=torch.float64)
        stats["class_sums"] = torch.zeros(num_classes, D, dtype=torch.float64)
        stats["feature_dim"] = D

    stats["sum_total"] += feat_cpu.sum(dim=0)

    binc = torch.bincount(y_cpu, minlength=num_classes)
    stats["class_counts"] += binc

    for c in range(num_classes):
        mask = (y_cpu == c)
        if mask.any():
            stats["class_sums"][c] += feat_cpu[mask].sum(dim=0)


def finalize_means(stats):
    """
    Returns:
      mu_g: [D]
      mu_c: [K, D]
      counts: [K]
    """
    N = stats["N"]
    counts = stats["class_counts"].clone()

    if N == 0:
        raise ValueError("No samples seen while finalizing means.")

    mu_g = stats["sum_total"] / N

    mu_c = torch.zeros_like(stats["class_sums"])
    nonzero = counts > 0
    mu_c[nonzero] = stats["class_sums"][nonzero] / counts[nonzero].unsqueeze(1).to(torch.float64)

    return mu_g, mu_c, counts


def compute_sb_trace(mu_g, mu_c, counts):
    """
    Between-class trace:
      (1/N) * sum_c n_c ||mu_c - mu_g||^2
    """
    centered = mu_c - mu_g.unsqueeze(0)
    sq = (centered * centered).sum(dim=1)
    sb_ss = (counts.to(torch.float64) * sq).sum().item()
    N = counts.sum().item()
    sb_trace = sb_ss / max(N, 1)
    return sb_trace, centered


def compute_etf_metrics(centered_means, counts):
    """
    centered_means: [K, D]
    """
    keep = counts > 0
    M = centered_means[keep]
    K = M.shape[0]

    if K <= 1:
        return {
            "etf_fro_deviation": float("nan"),
            "mean_abs_cosine_deviation": float("nan"),
            "max_abs_cosine_deviation": float("nan"),
        }

    row_norms = torch.norm(M, dim=1, keepdim=True)
    row_norms = torch.where(row_norms > 0, row_norms, torch.ones_like(row_norms))
    U = M / row_norms

    G = U @ U.t()
    target = (K / (K - 1)) * torch.eye(K, dtype=torch.float64) - (1 / (K - 1)) * torch.ones(K, K, dtype=torch.float64)

    fro_dev = torch.norm(G - target, p="fro") / torch.norm(target, p="fro")

    offdiag_mask = ~torch.eye(K, dtype=torch.bool)
    offdiag = G[offdiag_mask]
    ideal = -1.0 / (K - 1)

    mean_abs_cos_dev = torch.mean(torch.abs(offdiag - ideal)).item()
    max_abs_cos_dev = torch.max(torch.abs(offdiag - ideal)).item()

    return {
        "etf_fro_deviation": float(fro_dev.item()),
        "mean_abs_cosine_deviation": float(mean_abs_cos_dev),
        "max_abs_cosine_deviation": float(max_abs_cos_dev),
    }


# -----------------------------
# new: probe-NC3 helpers
# -----------------------------

def build_mean_subspace_basis(mu_g, mu_c, counts, rtol=1e-10):
    """
    Build an orthonormal basis B for the span of centered class means.

    Inputs:
      mu_g: [D]
      mu_c: [K, D]
      counts: [K]

    Returns:
      B: [D, r]
      M: [D, K_eff] centered mean matrix with only non-empty classes kept
      keep_idx: [K_eff] original class indices retained
    """
    keep = counts > 0
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    centered = mu_c[keep] - mu_g.unsqueeze(0)      # [K_eff, D]
    M = centered.t().contiguous()                  # [D, K_eff]

    if M.numel() == 0:
        return None, None, None

    if torch.allclose(M, torch.zeros_like(M)):
        return None, M, keep_idx

    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    if S.numel() == 0:
        return None, M, keep_idx

    thresh = rtol * max(float(S.max().item()), 1.0)
    r = int((S > thresh).sum().item())

    if r == 0:
        return None, M, keep_idx

    B = U[:, :r].contiguous()                      # [D, r]
    return B, M, keep_idx


@torch.no_grad()
def collect_projected_features_for_layer(model, loader, device, layer_key, mu_g, basis):
    """
    Collect low-dimensional projected features:
      z = (h - mu_g) @ basis

    Returns:
      Z: [N, r] float64 cpu
      y: [N] long cpu
    """
    z_chunks = []
    y_chunks = []

    for batch in tqdm(loader, desc=f"probe collect {layer_key}", leave=False):
        x = batch["image"].to(device)
        Bsz, P, C, H, W = x.shape

        outputs = model(x)
        y = outputs["labels"].long()
        perm_indices = outputs["perm_indices"].long()
        selected_perms = model.permutations[perm_indices].to(device)

        flat_x = x.reshape(Bsz * P, C, H, W)
        _, endpoints = model.backbone(flat_x, return_endpoints=True)

        feat = extract_layer_features_from_endpoints(
            endpoints=endpoints,
            layer_key=layer_key,
            B=Bsz,
            P=P,
            selected_perms=selected_perms,
        )

        feat_cpu = feat.detach().cpu().to(torch.float64)
        z = (feat_cpu - mu_g.unsqueeze(0)) @ basis

        z_chunks.append(z)
        y_chunks.append(y.detach().cpu())

    Z = torch.cat(z_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    return Z, y


def fit_ridge_probe_in_subspace(Z, y, num_classes, ridge):
    """
    Z: [N, r]
    y: [N]

    Solve:
      min_V || Z V - Y ||_F^2 + ridge ||V||_F^2

    Returns:
      V: [r, K]
      acc: scalar
    """
    N, r = Z.shape
    Y = torch.zeros(N, num_classes, dtype=torch.float64)
    Y[torch.arange(N), y] = 1.0

    A = Z.t() @ Z
    A = A + ridge * torch.eye(r, dtype=torch.float64)
    B = Z.t() @ Y

    V = torch.linalg.solve(A, B)                   # [r, K]
    logits = Z @ V
    pred = torch.argmax(logits, dim=1)
    acc = (pred == y).double().mean().item()

    return V, acc


def compute_probe_nc3_metrics_from_weights(W, M, keep_idx):
    """
    W: [D, K_full]
    M: [D, K_eff]
    keep_idx: indices of non-empty classes in K_full space

    Compare only on non-empty classes.

    Returns:
      probe_nc3_fro
      probe_nc3_mean_cosine
      probe_nc3_optimal_scale_error
    """
    Wk = W[:, keep_idx]                            # [D, K_eff]
    Mk = M                                         # [D, K_eff]

    eps = 1e-12

    W_f = torch.norm(Wk, p="fro")
    M_f = torch.norm(Mk, p="fro")

    if float(W_f.item()) <= eps or float(M_f.item()) <= eps:
        return {
            "probe_nc3_fro": float("nan"),
            "probe_nc3_mean_cosine": float("nan"),
            "probe_nc3_optimal_scale_error": float("nan"),
        }

    Wn = Wk / W_f
    Mn = Mk / M_f
    fro = torch.sum((Wn - Mn) ** 2).item()

    w_col_norms = torch.norm(Wk, dim=0)
    m_col_norms = torch.norm(Mk, dim=0)
    valid = (w_col_norms > eps) & (m_col_norms > eps)

    if valid.any():
        Wc = Wk[:, valid] / w_col_norms[valid].unsqueeze(0)
        Mc = Mk[:, valid] / m_col_norms[valid].unsqueeze(0)
        mean_cos = torch.sum(Wc * Mc, dim=0).mean().item()
    else:
        mean_cos = float("nan")

    alpha_num = torch.sum(Wk * Mk).item()
    alpha_den = torch.sum(Mk * Mk).item()
    if abs(alpha_den) <= eps:
        scale_err = float("nan")
    else:
        alpha = alpha_num / alpha_den
        scale_err = (torch.norm(Wk - alpha * Mk, p="fro") / (torch.norm(Mk, p="fro") + eps)).item()

    return {
        "probe_nc3_fro": float(fro),
        "probe_nc3_mean_cosine": float(mean_cos),
        "probe_nc3_optimal_scale_error": float(scale_err),
    }


@torch.no_grad()
def compute_probe_nc3_for_layer(model, loader, device, layer_key, means_for_layer, num_classes, ridge):
    """
    Proper comparable NC3-style metric for the nonlinear Google jigsaw stack:
      - build mean-subspace basis from centered class means
      - fit linear ridge probe in that subspace
      - lift weights back to ambient space
      - compare lifted weights to centered class means

    means_for_layer:
      {
        "mu_g": [D],
        "mu_c": [K, D],
        "counts": [K],
        ...
      }
    """
    mu_g = means_for_layer["mu_g"]
    mu_c = means_for_layer["mu_c"]
    counts = means_for_layer["counts"]

    basis, M, keep_idx = build_mean_subspace_basis(mu_g, mu_c, counts)

    if basis is None or M is None or keep_idx is None:
        return {
            "probe_nc3_fro": float("nan"),
            "probe_nc3_mean_cosine": float("nan"),
            "probe_nc3_optimal_scale_error": float("nan"),
            "probe_accuracy": float("nan"),
            "probe_subspace_rank": 0,
        }

    Z, y = collect_projected_features_for_layer(
        model=model,
        loader=loader,
        device=device,
        layer_key=layer_key,
        mu_g=mu_g,
        basis=basis,
    )

    V, probe_acc = fit_ridge_probe_in_subspace(
        Z=Z,
        y=y,
        num_classes=num_classes,
        ridge=ridge,
    )

    W = basis @ V                                  # [D, K]
    metrics = compute_probe_nc3_metrics_from_weights(W=W, M=M, keep_idx=keep_idx)

    metrics["probe_accuracy"] = float(probe_acc)
    metrics["probe_subspace_rank"] = int(basis.shape[1])
    return metrics


# -----------------------------
# existing NC passes
# -----------------------------

@torch.no_grad()
def first_pass_collect_means(model, loader, device, num_classes):
    """
    First pass:
      - collect global sums and class sums for each layer
      - compute head accuracy on the actual jigsaw logits
    """
    layer_stats = init_running_stats(num_classes)

    total_head_correct = 0
    total_head_count = 0
    total_head_loss = 0.0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")

    for batch in tqdm(loader, desc="first pass", leave=False):
        x = batch["image"].to(device)
        Bsz, P, C, H, W = x.shape

        outputs = model(x)
        logits = outputs["logits"]
        y = outputs["labels"].long()
        perm_indices = outputs["perm_indices"].long()
        selected_perms = model.permutations[perm_indices].to(device)

        total_head_loss += ce(logits, y).item()
        total_head_correct += (logits.argmax(dim=1) == y).sum().item()
        total_head_count += y.numel()

        flat_x = x.reshape(Bsz * P, C, H, W)
        _, endpoints = model.backbone(flat_x, return_endpoints=True)

        y_cpu = y.detach().cpu()

        for layer in LAYER_KEYS:
            feat = extract_layer_features_from_endpoints(
                endpoints=endpoints,
                layer_key=layer,
                B=Bsz,
                P=P,
                selected_perms=selected_perms,
            )

            feat_cpu = feat.detach().cpu().to(torch.float64)
            update_running_sums(layer_stats[layer], feat_cpu, y_cpu, num_classes)

    head_metrics = {
        "head_accuracy": total_head_correct / max(total_head_count, 1),
        "head_cross_entropy": total_head_loss / max(total_head_count, 1),
        "num_examples": total_head_count,
    }

    means = {}
    for layer in LAYER_KEYS:
        mu_g, mu_c, counts = finalize_means(layer_stats[layer])
        means[layer] = {
            "mu_g": mu_g,
            "mu_c": mu_c,
            "counts": counts,
            "feature_dim": layer_stats[layer]["feature_dim"],
            "num_examples": layer_stats[layer]["N"],
        }

    return means, head_metrics


@torch.no_grad()
def second_pass_compute_dispersion_and_ncc(model, loader, device, means_by_layer):
    """
    Second pass:
      - compute within-class trace
      - compute nearest-class-center accuracy
    """
    out = {}
    for layer in LAYER_KEYS:
        out[layer] = {
            "sw_ss": 0.0,
            "ncc_correct": 0,
            "ncc_total": 0,
        }

    for batch in tqdm(loader, desc="second pass", leave=False):
        x = batch["image"].to(device)
        Bsz, P, C, H, W = x.shape

        outputs = model(x)
        y = outputs["labels"].long()
        perm_indices = outputs["perm_indices"].long()
        selected_perms = model.permutations[perm_indices].to(device)

        flat_x = x.reshape(Bsz * P, C, H, W)
        _, endpoints = model.backbone(flat_x, return_endpoints=True)

        y_cpu = y.detach().cpu()

        for layer in LAYER_KEYS:
            feat = extract_layer_features_from_endpoints(
                endpoints=endpoints,
                layer_key=layer,
                B=Bsz,
                P=P,
                selected_perms=selected_perms,
            )
            feat_cpu = feat.detach().cpu().to(torch.float64)

            mu_c = means_by_layer[layer]["mu_c"]
            mu_y = mu_c[y_cpu]

            diffs = feat_cpu - mu_y
            out[layer]["sw_ss"] += (diffs * diffs).sum().item()

            dists = torch.cdist(feat_cpu, mu_c, p=2.0)
            pred = torch.argmin(dists, dim=1)

            out[layer]["ncc_correct"] += (pred == y_cpu).sum().item()
            out[layer]["ncc_total"] += y_cpu.numel()

    return out


def compute_nc_metrics_for_checkpoint(model, loader, device, args):
    num_classes = args.perm_subset_size

    means_by_layer, head_metrics = first_pass_collect_means(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
    )

    pass2 = second_pass_compute_dispersion_and_ncc(
        model=model,
        loader=loader,
        device=device,
        means_by_layer=means_by_layer,
    )

    layer_metrics = {}

    for layer in LAYER_KEYS:
        mu_g = means_by_layer[layer]["mu_g"]
        mu_c = means_by_layer[layer]["mu_c"]
        counts = means_by_layer[layer]["counts"]

        sb_trace, centered_means = compute_sb_trace(mu_g, mu_c, counts)
        sw_trace = pass2[layer]["sw_ss"] / max(means_by_layer[layer]["num_examples"], 1)

        if sb_trace > 0:
            nc1 = sw_trace / sb_trace
        else:
            nc1 = float("nan")

        ncc_acc = pass2[layer]["ncc_correct"] / max(pass2[layer]["ncc_total"], 1)
        etf_metrics = compute_etf_metrics(centered_means, counts)

        probe_metrics = compute_probe_nc3_for_layer(
            model=model,
            loader=loader,
            device=device,
            layer_key=layer,
            means_for_layer=means_by_layer[layer],
            num_classes=num_classes,
            ridge=PROBE_NC3_RIDGE,
        )

        layer_metrics[layer] = {
            "feature_dim": int(means_by_layer[layer]["feature_dim"]),
            "num_examples": int(means_by_layer[layer]["num_examples"]),
            "num_classes": int((counts > 0).sum().item()),
            "class_counts": [int(x) for x in counts.tolist()],
            "sw_trace": float(sw_trace),
            "sb_trace": float(sb_trace),
            "nc1_trace_ratio": float(nc1),
            "ncc_accuracy": float(ncc_acc),
            "etf_fro_deviation": float(etf_metrics["etf_fro_deviation"]),
            "mean_abs_cosine_deviation": float(etf_metrics["mean_abs_cosine_deviation"]),
            "max_abs_cosine_deviation": float(etf_metrics["max_abs_cosine_deviation"]),
            "probe_nc3_fro": float(probe_metrics["probe_nc3_fro"]),
            "probe_nc3_mean_cosine": float(probe_metrics["probe_nc3_mean_cosine"]),
            "probe_nc3_optimal_scale_error": float(probe_metrics["probe_nc3_optimal_scale_error"]),
            "probe_accuracy": float(probe_metrics["probe_accuracy"]),
            "probe_subspace_rank": int(probe_metrics["probe_subspace_rank"]),
        }

    return {
        "head_metrics": head_metrics,
        "layer_metrics": layer_metrics,
    }


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    return obj


# -----------------------------
# plotting
# -----------------------------

def plot_metric_vs_epoch(records, metric_name, out_path):
    plt.figure(figsize=(10, 6))

    epochs = [r["epoch"] for r in records]
    for layer in LAYER_KEYS:
        values = [r["layer_metrics"][layer][metric_name] for r in records]
        plt.plot(epochs, values, marker="o", label=layer)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_heatmap(records, metric_name, out_path):
    epochs = [r["epoch"] for r in records]
    Z = np.array([
        [r["layer_metrics"][layer][metric_name] for layer in LAYER_KEYS]
        for r in records
    ], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(Z, aspect="auto")
    plt.colorbar(im, label=metric_name)
    plt.xticks(range(len(LAYER_KEYS)), LAYER_KEYS, rotation=45)
    plt.yticks(range(len(epochs)), [f"{e:.3f}" for e in epochs])
    plt.xlabel("Layer")
    plt.ylabel("Epoch")
    plt.title(f"{metric_name} heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_head_metric(records, metric_name, out_path):
    epochs = [r["epoch"] for r in records]
    values = [r["head_metrics"][metric_name] for r in records]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    device = get_device()
    workdir = Path(WORKDIR)
    args = load_args(workdir)

    if USE_SPLIT == "train":
        split_name = args.train_split
    elif USE_SPLIT == "val":
        split_name = args.val_split
    else:
        split_name = USE_SPLIT

    loader = build_loader(args, split_name=split_name)
    ckpts = list_checkpoints_sorted(workdir, only_last=ONLY_LAST)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = workdir / f"neural_collapse_backbone_pretext_permutation_labels_probe_nc3_{split_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"workdir: {workdir}")
    print(f"split: {split_name}")
    print(f"output dir: {outdir}")
    print(f"num checkpoints: {len(ckpts)}")

    model = te.build_model(args).to(device)
    model.eval()

    all_records = []

    for ckpt_path in tqdm(ckpts, desc="checkpoints"):
        step, epoch, _ = te.load_checkpoint(
            model=model,
            optimizer=None,
            checkpoint_path=str(ckpt_path),
            device=device,
        )
        model.eval()

        metrics = compute_nc_metrics_for_checkpoint(
            model=model,
            loader=loader,
            device=device,
            args=args,
        )

        record = {
            "checkpoint_name": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "step": int(step),
            "epoch": float(epoch),
            "split": split_name,
            "probe_nc3_ridge": float(PROBE_NC3_RIDGE),
            "head_metrics": metrics["head_metrics"],
            "layer_metrics": metrics["layer_metrics"],
        }
        all_records.append(record)

        with open(outdir / f"partial_metrics_{ckpt_path.stem}.json", "w") as f:
            json.dump(sanitize_for_json(record), f, indent=2)

    all_records = sorted(all_records, key=lambda r: (r["epoch"], r["step"]))

    metrics_to_plot = [
        "nc1_trace_ratio",
        "sw_trace",
        "sb_trace",
        "ncc_accuracy",
        "etf_fro_deviation",
        "mean_abs_cosine_deviation",
        "max_abs_cosine_deviation",
        "probe_nc3_fro",
        "probe_nc3_mean_cosine",
        "probe_nc3_optimal_scale_error",
        "probe_accuracy",
        "probe_subspace_rank",
    ]

    for metric_name in metrics_to_plot:
        plot_metric_vs_epoch(
            records=all_records,
            metric_name=metric_name,
            out_path=outdir / f"{metric_name}_vs_epoch_layers.png",
        )
        plot_metric_heatmap(
            records=all_records,
            metric_name=metric_name,
            out_path=outdir / f"{metric_name}_layer_epoch_heatmap.png",
        )

    plot_head_metric(
        records=all_records,
        metric_name="head_accuracy",
        out_path=outdir / "head_accuracy_vs_epoch.png",
    )
    plot_head_metric(
        records=all_records,
        metric_name="head_cross_entropy",
        out_path=outdir / "head_cross_entropy_vs_epoch.png",
    )

    summary = {
        "workdir": str(workdir),
        "split": split_name,
        "timestamp_completed": timestamp,
        "device": str(device),
        "layer_keys": LAYER_KEYS,
        "probe_nc3_definition": "ridge linear probe in span(centered class means), compared against centered class means in ambient space",
        "probe_nc3_ridge": float(PROBE_NC3_RIDGE),
        "num_checkpoints_evaluated": len(all_records),
        "records": all_records,
    }

    json_path = outdir / f"raw_neural_collapse_backbone_pretext_probe_nc3_metrics_{split_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)

    print("done")
    print(f"saved json: {json_path}")
    print(f"saved plots under: {outdir}")


if __name__ == "__main__":
    main()
