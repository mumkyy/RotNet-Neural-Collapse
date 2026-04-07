#!/usr/bin/env python3

import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import train_and_eval as te


# ============================================================
# config
# ============================================================

WORKDIR = "/project/amr239/gma35/RotNet-Neural-Collapse/googleResultsRecreation/workdirs/jigsaw_imagenette_resnet50_NC10"

USE_SPLIT = "val"
ONLY_LAST = False

# special backbone endpoint targets computed from backbone(..., return_endpoints=True)
SPECIAL_ENDPOINT_TARGETS = [
    "after_root",
    "block1",
    "block2",
    "block3",
    "block4",
    "pre_logits",
]

# if True, auto-register all leaf modules under backbone + head
AUTO_DISCOVER_HOOK_TARGETS = True

# optional manual additions
EXTRA_HOOK_TARGETS = [
    # examples:
    # "block2.0.conv1",
    # "block2.1.conv1",
    # "block4.2.conv3",
    # "classifier",
    # "head.conv1",
    # "head.bn1",
    # "head.conv2",
]

# families to use for focused plots
FOCUS_PREFIXES = [
    "block4",
    "head",
    "classifier",
]


# ============================================================
# utils
# ============================================================

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


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


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


def gapify(x):
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 2:
        return x
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def is_leaf_module(module):
    return len(list(module.children())) == 0


def module_name_sort_key(name):
    return tuple(int(p) if p.isdigit() else p for p in name.replace("head.", "zzhead.").split("."))


# ============================================================
# target discovery / resolution
# ============================================================

def discover_hook_targets(model):
    targets = []

    # backbone leaf modules
    for name, module in model.backbone.named_modules():
        if name == "":
            continue
        if is_leaf_module(module):
            targets.append(name)

    # jigsaw head leaf modules
    if hasattr(model, "head"):
        for name, module in model.head.named_modules():
            if name == "":
                continue
            if is_leaf_module(module):
                targets.append(f"head.{name}")

    # de-dupe, sort
    targets = sorted(set(targets), key=module_name_sort_key)
    return targets


def resolve_module(model, name):
    if name.startswith("head."):
        cur = model
        for part in name.split("."):
            cur = getattr(cur, part)
        return cur

    # backbone shorthand
    cur = model.backbone
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


# ============================================================
# feature transforms
# ============================================================

def extract_layer_features_from_endpoints(endpoints, layer_key, B, P, selected_perms):
    """
    endpoints[layer_key]:
      after_root, block1..block4, pre_logits
    shape:
      [B*P, C, H, W] for jigsaw backbone with global_pool=False
    """
    feat = endpoints[layer_key]

    if feat.ndim != 4:
        raise ValueError(f"Expected 4D endpoint for {layer_key}, got {tuple(feat.shape)}")

    _, C, H, W = feat.shape
    feat = feat.view(B, P, C, H, W)
    concat_feat = te.permute_and_concat_batch_patches(feat, selected_perms)
    concat_feat = gapify(concat_feat)
    return concat_feat


def transform_hook_activation_to_examples(name, feat, B, P, selected_perms):
    """
    Convert hooked activations into permutation-example features suitable for NC1.

    backbone hook outputs:
      usually [B*P, ...]  -> reshape to [B, P, ...], permute+concat, GAP if needed

    head hook outputs:
      already [B*M, ...]  -> GAP if needed
    """
    if feat is None:
        raise ValueError(f"Hook activation for {name} is None.")

    # jigsaw head path already operates on permutation examples
    if name.startswith("head."):
        return gapify(feat)

    # backbone path is patch-level
    if feat.ndim == 4:
        _, C, H, W = feat.shape
        feat = feat.view(B, P, C, H, W)
        feat = te.permute_and_concat_batch_patches(feat, selected_perms)
        feat = gapify(feat)
        return feat

    if feat.ndim == 2:
        _, D = feat.shape
        feat = feat.view(B, P, D, 1, 1)
        feat = te.permute_and_concat_batch_patches(feat, selected_perms)
        feat = gapify(feat)
        return feat

    raise ValueError(f"Unsupported hooked tensor shape for {name}: {tuple(feat.shape)}")


# ============================================================
# running stats for NC1
# ============================================================

def init_running_stats(target_names, num_classes):
    stats = OrderedDict()
    for name in target_names:
        stats[name] = {
            "N": 0,
            "class_counts": torch.zeros(num_classes, dtype=torch.long),
            "sum_total": None,
            "class_sums": None,
            "feature_dim": None,
        }
    return stats


def update_running_sums(stats, feat_cpu, y_cpu, num_classes):
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
    centered = mu_c - mu_g.unsqueeze(0)
    sq = (centered * centered).sum(dim=1)
    sb_ss = (counts.to(torch.float64) * sq).sum().item()
    N = counts.sum().item()
    sb_trace = sb_ss / max(N, 1)
    return sb_trace


# ============================================================
# hook registration
# ============================================================

class HookBank:
    def __init__(self):
        self.outputs = {}
        self.handles = []

    def clear(self):
        self.outputs.clear()

    def add_forward_hook(self, name, module):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.outputs[name] = output.detach()
            else:
                raise ValueError(f"Hooked module {name} returned non-tensor output type {type(output)}")
        self.handles.append(module.register_forward_hook(hook))

    def add_forward_prehook(self, name, module):
        def hook(_module, inputs):
            x = inputs[0]
            if torch.is_tensor(x):
                self.outputs[name] = x.detach()
            else:
                raise ValueError(f"Prehooked module {name} got non-tensor input type {type(x)}")
        self.handles.append(module.register_forward_pre_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.outputs.clear()


# ============================================================
# NC1 computation with granular hooks
# ============================================================

@torch.no_grad()
def first_pass_collect_means(model, loader, device, num_classes, hook_targets, endpoint_targets):
    all_targets = endpoint_targets + hook_targets
    layer_stats = init_running_stats(all_targets, num_classes)

    total_head_correct = 0
    total_head_count = 0
    total_head_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")

    hook_bank = HookBank()
    for name in hook_targets:
        hook_bank.add_forward_hook(name, resolve_module(model, name))

    try:
        for batch in tqdm(loader, desc="first pass", leave=False):
            hook_bank.clear()

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

            y_cpu = y.detach().cpu()

            # special backbone endpoints
            if endpoint_targets:
                flat_x = x.reshape(Bsz * P, C, H, W)
                _, endpoints = model.backbone(flat_x, return_endpoints=True)
                for name in endpoint_targets:
                    feat = extract_layer_features_from_endpoints(
                        endpoints=endpoints,
                        layer_key=name,
                        B=Bsz,
                        P=P,
                        selected_perms=selected_perms,
                    )
                    feat_cpu = feat.detach().cpu().to(torch.float64)
                    update_running_sums(layer_stats[name], feat_cpu, y_cpu, num_classes)

            # hooked granular targets
            for name in hook_targets:
                if name not in hook_bank.outputs:
                    raise RuntimeError(f"Hook target {name} did not fire during forward pass.")
                feat = transform_hook_activation_to_examples(
                    name=name,
                    feat=hook_bank.outputs[name],
                    B=Bsz,
                    P=P,
                    selected_perms=selected_perms,
                )
                feat_cpu = feat.detach().cpu().to(torch.float64)
                update_running_sums(layer_stats[name], feat_cpu, y_cpu, num_classes)

    finally:
        hook_bank.close()

    head_metrics = {
        "head_accuracy": total_head_correct / max(total_head_count, 1),
        "head_cross_entropy": total_head_loss / max(total_head_count, 1),
        "num_examples": total_head_count,
    }

    means = OrderedDict()
    for name in all_targets:
        mu_g, mu_c, counts = finalize_means(layer_stats[name])
        means[name] = {
            "mu_g": mu_g,
            "mu_c": mu_c,
            "counts": counts,
            "feature_dim": layer_stats[name]["feature_dim"],
            "num_examples": layer_stats[name]["N"],
        }

    return means, head_metrics


@torch.no_grad()
def second_pass_compute_sw(model, loader, device, means_by_target, hook_targets, endpoint_targets):
    all_targets = endpoint_targets + hook_targets
    out = OrderedDict()
    for name in all_targets:
        out[name] = {"sw_ss": 0.0}

    hook_bank = HookBank()
    for name in hook_targets:
        hook_bank.add_forward_hook(name, resolve_module(model, name))

    try:
        for batch in tqdm(loader, desc="second pass", leave=False):
            hook_bank.clear()

            x = batch["image"].to(device)
            Bsz, P, C, H, W = x.shape

            outputs = model(x)
            y = outputs["labels"].long()
            perm_indices = outputs["perm_indices"].long()
            selected_perms = model.permutations[perm_indices].to(device)
            y_cpu = y.detach().cpu()

            if endpoint_targets:
                flat_x = x.reshape(Bsz * P, C, H, W)
                _, endpoints = model.backbone(flat_x, return_endpoints=True)

                for name in endpoint_targets:
                    feat = extract_layer_features_from_endpoints(
                        endpoints=endpoints,
                        layer_key=name,
                        B=Bsz,
                        P=P,
                        selected_perms=selected_perms,
                    )
                    feat_cpu = feat.detach().cpu().to(torch.float64)
                    mu_c = means_by_target[name]["mu_c"]
                    mu_y = mu_c[y_cpu]
                    diffs = feat_cpu - mu_y
                    out[name]["sw_ss"] += (diffs * diffs).sum().item()

            for name in hook_targets:
                if name not in hook_bank.outputs:
                    raise RuntimeError(f"Hook target {name} did not fire during forward pass.")
                feat = transform_hook_activation_to_examples(
                    name=name,
                    feat=hook_bank.outputs[name],
                    B=Bsz,
                    P=P,
                    selected_perms=selected_perms,
                )
                feat_cpu = feat.detach().cpu().to(torch.float64)
                mu_c = means_by_target[name]["mu_c"]
                mu_y = mu_c[y_cpu]
                diffs = feat_cpu - mu_y
                out[name]["sw_ss"] += (diffs * diffs).sum().item()

    finally:
        hook_bank.close()

    return out


# ============================================================
# direct NC3
# ============================================================

def compute_direct_nc3_from_weight_and_means(W, mu_c, counts, eps=1e-12):
    keep = counts > 0
    Wk = W[keep].to(torch.float64)
    Muk = mu_c[keep].to(torch.float64)

    if Wk.numel() == 0 or Muk.numel() == 0:
        return {
            "nc3": float("nan"),
            "mean_cosine": float("nan"),
            "optimal_scale_error": float("nan"),
            "num_classes_used": 0,
        }

    mu_g = Muk.mean(dim=0, keepdim=True)
    Mc = Muk - mu_g

    Wn = Wk / (torch.norm(Wk, p="fro") + eps)
    Mn = Mc / (torch.norm(Mc, p="fro") + eps)
    nc3 = torch.sum((Wn - Mn) ** 2).item()

    w_norms = torch.norm(Wk, dim=1)
    m_norms = torch.norm(Mc, dim=1)
    valid = (w_norms > eps) & (m_norms > eps)

    if valid.any():
        Wc = Wk[valid] / w_norms[valid].unsqueeze(1)
        Mcn = Mc[valid] / m_norms[valid].unsqueeze(1)
        mean_cos = torch.sum(Wc * Mcn, dim=1).mean().item()
    else:
        mean_cos = float("nan")

    alpha_num = torch.sum(Wk * Mc).item()
    alpha_den = torch.sum(Mc * Mc).item()
    if abs(alpha_den) <= eps:
        scale_err = float("nan")
    else:
        alpha = alpha_num / alpha_den
        scale_err = (torch.norm(Wk - alpha * Mc, p="fro") / (torch.norm(Mc, p="fro") + eps)).item()

    return {
        "nc3": float(nc3),
        "mean_cosine": float(mean_cos),
        "optimal_scale_error": float(scale_err),
        "num_classes_used": int(keep.sum().item()),
    }


def init_head_stat_dict(num_classes):
    return {
        "N": 0,
        "class_counts": torch.zeros(num_classes, dtype=torch.long),
        "class_sums": None,
        "feature_dim": None,
    }


def update_head_stats(stats, feat_cpu, y_cpu, num_classes):
    N, D = feat_cpu.shape
    stats["N"] += N

    if stats["class_sums"] is None:
        stats["class_sums"] = torch.zeros(num_classes, D, dtype=torch.float64)
        stats["feature_dim"] = D

    binc = torch.bincount(y_cpu, minlength=num_classes)
    stats["class_counts"] += binc

    for c in range(num_classes):
        mask = (y_cpu == c)
        if mask.any():
            stats["class_sums"][c] += feat_cpu[mask].sum(dim=0)


def finalize_head_means(stats):
    counts = stats["class_counts"].clone()
    mu_c = torch.zeros_like(stats["class_sums"])
    nonzero = counts > 0
    mu_c[nonzero] = stats["class_sums"][nonzero] / counts[nonzero].unsqueeze(1).to(torch.float64)
    return mu_c, counts


@torch.no_grad()
def compute_direct_nc3_jigsaw_head(model, loader, device, num_classes):
    if not hasattr(model, "head") or not hasattr(model.head, "conv2"):
        raise RuntimeError("Expected model.head.conv2 for jigsaw-head NC3.")

    bank = HookBank()
    bank.add_forward_prehook("head.conv2.input", model.head.conv2)

    stats = init_head_stat_dict(num_classes)
    total_correct = 0
    total_count = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")

    try:
        for batch in tqdm(loader, desc="direct nc3 jigsaw head", leave=False):
            bank.clear()

            x = batch["image"].to(device)
            outputs = model(x)
            logits = outputs["logits"]
            y = outputs["labels"].long()

            if "head.conv2.input" not in bank.outputs:
                raise RuntimeError("Jigsaw head conv2 prehook did not fire.")

            feat_cpu = gapify(bank.outputs["head.conv2.input"]).detach().cpu().to(torch.float64)
            y_cpu = y.detach().cpu()
            update_head_stats(stats, feat_cpu, y_cpu, num_classes)

            total_loss += ce(logits, y).item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += y.numel()

    finally:
        bank.close()

    mu_c, counts = finalize_head_means(stats)
    W = model.head.conv2.weight.detach().cpu().to(torch.float64).squeeze(-1).squeeze(-1)

    out = compute_direct_nc3_from_weight_and_means(W=W, mu_c=mu_c, counts=counts)
    out["feature_dim"] = int(stats["feature_dim"])
    out["class_counts"] = [int(x) for x in counts.tolist()]
    out["head_accuracy"] = total_correct / max(total_count, 1)
    out["head_cross_entropy"] = total_loss / max(total_count, 1)
    out["num_examples"] = total_count
    out["weight_shape"] = list(W.shape)
    return out


@torch.no_grad()
def compute_direct_nc3_backbone_classifier(model, loader, device, num_classes):
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "classifier"):
        raise RuntimeError("Expected model.backbone.classifier for backbone-classifier NC3.")

    bank = HookBank()
    bank.add_forward_prehook("classifier.input", model.backbone.classifier)

    stats = init_head_stat_dict(num_classes)

    try:
        for batch in tqdm(loader, desc="direct nc3 backbone classifier", leave=False):
            bank.clear()

            x = batch["image"].to(device)
            Bsz, P, _, _, _ = x.shape

            outputs = model(x)
            y = outputs["labels"].long()
            M = y.numel() // Bsz

            if "classifier.input" not in bank.outputs:
                raise RuntimeError("Backbone classifier prehook did not fire.")

            # patch-level [B*P, C, H, W] -> GAP -> [B*P, D]
            h_patch = gapify(bank.outputs["classifier.input"]).detach().cpu().to(torch.float64)
            h_patch = h_patch.view(Bsz, P, -1)

            # expand each image's patches across all permutation labels
            h_rep = h_patch.unsqueeze(1).expand(Bsz, M, P, h_patch.shape[-1]).reshape(Bsz * M * P, -1)
            y_rep = y.detach().cpu().view(Bsz, M).unsqueeze(-1).expand(Bsz, M, P).reshape(Bsz * M * P)

            update_head_stats(stats, h_rep, y_rep, num_classes)

    finally:
        bank.close()

    mu_c, counts = finalize_head_means(stats)
    W = model.backbone.classifier.weight.detach().cpu().to(torch.float64).squeeze(-1).squeeze(-1)

    out = compute_direct_nc3_from_weight_and_means(W=W, mu_c=mu_c, counts=counts)
    out["feature_dim"] = int(stats["feature_dim"])
    out["class_counts"] = [int(x) for x in counts.tolist()]
    out["num_examples"] = int(stats["N"])
    out["weight_shape"] = list(W.shape)
    return out


# ============================================================
# full checkpoint metrics
# ============================================================

def compute_metrics_for_checkpoint(model, loader, device, args, hook_targets, endpoint_targets):
    num_classes = args.perm_subset_size
    all_targets = endpoint_targets + hook_targets

    means_by_target, head_metrics = first_pass_collect_means(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        hook_targets=hook_targets,
        endpoint_targets=endpoint_targets,
    )

    pass2 = second_pass_compute_sw(
        model=model,
        loader=loader,
        device=device,
        means_by_target=means_by_target,
        hook_targets=hook_targets,
        endpoint_targets=endpoint_targets,
    )

    nc1_metrics = OrderedDict()
    for name in all_targets:
        mu_g = means_by_target[name]["mu_g"]
        mu_c = means_by_target[name]["mu_c"]
        counts = means_by_target[name]["counts"]

        sb_trace = compute_sb_trace(mu_g, mu_c, counts)
        sw_trace = pass2[name]["sw_ss"] / max(means_by_target[name]["num_examples"], 1)
        nc1 = sw_trace / sb_trace if sb_trace > 0 else float("nan")

        nc1_metrics[name] = {
            "feature_dim": int(means_by_target[name]["feature_dim"]),
            "num_examples": int(means_by_target[name]["num_examples"]),
            "num_classes": int((counts > 0).sum().item()),
            "class_counts": [int(x) for x in counts.tolist()],
            "sw_trace": float(sw_trace),
            "sb_trace": float(sb_trace),
            "nc1_trace_ratio": float(nc1),
            "target_type": (
                "endpoint" if name in endpoint_targets
                else "jigsaw_head_hook" if name.startswith("head.")
                else "backbone_hook"
            ),
        }

    jigsaw_head_nc3 = compute_direct_nc3_jigsaw_head(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
    )

    backbone_classifier_nc3 = compute_direct_nc3_backbone_classifier(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
    )

    return {
        "head_metrics": head_metrics,
        "nc1_metrics": nc1_metrics,
        "nc3_metrics": {
            "jigsaw_head_conv2": jigsaw_head_nc3,
            "backbone_classifier": backbone_classifier_nc3,
        },
    }


# ============================================================
# plotting
# ============================================================

def plot_bar(names, values, ylabel, title, out_path, figsize=(22, 8)):
    plt.figure(figsize=figsize)
    x = np.arange(len(names))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_line(xvals, yvals, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(xvals, yvals, marker="x")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overlay(xvals, series_dict, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(12, 7))
    for name, values in series_dict.items():
        plt.plot(xvals, values, marker="x", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(rows, cols, values_2d, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(max(12, len(cols) * 0.35), max(6, len(rows) * 0.25)))
    im = plt.imshow(values_2d, aspect="auto")
    plt.colorbar(im, label=title)
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(rows)), rows)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_family_groups(target_names):
    groups = OrderedDict()
    groups["all_granular"] = target_names

    for prefix in FOCUS_PREFIXES:
        groups[prefix] = [n for n in target_names if n == prefix or n.startswith(prefix + ".")]

    groups["jigsaw_head"] = [n for n in target_names if n.startswith("head.")]
    groups["block4_all"] = [n for n in target_names if n == "block4" or n.startswith("block4.")]
    groups["backbone_classifier"] = [n for n in target_names if n == "classifier"]

    return {k: v for k, v in groups.items() if len(v) > 0}


def produce_plots(records, outdir, target_names):
    family_groups = make_family_groups(target_names)

    # final-checkpoint NC1 bars
    final_rec = records[-1]
    final_nc1 = final_rec["nc1_metrics"]

    names = target_names
    vals = [final_nc1[n]["nc1_trace_ratio"] for n in names]
    plot_bar(
        names=names,
        values=vals,
        ylabel="NC1",
        title=f"NC1 - layerwise granular ({final_rec['checkpoint_name']})",
        out_path=outdir / "nc1_layerwise_granular_final.png",
        figsize=(max(24, len(names) * 0.35), 8),
    )

    for group_name, group_targets in family_groups.items():
        vals = [final_nc1[n]["nc1_trace_ratio"] for n in group_targets]
        plot_bar(
            names=group_targets,
            values=vals,
            ylabel="NC1",
            title=f"NC1 - {group_name} ({final_rec['checkpoint_name']})",
            out_path=outdir / f"nc1_{group_name}_final.png",
            figsize=(max(14, len(group_targets) * 0.45), 7),
        )

    # feature-dim companion plot
    dims = [final_nc1[n]["feature_dim"] for n in names]
    plot_bar(
        names=names,
        values=dims,
        ylabel="feature_dim",
        title=f"Feature dimension by target ({final_rec['checkpoint_name']})",
        out_path=outdir / "feature_dim_by_target_final.png",
        figsize=(max(24, len(names) * 0.35), 8),
    )

    # NC3 final bars
    nc3_final = final_rec["nc3_metrics"]
    nc3_names = ["backbone_classifier", "jigsaw_head_conv2"]
    nc3_vals = [nc3_final[n]["nc3"] for n in nc3_names]
    plot_bar(
        names=nc3_names,
        values=nc3_vals,
        ylabel="NC3",
        title=f"Direct NC3 by classifier ({final_rec['checkpoint_name']})",
        out_path=outdir / "nc3_final_bar.png",
        figsize=(8, 6),
    )

    mean_cos_vals = [nc3_final[n]["mean_cosine"] for n in nc3_names]
    plot_bar(
        names=nc3_names,
        values=mean_cos_vals,
        ylabel="mean cosine",
        title=f"Direct NC3 mean cosine by classifier ({final_rec['checkpoint_name']})",
        out_path=outdir / "nc3_mean_cosine_final_bar.png",
        figsize=(8, 6),
    )

    # if multiple checkpoints, produce epoch-based curves and heatmaps
    if len(records) > 1:
        epochs = [r["epoch"] for r in records]

        # requested style: per-layer epoch vs NC1
        for name in target_names:
            vals = [r["nc1_metrics"][name]["nc1_trace_ratio"] for r in records]
            safe_name = name.replace(".", "_")
            plot_line(
                xvals=epochs,
                yvals=vals,
                xlabel="epoch",
                ylabel="NC1",
                title=f"NC1 - {name} vs epoch",
                out_path=outdir / f"nc1_{safe_name}_vs_epoch.png",
            )

        # focused overlays
        for group_name, group_targets in family_groups.items():
            if len(group_targets) <= 1:
                continue
            series = OrderedDict()
            for name in group_targets:
                series[name] = [r["nc1_metrics"][name]["nc1_trace_ratio"] for r in records]
            plot_overlay(
                xvals=epochs,
                series_dict=series,
                xlabel="epoch",
                ylabel="NC1",
                title=f"NC1 - {group_name} vs epoch",
                out_path=outdir / f"nc1_{group_name}_vs_epoch_overlay.png",
            )

        # full NC1 heatmap
        Z = np.array(
            [[r["nc1_metrics"][name]["nc1_trace_ratio"] for name in target_names] for r in records],
            dtype=np.float64,
        )
        plot_heatmap(
            rows=[f"{e:.3f}" for e in epochs],
            cols=target_names,
            values_2d=Z,
            xlabel="target layer",
            ylabel="epoch",
            title="NC1 heatmap",
            out_path=outdir / "nc1_heatmap_epochs_targets.png",
        )

        # NC3 vs epoch
        series = OrderedDict()
        series["backbone_classifier"] = [r["nc3_metrics"]["backbone_classifier"]["nc3"] for r in records]
        series["jigsaw_head_conv2"] = [r["nc3_metrics"]["jigsaw_head_conv2"]["nc3"] for r in records]
        plot_overlay(
            xvals=epochs,
            series_dict=series,
            xlabel="epoch",
            ylabel="NC3",
            title="Direct NC3 vs epoch",
            out_path=outdir / "nc3_vs_epoch.png",
        )


# ============================================================
# main
# ============================================================

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
    outdir = workdir / f"granular_nc1_nc3_{split_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"workdir: {workdir}")
    print(f"split: {split_name}")
    print(f"output dir: {outdir}")
    print(f"num checkpoints: {len(ckpts)}")

    model = te.build_model(args).to(device)
    model.eval()

    if AUTO_DISCOVER_HOOK_TARGETS:
        hook_targets = discover_hook_targets(model)
    else:
        hook_targets = []

    hook_targets = sorted(set(hook_targets + EXTRA_HOOK_TARGETS), key=module_name_sort_key)

    endpoint_targets = [t for t in SPECIAL_ENDPOINT_TARGETS if t in getattr(model.backbone, "all_feat_names", [])]

    print(f"num endpoint targets: {len(endpoint_targets)}")
    print(f"num hook targets: {len(hook_targets)}")

    all_records = []

    for ckpt_path in tqdm(ckpts, desc="checkpoints"):
        step, epoch, _ = te.load_checkpoint(
            model=model,
            optimizer=None,
            checkpoint_path=str(ckpt_path),
            device=device,
        )
        model.eval()

        metrics = compute_metrics_for_checkpoint(
            model=model,
            loader=loader,
            device=device,
            args=args,
            hook_targets=hook_targets,
            endpoint_targets=endpoint_targets,
        )

        record = {
            "checkpoint_name": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "step": int(step),
            "epoch": float(epoch),
            "split": split_name,
            "head_metrics": metrics["head_metrics"],
            "nc1_metrics": metrics["nc1_metrics"],
            "nc3_metrics": metrics["nc3_metrics"],
        }
        all_records.append(record)

        with open(outdir / f"partial_metrics_{ckpt_path.stem}.json", "w") as f:
            json.dump(sanitize_for_json(record), f, indent=2)

    all_records = sorted(all_records, key=lambda r: (r["epoch"], r["step"]))

    target_names = endpoint_targets + hook_targets
    produce_plots(all_records, outdir, target_names)

    summary = {
        "workdir": str(workdir),
        "split": split_name,
        "timestamp_completed": timestamp,
        "device": str(device),
        "only_last": ONLY_LAST,
        "endpoint_targets": endpoint_targets,
        "hook_targets": hook_targets,
        "notes": {
            "nc1": "computed from hooked activations / special endpoints using permutation-example labels",
            "nc3_backbone_classifier": "direct NC3 using backbone.classifier weights and GAP(input to classifier), with current pretext permutation labels expanded over patch-level samples",
            "nc3_jigsaw_head": "direct NC3 using head.conv2 weights and GAP(input to conv2), with current jigsaw permutation labels",
            "epoch_plots": "produced only when more than one checkpoint is evaluated; with ONLY_LAST=True you will get final-checkpoint layerwise plots and raw granular json",
        },
        "num_checkpoints_evaluated": len(all_records),
        "records": all_records,
    }

    json_path = outdir / f"raw_granular_nc1_nc3_metrics_{split_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)

    print("done")
    print(f"saved json: {json_path}")
    print(f"saved plots under: {outdir}")


if __name__ == "__main__":
    main()