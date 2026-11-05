#!/usr/bin/env python3
"""Compare weight and representation projections with isotropic Gaussians.

Run from the repository root with ``python -m measurements.kurtosisMeasurements``.
Representation stable rank is computed from the globally centered N-by-D
activation matrix after spatial average pooling. Its leading ``round(sr)``
principal directions define the projection. Moments are computed over all
entries of the projected N-by-k matrix. The Gaussian reference consists of
k-dimensional vectors with the fitted coordinate means and one common variance.
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="Config path under --config-root, or a .py file")
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--exp-dir", "--exp_dir", dest="exp_dir", help="Default: experiments/<config stem>")
    parser.add_argument("--ckpt-glob", default="model_net_epoch*")
    parser.add_argument("--checkpoint", type=int, help="Epoch number; default: latest")
    parser.add_argument("--net-key", default="model")
    parser.add_argument("--arch-class", default="ResNet34_NIN_Style")
    parser.add_argument(
        "--layers",
        help="Comma-separated feature keys or exact Conv2d/Linear module names; default: all stages and head",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, help="Default: config batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--plot-max-entries", type=int, default=200000)
    parser.add_argument("--out-path", "--out_path", dest="out_path", default="./moments")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_root, exp):
    supplied = Path(exp)
    path = supplied if supplied.suffix == ".py" else Path(config_root) / f"{exp}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    spec = importlib.util.spec_from_file_location("measurement_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config, path


def build_model(config, net_key, class_name):
    networks = config.get("networks", {})
    if net_key not in networks:
        raise KeyError(f"Network '{net_key}' not found; available: {list(networks)}")
    net_config = networks[net_key]
    path = Path(net_config["def_file"])
    if not path.is_file():
        raise FileNotFoundError(f"Architecture not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {path}")
    return getattr(module, class_name)(net_config.get("opt", {}).copy())


def discover_checkpoint(exp_dir, pattern, requested_epoch):
    checkpoints = {}
    for path in exp_dir.glob(pattern):
        match = re.search(r"epoch(\d+)$", path.stem)
        if match:
            checkpoints[int(match.group(1))] = path
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints in {exp_dir} matching '{pattern}'")
    epoch = max(checkpoints) if requested_epoch is None else requested_epoch
    if epoch not in checkpoints:
        raise KeyError(f"Epoch {epoch} not found; available: {sorted(checkpoints)}")
    return epoch, checkpoints[epoch]


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "network", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=True)


def select_feature_keys(model, requested):
    if not hasattr(model, "all_feat_names"):
        raise RuntimeError("Model must expose all_feat_names and out_feat_keys")
    if requested:
        keys = [key.strip() for key in requested.split(",") if key.strip()]
    else:
        keys = ["conv1"]
        keys += [
            key for key in model.all_feat_names
            if re.fullmatch(r"conv[2-5]\.block\d+", key)
        ]
        keys += ["penult", "lin1", "lin2", "classifier"]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("--layers must contain unique feature keys")
    weight_names = {
        name for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    unknown = [
        key for key in keys
        if key not in model.all_feat_names and key not in weight_names
    ]
    if unknown:
        raise ValueError(f"Unknown layers: {unknown}; feature keys: {model.all_feat_names}")
    feature_keys = []
    for key in keys:
        feature_key = key if key in model.all_feat_names else weight_group_label(key)
        if feature_key not in model.all_feat_names:
            raise ValueError(f"Cannot map weight module {key} to a feature key")
        if feature_key not in feature_keys:
            feature_keys.append(feature_key)
    return feature_keys


def select_weight_modules(model, feature_keys, requested=None):
    modules = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    available = dict(modules)
    feature_map = getattr(model, "_feature_name_map", {})
    selected_ids = set()
    requested_keys = [key.strip() for key in requested.split(",") if key.strip()] if requested else feature_keys
    for key in requested_keys:
        if key in available:
            selected_ids.add(id(available[key]))
        elif key in feature_map:
            selected_ids.update(
                id(module) for module in feature_map[key].modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))
            )
    return [(name, module) for name, module in modules if id(module) in selected_ids]


def build_loader(config, args, exp_dir):
    """Match the repo's GenericDataset and callable DataLoader conventions."""
    from dataloader import GenericDataset, DataLoader as RotLoader

    options = config["data_train_opt" if args.split == "train" else "data_test_opt"]
    mode = options["pretext_mode"]
    fixed_perms = None
    if mode in ("jigsaw", "jigsaw_9"):
        path = exp_dir / "jigsaw_perms.json"
        if not path.is_file():
            raise FileNotFoundError(f"Training-time jigsaw permutations not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            fixed_perms = [tuple(permutation) for permutation in json.load(file)]
        if not fixed_perms:
            raise ValueError(f"No permutations found in {path}")

    dataset = GenericDataset(
        dataset_name=options["dataset_name"],
        split=args.split,
        random_sized_crop=options.get("random_sized_crop", False),
        num_imgs_per_cat=None,
        pretext_mode=mode,
        sigmas=None,
        kernel_sizes=None,
        patch_jitter=options.get("patch_jitter", 0),
        color_distort=options.get("color_distort", False),
        color_dist_strength=options.get("color_dist_strength", 0.0),
        fixed_perms=fixed_perms,
    )
    loader = RotLoader(
        dataset=dataset,
        batch_size=args.batch_size or options["batch_size"],
        unsupervised=options.get("unsupervised", True),
        epoch_size=None,
        num_workers=args.workers,
        shuffle=False,
    )
    return loader, options["dataset_name"], mode


def gapify(feature):
    if feature.ndim == 4:
        return feature.mean(dim=(2, 3))
    if feature.ndim == 2:
        return feature
    return feature.flatten(1)


@torch.no_grad()
def collect_representations(model, loader, keys, device, seed):
    """Collect one augmented training pass, so all layers see the same inputs."""
    set_seed(seed)
    buffers = {key: [] for key in keys}
    label_batches = []
    for images, labels in tqdm(loader(0), desc="Collect representations", unit="batch"):
        outputs = model(images.to(device), out_feat_keys=keys)
        if len(keys) == 1:
            outputs = [outputs]
        if len(outputs) != len(keys):
            raise RuntimeError("Model returned a different number of features than requested")
        for key, feature in zip(keys, outputs):
            buffers[key].append(gapify(feature).detach().to("cpu", dtype=torch.float32))
        label_batches.append(labels.to("cpu", dtype=torch.long))
    if not label_batches:
        raise RuntimeError("The data loader returned no examples")
    labels = torch.cat(label_batches)
    return buffers, labels


def canonicalize_columns(matrix, basis=None):
    """Fix arbitrary SVD/PCA signs for reproducible odd moments."""
    anchor_source = matrix if basis is None else basis
    anchor_rows = anchor_source.abs().argmax(dim=0)
    columns = torch.arange(matrix.shape[1], device=matrix.device)
    signs = torch.sign(anchor_source[anchor_rows, columns])
    signs[signs == 0] = 1.0
    return matrix * signs


def projected_moments(projected, mean_mode, seed, plot_max_entries):
    """Analyze an N-by-k matrix against N(mu, sigma^2 I_k)."""
    n, k = projected.shape
    if mean_mode == "scalar":
        means = projected.mean().expand(k)
    else:
        means = projected.mean(dim=0)
    centered = projected - means
    variance = centered.square().mean().item()
    raw = {str(order): projected.pow(order).mean().item() for order in range(1, 5)}
    gaussian_raw = {
        "1": means.mean().item(),
        "2": (means.square() + variance).mean().item(),
        "3": (means.pow(3) + 3 * means * variance).mean().item(),
        "4": (means.pow(4) + 6 * means.square() * variance + 3 * variance**2).mean().item(),
    }
    if variance > 0:
        skewness = centered.pow(3).mean().item() / variance**1.5
        kurtosis = centered.pow(4).mean().item() / variance**2
        covariance = centered.T @ centered / n
        norm_squared = covariance.square().sum().item()
        isotropy_score = covariance.trace().item()**2 / (k * norm_squared)
        isotropy_score = max(0.0, min(1.0, isotropy_score))
    else:
        skewness = kurtosis = isotropy_score = None

    rng = np.random.default_rng(seed)
    row_count = min(n, max(1, plot_max_entries // k))
    row_indices = rng.choice(n, size=row_count, replace=False)
    selected_rows = torch.as_tensor(row_indices, device=projected.device)
    empirical_plot = projected.index_select(0, selected_rows).detach().cpu().numpy().reshape(-1)
    gaussian_plot = rng.normal(
        loc=means.detach().cpu().numpy(),
        scale=variance**0.5,
        size=(row_count, k),
    ).reshape(-1)
    statistics = {
        "sample_count": n,
        "projected_dimensions": k,
        "fitted_coordinate_variance": variance,
        "variance": variance,
        "raw_moments": raw,
        "matched_isotropic_gaussian_raw_moments": gaussian_raw,
        "moment_differences": {
            order: raw[order] - gaussian_raw[order] for order in raw
        },
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3 if kurtosis is not None else None,
        "covariance_isotropy_score": isotropy_score,
    }
    return statistics, empirical_plot, gaussian_plot


def measure_weight(module, device, seed, plot_max_entries):
    weight = module.weight.detach().to(device=device, dtype=torch.float64)
    matrix = weight.reshape(weight.shape[0], -1)
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    frobenius_squared = singular_values.square().sum().item()
    spectral_squared = singular_values[0].square().item()
    stable_rank = frobenius_squared / spectral_squared if spectral_squared > 0 else 0.0
    k = min(singular_values.numel(), max(1, round(stable_rank)))
    projected = canonicalize_columns(u[:, :k] * singular_values[:k])
    statistics, empirical_plot, gaussian_plot = projected_moments(
        projected, "scalar", seed, plot_max_entries
    )
    statistics.update({
        "weight_shape": list(weight.shape),
        "matrix_shape": list(matrix.shape),
        "frobenius_norm": frobenius_squared**0.5,
        "stable_rank": stable_rank,
        "numerical_rank": int((singular_values > 1e-4).sum().item()),
    })
    return statistics, empirical_plot, gaussian_plot


def class_collapse_metrics(matrix, labels, num_classes):
    """Supplementary NC1/NC2 metrics on the unprojected representations."""
    counts = torch.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        return {"nc1_within_to_between_trace": None, "nc2_relative_etf_gram_error": None}
    sums = torch.zeros(num_classes, matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    sums.index_add_(0, labels, matrix)
    means = sums / counts[:, None]
    residuals = matrix - means.index_select(0, labels)
    within_trace = residuals.square().sum().item() / matrix.shape[0]
    centered_means = means - means.mean(dim=0, keepdim=True)
    between_trace = centered_means.square().sum(dim=1).mean().item()
    nc1 = within_trace / between_trace if between_trace > 0 else None

    ideal = torch.eye(num_classes, dtype=matrix.dtype, device=matrix.device)
    ideal -= torch.ones_like(ideal) / num_classes
    gram = centered_means @ centered_means.T
    scale = (gram * ideal).sum() / ideal.square().sum()
    etf_error = (
        (gram - scale * ideal).norm().item() / (scale * ideal).norm().item()
        if scale.item() > 0 else None
    )
    return {
        "nc1_within_to_between_trace": nc1,
        "nc2_relative_etf_gram_error": etf_error,
    }


def measure_representation(
    features_cpu, labels_cpu, num_classes, device, seed, plot_max_entries
):
    matrix = features_cpu.to(device=device, dtype=torch.float64)
    labels = labels_cpu.to(device=device, dtype=torch.long)
    class_metrics = class_collapse_metrics(matrix, labels, num_classes)

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    gram = centered.T @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    largest = eigenvalues[-1].item()
    frobenius_squared = eigenvalues.sum().item()
    stable_rank = frobenius_squared / largest if largest > 0 else 0.0
    squared_spectrum_sum = eigenvalues.square().sum().item()
    full_isotropy_score = (
        frobenius_squared**2 / (matrix.shape[1] * squared_spectrum_sum)
        if squared_spectrum_sum > 0 else None
    )
    if full_isotropy_score is not None:
        full_isotropy_score = max(0.0, min(1.0, full_isotropy_score))
    k = min(matrix.shape[1], max(1, round(stable_rank)))
    basis = eigenvectors[:, -k:].flip(dims=(1,))
    projected = canonicalize_columns(centered @ basis, basis)

    statistics, empirical_plot, gaussian_plot = projected_moments(
        projected, "coordinate", seed, plot_max_entries
    )

    # _, S, _ = torch.linalg.svd(centered)

    # svspec =  []
    # num_singularVals = S.size(0)
    # for i in range(len(num_singularVals)): 
    #     svspec.append(S[i])

    svspec = torch.linalg.svdvals(centered).detach().cpu().tolist()

    statistics.update({
        "representation_shape": list(matrix.shape),
        "frobenius_norm": frobenius_squared**0.5,
        "stable_rank": stable_rank,
        "numerical_rank": int((eigenvalues > largest * 1e-8).sum().item()) if largest > 0 else 0,
        "full_covariance_isotropy_score": full_isotropy_score,
        "singular_values": svspec,
        **class_metrics,
    })
    return statistics, empirical_plot, gaussian_plot, svspec


def weight_group_label(module_name):
    match = re.fullmatch(r"_feature_blocks\.(\d+)\.(.+)", module_name)
    if not match:
        return module_name
    stage = int(match.group(1))
    path = match.group(2)
    if stage == 0:
        return "conv1.Stem_Conv"
    if stage in (1, 2, 3, 4):
        return f"conv{stage + 1}.{path.split('.')[0]}"
    head = path.split(".")[0]
    return head.lower() if head == "Classifier" else head


def grouped_ticks(names, grouped):
    labels = [weight_group_label(name) if grouped else name for name in names]
    groups = []
    for index, label in enumerate(labels):
        if groups and groups[-1][0] == label:
            groups[-1][2] = index
        else:
            groups.append([label, index, index])
    positions = [(first + last) / 2 for _, first, last in groups]
    return positions, [label for label, _, _ in groups]


def label_axis(axis, positions, labels):
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    axis.set_xlabel("Network depth")


def plot_network(statistics, output_dir, title, grouped, filename_prefix):
    names = list(statistics)
    x = list(range(len(names)))
    positions, labels = grouped_ticks(names, grouped)
    width = max(15, 0.8 * len(labels))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(width, 13), constrained_layout=True)
    for order, axis in enumerate(axes, start=1):
        observed = [statistics[name]["raw_moments"][str(order)] for name in names]
        gaussian = [
            statistics[name]["matched_isotropic_gaussian_raw_moments"][str(order)]
            for name in names
        ]
        axis.plot(x, gaussian, "--", color="0.55", label="Fitted isotropic Gaussian")
        axis.plot(x, observed, "o-", color="#24578a", markersize=3, label="Observed")
        axis.set_title(f"Raw moment {order}: E[X^{order}]")
        axis.grid(axis="y", alpha=0.25)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        if order == 1:
            axis.legend()
    label_axis(axes[-1], positions, labels)
    fig.suptitle(f"{title}: projected moments across depth")
    fig.savefig(output_dir / f"{filename_prefix}_moments.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(width, 10), constrained_layout=True)
    shape_fields = (
        ("skewness", 0.0, "Projected skewness"),
        ("kurtosis", 3.0, "Projected kurtosis"),
        ("covariance_isotropy_score", 1.0, "Projected covariance isotropy score"),
    )
    for axis, (field, reference, name) in zip(axes, shape_fields):
        axis.axhline(reference, color="0.55", linestyle="--", label="Isotropic Gaussian")
        axis.plot(x, [statistics[key][field] for key in names], "o-", color="#24578a", markersize=3)
        if field == "covariance_isotropy_score" and not grouped:
            axis.plot(
                x,
                [statistics[key]["full_covariance_isotropy_score"] for key in names],
                "s-",
                color="#a05a2c",
                markersize=3,
                label="Before projection",
            )
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[-1].set_ylim(-0.05, 1.05)
    label_axis(axes[-1], positions, labels)
    fig.suptitle(f"{title}: Gaussian shape diagnostics across depth")
    fig.savefig(output_dir / f"{filename_prefix}_gaussian_shape.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(width, 10), constrained_layout=True)
    for axis, field, name in zip(
        axes,
        ("frobenius_norm", "stable_rank", "numerical_rank"),
        ("Frobenius norm", "Stable rank", "Numerical rank"),
    ):
        axis.plot(x, [statistics[key][field] for key in names], "o-", color="#24578a", markersize=3)
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.25)
    label_axis(axes[-1], positions, labels)
    fig.suptitle(f"{title}: rank and norm across depth")
    fig.savefig(output_dir / f"{filename_prefix}_rank_and_norm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(empirical, gaussian, statistics, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if statistics["fitted_coordinate_variance"] > 0:
        all_values = np.concatenate((empirical, gaussian))
        lower, upper = np.quantile(all_values, (0.001, 0.999))
        bins = np.linspace(lower, upper, 81) if upper > lower else 80
        axes[0].hist(empirical, bins=bins, density=True, alpha=0.45, label="Observed")
        axes[0].hist(gaussian, bins=bins, density=True, alpha=0.35, label="Isotropic Gaussian")
        quantiles = np.linspace(0.001, 0.999, 199)
        empirical_q = np.quantile(empirical, quantiles)
        gaussian_q = np.quantile(gaussian, quantiles)
        axes[1].plot(gaussian_q, empirical_q, color="#24578a")
        line_min = min(empirical_q[0], gaussian_q[0])
        line_max = max(empirical_q[-1], gaussian_q[-1])
        axes[1].plot((line_min, line_max), (line_min, line_max), "--", color="0.55")
        axes[1].set_xlabel("Gaussian quantile")
        axes[1].set_ylabel("Observed quantile")
    else:
        axes[0].axvline(empirical[0], color="#24578a", label="Degenerate projection")
        axes[1].text(0.5, 0.5, "Zero fitted variance", ha="center", va="center")
    axes[0].set_title("Flattened coordinates of k-dimensional vectors")
    axes[0].set_xlabel("Projected coordinate value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].set_title("Quantile comparison")
    fig.suptitle(
        f"{title} | k={statistics['projected_dimensions']} | "
        f"skew={statistics['skewness'] if statistics['skewness'] is not None else float('nan'):.3g} | "
        f"kurtosis={statistics['kurtosis'] if statistics['kurtosis'] is not None else float('nan'):.3g}"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def plot_rep_svd(svspec, out_path):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    indices = np.arange(1, len(svspec) + 1)

    ax.plot(indices, svspec)
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title(out_path.stem.replace("_svspec", ""))
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
def save_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def main():
    args = parse_args()
    if args.plot_max_entries < 1:
        raise ValueError("--plot-max-entries must be positive")
    set_seed(args.seed)
    config, config_path = load_config(args.config_root, args.exp)
    exp_dir = Path(args.exp_dir) if args.exp_dir else Path("experiments") / config_path.stem
    config["exp_dir"] = str(exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {exp_dir}")
    epoch, checkpoint = discover_checkpoint(exp_dir, args.ckpt_glob, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = build_model(config, args.net_key, args.arch_class)
    load_checkpoint(model, checkpoint)
    model.eval().to(device)
    feature_keys = select_feature_keys(model, args.layers)
    weight_modules = select_weight_modules(model, feature_keys, args.layers)
    if not weight_modules:
        raise RuntimeError("No Conv2d or Linear weights selected")

    output_root = Path(args.out_path)
    weights_dir = output_root / "weights"
    representations_dir = output_root / "representations"
    weight_layerwise = weights_dir / "layerwise"
    representation_layerwise = representations_dir / "layerwise"
    weight_layerwise.mkdir(parents=True, exist_ok=True)
    representation_layerwise.mkdir(parents=True, exist_ok=True)

    weight_results = {}
    for index, (name, module) in enumerate(tqdm(weight_modules, desc="Weight projections")):
        statistics, empirical, gaussian = measure_weight(
            module, device, args.seed + index, args.plot_max_entries
        )
        weight_results[name] = statistics
        plot_overlay(
            empirical, gaussian, statistics, name,
            weight_layerwise / f"{safe_name(name)}.png",
        )
    save_json(weights_dir / "rawText.json", {
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "method": "Stable-rank SVD projection of each Conv2d/Linear weight matrix",
        "gaussian_reference": "N(scalar mean * 1, pooled variance * I), fitted per layer",
        "note": "First two raw moments match the fitted Gaussian by construction.",
        "layers": weight_results,
    })
    plot_network(weight_results, weights_dir, "Weights", grouped=True, filename_prefix="layerwise")

    loader, dataset_name, pretext_mode = build_loader(config, args, exp_dir)
    buffers, labels = collect_representations(model, loader, feature_keys, device, args.seed)
    num_classes = config["networks"][args.net_key]["opt"]["num_classes"]
    if labels.ndim != 1 or ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError(f"Loader labels must be integers in [0, {num_classes})")
    class_counts = torch.bincount(labels, minlength=num_classes).tolist()

    representation_results = {}
    for index, key in enumerate(tqdm(feature_keys, desc="Representation projections")):
        feature_matrix = torch.cat(buffers.pop(key), dim=0)
        statistics, empirical, gaussian, svspec = measure_representation(
            feature_matrix,
            labels,
            num_classes,
            device,
            args.seed + 10000 + index,
            args.plot_max_entries,
        )
        representation_results[key] = statistics
        plot_overlay(
            empirical, gaussian, statistics, key,
            representation_layerwise / f"{safe_name(key)}.png",
        )
        plot_rep_svd(svspec, representation_layerwise / f"{safe_name(key)}_svspec.png")

        del feature_matrix

    save_json(representations_dir / "rawText.json", {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "dataset": dataset_name,
        "split": args.split,
        "pretext_mode": pretext_mode,
        "label_meaning": "Loader pretext labels; jigsaw_9 means permutation IDs",
        "class_counts": class_counts,
        "feature_processing": "Global average pool convolutional maps over spatial positions",
        "projection": "Center N-by-D representations; top k PCA directions, k=round(stable_rank)",
        "gaussian_reference": "N(projected coordinate means, pooled variance * I_k), fitted per layer",
        "moment_definition": "Average powers over all coordinates of projected vectors",
        "isotropy_score_definition": "tr(C)^2 / (k * ||C||_F^2); 1 is isotropic",
        "note": (
            "Matching scalar moments and a flat coordinate histogram do not prove joint "
            "Gaussianity. Projected isotropy is automatic when k=1; consult the full "
            "covariance score too. PCA signs are fixed by their largest loading."
        ),
        "layers": representation_results,
    })
    plot_network(
        representation_results,
        representations_dir,
        "Representations",
        grouped=False,
        filename_prefix="network",
    )
    print(f"Measured {len(weight_results)} weight matrices and {len(representation_results)} feature keys")
    print(f"Representations used {labels.numel()} examples from the {args.split} split")
    print(f"Results saved to {output_root}")


if __name__ == "__main__":
    main()
