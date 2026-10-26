"""Measure weight-matrix rank and the first four moments of each layer.

For each Conv2d or Linear weight matrix, retain the leading ``round(stable_rank)``
left-singular-vector coordinates, scaled by their singular values. Compare the
flattened coordinates with an isotropic Gaussian fitted to their mean and
variance. The fitted Gaussian has identical first and second moments by
construction; skewness and kurtosis reveal departures from Gaussian shape.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="Experiment name under config/, or path to a config .py file")
    parser.add_argument("--exp-dir", "--exp_dir", dest="exp_dir", help="Checkpoint directory")
    parser.add_argument("--net-key", default="model")
    parser.add_argument("--arch-class", default="NetworkInNetwork")
    parser.add_argument(
        "--layers",
        help=(
            "Comma-separated feature keys (such as conv2.block0) or exact weight "
            "module names; default: all Conv2d/Linear modules"
        ),
    )
    parser.add_argument("--ckpt-glob", default="model_net_epoch*")
    parser.add_argument("--out-path", "--out_path", dest="out_path", default="./moments")
    return parser.parse_args()


def load_config(exp):
    supplied = Path(exp)
    config_path = supplied if supplied.suffix == ".py" else Path("config") / f"{exp}.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config, config_path


def build_fresh_model(config, net_key, arch_class):
    networks = config.get("networks", {})
    if net_key not in networks:
        raise RuntimeError(f"Network '{net_key}' not found; available: {list(networks)}")
    net_config = networks[net_key]
    architecture_path = Path(net_config["def_file"])
    if not architecture_path.is_file():
        raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
    spec = importlib.util.spec_from_file_location(architecture_path.stem, architecture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = arch_class or net_config.get("arch", architecture_path.stem)
    if not hasattr(module, class_name):
        raise RuntimeError(f"Class '{class_name}' not found in {architecture_path}")
    return getattr(module, class_name)(net_config.get("opt", {}).copy()).cpu()


def discover_latest_checkpoint(exp_dir, pattern):
    checkpoints = {}
    for path in exp_dir.glob(pattern):
        match = re.search(r"epoch(\d+)$", path.stem)
        if match:
            checkpoints[int(match.group(1))] = path
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints in {exp_dir} matching '{pattern}'")
    return checkpoints[max(checkpoints)]


def load_state_dict(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "network", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=True)


def selected_weight_modules(model, names):
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    if names is None:
        return modules
    requested = [name.strip() for name in names.split(",") if name.strip()]
    available = dict(modules)
    feature_map = getattr(model, "_feature_name_map", {})
    selected_ids = set()
    unknown = []
    no_weights = []
    for key in requested:
        if key in available:
            selected_ids.add(id(available[key]))
        elif key in feature_map:
            matches = [
                module for module in feature_map[key].modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))
            ]
            if matches:
                selected_ids.update(id(module) for module in matches)
            else:
                no_weights.append(key)
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"Unknown layer(s): {unknown}. Feature keys: {list(feature_map)}; "
            f"weight modules: {list(available)}"
        )
    if no_weights:
        raise ValueError(f"These feature keys contain no Conv2d/Linear weights: {no_weights}")
    return [(name, module) for name, module in modules if id(module) in selected_ids]


def measure_layer(module):
    weight = module.weight.detach().cpu().to(torch.float64)
    matrix = weight.reshape(weight.shape[0], -1)
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    frobenius_squared = torch.sum(singular_values.square()).item()
    spectral_squared = singular_values[0].square().item()
    stable_rank = frobenius_squared / spectral_squared if spectral_squared > 0 else 0.0
    k = min(len(singular_values), max(1, round(stable_rank)))

    projected = u[:, :k] * singular_values[:k]
    # Fix the otherwise arbitrary SVD sign so odd moments are reproducible.
    max_indices = projected.abs().argmax(dim=0)
    column_indices = torch.arange(k)
    signs = torch.sign(projected[max_indices, column_indices])
    signs[signs == 0] = 1.0
    values = (projected * signs).reshape(-1)

    mean = values.mean().item()
    centered = values - mean
    variance = centered.square().mean().item()
    raw = {str(order): values.pow(order).mean().item() for order in range(1, 5)}

    # X ~ N(mean * 1, variance * I): all coordinates have the same variance.
    gaussian_raw = {
        "1": mean,
        "2": mean**2 + variance,
        "3": mean**3 + 3 * mean * variance,
        "4": mean**4 + 6 * mean**2 * variance + 3 * variance**2,
    }
    moment_differences = {
        order: raw[order] - gaussian_raw[order] for order in raw
    }
    if variance > 0:
        standardized = centered / variance**0.5
        skewness = standardized.pow(3).mean().item()
        kurtosis = standardized.pow(4).mean().item()
    else:
        skewness = None
        kurtosis = None

    return {
        "weight_shape": list(weight.shape),
        "matrix_shape": list(matrix.shape),
        "frobenius_norm": frobenius_squared**0.5,
        "stable_rank": stable_rank,
        "numerical_rank": int((singular_values > 1e-4).sum().item()),
        "projected_dimensions": k,
        "sample_count": values.numel(),
        "raw_moments": raw,
        "matched_isotropic_gaussian_raw_moments": gaussian_raw,
        "moment_differences": moment_differences,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3 if kurtosis is not None else None,
    }


def feature_block_label(module_name):
    """Turn an internal weight name into the feature key shown on the x axis."""
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


def grouped_axis_ticks(names):
    """Center one readable label over all measured weights in each feature block."""
    groups = []
    for index, name in enumerate(names):
        label = feature_block_label(name)
        if groups and groups[-1][0] == label:
            groups[-1][2] = index
        else:
            groups.append([label, index, index])

    positions = [(first + last) / 2 for _, first, last in groups]
    labels = [label for label, _, _ in groups]
    return positions, labels


def label_depth_axis(ax, positions, labels):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_xlabel("Feature block (each point is one weight matrix)")


def generate_plots(layers, out_path):
    names = list(layers)
    x = list(range(len(names)))
    tick_positions, tick_labels = grouped_axis_ticks(names)
    plot_width = max(15, 0.8 * len(tick_labels))

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(plot_width, 13),
        constrained_layout=True,
    )
    for order, ax in enumerate(axes, start=1):
        observed = [layers[name]["raw_moments"][str(order)] for name in names]
        gaussian = [
            layers[name]["matched_isotropic_gaussian_raw_moments"][str(order)]
            for name in names
        ]
        ax.plot(
            x,
            gaussian,
            "--",
            color="0.55",
            linewidth=1.3,
            label="Fitted isotropic Gaussian",
        )
        ax.plot(
            x,
            observed,
            "o-",
            color="#24578a",
            label="Layer",
            markersize=3,
        )
        ax.set_title(f"Raw moment {order}: E[X^{order}]")
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        if order == 1:
            ax.legend()

    label_depth_axis(axes[-1], tick_positions, tick_labels)
    fig.suptitle("Stable-rank projections vs matched isotropic Gaussians")
    fig.savefig(out_path / "layerwise_moments.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(plot_width, 8),
        constrained_layout=True,
    )
    for ax, field, reference, title in (
        (axes[0], "skewness", 0, "Skewness (Gaussian = 0)"),
        (axes[1], "kurtosis", 3, "Kurtosis (Gaussian = 3)"),
    ):
        ax.axhline(reference, color="0.55", linestyle="--", label="Gaussian")
        ax.plot(
            x,
            [layers[name][field] for name in names],
            "o-",
            color="#24578a",
            markersize=3,
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    label_depth_axis(axes[-1], tick_positions, tick_labels)
    fig.savefig(out_path / "layerwise_gaussian_shape.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config, config_path = load_config(args.exp)
    exp_dir = Path(args.exp_dir) if args.exp_dir else Path("experiments") / config_path.stem
    config["exp_dir"] = str(exp_dir)
    model = build_fresh_model(config, args.net_key, args.arch_class)
    checkpoint = discover_latest_checkpoint(exp_dir, args.ckpt_glob)
    load_state_dict(model, checkpoint)
    modules = selected_weight_modules(model, args.layers)
    if not modules:
        raise RuntimeError("No Conv2d or Linear weight modules selected")

    layers = {name: measure_layer(module) for name, module in modules}
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(checkpoint),
        "method": "Stable-rank SVD projection; moments over all projected coordinates",
        "gaussian_reference": "N(mean * 1, variance * I), fitted separately for each layer",
        "note": "The first two raw moments match the fitted Gaussian by construction.",
        "layers": layers,
    }
    with (out_path / "rawText.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    generate_plots(layers, out_path)
    print(f"Measured {len(layers)} layers from {checkpoint}")
    print(f"Saved results and plots to {out_path}")


if __name__ == "__main__":
    main()
