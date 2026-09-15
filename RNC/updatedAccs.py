#!/usr/bin/env python3

import re
import csv
import argparse
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

import matplotlib.pyplot as plt


# ============================================================
# REGEX
# ============================================================

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

EPOCH_RE = re.compile(
    r"Training epoch\s*\[\s*(\d+)\s*/\s*\d+\s*\]"
)

TRAIN_RE = re.compile(
    rf"Training stats:\s*\{{[^}}]*[\"']prec1[\"']:\s*({FLOAT_RE})"
)

EVAL_RE = re.compile(
    rf"Evaluation stats:\s*\{{[^}}]*[\"']prec1[\"']:\s*({FLOAT_RE})"
)


# ============================================================
# LOG PARSING
# ============================================================

def parse_log_epoch(path: Path, target_epoch: int):
    """
    Extract training and validation top-1 accuracy for one
    specific epoch.

    Returns:
        (train_acc, val_acc)

    Either value may be None if not found.
    """

    current_epoch = None
    train_acc = None
    val_acc = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            m = EPOCH_RE.search(line)

            if m:
                current_epoch = int(m.group(1))
                continue

            if current_epoch != target_epoch:
                continue

            m = TRAIN_RE.search(line)

            if m:
                train_acc = float(m.group(1))
                continue

            m = EVAL_RE.search(line)

            if m:
                val_acc = float(m.group(1))
                continue

    return train_acc, val_acc


# ============================================================
# EXPERIMENT DISCOVERY
# ============================================================

def discover_experiment_dirs(root: Path, checkpoint_name: str):
    """
    Recursively find every experiment directory containing
    the requested checkpoint.

    Example checkpoint:
        classifier_net_epoch100
    """

    experiment_dirs = set()

    for checkpoint in root.rglob(checkpoint_name):
        experiment_dirs.add(checkpoint.parent)

    return sorted(experiment_dirs)


# ============================================================
# LOG DISCOVERY
# ============================================================

def find_log_for_epoch(
    experiment_dir: Path,
    target_epoch: int,
    log_pattern: str = "*.txt"
):
    """
    Locate the log corresponding to an experiment and extract
    accuracy at target_epoch.

    Preference is given to:

        experiment_dir/logs/

    If multiple logs contain the requested epoch, the most
    recently modified one is selected.
    """

    log_dir = experiment_dir / "logs"

    if log_dir.exists():
        candidates = list(log_dir.rglob(log_pattern))
    else:
        candidates = list(experiment_dir.rglob(log_pattern))

    matches = []

    for log_file in candidates:

        train_acc, val_acc = parse_log_epoch(
            log_file,
            target_epoch
        )

        if train_acc is not None or val_acc is not None:
            matches.append(
                (
                    log_file,
                    train_acc,
                    val_acc
                )
            )

    if not matches:
        return None, None, None

    # Most recent matching log wins.
    matches.sort(
        key=lambda x: x[0].stat().st_mtime
    )

    log_file, train_acc, val_acc = matches[-1]

    return log_file, train_acc, val_acc


# ============================================================
# NETWORK KEY EXTRACTION
# ============================================================

def extract_network_key(
    experiment_dir: Path,
    network_name: str
):
    """
    Extract the network feature/layer key from the experiment
    directory name.

    Example:

    Imagenette_ConvClassifier_on_Jigsaw_10_classes_resnet34_conv3.block2_feats_Collapsed_MSE

    becomes:

    conv3.block2

    Example:

    Imagenette_ConvClassifier_on_Jigsaw_10_classes_resnet34_conv1.Stem_BN_feats_Collapsed_MSE

    becomes:

    conv1.Stem_BN
    """

    name = experiment_dir.name

    marker = f"{network_name}_"

    if marker not in name:
        return name

    tail = name.split(marker, 1)[1]

    if "_feats_" in tail:
        tail = tail.split("_feats_", 1)[0]

    return tail


# ============================================================
# RUN EXTRACTION
# ============================================================

def extract_run_name(path: Path, root: Path):
    """
    Find run1, run2, ..., runN in the experiment path.
    """

    current = path

    while True:

        if re.fullmatch(r"run\d+", current.name):
            return current.name

        if current == root:
            break

        if current.parent == current:
            break

        current = current.parent

    return "unknown_run"


# ============================================================
# NETWORK DEPTH SORTING
# ============================================================

def network_layer_sort_key(layer: str):
    """
    Sort ResNet layers in approximate forward-pass order.

    Desired ordering:

        conv1.Stem_Conv
        conv1.Stem_BN
        conv1.Stem_ReLU
        conv1.Stem_MaxPool
        conv1

        conv2.block0
        conv2.block1
        conv2.block2
        conv2

        conv3.block0
        ...
        conv3

        conv4.block0
        ...
        conv4

        conv5.block0
        ...
        conv5
    """

    stem_order = {
        "conv1.Stem_Conv": 0,
        "conv1.Stem_BN": 1,
        "conv1.Stem_ReLU": 2,
        "conv1.Stem_MaxPool": 3,
        "conv1": 4,
    }

    if layer in stem_order:
        return (1, stem_order[layer])

    # conv2.block0, conv3.block2, etc.
    m = re.fullmatch(
        r"conv(\d+)\.block(\d+)",
        layer
    )

    if m:
        stage = int(m.group(1))
        block = int(m.group(2))

        return (stage, block)

    # Overall stage output should come after stage blocks.
    m = re.fullmatch(
        r"conv(\d+)",
        layer
    )

    if m:
        stage = int(m.group(1))
        return (stage, 1000)

    # Optional later layers if present.
    tail_order = {
        "lin1": (100, 0),
        "lin2": (101, 0),
        "classifier": (102, 0),
    }

    if layer in tail_order:
        return tail_order[layer]

    return (999, layer)


# ============================================================
# COLLECT DATA
# ============================================================

def collect_results(
    root: Path,
    target_epoch: int,
    checkpoint_name: str,
    network_name: str,
    log_pattern: str,
):
    """
    Search the complete experiment tree.

    Returns:

        {
            "conv1.Stem_Conv": [
                {
                    "run": "run1",
                    "train": ...,
                    "val": ...,
                    "experiment_dir": ...,
                    "log_file": ...
                },
                ...
            ],
            ...
        }
    """

    experiment_dirs = discover_experiment_dirs(
        root,
        checkpoint_name
    )

    print(
        f"Found {len(experiment_dirs)} experiment(s) "
        f"containing {checkpoint_name}."
    )

    results = defaultdict(list)

    for experiment_dir in experiment_dirs:

        layer = extract_network_key(
            experiment_dir,
            network_name
        )

        run = extract_run_name(
            experiment_dir,
            root
        )

        log_file, train_acc, val_acc = find_log_for_epoch(
            experiment_dir,
            target_epoch,
            log_pattern
        )

        if log_file is None:

            print(
                f"WARNING: checkpoint exists but no epoch "
                f"{target_epoch} log data found:"
            )

            print(f"    {experiment_dir}")

            continue

        results[layer].append(
            {
                "run": run,
                "train": train_acc,
                "val": val_acc,
                "experiment_dir": experiment_dir,
                "log_file": log_file,
            }
        )

    return results


# ============================================================
# AGGREGATE RUNS
# ============================================================

def aggregate_results(
    results,
    metric="val"
):
    """
    Calculate mean and sample standard deviation across
    repeated runs for every network layer.
    """

    aggregate = {}

    for layer, records in results.items():

        values = [
            record[metric]
            for record in records
            if record[metric] is not None
        ]

        if not values:
            continue

        layer_mean = mean(values)

        layer_std = (
            stdev(values)
            if len(values) >= 2
            else 0.0
        )

        aggregate[layer] = {
            "mean": layer_mean,
            "std": layer_std,
            "n": len(values),
            "records": records,
        }

    return aggregate


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    aggregate,
    metric,
    epoch,
    expected_runs=None
):
    """
    Print aggregate accuracy statistics to stdout.
    """

    layers = sorted(
        aggregate,
        key=network_layer_sort_key
    )

    metric_name = (
        "Validation"
        if metric == "val"
        else "Training"
    )

    print()
    print(
        f"{metric_name} top-1 accuracy at epoch {epoch}"
    )

    print("=" * 95)

    print(
        f"{'Network key':<32}"
        f"{'N':>5}"
        f"{'Mean':>14}"
        f"{'Std Dev':>14}"
    )

    print("-" * 95)

    for layer in layers:

        data = aggregate[layer]

        warning = ""

        if (
            expected_runs is not None
            and data["n"] != expected_runs
        ):
            warning = (
                f"   <-- expected {expected_runs} runs"
            )

        print(
            f"{layer:<32}"
            f"{data['n']:>5}"
            f"{data['mean']:>13.4f}%"
            f"{data['std']:>13.4f}%"
            f"{warning}"
        )

    print("=" * 95)
    print()


# ============================================================
# CSV OUTPUT
# ============================================================

def save_csv(
    aggregate,
    output_path: Path,
    metric: str,
):
    """
    Save:

        network key
        N
        mean
        std dev
        individual run values
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_runs = sorted(
        {
            record["run"]
            for data in aggregate.values()
            for record in data["records"]
        },
        key=lambda x: (
            int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x)
            else 999999
        )
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "network_key",
                "n",
                "mean",
                "std_dev",
                *all_runs,
            ]
        )

        for layer in sorted(
            aggregate,
            key=network_layer_sort_key
        ):

            data = aggregate[layer]

            by_run = {
                record["run"]: record[metric]
                for record in data["records"]
                if record[metric] is not None
            }

            writer.writerow(
                [
                    layer,
                    data["n"],
                    f"{data['mean']:.6f}",
                    f"{data['std']:.6f}",
                    *[
                        (
                            f"{by_run[run]:.6f}"
                            if run in by_run
                            else ""
                        )
                        for run in all_runs
                    ]
                ]
            )


# ============================================================
# PLOTTING
# ============================================================

def plot_results(
    aggregate,
    output_path: Path,
    network_name: str,
    experiment_name: str,
    epoch: int,
    metric: str,
):
    """
    Plot average top-1 accuracy across network depth.

    Green bar:
        Mean accuracy across all repeated runs.

    Error bar:
        +/- 1 sample standard deviation.

    Each bar is labeled:
        mean +/- std

    The y-axis is tightened around the observed range so
    differences across network depth remain visually clear.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    layers = sorted(
        aggregate,
        key=network_layer_sort_key
    )

    means = [
        aggregate[layer]["mean"]
        for layer in layers
    ]

    stds = [
        aggregate[layer]["std"]
        for layer in layers
    ]

    x = list(range(len(layers)))

    fig, ax = plt.subplots(
        figsize=(18, 8)
    )

    # ========================================================
    # Mean bars with +/- standard deviation
    # ========================================================

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color="green",
        alpha=0.75,
        edgecolor="black",
        error_kw={
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
    )

    # ========================================================
    # Annotate each bar with mean +/- std
    # ========================================================

    for bar, avg, std in zip(
        bars,
        means,
        stds
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            avg + std + 0.15,
            f"{avg:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    # ========================================================
    # X axis
    # ========================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        layers,
        rotation=60,
        ha="right"
    )

    # ========================================================
    # Titles / labels
    # ========================================================

    metric_name = (
        "Validation"
        if metric == "val"
        else "Training"
    )

    ax.set_title(
        f"Top-1 Accuracy Across Network Depth\n"
        f"{network_name} | {experiment_name} | Epoch {epoch}",
        fontsize=15,
    )

    ax.set_xlabel(
        "Network Layer",
        fontsize=12
    )

    ax.set_ylabel(
        f"{metric_name} Top-1 Accuracy (%)",
        fontsize=12
    )

    # ========================================================
    # Tight y-axis around mean +/- std
    # ========================================================

    lower = min(
        avg - std
        for avg, std in zip(means, stds)
    )

    upper = max(
        avg + std
        for avg, std in zip(means, stds)
    )

    value_range = upper - lower

    # Give enough space above/below results without forcing
    # the plot to start from zero.
    padding = max(
        value_range * 0.15,
        1.0
    )

    y_min = max(
        0,
        lower - padding
    )

    y_max = min(
        100,
        upper + padding
    )

    # Edge case: nearly identical results at ~100%.
    if y_max <= y_min:
        y_min = max(0, lower - 1.0)
        y_max = min(100, upper + 1.0)

    ax.set_ylim(
        y_min,
        y_max
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    # ========================================================
    # Save
    # ========================================================

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Recursively analyze downstream experiments, "
            "extract top-1 accuracy at a specified epoch, "
            "aggregate repeated runs by network layer, "
            "and generate mean +/- standard deviation plots."
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "path",
        help=(
            "Root directory containing repeated downstream "
            "runs."
        )
    )

    # --------------------------------------------------------
    # Epoch / checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help=(
            "Epoch whose accuracy should be extracted. "
            "Default: 100"
        )
    )

    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help=(
            "Checkpoint required for an experiment to be "
            "included. Default: classifier_net_epoch<epoch>"
        )
    )

    # --------------------------------------------------------
    # Experiment metadata
    # --------------------------------------------------------

    parser.add_argument(
        "--network-name",
        default="resnet34",
        help=(
            "Network name used for layer extraction and "
            "plot title. Default: resnet34"
        )
    )

    parser.add_argument(
        "--experiment-name",
        default="experiment",
        help=(
            "Experiment/condition name used in plot title."
        )
    )

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    parser.add_argument(
        "--metric",
        choices=["val", "train"],
        default="val",
        help=(
            "Accuracy to aggregate. Default: val"
        )
    )

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    parser.add_argument(
        "--log-pattern",
        default="*.txt",
        help=(
            "Glob pattern for logs beneath each experiment. "
            "Default: *.txt"
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default="./accuracyFINAL5",
        help=(
            "Directory for output PNG and CSV."
        )
    )

    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help=(
            "Expected number of repetitions per network "
            "layer. A warning is shown if the discovered "
            "count differs."
        )
    )

    args = parser.parse_args()

    # ========================================================
    # Validate input
    # ========================================================

    root = Path(args.path).resolve()

    if not root.exists():
        parser.error(
            f"Path does not exist: {root}"
        )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_name = (
        args.checkpoint_name
        if args.checkpoint_name
        else f"classifier_net_epoch{args.epoch}"
    )

    # ========================================================
    # Collect
    # ========================================================

    results = collect_results(
        root=root,
        target_epoch=args.epoch,
        checkpoint_name=checkpoint_name,
        network_name=args.network_name,
        log_pattern=args.log_pattern,
    )

    if not results:
        raise SystemExit(
            "ERROR: No usable downstream results found."
        )

    # ========================================================
    # Aggregate
    # ========================================================

    aggregate = aggregate_results(
        results,
        metric=args.metric
    )

    if not aggregate:
        raise SystemExit(
            f"ERROR: No {args.metric} accuracy values found "
            f"for epoch {args.epoch}."
        )

    # ========================================================
    # Print results
    # ========================================================

    print_results(
        aggregate,
        metric=args.metric,
        epoch=args.epoch,
        expected_runs=args.expected_runs
    )

    # ========================================================
    # Output filenames
    # ========================================================

    prefix = (
        f"top1_epoch{args.epoch}_"
        f"{args.network_name}_"
        f"{args.experiment_name}"
    )

    csv_path = (
        output_dir /
        f"{prefix}.csv"
    )

    plot_path = (
        output_dir /
        f"{prefix}.png"
    )

    # ========================================================
    # Save
    # ========================================================

    save_csv(
        aggregate,
        csv_path,
        args.metric
    )

    plot_results(
        aggregate,
        plot_path,
        args.network_name,
        args.experiment_name,
        args.epoch,
        args.metric,
    )

    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
    