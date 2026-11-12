#!/usr/bin/env python3
"""
Layerwise collapsed downstream accuracy scatter plot.

Reads CSVs produced by updatedAccs.py (network_key, mean, ...).

Usage:
    python downstream_plot.py \
        --series path/to/a.csv RotNet CIFAR10 \
        --series path/to/b.csv Jigsaw Imagenette \
        --output collapsed_downstream.pdf
"""

import argparse
import csv

import matplotlib.pyplot as plt


def read_accs(path):
    with open(path, "r", encoding="utf-8") as f:
        return {row["network_key"]: float(row["mean"]) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", nargs=3, action="append", required=True,
                        metavar=("CSV", "TASK", "DATASET"),
                        help="Accuracy CSV, pretext task, and dataset. Repeat per series.")
    parser.add_argument("--output", default="collapsed_downstream_accuracy.pdf")
    args = parser.parse_args()

    series = [(read_accs(p), task, dataset) for p, task, dataset in args.series]

    # Layer order follows the CSVs (already depth-sorted by updatedAccs.py).
    layers = []
    for accs, _, _ in series:
        layers += [k for k in accs if k not in layers]
    x_of = {layer: i for i, layer in enumerate(layers)}

    fig, ax = plt.subplots(figsize=(12, 6))
    for accs, task, dataset in series:
        best = max(accs, key=lambda k: accs[k])
        color = ax.plot([x_of[k] for k in accs], list(accs.values()), marker="o", label=f"{task} ({dataset})")[0].get_color()
        ax.scatter(x_of[best], accs[best], s=250, facecolors="none", edgecolors=color, linewidths=2)
    ax.scatter([], [], s=250, facecolors="none", edgecolors="gray", linewidths=2, label="Best layer")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=55, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Collapsed Downstream Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
