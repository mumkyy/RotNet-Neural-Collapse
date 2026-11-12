#!/usr/bin/env python3
"""Plot layerwise collapsed downstream accuracy with standard deviation."""

import argparse
import csv

import matplotlib.pyplot as plt


def read_accs(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {
            row["network_key"]: (float(row["mean"]), float(row["std_dev"]))
            for row in csv.DictReader(f)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        nargs=3,
        action="append",
        required=True,
        metavar=("CSV", "TASK", "DATASET"),
        help="Accuracy CSV, pretext task, and dataset. Repeat per series.",
    )
    parser.add_argument("--output", default="collapsed_downstream_accuracy.pdf")
    args = parser.parse_args()

    series = [(read_accs(p), task, dataset) for p, task, dataset in args.series]

    # Layer order follows the CSVs.
    layers = []
    for accs, _, _ in series:
        layers += [k for k in accs if k not in layers]
    x_of = {layer: i for i, layer in enumerate(layers)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for accs, task, dataset in series:
        # Plot each series in the shared layer order.
        keys = [k for k in layers if k in accs]
        means = [accs[k][0] for k in keys]
        stds = [accs[k][1] for k in keys]

        plotted = ax.errorbar(
            [x_of[k] for k in keys],
            means,
            yerr=stds,
            fmt="-o",
            capsize=4,
            elinewidth=1.2,
            label=f"{task} ({dataset})",
        )
        color = plotted.lines[0].get_color()

        best = max(accs, key=lambda k: accs[k][0])
        ax.scatter(
            x_of[best],
            accs[best][0],
            s=250,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
            zorder=5,
        )

    ax.scatter(
        [], [],
        s=250,
        facecolors="none",
        edgecolors="gray",
        linewidths=2,
        label="Best layer",
    )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=55, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Collapsed Downstream Accuracy (Mean ± SD)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
