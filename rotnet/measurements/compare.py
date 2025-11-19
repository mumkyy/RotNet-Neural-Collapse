#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

def parse_kv_file(path):
    """Parse 'key: value' text file."""
    data = OrderedDict()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or 'metrics' in line.lower() or 'mismatch' in line.lower():
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    data[key.strip()] = float(value.strip())
                except ValueError:
                    continue
    return data

def plot_compare(layers, data_a, data_b, name_a, name_b, title, ylabel, outpath):
    """Create comparison plot."""
    plt.figure()
    x = list(range(len(layers)))
    plt.plot(x, data_a, 'o-', label=name_a)
    plt.plot(x, data_b, 's--', label=name_b)
    plt.xticks(x, layers, rotation=45)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netA-name", default="NetA")
    parser.add_argument("--netA-nc1", required=True, type=Path)
    parser.add_argument("--netA-mis", required=True, type=Path)
    parser.add_argument("--netB-name", default="NetB")
    parser.add_argument("--netB-nc1", required=True, type=Path)
    parser.add_argument("--netB-mis", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("results/compare_nc"))
    args = parser.parse_args()

    # Load data
    nc1_a = parse_kv_file(args.netA_nc1)
    mis_a = parse_kv_file(args.netA_mis)
    nc1_b = parse_kv_file(args.netB_nc1)
    mis_b = parse_kv_file(args.netB_mis)

    # Get common layers
    layers = list(nc1_a.keys())
    
    # Extract values in order
    nc1_vals_a = [nc1_a.get(k, 0) for k in layers]
    nc1_vals_b = [nc1_b.get(k, 0) for k in layers]
    mis_vals_a = [mis_a.get(k, 0) for k in layers]
    mis_vals_b = [mis_b.get(k, 0) for k in layers]

    # Make plots
    plot_compare(layers, nc1_vals_a, nc1_vals_b, args.netA_name, args.netB_name,
                "NC1 vs Layer", "NC1", args.outdir / "compare_nc1.pdf")
    
    plot_compare(layers, mis_vals_a, mis_vals_b, args.netA_name, args.netB_name,
                "NCC Mismatch vs Layer", "NCC Mismatch", args.outdir / "compare_mismatch.pdf")

    print(f"Saved plots to {args.outdir}")

if __name__ == "__main__":
    main()