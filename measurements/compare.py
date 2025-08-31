#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_nc1_mismatch.py
-----------------------
Read NC1 and NCC mismatch metric dumps for TWO networks and produce two comparison figures:
1) NC1 vs layer (both networks on the same axes)
2) NCC mismatch vs layer (both networks on the same axes)

Accepted file format:
- Plain text with one "layer: value" per line, e.g.:
    conv1: 281.885162
    conv2: 528.435486
    ...
    classifier: 0.033413

Usage:
  python compare_nc1_mismatch.py \
      --netA-name "Collapsed" \
      --netA-nc1 path/to/netA_nc1.txt \
      --netA-mis path/to/netA_mismatch.txt \
      --netB-name "NotCollapsed" \
      --netB-nc1 path/to/netB_nc1.txt \
      --netB-mis path/to/netB_mismatch.txt \
      --outdir out/plots
"""

import argparse
from pathlib import Path
from collections import OrderedDict
import re
import matplotlib.pyplot as plt

def parse_kv_file(path: Path) -> OrderedDict:
    """Parse 'key: value' text file into an OrderedDict[str, float]."""
    od = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([^:#]+)\s*:\s*([-+Ee0-9\.]+)\s*$', line)
            if not m:
                m = re.match(r'^([^:#]+)\s+([-+Ee0-9\.]+)\s*$', line)
            if not m:
                continue
            key = m.group(1).strip()
            val = float(m.group(2))
            od[key] = val
    return od

def unify_layer_order(*ods):
    canonical = ["conv1","conv2","conv3","conv4","conv5","classifier"]
    present = set()
    for od in ods:
        present.update(od.keys())
    if any(l in present for l in canonical):
        order = [l for l in canonical if l in present]
        seen = set(order)
        for od in ods:
            for k in od.keys():
                if k not in seen:
                    order.append(k); seen.add(k)
        return order
    # fallback: appearance order
    order, seen = [], set()
    for od in ods:
        for k in od.keys():
            if k not in seen:
                order.append(k); seen.add(k)
    return order

def extract_by_order(od: OrderedDict, order):
    return [od.get(k, None) for k in order]

def plot_compare(order, yA, yB, nameA, nameB, title, ylabel, outpath):
    x_labels, xa, xb = [], [], []
    for k, a, b in zip(order, yA, yB):
        if a is None and b is None:
            continue
        x_labels.append(k)
        xa.append(a)
        xb.append(b)
    x = list(range(len(x_labels)))
    plt.figure()
    plt.plot(x, xa, marker='o', label=nameA)
    plt.plot(x, xb, marker='s', label=nameB)
    plt.xticks(x, x_labels, rotation=20)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netA-name", default="NetA")
    ap.add_argument("--netA-nc1", required=True, type=Path)
    ap.add_argument("--netA-mis", required=True, type=Path, help="NCC mismatch file for NetA")
    ap.add_argument("--netB-name", default="NetB")
    ap.add_argument("--netB-nc1", required=True, type=Path)
    ap.add_argument("--netB-mis", required=True, type=Path, help="NCC mismatch file for NetB")
    ap.add_argument("--outdir", type=Path, default=Path("results/compare_nc"))
    args = ap.parse_args()

    nc1_a = parse_kv_file(args.netA_nc1)
    mis_a = parse_kv_file(args.netA_mis)
    nc1_b = parse_kv_file(args.netB_nc1)
    mis_b = parse_kv_file(args.netB_mis)

    order = unify_layer_order(nc1_a, nc1_b, mis_a, mis_b)

    A_nc1 = extract_by_order(nc1_a, order)
    B_nc1 = extract_by_order(nc1_b, order)
    A_mis = extract_by_order(mis_a, order)
    B_mis = extract_by_order(mis_b, order)

    out_nc1 = args.outdir / "compare_nc1.pdf"
    out_mis = args.outdir / "compare_mismatch.pdf"
    plot_compare(order, A_nc1, B_nc1, args.netA_name, args.netB_name,
                 title="NC1 vs Layer — Comparison",
                 ylabel="NC1 (Tr(Sw)/Tr(Sb))",
                 outpath=out_nc1)
    plot_compare(order, A_mis, B_mis, args.netA_name, args.netB_name,
                 title="NCC mismatch vs Layer — Comparison",
                 ylabel="NCC mismatch (fraction)",
                 outpath=out_mis)

    print("Saved:")
    print("  ", out_nc1)
    print("  ", out_mis)

if __name__ == "__main__":
    main()
