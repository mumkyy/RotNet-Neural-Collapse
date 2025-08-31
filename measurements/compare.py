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
import numpy as np

def parse_kv_file(path: Path) -> OrderedDict:
    """Parse 'key: value' text file into an OrderedDict[str, float]."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    od = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Try colon separator first
            m = re.match(r'^([^:#]+)\s*:\s*([-+Ee0-9\.]+)\s*$', line)
            if not m:
                # Try space separator as fallback
                m = re.match(r'^([^:#]+)\s+([-+Ee0-9\.]+)\s*$', line)
            
            if not m:
                print(f"Warning: Could not parse line {line_num} in {path}: '{line}'")
                continue
                
            key = m.group(1).strip()
            try:
                val = float(m.group(2))
                od[key] = val
            except ValueError as e:
                print(f"Warning: Could not convert value to float on line {line_num} in {path}: '{m.group(2)}'")
                continue
    
    if not od:
        raise ValueError(f"No valid data found in file: {path}")
    
    return od

def unify_layer_order(*ods):
    """Determine a unified layer ordering across all OrderedDicts."""
    canonical = ["conv1", "conv2", "conv3", "conv4", "conv5", "classifier"]
    present = set()
    for od in ods:
        present.update(od.keys())
    
    # If any canonical layers are present, use canonical order as base
    if any(l in present for l in canonical):
        order = [l for l in canonical if l in present]
        seen = set(order)
        # Add any non-canonical layers found
        for od in ods:
            for k in od.keys():
                if k not in seen:
                    order.append(k)
                    seen.add(k)
        return order
    
    # Fallback: use appearance order across all dicts
    order, seen = [], set()
    for od in ods:
        for k in od.keys():
            if k not in seen:
                order.append(k)
                seen.add(k)
    return order

def extract_by_order(od: OrderedDict, order):
    """Extract values from OrderedDict in specified order, None for missing keys."""
    return [od.get(k, None) for k in order]

def plot_compare(order, yA, yB, nameA, nameB, title, ylabel, outpath):
    """Create comparison plot for two networks."""
    # Filter out layers where both networks have None values
    x_labels, xa, xb = [], [], []
    for k, a, b in zip(order, yA, yB):
        if a is None and b is None:
            continue
        x_labels.append(k)
        xa.append(a)
        xb.append(b)
    
    if not x_labels:
        print(f"Warning: No valid data to plot for {title}")
        return
    
    x = list(range(len(x_labels)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines with different styles for better distinction
    line1 = ax.plot(x, xa, marker='o', linewidth=2, markersize=6, label=nameA)
    line2 = ax.plot(x, xb, marker='s', linewidth=2, markersize=6, label=nameB, linestyle='--')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {outpath}")

def validate_files(args):
    """Validate that all required files exist."""
    files_to_check = [
        (args.netA_nc1, "NetA NC1 file"),
        (args.netA_mis, "NetA mismatch file"),
        (args.netB_nc1, "NetB NC1 file"),
        (args.netB_mis, "NetB mismatch file")
    ]
    
    missing_files = []
    for file_path, description in files_to_check:
        if not file_path.exists():
            missing_files.append(f"{description}: {file_path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    return True

def main():
    ap = argparse.ArgumentParser(description="Compare NC1 and NCC mismatch metrics between two networks")
    ap.add_argument("--netA-name", default="NetA", help="Display name for first network")
    ap.add_argument("--netA-nc1", required=True, type=Path, help="NC1 metrics file for NetA")
    ap.add_argument("--netA-mis", required=True, type=Path, help="NCC mismatch file for NetA")
    ap.add_argument("--netB-name", default="NetB", help="Display name for second network")
    ap.add_argument("--netB-nc1", required=True, type=Path, help="NC1 metrics file for NetB")
    ap.add_argument("--netB-mis", required=True, type=Path, help="NCC mismatch file for NetB")
    ap.add_argument("--outdir", type=Path, default=Path("results/compare_nc"), 
                    help="Output directory for plots")
    
    args = ap.parse_args()
    
    # Validate input files exist
    if not validate_files(args):
        return 1
    
    try:
        # Parse all metric files
        print("Loading data files...")
        nc1_a = parse_kv_file(args.netA_nc1)
        mis_a = parse_kv_file(args.netA_mis)
        nc1_b = parse_kv_file(args.netB_nc1)
        mis_b = parse_kv_file(args.netB_mis)
        
        print(f"Loaded {len(nc1_a)} NC1 values for {args.netA_name}")
        print(f"Loaded {len(mis_a)} mismatch values for {args.netA_name}")
        print(f"Loaded {len(nc1_b)} NC1 values for {args.netB_name}")
        print(f"Loaded {len(mis_b)} mismatch values for {args.netB_name}")
        
        # Determine unified layer ordering
        order = unify_layer_order(nc1_a, nc1_b, mis_a, mis_b)
        print(f"Layer order: {order}")
        
        # Extract values in unified order
        A_nc1 = extract_by_order(nc1_a, order)
        B_nc1 = extract_by_order(nc1_b, order)
        A_mis = extract_by_order(mis_a, order)
        B_mis = extract_by_order(mis_b, order)
        
        # Create output paths
        out_nc1 = args.outdir / "compare_nc1.pdf"
        out_mis = args.outdir / "compare_mismatch.pdf"
        
        # Generate comparison plots
        print("Generating plots...")
        plot_compare(order, A_nc1, B_nc1, args.netA_name, args.netB_name,
                     title="NC1 vs Layer — Comparison",
                     ylabel="NC1 (Tr(Sw)/Tr(Sb))",
                     outpath=out_nc1)
        
        plot_compare(order, A_mis, B_mis, args.netA_name, args.netB_name,
                     title="NCC Mismatch vs Layer — Comparison", 
                     ylabel="NCC Mismatch (fraction)",
                     outpath=out_mis)
        
        print("\nComparison complete!")
        print("Saved plots:")
        print(f"  - {out_nc1}")
        print(f"  - {out_mis}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())