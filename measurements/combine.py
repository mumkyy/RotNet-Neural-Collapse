#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle
import sys
import matplotlib.pyplot as plt
from measurements import Measurements  # regular import, as requested

def parse_args():
    p = argparse.ArgumentParser("Compare two RotNet measurement runs")
    p.add_argument('--runA', required=True,
                   help='Path to run A results dir OR its metrics.pkl (e.g., results/.../bsXXX_epochsY-Z/ or .../metrics.pkl)')
    p.add_argument('--runB', required=True,
                   help='Path to run B results dir OR its metrics.pkl')
    p.add_argument('--out',  required=True,
                   help='Directory to write combined plots (e.g., results/Compare_Collapse_vs_NotCollapsed)')
    p.add_argument('--label-a', default='Collapsed', help='Legend label for run A')
    p.add_argument('--label-b', default='Not Collapsed', help='Legend label for run B')
    return p.parse_args()

def resolve_pkl(pathlike: str) -> Path:
    p = Path(pathlike)
    if p.is_file():
        return p
    direct = p / 'metrics.pkl'
    if direct.exists():
        return direct
    candidates = list(p.rglob('metrics.pkl'))
    if not candidates:
        sys.exit(f"metrics.pkl not found under: {p}")
    if len(candidates) > 1:
        print(f"[info] Multiple metrics.pkl found under {p}, using: {candidates[0]}")
    return candidates[0]

def main():
    args = parse_args()

    out_dir = Path(args.out) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    pklA = resolve_pkl(args.runA)
    pklB = resolve_pkl(args.runB)

    with open(pklA, 'rb') as f:
        dataA = pickle.load(f)
    with open(pklB, 'rb') as f:
        dataB = pickle.load(f)

    epochsA = list(map(int, dataA['epochs']))
    epochsB = list(map(int, dataB['epochs']))
    mA: Measurements = dataA['metrics']
    mB: Measurements = dataB['metrics']

    # Align to common epochs
    common = sorted(set(epochsA).intersection(epochsB))
    if not common:
        sys.exit("No overlapping epochs between runs.")
    idxA = [epochsA.index(e) for e in common]
    idxB = [epochsB.index(e) for e in common]

    def sel(vals, idx):
        return [vals[i] for i in idx]

    def plot_metric(name, yA, yB, ylabel=None, logy=False, filename=None):
        plt.figure(figsize=(8,6))
        if logy:
            plt.semilogy(common, yA, 'bx-', label=args.label_a)
            plt.semilogy(common, yB, 'ro-', label=args.label_b)
        else:
            plt.plot(common, yA, 'bx-', label=args.label_a)
            plt.plot(common, yB, 'ro-', label=args.label_b)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel or name)
        plt.title(f"{ylabel or name} vs Epoch")
        plt.legend(frameon=False)
        plt.grid(True, which='both' if logy else 'major', ls='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir / (filename or f"{name}.pdf"))
        plt.close()

    # 1) Training loss (linear)
    plot_metric('loss',
                sel(mA.loss, idxA),
                sel(mB.loss, idxB),
                ylabel='Training Loss',
                filename='training_loss.pdf')

    # 2) Training accuracy (%)
    plot_metric('accuracy',
                [v*100 for v in sel(mA.accuracy, idxA)],
                [v*100 for v in sel(mB.accuracy, idxB)],
                ylabel='Training Accuracy (%)',
                filename='training_accuracy.pdf')

    # 3) tr(Sw)/tr(Sb) (log-scale)
    plot_metric('trSwtrSb',
                sel(mA.trSwtrSb, idxA),
                sel(mB.trSwtrSb, idxB),
                ylabel='NC1 tr(Sw)/tr(Sb)',
                logy=True,
                filename='trace_ratio_log.pdf')

    # 4) NC-3 (W_M_dist) (linear)
    plot_metric('W_M_dist',
                sel(mA.W_M_dist, idxA),
                sel(mB.W_M_dist, idxB),
                ylabel='NC3 (‖W/‖W‖ − M_c/‖M_c‖‖²)',
                filename='nc3.pdf')

    print("✓  Done – results in", out_dir)

if __name__ == "__main__":
    main()
