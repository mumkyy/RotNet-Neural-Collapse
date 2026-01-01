#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle
import sys
import matplotlib.pyplot as plt

# REMOVED: from measurements.measurements import Measurements 
# (Not needed as we are working with standard python dicts now)

def parse_args():
    p = argparse.ArgumentParser("Compare two RotNet measurement runs")
    p.add_argument('--runA', required=True,
                   help='Path to run A results dir OR its metrics.pkl')
    p.add_argument('--runB', required=True,
                   help='Path to run B results dir OR its metrics.pkl')
    p.add_argument('--out',  required=True,
                   help='Directory to write combined plots')
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

def get_nc1_data(data_dict):
    """
    Helper to extract NC1. 
    measurementsFix.py saves 'nc1_by_layer'. 
    We automatically pick the LAST layer (deepest representation), 
    which is standard for Neural Collapse analysis.
    """
    if 'nc1_by_layer' in data_dict:
        layers_dict = data_dict['nc1_by_layer']
        # Get the last key (e.g., 'conv4' or 'layer4')
        last_layer = list(layers_dict.keys())[-1]
        print(f"[info] Using layer '{last_layer}' for NC1 comparison.")
        return layers_dict[last_layer]
    elif 'trSwtrSb' in data_dict:
        # Legacy support if using old pickles
        return data_dict['trSwtrSb']
    else:
        print("[warning] Could not find NC1 data (nc1_by_layer or trSwtrSb)")
        return [0] * len(data_dict.get('epochs', []))

def main():
    args = parse_args()

    out_dir = Path(args.out) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    pklA = resolve_pkl(args.runA)
    pklB = resolve_pkl(args.runB)

    print(f"Loading A: {pklA}")
    with open(pklA, 'rb') as f:
        dataA = pickle.load(f)
        
    print(f"Loading B: {pklB}")
    with open(pklB, 'rb') as f:
        dataB = pickle.load(f)

    # Handle structure mismatch: 
    # If 'payload' exists, unwrap it (legacy). If not, use dict directly (new).
    if 'payload' in dataA:
        print("[info] Detected legacy format for Run A (unwrapping payload)")
        # This converts the legacy object to a dict if possible, or keeps the object
        # But based on your error, you are likely using the dictionary format now.
        # We will assume dataA is the Dictionary from measurementsFix.py
        pass 
        
    # Extract Epochs
    epochsA = list(map(int, dataA['epochs']))
    epochsB = list(map(int, dataB['epochs']))

    # Align to common epochs
    common = sorted(set(epochsA).intersection(epochsB))
    if not common:
        sys.exit("No overlapping epochs between runs.")
    
    print(f"Comparing {len(common)} common epochs...")
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

    # --- EXTRACT METRICS SAFELY ---
    
    # 1) Training loss (linear)
    # New format key: 'loss', Legacy obj attr: .loss
    lossA = dataA['loss']
    lossB = dataB['loss']
    plot_metric('loss',
                sel(lossA, idxA),
                sel(lossB, idxB),
                ylabel='Training Loss',
                filename='training_loss.pdf')

    # 2) Training accuracy (%)
    # New format key: 'accuracy', Legacy obj attr: .accuracy
    accA = [v*100 for v in sel(dataA['accuracy'], idxA)]
    accB = [v*100 for v in sel(dataB['accuracy'], idxB)]
    plot_metric('accuracy',
                accA,
                accB,
                ylabel='Training Accuracy (%)',
                filename='training_accuracy.pdf')

    # 3) NC1 (log-scale)
    # New format key: 'nc1_by_layer' (dict), Legacy obj attr: .trSwtrSb
    nc1_A_full = get_nc1_data(dataA)
    nc1_B_full = get_nc1_data(dataB)
    
    plot_metric('nc1',
                sel(nc1_A_full, idxA),
                sel(nc1_B_full, idxB),
                ylabel='NC1',
                logy=True,
                filename='nc1.pdf')

    # 4) NC3 (linear)
    # New format key: 'nc3', Legacy obj attr: .W_M_dist
    # Note: measurementsFix.py calls it 'nc3', old calls it 'W_M_dist'
    nc3_A_full = dataA.get('nc3', dataA.get('W_M_dist', []))
    nc3_B_full = dataB.get('nc3', dataB.get('W_M_dist', []))
    
    plot_metric('nc3',
                sel(nc3_A_full, idxA),
                sel(nc3_B_full, idxB),
                ylabel='NC3',
                filename='nc3.pdf')

    # 5) Combined NC1 and NC3 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # NC1 plot (log scale)
    nc1_A = sel(nc1_A_full, idxA)
    nc1_B = sel(nc1_B_full, idxB)
    ax1.semilogy(common, nc1_A, 'bx-', label=args.label_a)
    ax1.semilogy(common, nc1_B, 'ro-', label=args.label_b)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NC1')
    ax1.set_title('NC1 vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.6)
    
    # NC3 plot (linear scale)
    nc3_A = sel(nc3_A_full, idxA)
    nc3_B = sel(nc3_B_full, idxB)
    ax2.plot(common, nc3_A, 'bx-', label=args.label_a)
    ax2.plot(common, nc3_B, 'ro-', label=args.label_b)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NC3')
    ax2.set_title('NC3 vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'nc1_nc3_comparison.pdf')
    plt.close()

    print("✓  Done – results in", out_dir)

if __name__ == "__main__":
    main()