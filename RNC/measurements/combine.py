#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle
import sys
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("Compare two measurement runs (dict pickles)")
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


def sel(vals, idx):
    return [vals[i] for i in idx]


def plot_metric(out_dir: Path, common_epochs, name, yA, yB, label_a, label_b,
                ylabel=None, logy=False, filename=None):
    plt.figure(figsize=(8, 6))
    if logy:
        plt.semilogy(common_epochs, yA, 'bx-', label=label_a)
        plt.semilogy(common_epochs, yB, 'ro-', label=label_b)
    else:
        plt.plot(common_epochs, yA, 'bx-', label=label_a)
        plt.plot(common_epochs, yB, 'ro-', label=label_b)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel or name)
    plt.title(f"{ylabel or name} vs Epoch")
    plt.legend(frameon=False)
    plt.grid(True, which='both' if logy else 'major', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / (filename or f"{name}.pdf"))
    plt.close()


def get_nc1_deepest_series(data_dict):
    """
    measurementsFix.py stores:
      - nc1_by_layer: dict[layer_key] -> List[float]
      - layer_keys:  List[str]  (ordered exactly as requested in CLI)
    We choose the "deepest" as layer_keys[-1] (not dict key order).
    """
    if 'nc1_by_layer' not in data_dict:
        print("[warning] Could not find nc1_by_layer.")
        return [0] * len(data_dict.get('epochs', []))

    layers_dict = data_dict['nc1_by_layer']

    # Preferred: use explicit ordering
    layer_keys = data_dict.get('layer_keys', None)
    if isinstance(layer_keys, list) and len(layer_keys) > 0:
        deepest = layer_keys[-1]
        if deepest in layers_dict:
            print(f"[info] Using deepest layer '{deepest}' for NC1 comparison (from layer_keys).")
            return layers_dict[deepest]
        else:
            print(f"[warning] layer_keys[-1] = '{deepest}' not found in nc1_by_layer keys.")

    # Fallback: try to pick something sensible
    keys = list(layers_dict.keys())
    keys = [k for k in keys if k != 'classifier']
    if not keys:
        print("[warning] nc1_by_layer has no usable keys.")
        return [0] * len(data_dict.get('epochs', []))

    # deterministic fallback sort
    keys_sorted = sorted(keys)
    deepest = keys_sorted[-1]
    print(f"[info] Using fallback deepest layer '{deepest}' (sorted key fallback).")
    return layers_dict[deepest]


def get_nc1_by_layer_at(data_dict, epoch_idx):
    if 'nc1_by_layer' not in data_dict:
        print("[warning] nc1_by_layer not found; skipping per-layer comparison.")
        return {}
    layers_dict = data_dict['nc1_by_layer']
    out = {}
    for k, series in layers_dict.items():
        if epoch_idx < len(series):
            out[k] = series[epoch_idx]
    return out


def plot_nc1_by_layer_compare(out_dir, layers, vals_a, vals_b, label_a, label_b, epoch):
    plt.figure(figsize=(10, 5))
    x = list(range(len(layers)))

    plt.semilogy(x, vals_a, 'bx-', label=label_a)
    plt.semilogy(x, vals_b, 'ro-', label=label_b)
    plt.xticks(x, layers, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('NC1')
    plt.title(f'NC1 across layers at epoch {epoch}')
    plt.legend(frameon=False)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / 'nc1_layers_final_compare.pdf')
    plt.close()


def plot_nc4_triple(out_dir: Path, common_epochs, dataA, dataB, idxA, idxB, label_a, label_b):
    """
    If nc4 metrics exist in both runs, plot:
      nc4_match, nc4_mismatch, ncc_acc
    """
    needed = ['nc4_match', 'nc4_mismatch', 'ncc_acc']
    if not all(k in dataA for k in needed) or not all(k in dataB for k in needed):
        print("[info] NC4 metrics not present in both runs; skipping NC4 plots.")
        return

    # Individual plots
    plot_metric(out_dir, common_epochs, 'nc4_match',
                sel(dataA['nc4_match'], idxA),
                sel(dataB['nc4_match'], idxB),
                label_a, label_b,
                ylabel='NC4 match (net == NCC)',
                filename='nc4_match.pdf')

    plot_metric(out_dir, common_epochs, 'nc4_mismatch',
                sel(dataA['nc4_mismatch'], idxA),
                sel(dataB['nc4_mismatch'], idxB),
                label_a, label_b,
                ylabel='NC4 mismatch (1 - match)',
                filename='nc4_mismatch.pdf')

    plot_metric(out_dir, common_epochs, 'ncc_acc',
                sel(dataA['ncc_acc'], idxA),
                sel(dataB['ncc_acc'], idxB),
                label_a, label_b,
                ylabel='NCC accuracy',
                filename='ncc_acc.pdf')

    # Combined single figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(common_epochs, sel(dataA['nc4_match'], idxA), 'bx-', label=label_a)
    axes[0].plot(common_epochs, sel(dataB['nc4_match'], idxB), 'ro-', label=label_b)
    axes[0].set_title('NC4 match')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.6)

    axes[1].plot(common_epochs, sel(dataA['nc4_mismatch'], idxA), 'bx-', label=label_a)
    axes[1].plot(common_epochs, sel(dataB['nc4_mismatch'], idxB), 'ro-', label=label_b)
    axes[1].set_title('NC4 mismatch')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.6)

    axes[2].plot(common_epochs, sel(dataA['ncc_acc'], idxA), 'bx-', label=label_a)
    axes[2].plot(common_epochs, sel(dataB['ncc_acc'], idxB), 'ro-', label=label_b)
    axes[2].set_title('NCC accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.6)

    for ax in axes:
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_dir / 'nc4_comparison.pdf')
    plt.close()


# ==================== NEW: NC4 layerwise plotting ====================

def plot_nc4_layerwise(out_dir: Path, common_epochs, dataA, dataB, idxA, idxB, label_a, label_b):
    """
    If nc4_layerwise exists in both runs, plot *one* comparison figure:
      - per-layer match over epochs
      - per-layer ncc_acc over epochs

    Expected structure in metrics.pkl:
      data['nc4_layerwise'] = {
         layer_key: {'match':[...], 'mismatch':[...], 'ncc_acc':[...]}
      }
      and optionally data['layer_keys'] for ordering.
    """
    if 'nc4_layerwise' not in dataA or 'nc4_layerwise' not in dataB:
        print("[info] NC4 layerwise not present in both runs; skipping layerwise NC4 plot.")
        return

    lwA = dataA['nc4_layerwise']
    lwB = dataB['nc4_layerwise']

    # Decide layer order
    layer_order = None
    if isinstance(dataA.get('layer_keys', None), list) and dataA['layer_keys']:
        layer_order = [k for k in dataA['layer_keys'] if k in lwA and k in lwB]
    if not layer_order:
        layer_order = sorted(set(lwA.keys()).intersection(lwB.keys()))

    if not layer_order:
        print("[warning] No overlapping layers found in nc4_layerwise; skipping.")
        return

    # Validate expected subkeys
    for k in layer_order:
        if not isinstance(lwA.get(k, None), dict) or not isinstance(lwB.get(k, None), dict):
            print(f"[warning] Bad nc4_layerwise format for layer {k}; skipping.")
            return
        for sub in ('match', 'ncc_acc'):
            if sub not in lwA[k] or sub not in lwB[k]:
                print(f"[warning] Missing '{sub}' in nc4_layerwise for layer {k}; skipping.")
                return

    # Plot: match curves per layer
    plt.figure(figsize=(10, 6))
    for k in layer_order:
        yA = sel(lwA[k]['match'], idxA)
        yB = sel(lwB[k]['match'], idxB)
        plt.plot(common_epochs, yA, marker='o', linestyle='-', label=f"{label_a} {k}")
        plt.plot(common_epochs, yB, marker='o', linestyle='--', label=f"{label_b} {k}")
    plt.xlabel("Epoch")
    plt.ylabel("NC4 layerwise match (net == NCC@layer)")
    plt.title("NC4 layerwise match vs Epoch (per layer)")
    plt.legend(frameon=False, ncol=2, fontsize=9)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "nc4_layerwise_match.pdf")
    plt.close()

    # Plot: ncc_acc curves per layer
    plt.figure(figsize=(10, 6))
    for k in layer_order:
        yA = sel(lwA[k]['ncc_acc'], idxA)
        yB = sel(lwB[k]['ncc_acc'], idxB)
        plt.plot(common_epochs, yA, marker='o', linestyle='-', label=f"{label_a} {k}")
        plt.plot(common_epochs, yB, marker='o', linestyle='--', label=f"{label_b} {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Layerwise NCC accuracy")
    plt.title("Layerwise NCC accuracy vs Epoch (per layer)")
    plt.legend(frameon=False, ncol=2, fontsize=9)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "nc4_layerwise_ncc_acc.pdf")
    plt.close()

    # Final-epoch summary (match + ncc_acc across layers)
    last_epoch = common_epochs[-1]
    plt.figure(figsize=(10, 5))
    x = list(range(len(layer_order)))

    matchA = [lwA[k]['match'][idxA[-1]] for k in layer_order]
    matchB = [lwB[k]['match'][idxB[-1]] for k in layer_order]
    accA = [lwA[k]['ncc_acc'][idxA[-1]] for k in layer_order]
    accB = [lwB[k]['ncc_acc'][idxB[-1]] for k in layer_order]

    plt.plot(x, matchA, 'bx-', label=f"{label_a} match")
    plt.plot(x, matchB, 'ro-', label=f"{label_b} match")
    plt.plot(x, accA, 'b^-', label=f"{label_a} ncc_acc")
    plt.plot(x, accB, 'r^-', label=f"{label_b} ncc_acc")

    plt.xticks(x, layer_order, rotation=45, ha='right')
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title(f"NC4 layerwise summary at epoch {last_epoch}")
    plt.legend(frameon=False, ncol=2, fontsize=9)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "nc4_layerwise_final_compare.pdf")
    plt.close()


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

    # Extract epochs
    epochsA = list(map(int, dataA['epochs']))
    epochsB = list(map(int, dataB['epochs']))

    # Align to common epochs
    common = sorted(set(epochsA).intersection(epochsB))
    if not common:
        sys.exit("No overlapping epochs between runs.")

    print(f"Comparing {len(common)} common epochs...")
    idxA = [epochsA.index(e) for e in common]
    idxB = [epochsB.index(e) for e in common]

    # 1) Loss
    plot_metric(out_dir, common, 'loss',
                sel(dataA['loss'], idxA),
                sel(dataB['loss'], idxB),
                args.label_a, args.label_b,
                ylabel='Loss',
                filename='training_loss.pdf')

    # 2) Accuracy (%)
    accA = [v * 100 for v in sel(dataA['accuracy'], idxA)]
    accB = [v * 100 for v in sel(dataB['accuracy'], idxB)]
    plot_metric(out_dir, common, 'accuracy',
                accA, accB,
                args.label_a, args.label_b,
                ylabel='Accuracy (%)',
                filename='training_accuracy.pdf')

    # 3) NC1 (deepest layer, log scale)
    nc1_A_full = get_nc1_deepest_series(dataA)
    nc1_B_full = get_nc1_deepest_series(dataB)
    plot_metric(out_dir, common, 'nc1',
                sel(nc1_A_full, idxA),
                sel(nc1_B_full, idxB),
                args.label_a, args.label_b,
                ylabel='NC1 (deepest layer)',
                logy=True,
                filename='nc1.pdf')

    # 4) NC3
    nc3_A_full = dataA.get('nc3', dataA.get('W_M_dist', []))
    nc3_B_full = dataB.get('nc3', dataB.get('W_M_dist', []))
    if not nc3_A_full or not nc3_B_full:
        print("[warning] NC3 missing in one run; skipping NC3 plots.")
    else:
        plot_metric(out_dir, common, 'nc3',
                    sel(nc3_A_full, idxA),
                    sel(nc3_B_full, idxB),
                    args.label_a, args.label_b,
                    ylabel='NC3',
                    filename='nc3.pdf')

    # 5) Combined NC1+NC3 figure (if NC3 exists)
    if nc3_A_full and nc3_B_full:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.semilogy(common, sel(nc1_A_full, idxA), 'bx-', label=args.label_a)
        ax1.semilogy(common, sel(nc1_B_full, idxB), 'ro-', label=args.label_b)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('NC1')
        ax1.set_title('NC1 vs Epoch')
        ax1.legend(frameon=False)
        ax1.grid(True, alpha=0.6)

        ax2.plot(common, sel(nc3_A_full, idxA), 'bx-', label=args.label_a)
        ax2.plot(common, sel(nc3_B_full, idxB), 'ro-', label=args.label_b)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('NC3')
        ax2.set_title('NC3 vs Epoch')
        ax2.legend(frameon=False)
        ax2.grid(True, alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_dir / 'nc1_nc3_comparison.pdf')
        plt.close()

    # 6) NC1 per-layer at final common epoch
    last_common_epoch = common[-1]
    nc1_layers_A = get_nc1_by_layer_at(dataA, idxA[-1])
    nc1_layers_B = get_nc1_by_layer_at(dataB, idxB[-1])
    if nc1_layers_A and nc1_layers_B:
        # preserve layer ordering if possible
        layer_order = dataA.get("layer_keys", None)
        if isinstance(layer_order, list) and layer_order:
            layers = [k for k in layer_order if k in nc1_layers_A and k in nc1_layers_B]
        else:
            layers = [k for k in nc1_layers_A.keys() if k in nc1_layers_B]

        if layers:
            vals_a = [nc1_layers_A[k] for k in layers]
            vals_b = [nc1_layers_B[k] for k in layers]
            plot_nc1_by_layer_compare(out_dir, layers, vals_a, vals_b,
                                      args.label_a, args.label_b, last_common_epoch)
        else:
            print("[warning] No overlapping layer keys for nc1_by_layer; skipping per-layer plot.")

    # 7) NC4 plots (if present)
    plot_nc4_triple(out_dir, common, dataA, dataB, idxA, idxB, args.label_a, args.label_b)

    # 8) NEW: NC4 layerwise plots (if present)
    plot_nc4_layerwise(out_dir, common, dataA, dataB, idxA, idxB, args.label_a, args.label_b)

    print("✓ Done – results in", out_dir)


if __name__ == "__main__":
    main()
