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
    p.add_argument('--out', required=True,
                   help='Directory to write combined plots')
    p.add_argument('--label-a', default='Collapsed', help='Legend label for run A')
    p.add_argument('--label-b', default='Not Collapsed', help='Legend label for run B')
    p.add_argument('--epoch', type=int, default=200,
                   help='Epoch to plot layerwise NC4 mismatch for (default: 200)')
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


def plot_nc4_layerwise_mismatch_at_epoch(out_dir: Path, dataA, dataB, label_a: str, label_b: str, epoch: int):
    """
    Plot ONLY: NC4 layerwise mismatch at a single epoch.

    Expected structure in metrics.pkl:
      data['epochs'] = [ ... ]
      data['nc4_layerwise'] = {
         layer_key: {'match':[...], 'mismatch':[...], 'ncc_acc':[...]}
      }
      and optionally:
      data['layer_keys'] = [layer ordering...]
    """
    if 'nc4_layerwise' not in dataA or 'nc4_layerwise' not in dataB:
        sys.exit("[error] nc4_layerwise not present in BOTH runs. Re-run measurements with layerwise NC4 enabled.")

    epochsA = list(map(int, dataA.get('epochs', [])))
    epochsB = list(map(int, dataB.get('epochs', [])))
    if epoch not in epochsA or epoch not in epochsB:
        sys.exit(f"[error] Requested epoch {epoch} not found in both runs. "
                 f"RunA epochs range: {min(epochsA) if epochsA else 'NA'}..{max(epochsA) if epochsA else 'NA'}, "
                 f"RunB epochs range: {min(epochsB) if epochsB else 'NA'}..{max(epochsB) if epochsB else 'NA'}")

    iA = epochsA.index(epoch)
    iB = epochsB.index(epoch)

    lwA = dataA['nc4_layerwise']
    lwB = dataB['nc4_layerwise']

    # ordering: prefer layer_keys from runA if available
    layer_order = None
    if isinstance(dataA.get('layer_keys', None), list) and dataA['layer_keys']:
        layer_order = [k for k in dataA['layer_keys'] if k in lwA and k in lwB]
    if not layer_order:
        layer_order = sorted(set(lwA.keys()).intersection(lwB.keys()))

    if not layer_order:
        sys.exit("[error] No overlapping layer keys between runs in nc4_layerwise.")

    # extract mismatch values at epoch
    valsA = []
    valsB = []
    kept_layers = []
    for k in layer_order:
        if not isinstance(lwA.get(k, None), dict) or not isinstance(lwB.get(k, None), dict):
            continue
        if 'mismatch' not in lwA[k] or 'mismatch' not in lwB[k]:
            continue
        seriesA = lwA[k]['mismatch']
        seriesB = lwB[k]['mismatch']
        if iA >= len(seriesA) or iB >= len(seriesB):
            continue
        kept_layers.append(k)
        valsA.append(seriesA[iA])
        valsB.append(seriesB[iB])

    if not kept_layers:
        sys.exit("[error] Found no layers with usable 'mismatch' series in both runs.")

    # plot
    plt.figure(figsize=(10, 5))
    x = list(range(len(kept_layers)))

    plt.plot(x, valsA, 'bx-', label=f"{label_a}")
    plt.plot(x, valsB, 'ro-', label=f"{label_b}")

    plt.xticks(x, kept_layers, rotation=45, ha='right')
    plt.xlabel('Layer')
    plt.ylabel('NC4 mismatch (1 - match)')
    plt.title(f'NC4 mismatch per-layer at epoch {epoch}')
    plt.legend(frameon=False)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / f'nc4_layerwise_mismatch_epoch{epoch}.pdf')
    plt.close()

    # also print a quick text summary for sanity
    print(f"[info] Saved: {out_dir / f'nc4_layerwise_mismatch_epoch{epoch}.pdf'}")
    for k, a, b in zip(kept_layers, valsA, valsB):
        print(f"  {k:>24s} | {label_a}: {a:.6f}  {label_b}: {b:.6f}")


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

    plot_nc4_layerwise_mismatch_at_epoch(
        out_dir=out_dir,
        dataA=dataA,
        dataB=dataB,
        label_a=args.label_a,
        label_b=args.label_b,
        epoch=args.epoch,
    )

    print("✓ Done – results in", out_dir)


if __name__ == "__main__":
    main()
