#!/usr/bin/env python3
"""
parse_rotnet_log.py

Extract <epoch, prec1> from a RotNet-Neural-Collapse training log and plot accuracy vs. epoch.

$ python accuracy.py path/to/LOG_INFO.txt
"""
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_log(path):
    """Return a DataFrame with columns epoch and prec1."""
    ep_re   = re.compile(r"Training epoch \[\s*(\d+)")
    res_re  = re.compile(r"Results:.*'prec1':\s*([\d.]+)")

    current_epoch = None
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ep_match = ep_re.search(line)
            if ep_match:                       # found a new epoch header
                current_epoch = int(ep_match.group(1))
                continue

            res_match = res_re.search(line)
            if res_match and current_epoch is not None:
                prec1 = float(res_match.group(1))
                rows.append((current_epoch, prec1))
                current_epoch = None           # reset until next epoch

    if not rows:
        raise ValueError("No epoch/prec1 pairs found – check the regexes.")

    df = pd.DataFrame(rows, columns=["epoch", "prec1"]).sort_values("epoch")
    return df


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python parse_rotnet_log.py <logfile>")

    log_path = Path(sys.argv[1])
    df = parse_log(log_path)
    print(df)
    
    visuals = Path("visuals")
    visuals.mkdir(exist_ok=True)
    out_png = visuals / f"{log_path.stem}_accuracy.png"

    df.plot(x="epoch", y="prec1", legend=False, marker="o")
    plt.title(log_path.name)
    plt.ylabel("top-1 accuracy (%)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
