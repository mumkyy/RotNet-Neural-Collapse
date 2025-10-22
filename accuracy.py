#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from statistics import mean, stdev

def parse_log(path):
    train_acc  = []  # list of (epoch, prec1)
    val_acc    = []  # list of (epoch, prec1)
    current_epoch = None

    # regexes
    epoch_re   = re.compile(r"Training epoch\s*\[\s*(\d+)\s*/\s*\d+\]")
    train_re   = re.compile(r"Training stats: \{[^}]*'prec1':\s*([\d.]+)")
    eval_re    = re.compile(r"Evaluation stats: \{[^}]*'prec1':\s*([\d.]+)")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # detect epoch number
            m = epoch_re.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue

            # detect train stats
            m = train_re.search(line)
            if m and current_epoch is not None:
                train_acc.append((current_epoch, float(m.group(1))))
                continue

            # detect eval stats
            m = eval_re.search(line)
            if m and current_epoch is not None:
                val_acc.append((current_epoch, float(m.group(1))))
                continue

    return train_acc, val_acc

def best_entry(acc_list):
    if not acc_list:
        return None
    return max(acc_list, key=lambda x: x[1])  # (epoch, best_acc)

def report_file(path):
    train_acc, val_acc = parse_log(path)
    best_train = best_entry(train_acc)
    best_val   = best_entry(val_acc)
    return best_train, best_val

def summarize(values, label):
    if not values:
        print(f"No {label} values found across files.")
        return
    if len(values) == 1:
        print(f"{label}: mean = {values[0]:.4f}%, stdev = 0.0000% (only one file)")
    else:
        print(f"{label}: mean = {mean(values):.4f}%, stdev = {stdev(values):.4f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Scan log file(s), report best train/val top-1 per file, and aggregate mean/stdev."
    )
    parser.add_argument(
        "path",
        help="Path to a single log file or a directory containing log files."
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*.txt",
        help="Glob pattern to match log files within a directory (default: *.txt)."
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search directories for matching log files."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file lines; only print aggregate stats."
    )
    parser.add_argument(
        "--mean-stdev",
        action="store_true",
        help="report mean and stdev"
    )
    args = parser.parse_args()

    p = Path(args.path)
    files = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        if args.recursive:
            files = sorted(p.rglob(args.pattern))
        else:
            files = sorted(p.glob(args.pattern))
    else:
        parser.error(f"Path not found: {p}")

    if not files:
        print("No log files matched.")
        return

    per_file_train = []
    per_file_val   = []

    if not args.quiet:
        print(f"Found {len(files)} file(s). Reporting best per file:\n")

    for f in files:
        best_train, best_val = report_file(f)

        if not args.quiet:
            print(f"[{f}]")
            if best_train:
                tepoch, tacc = best_train
                print(f"  Best training top-1 = {tacc:.4f}% at epoch {tepoch}")
            else:
                print("  No training entries found.")
            if best_val:
                vepoch, vacc = best_val
                print(f"  Best validation top-1 = {vacc:.4f}% at epoch {vepoch}")
            else:
                print("  No validation entries found.")
            print()

        if best_train:
            per_file_train.append(best_train[1])
        if best_val:
            per_file_val.append(best_val[1])

    if args.mean_stdev:
        print("Aggregate across files:")
        summarize(per_file_train, "Training top-1")
        summarize(per_file_val,   "Validation top-1")

if __name__ == "__main__":
    main()
