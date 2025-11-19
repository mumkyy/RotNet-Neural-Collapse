#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

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
        print(f"No {label} values found.")
        return
    if len(values) == 1:
        print(f"{label}: {values[0]:.4f}% (only one file)")
    else:
        print(f"{label}: mean = {mean(values):.4f}%, stdev = {stdev(values):.4f}%")

def detect_category(categories, path: Path) -> str:
    s = str(path).lower()
    for x in categories:
        if x in s:
            return x
    return "other"

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
        help="Report mean and stdev across all files (assumes same experiment repeated)"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",             
        default=None,           
        help="Category keywords to group by (e.g. --categories NIN4blocks conv2 conv4). "
            "Shows best performer in each category."
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

    # Always list best-per-file first (unless quiet), and cache results
    per_file_results = {}
    for f in files:
        per_file_results[f] = report_file(f)

    if not args.quiet:
        print(f"Found {len(files)} file(s). Reporting best per file:\n")
        for f in files:
            best_train, best_val = per_file_results[f]
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

    # Decide mode: category grouping or mean/stdev aggregation
    if args.categories:
        # Category mode: group files by category and show best in each
        categories_lower = [c.lower() for c in args.categories]
        category_results = defaultdict(list)  # category -> [(file, best_train, best_val)]
        
        for f in files:
            cat = detect_category(categories_lower, f)
            if cat != "other":
                best_train, best_val = per_file_results[f]
                category_results[cat].append((f, best_train, best_val))
        
        if not category_results:
            print("No files matched any category.")
            return
        
        print(f"Found {sum(len(v) for v in category_results.values())} file(s) across {len(category_results)} categories.\n")
        
        for cat in categories_lower:
            if cat not in category_results:
                continue
            
            results = category_results[cat]
            print(f"=== Category: {cat} ({len(results)} files) ===")
            
            # Find best by validation accuracy
            best_file = None
            best_val_acc = -1
            for f, best_train, best_val in results:
                if best_val and best_val[1] > best_val_acc:
                    best_val_acc = best_val[1]
                    best_file = (f, best_train, best_val)
            
            if best_file:
                f, best_train, best_val = best_file
                print(f"Best performer: {f}")
                if best_train:
                    print(f"  Training top-1 = {best_train[1]:.4f}% at epoch {best_train[0]}")
                if best_val:
                    print(f"  Validation top-1 = {best_val[1]:.4f}% at epoch {best_val[0]}")
            else:
                print("No validation results found in this category.")
            print()
    
    elif args.mean_stdev:
        # Aggregate mode: compute mean/stdev across all files
        per_file_train = []
        per_file_val   = []
        
        for f in files:
            best_train, best_val = per_file_results[f]
            if best_train:
                per_file_train.append(best_train[1])
            if best_val:
                per_file_val.append(best_val[1])
        
        print("Aggregate statistics:")
        summarize(per_file_train, "Training top-1")
        summarize(per_file_val,   "Validation top-1")
    
    return

if __name__ == "__main__":
    main()
