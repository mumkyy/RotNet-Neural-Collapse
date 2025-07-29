#!/usr/bin/env python3
import re
import argparse

def parse_log(path):
    train_acc  = []  # list of (epoch, prec1)
    val_acc    = []  # list of (epoch, prec1)
    current_epoch = None

    # regexes
    epoch_re   = re.compile(r"Training epoch\s*\[\s*(\d+)\s*/\s*\d+\]")
    train_re   = re.compile(r"Training stats: \{[^}]*'prec1':\s*([\d.]+)")
    eval_re    = re.compile(r"Evaluation stats: \{[^}]*'prec1':\s*([\d.]+)")

    with open(path, 'r') as f:
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

def report_max(acc_list, name):
    if not acc_list:
        print(f"No {name} entries found.")
        return
    # find entry with max accuracy
    epoch, best = max(acc_list, key=lambda x: x[1])
    print(f"Best {name} top-1 accuracy = {best:.4f}% at epoch {epoch}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract highest train/val top-1 accuracy from log")
    parser.add_argument("logfile", help="path to the training log file")
    args = parser.parse_args()

    train_acc, val_acc = parse_log(args.logfile)
    report_max(train_acc, "training")
    report_max(val_acc,   "validation")

if __name__ == "__main__":
    main()
