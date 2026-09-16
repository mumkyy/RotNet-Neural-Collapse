import re, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = "/mnt/user-data/uploads"
OUT = "/mnt/user-data/outputs"
FILES = [
    ("collapsed_highWD5e3",           "rank_collapsed_highWD5e3.txt"),
    ("nc3_nc1_inverse_noWD_standard", "rank_nc3_nc1_inverse_noWD_standard.txt"),
    ("not_collapsed_noWD_standard",   "rank_not_collapsed_noWD_standard.txt"),
]

entry_re = re.compile(
    r"'([^']+)':\s*\{'froNorm':\s*tensor\(([-\d.eE+]+).*?"
    r"'stableRank':\s*tensor\(([-\d.eE+]+).*?"
    r"'Rank':\s*tensor\((\d+)\).*?"
    r"'weightShape':\s*torch\.Size\(\[(\d+),\s*(\d+)\]\)\}", re.S)

def parse(path):
    line = [l for l in open(path).read().strip().splitlines()
            if l.strip() and set(l.strip()) != {"."}][-1]
    return {n: dict(fro=float(f), sr=float(s), rank=int(r), shape=(int(a), int(b)))
            for n, f, s, r, a, b in entry_re.findall(line)}

runs = {label: parse(os.path.join(UP, fn)) for label, fn in FILES}
layers = list(runs[FILES[0][0]].keys())
x = list(range(len(layers)))
short = [l.replace("_feature_blocks.", "").replace("block", "b").replace("conv", "c")
          .replace(".shortcut.0", ".sc") for l in layers]
maxrank = [min(runs[FILES[0][0]][l]["shape"]) for l in layers]

# ---- validation accuracies, earliest log file per probe ----------------------
acc = collections.defaultdict(lambda: collections.defaultdict(list))
sec = name = stamp = None
for line in open(os.path.join(UP, "accrank__2_.txt")):
    m = re.match(r"=====\s*(.+?)\s*=====", line)
    if m: sec = m.group(1); continue
    m = re.search(r".*resnet34_(.+?)_feats_", line)  # last "resnet34_" in the path
    if m:
        name = m.group(1)
        stamp = re.search(r"LOG_INFO_([\d_-]+)\.txt", line).group(1)
        continue
    m = re.search(r"Last validation top-1 = ([\d.]+)", line)
    if m: acc[sec][name].append((stamp, float(m.group(1))))

def probe_to_layer(p):
    if p == "conv1.Stem_Conv":
        return "_feature_blocks.0.Stem_Conv"
    m = re.match(r"conv(\d+)\.block(\d+)$", p)
    if m: return "_feature_blocks.%d.block%s.conv2" % (int(m.group(1)) - 1, m.group(2))
    return None

# the accuracy file names the third run differently from the rank file
ACC_SECTION = {"collapsed_highWD5e3": "collapsed_highWD5e3",
               "nc3_nc1_inverse_noWD_standard": "nc3_nc1_inverse_noWD_standard",
               "not_collapsed_noWD_standard": "not_collapsed_justNoWD"}

acc_xy = {}
for label, _ in FILES:
    pts = [(layers.index(probe_to_layer(p)), min(v)[1])
           for p, v in acc.get(ACC_SECTION[label], {}).items()
           if probe_to_layer(p) in layers]
    if pts: acc_xy[label] = sorted(pts)

marks = [dict(marker="o", ms=11, mfc="none", mew=1.5),
         dict(marker="s", ms=6),
         dict(marker="x", ms=5, mew=1.5)]
metrics = [("fro", "Frobenius norm"), ("sr", "Stable rank"), ("rank", "Rank")]
# stage boundaries (first layer of each _feature_blocks.N)
bounds = [i for i, l in enumerate(layers)
          if i and l.split(".")[1] != layers[i-1].split(".")[1]]

def draw_metric(ax, key, title):
    lines = []
    if key == "rank":
        ax.plot(x, maxrank, color="gray", label="min(weightShape)")
    for (label, _), m in zip(FILES, marks):
        lines += ax.plot(x, [runs[label][l][key] for l in layers], label=label, **m)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    for b in bounds:
        ax.axvline(b - 0.5, color="lightgray", lw=1)
    return lines

def draw_acc(ax, lines, legend_prefix=" val top-1"):
    for (label, _), m, ln in zip(FILES, marks, lines):
        if label in acc_xy:
            xs, ys = zip(*acc_xy[label])
            ax.plot(xs, ys, ls="--", alpha=0.6, color=ln.get_color(),
                    label=label + legend_prefix, **m)
    ax.set_ylim(0, 100)

def finish(fig, axes):
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short, rotation=90, fontsize=7)
    axes[-1].set_xlabel("layer")
    fig.tight_layout()

# ---- version A: dual axis ---------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
for ax, (key, title) in zip(axes, metrics):
    lines = draw_metric(ax, key, title)
    ax2 = ax.twinx()
    draw_acc(ax2, lines)
    ax2.set_ylabel("val top-1 (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
finish(fig, axes)
fig.savefig(os.path.join(OUT, "rank_metrics_comparison.png"), dpi=150)

# ---- version B: accuracy in its own panel -----------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
lines = None
for ax, (key, title) in zip(axes, metrics):
    lines = draw_metric(ax, key, title)
    ax.legend(fontsize=8, loc="upper left")
ax = axes[-1]
draw_acc(ax, lines, legend_prefix="")
ax.set_ylabel("val top-1 (%)")
ax.grid(True, alpha=0.3)
for b in bounds:
    ax.axvline(b - 0.5, color="lightgray", lw=1)
ax.legend(fontsize=8, loc="upper left")
finish(fig, axes)
fig.savefig(os.path.join(OUT, "rank_metrics_comparison_4panel.png"), dpi=150)