import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# map layer name -> results directory from measurements.py runs
LAYER_RUNS = {
    "conv1": "results/EXP_conv1_run_dir/metrics.pkl",
    "conv2": "results/EXP_conv2_run_dir/metrics.pkl",
    "conv3": "results/EXP_conv3_run_dir/metrics.pkl",
    "conv4": "results/EXP_conv4_run_dir/metrics.pkl",
    "conv5": "results/EXP_conv5_run_dir/metrics.pkl",
    "conv6": "results/EXP_conv6_run_dir/metrics.pkl",
    "conv7": "results/EXP_conv7_run_dir/metrics.pkl",
    "fc6"  : "results/EXP_fc6_run_dir/metrics.pkl",
}

nc1_vals = []
layer_order = []

for layer, pkl_path in LAYER_RUNS.items():
    p = Path(pkl_path)
    with open(p, "rb") as f:
        data = pickle.load(f)
    metrics = data["metrics"]
    # trSwtrSb is NC1; for a single checkpoint run, take the first (and only) value
    nc1 = metrics.trSwtrSb[-1]
    layer_order.append(layer)
    nc1_vals.append(nc1)

plt.figure()
plt.plot(layer_order, nc1_vals, "bx-")
plt.xlabel("layer")
plt.ylabel("NC1 = Tr(Sw)/Tr(Sb)")
plt.title("NC1 per layer (single checkpoint)")
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("NC1_per_layer.pdf")
