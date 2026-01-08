#!/usr/bin/env python
# coding: utf-8
"""
Latent space visualization (PCA + UMAP) in 2D and 3D.

Supports your repo layout variants:

Config:
  config/<exp>.py
where <exp> can be nested like:
  CIFAR10/jigsaw/MSE/collapsed/backbone/NAME

Checkpoints (two common patterns):
A) experiments/<exp>/model_net_epochXX            (nested to match config exp)
B) experiments/<folder_name>/model_net_epochXX    (flat folder name)

✅ Recommended: pass --exp_dir to explicitly point to the checkpoint folder
(e.g. experiments/CIFAR10_Jigsaw_NIN4blocks_...).

Also supports your custom dataloader.py:
- GenericDataset(dataset_name, split, ...)
- DataLoader(dataset, ...) returning an iterator via loader(epoch)

Downstream CIFAR10 class separability:
- We build dataset using dt from config (often data_test_opt),
- BUT we force RotLoader(unsupervised=False) so y = CIFAR10 class labels.

Outputs:
- results/<exp_tag>_<ModelName>/bs<batch_size>/plots/
  - latent_PCA2D_layer=..._ckptXX.png
  - latent_PCA3D_layer=..._ckptXX.png
  - latent_UMAP2D_layer=..._ckptXX.png  (if umap installed)
  - latent_UMAP3D_layer=..._ckptXX.png  (if umap installed)
"""

import argparse
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Optional: UMAP (pip install umap-learn)
try:
    import umap.umap_ as umap_lib
    _HAS_UMAP = True
except Exception:
    umap_lib = None
    _HAS_UMAP = False


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Feature extraction helpers
# ----------------------------
def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    candidates = [n for n, _ in model.named_modules() if name in n][:25]
    raise KeyError(
        f"Layer '{name}' not found in named_modules(). "
        f"Candidates containing '{name}': {candidates}"
    )


def _feat_to_vec(feat: torch.Tensor, pool: str) -> torch.Tensor:
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3)) if pool == "gap" else feat.flatten(1)
    if feat.dim() == 2:
        return feat
    return feat.flatten(1)


def _label_to_numpy(y_batch):
    # Your dataloader sometimes returns LongTensor([label]) etc.
    if isinstance(y_batch, (tuple, list)):
        y_batch = y_batch[0]
    if torch.is_tensor(y_batch):
        y_np = y_batch.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_batch)
    return y_np.reshape(-1)


@torch.no_grad()
def collect_embeddings_hook(
    model: nn.Module,
    loader,
    module_name: str,
    device: torch.device,
    pool: str = "gap",
    max_batches: int = 100,
    max_points: int = 5000,
):
    """Generic hook-based extractor: module_name must exist in model.named_modules()."""
    model.eval()
    stash = {"feat": None}

    def hook_fn(module, inp, out):
        stash["feat"] = out

    target_module = get_module_by_name(model, module_name)
    hook_handle = target_module.register_forward_hook(hook_fn)

    X_list, y_list = [], []
    seen = 0

    for b, batch in enumerate(loader):
        if max_batches != -1 and b >= max_batches:
            break
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("Dataloader batch is not (x, y). Adapt this part.")

        x_batch, y_batch = batch[0], batch[1]
        x_batch = x_batch.to(device, non_blocking=True)

        stash["feat"] = None
        _ = model(x_batch)

        feat = stash["feat"]
        if feat is None:
            raise RuntimeError("Hook did not capture features. Check that the module is used in forward.")

        feat_vec = _feat_to_vec(feat, pool).detach().cpu().to(torch.float32).numpy()
        y_np = _label_to_numpy(y_batch)

        X_list.append(feat_vec)
        y_list.append(y_np)

        seen += feat_vec.shape[0]
        if seen >= max_points:
            break

    hook_handle.remove()
    X = np.concatenate(X_list, axis=0)[:max_points]
    y = np.concatenate(y_list, axis=0)[:max_points]
    return X, y


@torch.no_grad()
def collect_embeddings_feature_api(
    model: nn.Module,
    loader,
    feature_key: str,
    device: torch.device,
    pool: str = "gap",
    max_batches: int = 100,
    max_points: int = 5000,
):
    """
    NetworkInNetwork-style extractor: uses model(x, out_feat_keys=[feature_key]).
    feature_key should be one of model.all_feat_names: e.g. 'conv1','conv2','conv3','conv4','penult',...
    """
    model.eval()
    X_list, y_list = [], []
    seen = 0

    for b, batch in enumerate(loader):
        if max_batches != -1 and b >= max_batches:
            break
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("Dataloader batch is not (x, y). Adapt this part.")

        x_batch, y_batch = batch[0], batch[1]
        x_batch = x_batch.to(device, non_blocking=True)

        feat = model(x_batch, out_feat_keys=[feature_key])

        feat_vec = _feat_to_vec(feat, pool).detach().cpu().to(torch.float32).numpy()
        y_np = _label_to_numpy(y_batch)

        X_list.append(feat_vec)
        y_list.append(y_np)

        seen += feat_vec.shape[0]
        if seen >= max_points:
            break

    X = np.concatenate(X_list, axis=0)[:max_points]
    y = np.concatenate(y_list, axis=0)[:max_points]
    return X, y


# ----------------------------
# PCA / UMAP
# ----------------------------
def pca_nd(X: np.ndarray, n_components: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def umap_nd(
    X: np.ndarray,
    n_components: int,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    if not _HAS_UMAP:
        raise ImportError("UMAP requested but umap-learn is not installed. Install with: pip install umap-learn")
    reducer = umap_lib.UMAP(
        n_components=n_components,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
    )
    return reducer.fit_transform(X)


# ----------------------------
# Plotting
# ----------------------------
def _make_label_colors(y: np.ndarray):
    classes = np.unique(y)
    classes_sorted = np.sort(classes)
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    color_idx = np.array([class_to_idx[c] for c in y], dtype=int)
    return classes_sorted, color_idx


def plot_scatter_2d(Z: np.ndarray, y: np.ndarray, title: str, save_path: Path, max_classes: int = 50):
    classes, color_idx = _make_label_colors(y)
    if classes.size > max_classes:
        keep = np.isin(y, classes[:max_classes])
        Z, y = Z[keep], y[keep]
        classes, color_idx = _make_label_colors(y)

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=color_idx, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    handles, labels = [], []
    denom = max(1, (len(classes) - 1))
    for i, c in enumerate(classes):
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=6,
                markerfacecolor=sc.cmap(i / denom), markeredgecolor="none"
            )
        )
        labels.append(str(c))
    plt.legend(handles, labels, fontsize=8, frameon=False, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scatter_3d(Z: np.ndarray, y: np.ndarray, title: str, save_path: Path, max_classes: int = 50):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    classes, color_idx = _make_label_colors(y)
    if classes.size > max_classes:
        keep = np.isin(y, classes[:max_classes])
        Z, y = Z[keep], y[keep]
        classes, color_idx = _make_label_colors(y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=color_idx, s=10, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    handles, labels = [], []
    denom = max(1, (len(classes) - 1))
    for i, c in enumerate(classes):
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=6,
                markerfacecolor=sc.cmap(i / denom), markeredgecolor="none"
            )
        )
        labels.append(str(c))
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Path resolution
# ----------------------------
def resolve_exp_dir(exp_root: Path, exp: str, exp_dir_override: str = None) -> Path:
    """
    Priority:
      1) --exp_dir (explicit)
      2) experiments/<exp>                 (nested pattern A)
      3) experiments/<Path(exp).name>      (flat pattern B)  <-- your case

    We return the first directory that exists. If none exist, return the nested default
    so the error message shows where we looked.
    """
    if exp_dir_override is not None:
        p = Path(exp_dir_override)
        p = p if p.is_absolute() else (Path.cwd() / p)
        return p.resolve()

    cand_nested = (exp_root / exp).resolve()
    cand_flat = (exp_root / Path(exp).name).resolve()

    if cand_nested.is_dir():
        return cand_nested
    if cand_flat.is_dir():
        return cand_flat

    # Return nested default for clearer messaging downstream
    return cand_nested


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Graphing the latent space (PCA + UMAP, 2D + 3D)")
    p.add_argument(
        "--exp",
        required=True,
        help="config path: config/<exp>.py (can include CIFAR10/jigsaw/...)",
    )
    p.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="epoch id XX for model_net_epochXX",
    )

    # Checkpoint folder controls
    p.add_argument(
        "--exp_root",
        type=str,
        default="experiments",
        help="root experiments folder (default: experiments)",
    )
    p.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="(RECOMMENDED for flat folders) explicit checkpoint directory containing model_net_epochXX",
    )

    # Feature selection
    p.add_argument(
        "--layer",
        required=True,
        help="If --feature_api: feature key ('conv1','conv2','conv3','conv4','penult'). "
             "Else: module name from model.named_modules().",
    )
    p.add_argument(
        "--feature_api",
        action="store_true",
        help="Use NetworkInNetwork feature API: model(x, out_feat_keys=[layer]).",
    )

    # Vectorization & sampling
    p.add_argument("--pool", type=str, default="gap", choices=["gap", "flatten"])
    p.add_argument("--max_batches", type=int, default=100)
    p.add_argument("--max_points", type=int, default=5000)
    p.add_argument("--standardize", action="store_true")

    # UMAP params
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1) load config
    cfg_file = Path("config") / f"{args.exp}.py"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config

    # 2) resolve checkpoint directory + set exp_dir for Algorithm
    exp_root = Path(args.exp_root).resolve()
    exp_dir = resolve_exp_dir(exp_root=exp_root, exp=args.exp, exp_dir_override=args.exp_dir)
    config["exp_dir"] = str(exp_dir)

    # 3) sanity: checkpoint file must exist
    ckpt_path = exp_dir / f"model_net_epoch{args.checkpoint}"
    if not ckpt_path.is_file():
        nearby = sorted(exp_dir.glob("model_net_epoch*"))
        preview = "\n".join([f"  - {p.name}" for p in nearby[:25]])
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Resolved exp_dir: {exp_dir}\n"
            f"Found {len(nearby)} files matching model_net_epoch* in {exp_dir}.\n"
            f"First few:\n{preview}"
        )

    print(f"[Checkpoint dir] {exp_dir}")
    print(f"[Checkpoint]     {ckpt_path.name}")

    # 4) build loader using YOUR dataloader.py API
    from dataloader import GenericDataset, DataLoader as RotLoader

    dt = config.get("data_test_opt", config["data_train_opt"])
    batch_size = int(dt.get("batch_size", 128))

    dataset = GenericDataset(
        dataset_name=dt.get("dataset_name", "cifar10"),
        split=dt.get("split", "test"),
        random_sized_crop=dt.get("random_sized_crop", False),
        num_imgs_per_cat=dt.get("num_imgs_per_cat", None),
        pretext_mode=dt.get("pretext_mode", "rotation"),
        sigmas=dt.get("sigmas", None),
        kernel_sizes=dt.get("kernel_sizes", None),
        patch_jitter=dt.get("patch_jitter", 0),
        color_distort=dt.get("color_distort", False),
        color_dist_strength=dt.get("color_dist_strength", 1.0),
        fixed_perms=dt.get("fixed_perms", None),
    )

    # downstream separability (CIFAR10 classes): FORCE supervised labels
    rot_loader = RotLoader(
        dataset,
        batch_size=batch_size,
        unsupervised=False,  # y = CIFAR10 class label
        epoch_size=dt.get("epoch_size", None),
        num_workers=4,
        shuffle=False,
    )
    loader = rot_loader(epoch=0)

    # 5) instantiate algorithm + load checkpoint
    import algorithms as alg

    algo = getattr(alg, config["algorithm_type"])(config)
    if use_cuda:
        algo.load_to_gpu()

    # uses config['exp_dir'] internally
    algo.load_checkpoint(args.checkpoint, train=False)

    feat_extractor = algo.networks.get(
        "feat_extractor",
        algo.networks.get("model", list(algo.networks.values())[0]),
    )
    model = feat_extractor.to(device)

    if args.feature_api and hasattr(model, "all_feat_names"):
        print("Available feature keys:")
        print(model.all_feat_names)

    # 6) collect embeddings
    if args.feature_api:
        X, y = collect_embeddings_feature_api(
            model=model,
            loader=loader,
            feature_key=args.layer,
            device=device,
            pool=args.pool,
            max_batches=args.max_batches,
            max_points=args.max_points,
        )
    else:
        X, y = collect_embeddings_hook(
            model=model,
            loader=loader,
            module_name=args.layer,
            device=device,
            pool=args.pool,
            max_batches=args.max_batches,
            max_points=args.max_points,
        )

    X = X.astype(np.float32)
    if args.standardize:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-5)

    # 7) reductions
    Z_pca_2 = pca_nd(X, 2)
    Z_pca_3 = pca_nd(X, 3)

    if _HAS_UMAP:
        Z_umap_2 = umap_nd(X, 2, seed=42, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
        Z_umap_3 = umap_nd(X, 3, seed=42, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
    else:
        Z_umap_2 = None
        Z_umap_3 = None
        print("WARNING: umap-learn not installed; skipping UMAP plots. Install with: pip install umap-learn")

    # 8) save plots
    exp_tag = args.exp.replace("/", "__")
    save_dir = Path("results") / f"{exp_tag}_{type(model).__name__}" / f"bs{batch_size}"
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    layer_tag = args.layer.replace(".", "_")
    ckpt_tag = f"ckpt{args.checkpoint}"

    pca2_path = plots_dir / f"latent_PCA2D_layer={layer_tag}_{ckpt_tag}.png"
    pca3_path = plots_dir / f"latent_PCA3D_layer={layer_tag}_{ckpt_tag}.png"
    plot_scatter_2d(Z_pca_2, y, f"PCA 2D | layer={args.layer} | {ckpt_tag}", pca2_path)
    plot_scatter_3d(Z_pca_3, y, f"PCA 3D | layer={args.layer} | {ckpt_tag}", pca3_path)

    if Z_umap_2 is not None:
        umap2_path = plots_dir / f"latent_UMAP2D_layer={layer_tag}_{ckpt_tag}.png"
        umap3_path = plots_dir / f"latent_UMAP3D_layer={layer_tag}_{ckpt_tag}.png"
        plot_scatter_2d(Z_umap_2, y, f"UMAP 2D | layer={args.layer} | {ckpt_tag}", umap2_path)
        plot_scatter_3d(Z_umap_3, y, f"UMAP 3D | layer={args.layer} | {ckpt_tag}", umap3_path)

    print("\n✓ Saved plots:")
    print(" ", pca2_path)
    print(" ", pca3_path)
    if Z_umap_2 is not None:
        print(" ", umap2_path)
        print(" ", umap3_path)
