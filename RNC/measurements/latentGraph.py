import torch
import torch.nn as nn

import argparse, importlib.util, random
from pathlib import Path 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# set the seed
def set_seed(seed: int): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def collect_embeddings(model: nn.Module, loader, layer_name: str, device: torch.device,
                       pool: str = "gap", max_batches: int = 100, max_points: int = 5000):
    model.eval()
    stash = {"feat": None}

    def hook_fn(module, inp, out):
        stash["feat"] = out

    target_module = get_module_by_name(model, layer_name)
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
            raise RuntimeError("Hook did not capture features. Check that the layer is used in forward.")

        if feat.dim() == 4:
            feat_vec = feat.mean(dim=(2, 3)) if pool == "gap" else feat.flatten(1)
        elif feat.dim() == 2:
            feat_vec = feat
        else:
            feat_vec = feat.flatten(1)

        feat_vec = feat_vec.detach().cpu().to(torch.float32).numpy()

        if torch.is_tensor(y_batch):
            y_np = y_batch.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_batch)
        y_np = y_np.reshape(-1)

        X_list.append(feat_vec)
        y_list.append(y_np)

        seen += feat_vec.shape[0]
        if seen >= max_points:
            break

    hook_handle.remove()

    X = np.concatenate(X_list, axis=0)[:max_points]
    y = np.concatenate(y_list, axis=0)[:max_points]
    return X, y

def pca_2d(X: np.ndarray):
    # Center
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD PCA (stable)
    # Xc = U S V^T, principal directions are rows of V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T  # (N,2)
    return Z

def plot_2d(Z: np.ndarray, y: np.ndarray, title: str, save_path: Path, max_classes: int = 10):
    plt.figure(figsize=(8, 6))
    classes = np.unique(y)
    # If too many unique labels, just plot first max_classes for readability
    classes = classes[:max_classes]

    for c in classes:
        idx = (y == c)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.6, label=str(c))

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args(): 

    p = argparse.ArgumentParser("Graphing the latent space")
    p.add_argument('--exp', required = True,  help = 'experiment that you would like to visualize the latent space for')
    #p.add_argument('--mode', required = False, type = str, default ='pca' , help = 'type of analysis (pca/umap)')
    p.add_argument('--checkpoint', type=int, default=0, help='epoch id of model_net_epochXX to load')

    p.add_argument('--layer', required=True,
                   help="module name to hook (must appear in model.named_modules())")

    p.add_argument('--pool', type=str, default='gap', choices=['gap', 'flatten'],
                   help="How to turn conv maps into vectors: global-average-pool ('gap') or flatten")
    p.add_argument('--max_batches', type=int, default=100,
                   help="limit batches for speed; use -1 for full loader")
    p.add_argument('--max_points', type=int, default=5000,
                   help="cap total samples for plotting")

    return p.parse_args()

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise KeyError(f"Layer '{name}' not found. Check model.named_modules() output.")

if __name__ == '__main__': 
    args = parse_args()
    set_seed(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    cfg_file = Path('config') / f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config

    # 2) build loader
    from dataloader import GenericDataset, DataLoader as RotLoader

    dt = config.get('data_test_opt', config['data_train_opt'])
    batch_size = dt['batch_size']


    dataset = GenericDataset(dt)
    loader = RotLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3) instantiate alg + load checkpoint
    import algorithms as alg
    algo = getattr(alg, config['algorithm_type'])(config)

    if use_cuda:
        algo.load_to_gpu()

    algo.load_checkpoint(args.checkpoint, train=False)

    feat_extractor = algo.networks.get(
        'feat_extractor',
        algo.networks.get('model', list(algo.networks.values())[0])
    )
    model = feat_extractor.to(device)


    print("\n[Modules] Example module names:")
    for i, (n, _) in enumerate(model.named_modules()):
        if i < 60:
            print(" ", n)
    print("...")

    X, y = collect_embeddings(
        model=model,
        loader=loader,
        layer_name=args.layer,
        device=device,
        pool=args.pool,
        max_batches=args.max_batches,
        max_points=args.max_points
    )

    Z = pca_2d(X)

    save_dir = Path('results') / f"{args.exp}_{type(model).__name__}" / f"bs{batch_size}"
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_path = plots_dir / f"latent_pca_layer={args.layer.replace('.', '_')}_ckpt{args.checkpoint}.png"
    plot_2d(
        Z, y,
        title=f"Latent PCA: layer={args.layer} (ckpt {args.checkpoint})",
        save_path=out_path
    )
    print(f"\n✓ Saved latent PCA plot to: {out_path}\n")
