#!/usr/bin/env python
# coding: utf-8

"""

NC1     = Tr(S_W)/Tr(S_B)
NC2     = CoV of class norm means (M) and weights (W) plus average off diagonal coherence for both
NC3     = || W / ||W||_F - M_c / ||M_c||_F ||_F^2
NCC Mismatch = 1 -P(argmax(net) == argmin NCC-distance)


python measurements.py --exp CIFAR10_RotNet_NIN4blocks --ckpt-glob "model_net_epoch*"
--arch-class NetworkInNetwork


"""

import argparse 
import importlib.util
import os
import pickle 
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple 

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.sparse.linalg import svds, ArpackError

def set_seed(seed: int) ->  None:  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Measurements:

    def __init__(self) -> None: 
        self.accuracy: List[float]      = []
        self.loss: List[float]          = []
        self.trSwtrSb: List[float]      = [] #NC1 : Tr(Sw)/Tr(Sb)
        self.norm_M_CoV: List[float]    = [] #CoV of ||avg(class) - avg(global)||
        self.norm_W_CoV: List[float]    = [] #cov of ||within class||
        self.cos_m: List[float]         = [] #coherence of M_c : Angle Uniformity (NC-2) for Features.
        self.cos_W: List[float]         = [] #coherence of W : Angle Uniformity (NC-2) for Weights
        self.W_M_dist: List[float]      = [] #Alignment of Features and Weights.
        self.NCC_mismatch: List[float]  = [] #Classifier Consistency.

def build_loss_from_config(config: dict) -> Optional[nn.Module]:

    """
    Config.py should have:             
        criterions = {}
        criterions['loss'] = {'ctype':'MSELoss', 'opt':None}
        config['criterions'] = criterions
        config['algorithm_type'] = 'ClassificationModel'

    we want to use this to construct loss 
    """
    cirterion_cfg = config.get('criterions', {}).get('loss',None)

    if cirterion_cfg is None: 
        print("[Measurement] No loss configured, loss curve will be skipped")
        return None

    #grab the loss type 
    ctype = cirterion_cfg.get('ctype', 'CrossEntropyLoss')
    opt = cirterion_cfg.get('opt') or {}

#basically take anything that is not CE and turn it into CE , i.e Using Ce as loss_fn 
    if hasattr(nn, ctype):
        loss_cls = getattr(nn, ctype)

        if ctype in ('CrossEntropyLoss', 'NLLLoss', 'NLLLoss2d'):
            return loss_cls(**opt)
        else: 
            print(f"[Measurement] configure loss '{ctype}' may not be compatible with integer labells, using CE for eval")
    else: 
        print(f"[Measurement] Unknown loss type '{ctype}'; using CELoss for evaluation instead")

    return nn.CrossEntropyLoss()
def infer_num_classes(model: nn.Module, config: dict, net_key: str) -> int:
    """
    Infer number of classes C from either:
      1) config['networks'][net_key]['opt']['num_classes'], or
      2) out_features of the last nn.Linear layer in the model.
    """
    try:
        C_cfg = (
            config.get('networks', {})
                  .get(net_key, {})
                  .get('opt', {})
                  .get('num_classes', None)
        )
    except Exception:
        C_cfg = None

    if C_cfg is not None:
        return int(C_cfg)

    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is not None:
        return int(last_linear.out_features)

    raise RuntimeError(
        "Could not infer num_classes, no 'num_classes' in config for "
        f"net_key='{net_key}' and no nn.Linear layers found in the model"
    )


def find_classification_layer(model: nn.Module) -> nn.Linear: 
    #this will return the last linear layer of the model
    cls_layer = None
    for m in model.modules(): 
        if isinstance(m, nn.Linear): 
            cls_layer = m 
    if cls_layer is None: 
        raise RuntimeError("No nn.Linear layer found in the model")
    return cls_layer

def find_named_module(model: nn.Module, name: str ) -> nn.Module: 
    #this will find the .named_modules () like 'layer4'/'block4'/'conv4' etc...
    for n,m in model.named_modules() : 
        if n == name: 
            return m 
    raise RuntimeError(f"could not find module with name {name} in model.named_modules()")

def build_fresh_model(
    config: dict,
    net_key: str,
    arch_class: Optional[str],
    use_cuda: bool,
) -> nn.Module:
    """
    Instantiate a fresh model according to config['networks'][net_key].
    """
    net_cfg_all = config.get('networks', {})
    if net_key not in net_cfg_all:
        raise RuntimeError(
            f"net_key {net_key} not found in config['networks']; "
            f"available keys: {list(net_cfg_all.keys())}"
        )

    net_cfg = net_cfg_all[net_key]

    def_file = net_cfg['def_file']          # path to architecture file
    opt_dict = net_cfg.get('opt', {})       # kwargs for model constructor

    module_path = Path(def_file)
    if not module_path.is_file():
        raise FileNotFoundError(f"architecture file {module_path} not found")

    # dynamically import the model definition file
    spec_model = importlib.util.spec_from_file_location(
        module_path.stem, module_path
    )
    mod_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(mod_model)  # type: ignore

    cls_name = arch_class or net_cfg.get('arch', module_path.stem)

    if not hasattr(mod_model, cls_name):
        raise RuntimeError(
            f"Could not find class '{cls_name}' in {def_file}. "
            "Either pass --arch-class, or add 'arch' to "
            f"config['networks']['{net_key}']."
        )

    ModelCls = getattr(mod_model, cls_name)
    model = ModelCls(**opt_dict)

    if use_cuda:
        model = model.cuda()

    return model



@torch.no_grad() 
def compute_metrics( 
    M:Measurements,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    C: int,
    loss_fn: Optional[nn.Module],
    feat_module_name: Optional[str] = None, 
    
) -> None: 
    
    """
    
    Compute NC metrics, loss, and accuracy for a single model (single epoch).

    Assumptions (kept very generic but explicit):
      - model:   x -> logits of shape (N, C), classification-style.
      - loader:  yields (x, y) where y are integer labels in [0, C-1].
      - there is an nn.Linear classification head with out_features = C.

    feat_module_name controls *which* feature layer h we measure NC on:

      - If feat_module_name is None:
            use the INPUT to the final nn.Linear (penultimate feature).
      - If feat_module_name is set:
            use the OUTPUT of that named module as features, but still
            use the final classifier weights W for NC2/NC3.

    
    """
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

    model.eval().to(device)

    feats: Dict[str, torch.Tensor] = {}

    if feat_module_name is None: 

        cls_layer = find_classification_layer(model)

        def hook(module, inp, out):
            # inp[0]: tensor of shape (N, D)
            feats['h'] = inp[0].detach().cpu()

        handle = cls_layer.register_forward_hook(hook)
    else:
        # Hook the output of a specific named module
        feat_module = find_named_module(model, feat_module_name)

        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                feats['h'] = out.detach().cpu().view(out.size(0), -1)
            else:
                raise RuntimeError(
                    f"Hooked module '{feat_module_name}' did not return a "
                    "Tensor; cannot use as feature layer."
                )

        handle = feat_module.register_forward_hook(hook)

    # --- PASS 1: class means, loss, accuracy --------------------------------
    N_per_class = torch.zeros(C, dtype=torch.long)
    sum_per_class: List[Optional[torch.Tensor]] = [None for _ in range(C)]
    total_ss = 0.0      # accumulates squared distances for trace(Sw)
    total_loss = 0.0
    net_correct = 0
    NCC_match = 0

    print("[Measurement] PASS 1: computing class means, accuracy, and loss...")
    for batch in tqdm(loader, desc="PASS 1", unit="batch"):
        if len(batch) == 2:
            # Standard classifier: (x, y)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x)
        elif len(batch) == 3:
            # Context-pred pretext: (uniform_patch, random_patch, y)
            u, r, y = batch
            u = u.to(device)
            r = r.to(device)
            y = y.to(device)
            out, _, _ = model(u, r)
        else:
            raise RuntimeError(f"Unexpected batch structure len={len(batch)}")

        if 'h' not in feats:
            raise RuntimeError(
                "Feature hook did not run; check feat_module_name and model."
            )
        h = feats['h'].view(len(y), -1)  # (N, D)

        # Loss
        
        if loss_fn is not None:
            N_batch = y.size(0)
            total_loss += loss_fn(out, y).item() * N_batch

        # Accuracy: argmax over classes
        net_correct += (out.argmax(1).cpu() == y.cpu()).sum().item()

        # Accumulate per-class sums
        y_cpu = y.cpu()
        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            hc = h[idx]  # (n_c, D)
            if sum_per_class[c] is None:
                sum_per_class[c] = hc.sum(0)
            else:
                sum_per_class[c] += hc.sum(0)
            N_per_class[c] += hc.size(0)

    # Build class means μ_c
    means: List[torch.Tensor] = []
    for c in range(C):
        n_c = int(N_per_class[c].item())
        if n_c == 0:
            raise RuntimeError(
                f"Class {c} has N_per_class=0; data split does not contain "
                "all classes, cannot compute NC metrics robustly."
            )
        means.append(sum_per_class[c] / float(n_c))  # type: ignore

    Mmat = torch.stack(means, dim=1)  # D x C
    muG  = Mmat.mean(dim=1, keepdim=True)  # D x 1

    # --- PASS 2: within-class scatter (Sw) and NCC mismatch -----------------
    print("[Measurement] PASS 2: computing within-class scatter and NCC mismatch...")
    for batch in tqdm(loader, desc="PASS 2", unit="batch"):
        if len(batch) == 2:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x)
        elif len(batch) == 3:
            u, r, y = batch
            u = u.to(device)
            r = r.to(device)
            y = y.to(device)
            out, _, _ = model(u, r)
        else:
            raise RuntimeError(f"Unexpected batch structure len={len(batch)}")

        if 'h' not in feats:
            raise RuntimeError(
                "Feature hook did not run in PASS 2; check feat_module_name."
            )
        h = feats['h'].view(len(y), -1)  # (N, D)


        y_cpu = y.cpu()
        for c in range(C):
            idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            hc = h[idx]  # (n_c, D)
            z = hc - means[c]  # centered features

            # Accumulate ||x - μ_y||^2 for trace(Sw)
            total_ss += z.pow(2).sum(dim=1).sum().item()

            # NCC predictions: nearest class mean
            dists = torch.norm(
                hc.unsqueeze(1) - Mmat.T.unsqueeze(0),
                dim=2
            )  # (n_c, C)
            NCCp = dists.argmin(dim=1)              # NCC predicted class
            netp = out[idx].argmax(dim=1).cpu()     # net predicted class

            NCC_match += (NCCp == netp).sum().item()

    # --- Finalize scalar metrics --------------------------------------------
    print("[Measurement] Finalizing metrics for this epoch...")

    N = int(N_per_class.sum().item())
    if N == 0:
        raise RuntimeError("No samples seen in loader; cannot compute metrics.")

    # Loss & accuracy
    if loss_fn is not None:
        loss = total_loss / float(N)
    else:
        loss = float('nan')
    acc = net_correct / float(N)

    NCCm = 1.0 - NCC_match / float(N)

    # NC-1: trace(Sw) / trace(Sb)
    trace_Sw = total_ss / float(N)

    M_centered = Mmat - muG  # D x C
    Sb = (M_centered @ M_centered.T) / float(C)  # D x D
    trace_Sb = Sb.trace().item()

    eps = 1e-12
    nc1_ratio = trace_Sw / (trace_Sb + eps)

    # NC-2: CoV of norms + coherence
    cls_layer = find_classification_layer(model)
    W = cls_layer.weight.T  # (D, C)

    M_c = M_centered.to(W.device)  # (D, C)

    Mn = M_c.norm(p=2, dim=0)  # (C,)
    Wn = W.norm(p=2, dim=0)    # (C,)

    covM = (Mn.std(unbiased=False) / (Mn.mean() + eps)).item()
    covW = (Wn.std(unbiased=False) / (Wn.mean() + eps)).item()

    def coherence(V: torch.Tensor) -> float:
        """
        Average off-diagonal magnitude of normalized Gram matrix.

        Steps:
          1. G = V^T V   (C x C)
          2. Normalize by L1 norm.
          3. Zero diagonal.
          4. Return mean |off-diagonal|.
        """
        G = V.T @ V  # (C, C)
        G = G / (G.norm(1, keepdim=True) + 1e-9)
        G.fill_diagonal_(0.0)
        return (G.abs().sum() / float(C * (C - 1))).item()

    cosM = coherence(M_c / (Mn + eps))
    cosW = coherence(W / (Wn + eps))

    # NC-3: distance between normalized W and normalized M_c
    W_normed  = W / (W.norm() + eps)
    Mc_normed = M_c / (M_c.norm() + eps)
    W_M_dist = (W_normed - Mc_normed).norm().pow(2).item()  # Frobenius^2

    # Store
    M.accuracy.append(acc)
    M.loss.append(loss)
    M.trSwtrSb.append(nc1_ratio)
    M.norm_M_CoV.append(covM)
    M.norm_W_CoV.append(covW)
    M.cos_m.append(cosM)
    M.cos_W.append(cosW)
    M.W_M_dist.append(W_M_dist)
    M.NCC_mismatch.append(NCCm)

    # Clean up hook
    handle.remove()

def parse_args(): 
    p = argparse.ArgumentParser(
        "Measure NC metrics on experiment (raw Pytorch)"
    )

    p.add_argument(
        '--exp', required=True,
        help = 'Experiment name, used to find config/<exp>.py and experiments/<exp>/'
    )
    p.add_argument(
        '--exp-dir', default=None,
        help='Optional: full path to the experiment directory; '
             'overrides experiments/<exp>'
    )
    p.add_argument(
        '--checkpoint', type=int, default=None,
        help='If set, analyse ONLY this epoch ID (e.g. 200).'
    )
    p.add_argument(
        '--start-epoch', type=int, default=None,
        help='First epoch to analyse (inclusive).'
    )
    p.add_argument(
        '--end-epoch', type=int, default=None,
        help='Last epoch to analyse (inclusive).'
    )
    p.add_argument(
        '--split', type=str, default='train', choices=['train', 'test'],
        help='Which data split to use (train/test).'
    )
    p.add_argument(
        '--workers', type=int, default=4,
        help='Dataloader workers.'
    )
    p.add_argument(
        '--no_cuda', action='store_true',
        help='Force CPU evaluation even if a GPU is visible.'
    )
    p.add_argument(
        '--net-key', type=str, default='model',
        help="Key in config['networks'] to treat as the primary model "
             "(default: 'model')."
    )
    p.add_argument(
        '--feat-module-name', type=str, default=None,
        help="Optional: name of a module in model.named_modules() whose "
             "OUTPUT should be used as the feature layer. If omitted, "
             "the INPUT to the last nn.Linear is used (penultimate features)."
    )
    p.add_argument(
        '--arch-class', type=str, default=None,
        help="Optional: explicit class name inside def_file for the model "
             "architecture. If omitted, use config['networks'][net_key]['arch'] "
             "or fall back to the def_file stem."
    )
    p.add_argument(
        '--ckpt-glob', type=str, default=None,
        help="Glob pattern (relative to exp_dir) for checkpoint files, e.g. "
             "'model_net_epoch*'. If omitted, defaults to "
             "f'{net_key}_net_epoch*'."
    )
    p.add_argument(
        '--config-root', type=str, default='config',
        help='Root directory containing <exp>.py config files '
             '(default: "./config"). For external projects, pass a path '
             'like "../context-pred/configs".'
    )

    return p.parse_args()

if __name__ == '__main__': 
    args = parse_args() 

    use_cuda = (not args.no_cuda) and torch.cuda.is_available() 

    if not use_cuda: 
        _orig_load = torch.load 
        torch.load = lambda f, **kw: _orig_load(
            f, map_location=torch.device('cpu'), **kw
        )
    set_seed(42)
    try:
        from torchvision import datasets, transforms
        has_imagenette = hasattr(datasets, "Imagenette")

        if not has_imagenette:
            print("[Warning] torchvision.datasets.Imagenette not found — applying monkey-patch.")

            class Imagenette(datasets.ImageFolder):
                """
                Minimal replacement for torchvision.datasets.Imagenette.
                The dataset root should contain:
                    imagenette2-160/train/
                    imagenette2-160/val/
                """

                def __init__(self, root, split="train", transform=None, size="160px", download=False):
                    if size not in ("160px", "320px"):
                        raise ValueError("Imagenette monkey-patch supports size='160px' or '320px'.")

                    # Directory names used by the official dataset
                    possible_dirs = [
                        os.path.join(root, "imagenette2-160"),
                        os.path.join(root, "imagenette2"),
                        os.path.join(root, "imagenette2-320"),
                    ]

                    dataset_root = None
                    for d in possible_dirs:
                        if os.path.isdir(d):
                            dataset_root = d
                            break

                    if dataset_root is None:
                        raise FileNotFoundError(
                            "Could not find Imagenette directory under root '{}'. "
                            "Expected one of: {}".format(root, possible_dirs)
                        )

                    split_dir = os.path.join(dataset_root, split)
                    if not os.path.isdir(split_dir):
                        raise FileNotFoundError(f"Split directory not found: {split_dir}")

                    super().__init__(split_dir, transform=transform)

            datasets.Imagenette = Imagenette

    except Exception as e:
        print(f"[Warning] Imagenette monkey-patch failed: {e}")
    cfg_root = Path(args.config_root)
    cfg_file = cfg_root / f"{args.exp}.py"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config

    # Experiment directory (where checkpoints live)
    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = Path('experiments') / args.exp
    config['exp_dir'] = str(exp_dir)

    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        
    from dataloader import GenericDataset, DataLoader as RotLoader

    if args.split == 'train':
        dt = config['data_train_opt']
    else:
        dt = config['data_test_opt']

    # -------------------------------------------------------------------------
    # Decide which data pipeline to use:
    #   - RNC GenericDataset (CIFAR RotNet etc.)
    #   - context-pred Imagenette patch loader (AlexNetwork pretext)
    # -------------------------------------------------------------------------
    is_context_pred = (
        'dataset_name' in dt
        and dt['dataset_name'].lower() == 'imagenette'
        and 'patch_dim' in dt
    )

    if is_context_pred:
        # Import context-pred's data.py by path and use get_loaders
        repo_root = Path(__file__).resolve().parents[2]
        ctx_dir = repo_root / 'context-pred' / 'context-pred'
        data_py = ctx_dir / 'data.py'
        if not data_py.is_file():
            raise FileNotFoundError(f"Context-pred data.py not found at {data_py}")

        spec_data = importlib.util.spec_from_file_location("context_pred_data", data_py)
        ctx_mod = importlib.util.module_from_spec(spec_data)
        spec_data.loader.exec_module(ctx_mod)  # type: ignore

        net_cfg = config.get('networks', {}).get(args.net_key, {})
        def_file_cfg = net_cfg.get('def_file', None)
        if def_file_cfg in (None, 'model.py', './model.py'):
            abs_model_path = ctx_dir / 'model.py'
            if not abs_model_path.is_file():
                raise FileNotFoundError(f"Expected AlexNetwork model at {abs_model_path}")
            net_cfg['def_file'] = str(abs_model_path)

        # Map mode string -> Modes enum
        mode_str = dt.get('mode', 'EIGHT').upper()
        if mode_str == 'QUAD':
            mode = ctx_mod.Modes.QUAD
        elif mode_str == 'EIGHT':
            mode = ctx_mod.Modes.EIGHT
        else:
            raise ValueError(f"Unsupported context-pred mode '{mode_str}'")

        patch_dim  = dt['patch_dim']
        batch_size = dt['batch_size']

        # dataset_root from context-pred config.
        # If it's relative (e.g. "data"), interpret it relative to ctx_dir
        # so we actually look under:  <repo_root>/context-pred/context-pred/data
        root_cfg = dt.get('dataset_root', 'data')
        root_path = Path(root_cfg)
        if not root_path.is_absolute():
            root_path = ctx_dir / root_path
        root = str(root_path)

        gap       = dt.get('gap', None)
        chromatic = dt.get('chromatic', True)
        jitter    = dt.get('jitter', True)

        train_loader, val_loader = ctx_mod.get_loaders(
            mode      = mode,
            patch_dim = patch_dim,
            batch_size= batch_size,
            num_workers=args.workers,
            root      = root,
            gap       = gap,
            chromatic = chromatic,
            jitter    = jitter,
        )

        loader = train_loader if args.split == 'train' else val_loader

    else:
        # Original RNC path (CIFAR RotNet etc.)
        dataset_kwargs = {
            k: v for k, v in dt.items()
            if k not in ('batch_size', 'unsupervised', 'epoch_size')
        }

        ds = GenericDataset(**dataset_kwargs)

        loader = RotLoader(
            dataset=ds,
            unsupervised=dt.get('unsupervised', False),
            epoch_size=dt.get('epoch_size', None),
            num_workers=args.workers,
            shuffle=False
        )(0)


    # Build evaluation loss
    loss_fn = build_loss_from_config(config)

    # ---------------------------------------------------------------------
    # 3) Discover checkpoints and map epoch -> path
    # ---------------------------------------------------------------------
    pattern = args.ckpt_glob or f"{args.net_key}_net_epoch*"
    ckpt_files = list(exp_dir.glob(pattern))
    if not ckpt_files:
        raise RuntimeError(
            f"No checkpoints found in {exp_dir} matching pattern '{pattern}'."
        )

    epoch_to_path: Dict[int, Path] = {}
    for f in ckpt_files:
        stem = f.stem  # e.g. 'model_net_epoch200'
        try:
            ep_str = stem.split('epoch')[-1]
            epoch = int(ep_str)
            epoch_to_path[epoch] = f
        except ValueError:
            continue

    if not epoch_to_path:
        raise RuntimeError(
            f"Could not parse any epoch numbers from checkpoints in {exp_dir} "
            f"matching pattern '{pattern}'."
        )

    all_epochs = sorted(epoch_to_path.keys())
    min_epoch, max_epoch = min(all_epochs), max(all_epochs)

    if args.checkpoint is not None:
        if args.checkpoint not in all_epochs:
            raise RuntimeError(
                f"Requested checkpoint epoch {args.checkpoint} not found "
                f"in {exp_dir}; available: {all_epochs}"
            )
        epoch_list = [args.checkpoint]
    elif args.start_epoch is not None or args.end_epoch is not None:
        start = args.start_epoch if args.start_epoch is not None else min_epoch
        end   = args.end_epoch   if args.end_epoch   is not None else max_epoch
        epoch_list = [e for e in all_epochs if start <= e <= end]
        if not epoch_list:
            raise RuntimeError(
                f"No checkpoints found between epochs {start}–{end} "
                f"in {exp_dir}"
            )
    else:
        epoch_list = all_epochs

    print(f"[Measurement] Found epochs: {all_epochs}")
    print(f"[Measurement] Analysing epochs: {epoch_list}")

    metrics = Measurements()
    batch_size = dt.get('batch_size', 0)

    last_model: Optional[nn.Module] = None

    for e in epoch_list:
        print(f"\n[Measurement] Loading checkpoint epoch {e}")
        ckpt_path = epoch_to_path[e]

        # Build fresh model from config
        model = build_fresh_model(
            config=config,
            net_key=args.net_key,
            arch_class=args.arch_class,
            use_cuda=use_cuda
        )

        # Load state dict
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)

        if use_cuda:
            model = model.cuda()

        # Infer number of classes C for NC metrics
        C = infer_num_classes(model, config, args.net_key)

        # Compute & append metrics for this epoch
        compute_metrics(
            M=metrics,
            model=model,
            loader=loader,
            C=C,
            loss_fn=loss_fn,
            feat_module_name=args.feat_module_name,
        )

        last_model = model

    if last_model is None:
        raise RuntimeError("No models were evaluated; something went wrong.")
    
    save_dir = (
        Path('results')
        / f"{args.exp}_{type(last_model).__name__}"
        / f"bs{batch_size}_epochs{epoch_list[0]}-{epoch_list[-1]}_{args.split}"
    )
    (save_dir / 'plots').mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump({'epochs': epoch_list, 'metrics': metrics}, f)

    # Simple 1D curves vs epoch
    for name in vars(metrics):
        curve = getattr(metrics, name)
        if not curve:
            continue
        plt.figure()
        plt.plot(epoch_list, curve, 'bx-')
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.title(f"{name} vs epoch")
        plt.tight_layout()
        plt.savefig(save_dir / 'plots' / f"{name}.pdf")
        plt.close()

    print("✓ Done – results in", save_dir)