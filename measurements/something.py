import argparse, importlib.util, os, pickle, random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm  # added for progress bars
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds, ArpackError
from torch.utils.data import DataLoader
from measurements import Measurements, compute_metrics, parse_args, set_seed

def __main__():
     args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:                       # i.e. you passed --no_cuda
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, map_location=torch.device('cpu'))
    set_seed(42)
    

    # 1) load config
    cfg_file = Path('config')/f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
       exp_dir = Path('experiments') / args.exp    
    config['exp_dir'] = str(exp_dir)

    # 2) build loader
    from dataloader import GenericDataset, DataLoader as RotLoader
    dt = config['data_train_opt']
    ds_train = GenericDataset(
        dataset_name=dt['dataset_name'],
        split=dt['split'],
        random_sized_crop=dt['random_sized_crop'],
        num_imgs_per_cat=dt.get('num_imgs_per_cat')
    )
    loader = RotLoader(
        dataset=ds_train,
        batch_size=dt['batch_size'],
        unsupervised=dt['unsupervised'],
        epoch_size=dt['epoch_size'],
        num_workers=args.workers,
        shuffle=False
    )(0)

    C = 4  # RotNet 4 rotations
    feat_layer = config.get('feature_layer', 'encoder')

    # 3) instantiate alg + load checkpoint
    import algorithms as alg
    algo = getattr(alg, config['algorithm_type'])(config)
    if use_cuda:
        algo.load_to_gpu()
    algo.load_checkpoint(args.checkpoint, train=False)

    # figure out which epochs to process
    exp_dir = Path(config['exp_dir'])
    # grab all "model_net_epochXX" files and extract XX
    all_files = list(exp_dir.glob('*_net_epoch*'))
    all_epochs = sorted(int(f.stem.split('epoch')[-1]) for f in all_files)
    start = args.start_epoch
    end   = args.end_epoch or max(all_epochs)
    epoch_list = [e for e in all_epochs if start <= e <= end]
    if not epoch_list:
        raise RuntimeError(f"No checkpoints found in {exp_dir} between epochs {start}–{end}")

    # 4) measure over all epochs
    metrics = Measurements()
    for e in epoch_list:
        print(f"\n[Measurement] Loading checkpoint epoch {e}")
        algo.load_checkpoint(e, train=False)
        # pull out the nn.Module exactly as before...
        # (your existing fallback logic that sets `model` from `algo`)
        model = None
        if hasattr(algo, 'networks'):
            for k,v in algo.networks.items():
                if isinstance(v, nn.Module):
                    model = v; break
        if model is None:
            for name in ('model','net'):
                cand = getattr(algo, name, None)
                if isinstance(cand, nn.Module):
                    model = cand; break
        if model is None:
            raise RuntimeError("Couldn't find a torch.nn.Module in algo")

        # now compute and append a single point
        compute_metrics(metrics, model, loader, C, feat_layer)

    # 5) save  plot curves vs epoch_list (instead of len=1)
    save_dir = Path('results')/f"{args.exp}_{type(model).__name__}"/f"bs{args.batch_size}_epochs{start}-{end}"
    (save_dir/'plots').mkdir(parents=True, exist_ok=True)
    with open(save_dir/'metrics.pkl','wb') as f:
        pickle.dump({'epochs': epoch_list, 'metrics': metrics}, f)

    # plotting
    for name in vars(metrics):
        plt.figure()
        plt.plot(epoch_list, getattr(metrics, name), 'bx-')
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.title(f"{name} vs epoch")
        plt.tight_layout()
        plt.savefig(save_dir/'plots'/f"{name}.pdf")
        plt.close()

    print("✓  Done – results in", save_dir)

if __name__=='__main__':
    main()