#!/usr/bin/env python3
# coding: utf-8

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from data import get_loaders, Modes
from model import AlexNetwork


def _load_measurements_fix():
    root = Path(__file__).resolve().parents[2]
    mf_path = root / "RNC" / "measurements" / "measurementsFix.py"
    spec = importlib.util.spec_from_file_location("measurementsFix", mf_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load measurementsFix from {mf_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_config(cfg_name: str) -> dict:
    cfg_mod = importlib.import_module(f"configs.{cfg_name}")
    return cfg_mod.config


def _load_state_dict(model: nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        sd = state.get("state_dict") or state.get("network") or state.get("model") or state
    else:
        sd = state
    model.load_state_dict(sd, strict=True)


def _split_pair(x):
    if isinstance(x, (tuple, list)):
        return x[0], x[1]
    if x.dim() == 5:
        return x[:, 0], x[:, 1]
    raise ValueError("Expected input as (N,2,C,H,W) or (uniform, random) tuple.")


class PairLoader:
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __call__(self, _epoch=0):
        for uniform_patch, random_patch, y in self.base_loader:
            x = torch.stack([uniform_patch, random_patch], dim=1)
            yield x, y


class ContextPredWrapper(nn.Module):
    def __init__(self, base: AlexNetwork):
        super().__init__()
        self.base = base
        self.all_feat_names = [
            "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "fc6", "classifier"
        ]

        class _Block(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.Classifier = classifier

        self._feature_blocks = nn.ModuleList([_Block(self.base.fc[-1])])

    def forward(self, x, out_feat_keys=None):
        uniform, random = _split_pair(x)
        if out_feat_keys is None:
            logits, _, _ = self.base(uniform, random)
            return logits

        feat_keys = [k for k in out_feat_keys if k != "classifier"]
        feats_u = self.base.extract_features(uniform, out_keys=feat_keys)
        feats_r = self.base.extract_features(random, out_keys=feat_keys)

        outs = []
        for k in out_feat_keys:
            if k == "classifier":
                logits, _, _ = self.base(uniform, random)
                outs.append(logits)
            else:
                outs.append(torch.cat([feats_u[k], feats_r[k]], dim=1))
        return outs


def main():
    p = argparse.ArgumentParser("Context-pred NC metrics (measurementsFix-like)")
    p.add_argument("--config", required=True, help="configs.<name> (e.g. CIFAR10.4_way.backbone.aug.CIFAR10_4_way_collapsed_backbone)")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--layers", default="conv1,conv2,conv3,conv4")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--nc4", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    cfg = _load_config(args.config)
    train_opt = cfg["data_train_opt"]
    net_cfg = cfg["networks"]["model"]

    if net_cfg.get("arch", "AlexNetwork") != "AlexNetwork":
        raise RuntimeError("This script expects an AlexNetwork pretext config.")

    base = AlexNetwork(**net_cfg.get("opt", {}))
    _load_state_dict(base, Path(args.ckpt))
    model = ContextPredWrapper(base)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    mode_str = train_opt.get("mode", "EIGHT").upper()
    mode = Modes.QUAD if mode_str == "QUAD" else Modes.EIGHT

    train_loader, val_loader = get_loaders(
        mode=mode,
        patch_dim=train_opt["patch_dim"],
        batch_size=args.batch_size or train_opt["batch_size"],
        num_workers=args.num_workers,
        root=train_opt["dataset_root"],
        gap=train_opt.get("gap", None),
        chromatic=train_opt.get("chromatic", True),
        jitter=train_opt.get("jitter", True),
        dataset_name=train_opt.get("dataset_name", "Imagenette"),
    )

    base_loader = train_loader if args.split == "train" else val_loader
    loader = PairLoader(base_loader)

    mf = _load_measurements_fix()
    layer_keys = [k.strip() for k in args.layers.split(",") if k.strip()]

    if args.nc4:
        nc1_by_layer, acc, loss, nc3, means_penult = mf.compute_epoch_metrics_multilayer(
            model=model,
            loader=loader,
            num_classes=args.num_classes,
            layer_keys=layer_keys,
            device=device,
            return_means_penult=True,
        )
        nc4_match, nc4_mismatch, ncc_acc = mf.nc4Fun(
            model=model,
            loader=loader,
            means_penult=means_penult,
            num_classes=args.num_classes,
            device=device,
        )
        print(f"acc={acc:.4f} loss={loss:.4f} nc3={nc3:.6f}")
        print("nc1:", " | ".join([f"{k}={nc1_by_layer[k]:.6f}" for k in layer_keys]))
        print(f"nc4_match={nc4_match:.4f} nc4_mismatch={nc4_mismatch:.4f} ncc_acc={ncc_acc:.4f}")
    else:
        nc1_by_layer, acc, loss, nc3 = mf.compute_epoch_metrics_multilayer(
            model=model,
            loader=loader,
            num_classes=args.num_classes,
            layer_keys=layer_keys,
            device=device,
        )
        print(f"acc={acc:.4f} loss={loss:.4f} nc3={nc3:.6f}")
        print("nc1:", " | ".join([f"{k}={nc1_by_layer[k]:.6f}" for k in layer_keys]))


if __name__ == "__main__":
    main()
