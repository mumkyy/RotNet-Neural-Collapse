#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IsoScore + LeJEPA Epps-Pulley on hidden representations, measured on the PRETEXT TASK
distribution (e.g. all 4 rotations with their rotation labels), for every checkpoint of
a NetworkInNetwork-style model.

Inputs come from the project's own GenericDataset + DataLoader (dataloader.py), in
unsupervised mode, so images, rotations and normalization match training exactly.

Views per layer (labels = pretext classes, e.g. rot0/rot90/rot180/rot270):
  pooled     all pretext classes together
  <class>    each pretext class separately
  within     each sample minus its class mean, pooled over classes

IsoScore (Rudman et al. 2022), compared with the IsoScore of true N(0, I) at the same
N and d. Identity: IsoScore = (PR - 1)/(d - 1), PR = participation ratio.

Epps-Pulley: LeJEPA's implementation, used the way the LeJEPA README uses it
(SlicingUnivariateTest(EppsPulley(...), num_slices=1024), reduction='mean'):
  - features are projected on num_slices random unit directions (no whitening),
  - each projection is compared with EXACTLY N(0, 1) (no centering/standardization),
  - T = N * int |phi_N(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt  (trapezoid, t in [0, t_max]),
  - reported value = mean of T over slices.
  This is a statistic (LeJEPA uses it as a loss), not a hypothesis test: no p-value.
  It measures distance from N(0, I) jointly in mean, scale, isotropy and shape, and grows
  with N for non-Gaussian data, so compare values only at the same --ep-n.
  ep_lejepa_gaussian_ref = the same computation on true N(0, I) data of the same shape.
  ep_shape = the same statistic with each 1-D projection centered and scaled to unit variance
  first: it compares only the SHAPE of each slice with N(0, 1), so activation scale and mean
  (e.g. from different weight decay) do not enter. ep_shape_gaussian_ref is its N(0, I) value.

Usage (from the repo root, like the rank script)
-----
  python measure_iso_gauss.py --exp <cfg> --exp-dir <dir> --last-only
  python measure_iso_gauss.py --exp <cfg> --exp-dir <dir> --epochs 1,50,100,last
  python measure_iso_gauss.py --exp <cfg> --exp-dir <dir> --list-modules
  python measure_iso_gauss.py --exp <cfg> --exp-dir <dir> --hook-modules "<relu1>,..." --hook-input
"""

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

DEFAULT_LAYERS = ("conv2.block0,conv2.block1,conv2.block2,conv3.block0,conv3.block1,conv3.block2,"
                  "conv3.block3,conv4.block0,conv4.block1,conv4.block2,conv4.block3,conv4.block4,"
                  "conv4.block5,conv5.block0,conv5.block1,conv5.block2,lin1,lin2,classifier")
ROT_NAMES = {0: "rot0", 1: "rot90", 2: "rot180", 3: "rot270"}


# =============================================================================
# IsoScore (unchanged)
# =============================================================================

def isoscore_from_cov(C):
    """IsoScore (Rudman et al. 2022, Alg. 1). Steps 2-3 == eigenvalues of C."""
    d = C.shape[0]
    if d < 2:
        return float("nan")
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    if lam.sum() <= 0:
        return float("nan")
    sig = np.sqrt(d) * lam / np.linalg.norm(lam)                          # step 4
    delta = np.linalg.norm(sig - 1) / np.sqrt(2 * (d - np.sqrt(d)))       # step 5
    phi = (d - delta ** 2 * (d - np.sqrt(d))) ** 2 / d ** 2               # step 6
    return float((d * phi - 1) / (d - 1))                                 # step 7


def spectrum_stats(C):
    d = C.shape[0]
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    tot = lam.sum()
    if tot <= 0:
        return dict(eff_rank_frac=float("nan"), pr_frac=float("nan"), top1_frac=float("nan"))
    p = lam / tot
    pz = p[p > 0]
    return dict(eff_rank_frac=float(np.exp(-(pz * np.log(pz)).sum())) / d,
                pr_frac=float(tot ** 2 / (lam ** 2).sum()) / d,
                top1_frac=float(p[0]))


class IsoNullCache:
    """IsoScore of true N(0, I_d) at the same N: the realistic 'perfect' value."""

    def __init__(self, seed, reps=3):
        self.seed, self.reps, self.store = seed, reps, {}

    def get(self, n_points, d, chunk=20000):
        if (n_points, d) not in self.store:
            rng = np.random.default_rng(self.seed + d)
            vals = []
            for _ in range(self.reps):
                s, ss, left = np.zeros(d), np.zeros((d, d)), n_points
                while left > 0:
                    m = min(chunk, left)
                    Z = rng.standard_normal((m, d))
                    s += Z.sum(0)
                    ss += Z.T @ Z
                    left -= m
                mu = s / n_points
                vals.append(isoscore_from_cov((ss - n_points * np.outer(mu, mu)) / (n_points - 1)))
            self.store[(n_points, d)] = float(np.mean(vals))
        return self.store[(n_points, d)]


# =============================================================================
# Epps-Pulley: LeJEPA implementation
#   source: github.com/galilai-group/lejepa @ c293d291ca87cd4fddee9d3fffe4e914c7272052
#     lejepa/univariate/epps_pulley.py  (class EppsPulley)
#     lejepa/multivariate/slicing.py    (class SlicingUnivariateTest)
#   Adaptations (math unchanged):
#     - distributed all_reduce removed (single process: identity, world_size = 1)
#     - slicing seed passed explicitly instead of the internal global_step counter,
#       so every layer/view/network with the same d uses the same directions
# =============================================================================

class EppsPulley(torch.nn.Module):
    """T = N * int |phi_empirical(t) - phi_normal(t)|^2 w(t) dt against standard N(0, 1).
    Input (*, N, K): N samples, K slices. Output (*, K)."""

    def __init__(self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)
        cos_mean = cos_vals.mean(-3)  # (*, K, n_points)
        sin_mean = sin_vals.mean(-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N  # world_size = 1


class SlicingUnivariateTest(torch.nn.Module):
    """Project (*, N, D) samples on num_slices random unit directions, apply a univariate test."""

    def __init__(self, univariate_test, num_slices: int, reduction: str = "mean", clip_value: float = None):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.univariate_test = univariate_test
        self.clip_value = clip_value

    def forward(self, x, seed: int):
        with torch.no_grad():
            g = torch.Generator(device=x.device)
            g.manual_seed(seed)
            A = torch.randn((x.size(-1), self.num_slices), device=x.device, generator=g)
            A /= A.norm(p=2, dim=0)
        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats


def lejepa_ep(X, slicer, seed, device, chunk=128, standardize_slices=False):
    """Mean sliced Epps-Pulley statistic of X (n, d), computed in slice chunks to bound memory.
    standardize_slices=False: identical to LeJEPA's slicer(X), reduction='mean' (same directions).
    standardize_slices=True : each 1-D projection is centered and scaled to unit variance before
    EppsPulley, so only its SHAPE is compared with N(0, 1) (mean and scale removed)."""
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn((x.size(-1), slicer.num_slices), device=device, generator=g)
    A /= A.norm(p=2, dim=0)
    ep = slicer.univariate_test.to(device)
    def proj(s):
        P = x @ A[:, s:s + chunk]
        if standardize_slices:
            P = (P - P.mean(0)) / P.std(0).clamp_min(1e-12)
        return P
    stats = torch.cat([ep(proj(s)) for s in range(0, A.shape[1], chunk)])
    if slicer.clip_value is not None:
        stats[stats < slicer.clip_value] = 0
    return float(stats.mean())


# =============================================================================
# Model + data plumbing (mirrors the rank script; data from the project's dataloader.py)
# =============================================================================

def build_fresh_model(config, net_key, arch_class):
    net_cfg_all = config.get("networks", {})
    if net_key not in net_cfg_all:
        raise RuntimeError(f"net_key '{net_key}' not in config['networks']. Keys: {list(net_cfg_all.keys())}")
    net_cfg = net_cfg_all[net_key]
    def_file = Path(net_cfg["def_file"])
    if not def_file.is_file():
        raise FileNotFoundError(f"Architecture file not found: {def_file}")
    spec_model = importlib.util.spec_from_file_location(def_file.stem, def_file)
    mod_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(mod_model)
    cls_name = arch_class or net_cfg.get("arch", def_file.stem)
    if not hasattr(mod_model, cls_name):
        raise RuntimeError(f"Class '{cls_name}' not found in {def_file}")
    return getattr(mod_model, cls_name)(net_cfg.get("opt", {}).copy()).cpu()


def load_state_dict(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state
    if isinstance(state, dict):
        for k in ("state_dict", "network", "model"):
            if k in state:
                sd = state[k]
                break
    model.load_state_dict(sd, strict=True)


def discover_checkpoints(exp_dir, ckpt_glob):
    files = list(exp_dir.glob(ckpt_glob))
    if not files:
        raise RuntimeError(f"No ckpts found in {exp_dir} matching '{ckpt_glob}'")
    out = {}
    for f in files:
        if "epoch" not in f.stem:
            continue
        try:
            out[int(f.stem.split("epoch")[-1])] = f
        except ValueError:
            continue
    if not out:
        raise RuntimeError("Found checkpoint files but failed to parse epoch numbers.")
    return out


def make_batch_source(args, config):
    """Returns (factory, info). factory() yields (x, y, n_imgs) per batch, where x holds
    R pretext views of each of n_imgs images, ordered image-major like the project collate."""

    if args.dataset_name == "synthetic":                       # smoke tests only
        g = torch.Generator().manual_seed(args.seed)
        imgs = torch.randn(args.num_images, 3, args.img_size, args.img_size, generator=g)

        def factory():
            for s in range(0, len(imgs), args.batch_size):
                b = imgs[s:s + args.batch_size]
                x = torch.stack([torch.rot90(b, k, (2, 3)) for k in range(4)], 1)
                yield x.flatten(0, 1), torch.arange(4).repeat(len(b)), len(b)
        return factory, dict(dataset_name="synthetic", split="-", pretext_mode="rotation",
                             n_images=len(imgs), class_names=ROT_NAMES)

    sys.path.insert(0, os.getcwd())
    from dataloader import GenericDataset, DataLoader          # the project's dataloader.py

    # dataset options: config['data_test_opt'], then command-line overrides
    opt = dict(config.get("data_test_opt") or {})
    for key, val in (("dataset_name", args.dataset_name), ("split", args.split),
                     ("pretext_mode", args.pretext_mode)):
        if val is not None:
            opt[key] = val
    opt.setdefault("split", "test")
    opt.setdefault("pretext_mode", "rotation")
    if "dataset_name" not in opt:
        raise RuntimeError("No dataset_name in config['data_test_opt']; pass --dataset-name.")
    if args.sigmas is not None:
        opt["sigmas"] = args.sigmas
    if args.kernel_sizes is not None:
        opt["kernel_sizes"] = args.kernel_sizes
    if args.patch_jitter is not None:
        opt["patch_jitter"] = args.patch_jitter
    # dataloader.py stores sigmas/kernel_sizes=None by default and DataLoader then crashes on
    # list(None); fill the DataLoader's own fallback values, but they must match training.
    if opt["pretext_mode"] == "gaussian_noise" and opt.get("sigmas") is None:
        opt["sigmas"] = [1e-3, 1e-2, 1e-1, 1.0]
        warnings.warn(f"No sigmas in config/CLI; using {opt['sigmas']}. Must match training (--sigmas).")
    if opt["pretext_mode"] == "gaussian_blur" and opt.get("kernel_sizes") is None:
        opt["kernel_sizes"] = [3, 5, 7, 9]
        warnings.warn(f"No kernel_sizes in config/CLI; using {opt['kernel_sizes']}. Must match training (--kernel-sizes).")
    if opt["split"].lower() == "train":
        warnings.warn("split=train applies training augmentation (random crop/flip) to the measured inputs.")
    accepted = set(inspect.signature(GenericDataset.__init__).parameters) - {"self"}
    ds_kwargs = {k: v for k, v in opt.items() if k in accepted}
    ds = GenericDataset(**ds_kwargs)
    mode = ds.pretext_mode
    if mode.startswith("jigsaw") and ds_kwargs.get("fixed_perms") is None:
        warnings.warn("Jigsaw with no fixed_perms in config['data_test_opt']: permutations are regenerated "
                      "and must match the ones used in training, or labels are meaningless.")
    n = min(args.num_images, len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, unsupervised=True,
                        epoch_size=n, num_workers=args.workers, shuffle=False)

    def factory():
        np.random.seed(args.seed)                              # gaussian_noise draws from np.random
        torch.manual_seed(args.seed)
        seen = 0
        for x, y in loader(0):
            n_imgs = min(args.batch_size, n - seen)
            seen += n_imgs
            yield x, y, n_imgs

    if mode == "rotation":
        names = ROT_NAMES
    elif mode == "gaussian_noise":
        names = {i: f"noise{s:g}" for i, s in enumerate(getattr(ds, "sigmas", None) or [1e-3, 1e-2, 1e-1, 1.0])}
    elif mode == "gaussian_blur":
        names = {i: f"blurk{k}" for i, k in enumerate(getattr(ds, "kernel_sizes", None) or [3, 5, 7, 9])}
    elif mode.startswith("jigsaw"):
        names = {i: f"perm{i}" for i in range(len(ds.jigsaw_perms))}
    else:
        names = {}
    info = dict(dataset_name=ds_kwargs.get("dataset_name"), split=ds_kwargs.get("split"),
                pretext_mode=mode, n_images=n, class_names=names,
                mean_pix=getattr(ds, "mean_pix", None), std_pix=getattr(ds, "std_pix", None))
    return factory, info


class FeatureTap:
    """Feature keys via model(x, out_feat_keys=...), or forward hooks on module names."""

    def __init__(self, model, feat_keys, hook_modules, hook_input=False):
        self.model = model
        self.hook_input = hook_input
        self.keys = hook_modules or feat_keys
        self.handles, self.buf = [], {}
        if hook_modules:
            mods = dict(model.named_modules())
            for name in hook_modules:
                if name not in mods:
                    raise RuntimeError(f"Module '{name}' not found; run with --list-modules")
                self.handles.append(mods[name].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def f(_m, inp, out):
            src = inp if self.hook_input else out                  # input = pre-activation of a ReLU
            self.buf[name] = src[0] if isinstance(src, (tuple, list)) else src
        return f

    def __call__(self, x):
        if self.handles:
            self.buf = {}
            self.model(x)
            return dict(self.buf)
        out = self.model(x, out_feat_keys=list(self.keys))
        if not isinstance(out, (list, tuple)):
            out = [out]
        return dict(zip(self.keys, out))


class CovAcc:
    """Streaming sum / sum of outer products (float64)."""

    def __init__(self, d=None, device=None):
        self.n, self.s, self.ss = 0, None, None

    def add(self, x):
        if x.shape[0] == 0:
            return
        if self.s is None:
            self.s = x.new_zeros(x.shape[1])
            self.ss = x.new_zeros(x.shape[1], x.shape[1])
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += x.T @ x

    def np(self):
        return self.n, self.s.cpu().numpy(), self.ss.cpu().numpy()


def cov_from(n, s, ss):
    mu = s / n
    return mu, (ss - n * np.outer(mu, mu)) / (n - 1)


def within_cov(parts):
    """parts: list of (n, s, ss) per class -> pooled within-class covariance, total n."""
    N = sum(p[0] for p in parts)
    S = sum(p[2] - np.outer(p[1], p[1]) / p[0] for p in parts)
    return S / (N - len(parts)), N


class LayerCollector:
    def __init__(self, k, border, max_ep_imgs):
        self.k, self.border, self.max_ep_imgs = k, border, max_ep_imgs
        self.acc = {}                    # (label, parity) -> CovAcc
        self.ep_x, self.ep_y, self.ep_img = [], [], []
        self.shape = None

    def add(self, f, labels, img_ids, gen):
        f = f.detach()
        self.shape = tuple(f.shape[1:])
        M = f.shape[0]
        if f.dim() == 4:
            b = self.border
            if b and min(f.shape[2], f.shape[3]) > 2 * b + 1:
                f = f[:, :, b:-b, b:-b]
            C, HW = f.shape[1], f.shape[2] * f.shape[3]
            kk = min(self.k, HW)
            order = torch.rand(M, HW, generator=gen).argsort(1)[:, :kk].to(f.device)
            pts = torch.gather(f.flatten(2), 2, order[:, None, :].expand(-1, C, -1)).permute(0, 2, 1)
        elif f.dim() == 2:
            pts = f[:, None, :]
        else:
            raise RuntimeError(f"Unsupported feature shape {tuple(f.shape)}")
        pts = pts.double()                                        # (M, k, C)
        lab = labels.to(f.device)
        # pseudo-random image halves (index parity would align with jigsaw test labels = idx % K)
        par = ((img_ids * 2654435761 % 4294967296) // 65536 % 2).to(f.device)
        for c in torch.unique(lab).tolist():
            for p in (0, 1):
                m = (lab == c) & (par == p)
                if m.any():
                    self.acc.setdefault((c, p), CovAcc()).add(pts[m].reshape(-1, pts.shape[-1]))
        keep = img_ids < self.max_ep_imgs                          # one position per sample
        if keep.any():
            self.ep_x.append(pts[keep.to(f.device), 0, :].float().cpu())
            self.ep_y.append(labels[keep].cpu())
            self.ep_img.append(img_ids[keep].cpu())

    def classes(self):
        return sorted({c for c, _ in self.acc})

    def parts(self, classes, parities=(0, 1)):
        out = []
        for c in classes:
            accs = [self.acc[(c, p)] for p in parities if (c, p) in self.acc]
            if not accs:
                continue
            n = sum(a.n for a in accs)
            s = sum(a.s for a in accs).cpu().numpy()
            ss = sum(a.ss for a in accs).cpu().numpy()
            out.append((n, s, ss))
        return out


def view_covs(col, view, classes):
    """Return (C_full, n_points, C_half0, C_half1) for a view."""
    def build(parities):
        if view == "pooled":
            n, s, ss = [sum(x) for x in zip(*col.parts(classes, parities))]
            return cov_from(n, s, ss)[1], n
        if view == "within":
            return within_cov(col.parts(classes, parities))
        n, s, ss = col.parts([view], parities)[0]
        return cov_from(n, s, ss)[1], n
    def half(p):
        try:
            return build((p,))[0]
        except (IndexError, ValueError, ZeroDivisionError):
            return None
    C, n = build((0, 1))
    return C, n, half(0), half(1)


def make_sampler(view, X, y, img, class_means, alive):
    """Subsampler for EP. pooled/within: one random pretext class per image (independence)."""
    if view in ("pooled", "within"):
        Xv = X - class_means[y] if view == "within" else X

        def sampler(rng, n):
            perm = rng.permutation(len(img))
            _, first = np.unique(img[perm], return_index=True)    # one random row per image
            rows = perm[first]
            return Xv[rng.choice(rows, n, replace=False)][:, alive]
        n_avail = len(np.unique(img))
    else:
        rows_c = np.flatnonzero(y == view)

        def sampler(rng, n):
            return X[rng.choice(rows_c, n, replace=False)][:, alive]
        n_avail = len(rows_c)
    return sampler, n_avail


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("IsoScore + LeJEPA Epps-Pulley on the pretext-task distribution")
    p.add_argument("--exp", help="config name (config/<exp>.py)")
    p.add_argument("--exp-dir", help="directory with checkpoints")
    p.add_argument("--net-key", default="model")
    p.add_argument("--arch-class", default="NetworkInNetwork")
    p.add_argument("--layers", default=DEFAULT_LAYERS, help="feature keys passed as out_feat_keys")
    p.add_argument("--hook-modules", default=None,
                   help="Comma list of module names to hook instead of feature keys")
    p.add_argument("--hook-input", action="store_true",
                   help="record the INPUT of each hooked module instead of its output. Hook a block's final "
                        "ReLU to get its pre-activation (in a ResNet block: BN(conv) + shortcut)")
    p.add_argument("--list-modules", action="store_true")
    p.add_argument("--ckpt-glob", default="model_net_epoch*")
    p.add_argument("--epochs", default="all", help='"all" or comma list, e.g. "1,50,100,last"')
    p.add_argument("--last-only", action="store_true")
    # data: dataloader.py (GenericDataset + DataLoader), options from config['data_test_opt']
    p.add_argument("--pretext-mode", default=None,
                   choices=["rotation", "gaussian_noise", "gaussian_blur", "jigsaw", "jigsaw_9"],
                   help="override; default from config['data_test_opt'], else rotation")
    p.add_argument("--sigmas", type=float, nargs="+", default=None, help="gaussian_noise levels (override)")
    p.add_argument("--kernel-sizes", type=int, nargs="+", default=None, help="gaussian_blur kernels (override)")
    p.add_argument("--patch-jitter", type=int, default=None, help="jigsaw patch jitter (override)")
    p.add_argument("--dataset-name", default=None, help="override, e.g. cifar100 (or 'synthetic' for smoke tests)")
    p.add_argument("--split", default=None, help="override; default from config, else 'test'")
    p.add_argument("--num-images", type=int, default=10000, help="images (each gives R pretext samples)")
    p.add_argument("--batch-size", type=int, default=128, help="images per batch (x R pretext samples)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=32, help="synthetic only")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    # IsoScore
    p.add_argument("--views", default="pooled,per_class,within", help="subset of pooled,per_class,within")
    p.add_argument("--positions-per-image", type=int, default=16, help="IsoScore points per sample (conv maps)")
    p.add_argument("--border", type=int, default=1, help="Drop this many border pixels")
    # LeJEPA Epps-Pulley (defaults: EppsPulley() and the README's num_slices=1024)
    p.add_argument("--ep-n", type=int, default=2000, help="samples N per view (statistic scales with N)")
    p.add_argument("--ep-reps", type=int, default=1, help="repeat on fresh subsamples; reports mean and std")
    p.add_argument("--ep-max-images", type=int, default=5000, help="images kept in memory for EP")
    p.add_argument("--ep-slices", type=int, default=1024, help="num_slices")
    p.add_argument("--ep-t-max", type=float, default=3.0, help="EppsPulley t_max")
    p.add_argument("--ep-n-points", type=int, default=17, help="EppsPulley n_points (odd)")
    p.add_argument("--out-dir", default=None, help="Defaults to <exp_dir>/iso_gauss")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.exp or not args.exp_dir:
        raise SystemExit("--exp and --exp-dir are required")

    cfg_file = Path("config") / f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config
    exp_dir = Path(args.exp_dir)
    config["exp_dir"] = str(exp_dir)

    model = build_fresh_model(config, args.net_key, args.arch_class)
    if args.list_modules:
        for n, m in model.named_modules():
            print(f"{n:60s} {type(m).__name__}")
        return

    hook_modules = [s.strip() for s in args.hook_modules.split(",")] if args.hook_modules else None
    feat_keys = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not hook_modules:
        if not hasattr(model, "all_feat_names"):
            raise RuntimeError("Model missing all_feat_names; use --hook-modules instead.")
        for k in feat_keys:
            if k not in model.all_feat_names:
                raise RuntimeError(f"Layer key '{k}' not in model.all_feat_names: {model.all_feat_names}")

    epoch_to_path = discover_checkpoints(exp_dir, args.ckpt_glob)
    all_epochs = sorted(epoch_to_path)
    if args.last_only:
        epochs = [all_epochs[-1]]
    elif args.epochs == "all":
        epochs = all_epochs
    else:
        epochs = sorted({all_epochs[-1] if t.strip() == "last" else int(t) for t in args.epochs.split(",")})
        missing = [e for e in epochs if e not in epoch_to_path]
        if missing:
            raise RuntimeError(f"No checkpoint for epochs {missing}. Available: {all_epochs}")

    views_req = {v.strip() for v in args.views.split(",")}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    factory, info = make_batch_source(args, config)
    print(f"[data] {info}")
    class_names = info.get("class_names") or {}
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "iso_gauss"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps({**vars(args), "data": info}, indent=2, default=str))
    iso_null = IsoNullCache(args.seed)
    slicer = SlicingUnivariateTest(EppsPulley(t_max=args.ep_t_max, n_points=args.ep_n_points),
                                   num_slices=args.ep_slices, reduction="mean")
    ep_ref_cache = {}

    def ep_reference(n, d, standardize_slices=False):
        """Same statistic on true N(0, I_d) samples of size n (same slices)."""
        key = (n, d, standardize_slices)
        if key not in ep_ref_cache:
            r = np.random.default_rng(args.seed + 1000003 + d)
            vals = [lejepa_ep(r.standard_normal((n, d)), slicer, args.seed + d, device,
                              standardize_slices=standardize_slices)
                    for _ in range(max(3, args.ep_reps))]
            ep_ref_cache[key] = float(np.mean(vals))
        return ep_ref_cache[key]

    all_stats, rows = {}, []
    for ep in epochs:
        t0 = time.time()
        load_state_dict(model, epoch_to_path[ep])
        model.to(device).eval()
        tap = FeatureTap(model, feat_keys, hook_modules, hook_input=args.hook_input)
        cols = {k: LayerCollector(args.positions_per_image, args.border, args.ep_max_images) for k in tap.keys}
        gen = torch.Generator().manual_seed(args.seed)             # same positions every epoch
        img_offset = 0
        with torch.no_grad():
            for x, y, n_imgs in factory():
                R = x.shape[0] // n_imgs
                img_ids = img_offset + torch.arange(n_imgs).repeat_interleave(R)
                img_offset += n_imgs
                feats = tap(x.to(device))
                for k, f in feats.items():
                    cols[k].add(f, y.long(), img_ids, gen)
        for h in tap.handles:
            h.remove()
        print(f"[fwd] epoch {ep}: {img_offset} images x {R} pretext samples ({time.time() - t0:.0f}s)", flush=True)

        all_stats[ep] = {}
        for k, col in cols.items():
            rng = np.random.default_rng(args.seed)                 # same subsamples for every layer
            classes = col.classes()
            cname = lambda c: class_names.get(c, f"class{c}")
            if k == "classifier" and len(col.shape) == 1 and col.shape[0] != len(classes):
                warnings.warn(f"classifier has {col.shape[0]} outputs but the {info['pretext_mode']} pretext "
                              f"gives {len(classes)} classes: is --pretext-mode the task this model was trained on?")
            views = (["pooled"] if "pooled" in views_req else []) \
                + (classes if "per_class" in views_req else []) \
                + (["within"] if "within" in views_req and len(classes) > 1 else [])
            X = torch.cat(col.ep_x).numpy().astype(np.float64)
            yv = torch.cat(col.ep_y).numpy()
            img = torch.cat(col.ep_img).numpy()
            class_means = np.zeros((max(classes) + 1, X.shape[1]))
            for c in classes:                                   # class means from full stream
                n_c, s_c, _ = col.parts([c])[0]
                class_means[c] = s_c / n_c

            all_stats[ep][k] = {}
            line = []
            for v in views:
                vname = v if isinstance(v, str) else cname(v)
                # ---- IsoScore (unchanged) ----
                C, n_pts, Ch0, Ch1 = view_covs(col, v, classes)
                d = C.shape[0]
                var = np.diag(C)
                alive = var > max(1e-12, 1e-8 * var.max())
                r = dict(epoch=ep, layer=k, view=vname, shape=list(col.shape), d=d,
                         n_dead=int((~alive).sum()), n_iso_points=int(n_pts))
                if n_pts <= 5 * d:
                    warnings.warn(f"{k}/{vname}: {n_pts} IsoScore points for d={d}; IsoScore biased low.")
                r["isoscore"] = isoscore_from_cov(C)
                r["isoscore_alive"] = isoscore_from_cov(C[np.ix_(alive, alive)])
                r["isoscore_null"] = iso_null.get(int(n_pts), d)
                r["isoscore_half_a"] = isoscore_from_cov(Ch0) if Ch0 is not None else float("nan")
                r["isoscore_half_b"] = isoscore_from_cov(Ch1) if Ch1 is not None else float("nan")
                r["participation_ratio"] = r["isoscore"] * (d - 1) + 1     # exact identity
                r.update(spectrum_stats(C))

                # ---- LeJEPA sliced Epps-Pulley on the raw features (all channels) ----
                sampler, n_avail = make_sampler(v, X, yv, img, class_means, np.ones(d, bool))
                n_ep = min(args.ep_n, n_avail)
                if n_ep < args.ep_n:
                    warnings.warn(f"{k}/{vname}: only {n_ep} samples for EP (asked {args.ep_n}); "
                                  f"the statistic scales with N, so this row is not comparable.")
                samples = [sampler(rng, n_ep) for _ in range(args.ep_reps)]
                vals = [lejepa_ep(S, slicer, args.seed + d, device) for S in samples]
                vals_shape = [lejepa_ep(S, slicer, args.seed + d, device, standardize_slices=True) for S in samples]
                r["ep_n"] = n_ep
                r["ep_slices"] = args.ep_slices
                # LeJEPA as-is: distance from exactly N(0, I) (mean + scale + isotropy + shape)
                r["ep_lejepa"] = float(np.mean(vals))
                r["ep_lejepa_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                r["ep_lejepa_gaussian_ref"] = ep_reference(n_ep, d)
                # same statistic, each slice standardized first: shape only
                r["ep_shape"] = float(np.mean(vals_shape))
                r["ep_shape_std"] = float(np.std(vals_shape, ddof=1)) if len(vals_shape) > 1 else float("nan")
                r["ep_shape_gaussian_ref"] = ep_reference(n_ep, d, standardize_slices=True)

                all_stats[ep][k][vname] = r
                rows.append(r)
                line.append(f"{vname}: Iso={r['isoscore']:.3f}/{r['isoscore_null']:.3f} "
                            f"EP={r['ep_lejepa']:.4g}/{r['ep_lejepa_gaussian_ref']:.3g} "
                            f"EPshape={r['ep_shape']:.4g}/{r['ep_shape_gaussian_ref']:.3g}")
            print(f"  ep{ep} {k:16s} d={col.shape[0]:4d} | " + " | ".join(line), flush=True)
        print(f"[done] epoch {ep} ({time.time() - t0:.0f}s)", flush=True)

        keys = list(dict.fromkeys(kk for row in rows for kk in row))
        with open(out_dir / "iso_gauss_over_epochs.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        (out_dir / "iso_gauss_over_epochs.json").write_text(json.dumps(
            {str(e): v for e, v in all_stats.items()}, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()