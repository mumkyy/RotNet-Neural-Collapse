#!/usr/bin/env python3
# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division

import argparse
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from models.utils import get_net


TPU_ITERATIONS_PER_LOOP = 500


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_int_or_pair(value):
    if value is None:
        return None
    parts = str(value).split(",")
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise argparse.ArgumentTypeError(f"Expected int or pair, got: {value}")


def parse_int_tuple(value):
    if value is None or value == "":
        return None
    return tuple(int(x) for x in str(value).split(","))


def parse_shape(value):
    if value is None:
        return None
    out = []
    for x in str(value).split(","):
        x = x.strip()
        out.append(None if x == "None" else int(x))
    return tuple(out)


def set_random_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(num_gpus=1, use_tpu=False):
    del use_tpu
    if torch.cuda.is_available() and num_gpus > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def build_data_cfg(args):
    preprocessing = args.preprocessing
    if getattr(args, "task", None) == "downstream" and args.downstream_preprocessing is not None:
        preprocessing = args.downstream_preprocessing

    return SimpleNamespace(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        preprocessing=preprocessing,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        resize_size=args.resize_size if args.resize_size is not None else (292, 292),
        crop_size=args.crop_size if args.crop_size is not None else 255,
        grayscale_probability=args.grayscale_probability if args.grayscale_probability is not None else 1.0,
        splits_per_side=args.splits_per_side if args.splits_per_side is not None else 3,
        patch_jitter=args.patch_jitter if args.patch_jitter is not None else 0,
        smaller_size=args.smaller_size if args.smaller_size is not None else 256,
    )


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_milestones(args):
    if args.decay_epochs is None:
        return []
    return list(args.decay_epochs)


def maybe_adjust_learning_rate(optimizer, base_lr, epoch_float, args):
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else 0
    lr_decay_factor = args.lr_decay_factor if args.lr_decay_factor is not None else 0.1
    milestones = get_milestones(args)

    lr = base_lr

    if warmup_epochs > 0 and epoch_float < warmup_epochs:
        lr = base_lr * (epoch_float / warmup_epochs)

    for milestone in milestones:
        if epoch_float >= milestone:
            lr *= lr_decay_factor

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def list_checkpoints(workdir):
    ckpt_dir = Path(workdir)
    if not ckpt_dir.exists():
        return []
    return sorted(
        [p for p in ckpt_dir.glob("checkpoint_step_*.pt")] + [p for p in ckpt_dir.glob("checkpoint_last.pt")],
        key=lambda p: p.stat().st_mtime,
    )


def save_checkpoint(workdir, step, epoch, model, optimizer, best_metric=None, name=None):
    ckpt_dir = Path(workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = name or f"checkpoint_step_{step}.pt"
    path = ckpt_dir / filename

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
    }
    torch.save(payload, path)
    return str(path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)

    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = payload.get("step", 0)
    epoch = payload.get("epoch", 0.0)
    best_metric = payload.get("best_metric", None)
    return step, epoch, best_metric


def compute_patch_size(crop_size, splits_per_side, patch_jitter):
    if isinstance(crop_size, tuple):
        crop_h, crop_w = crop_size
        if crop_h != crop_w:
            raise ValueError("This export helper expects square crop_size for jigsaw.")
        crop_size = crop_h
    grid = crop_size // splits_per_side
    patch_size = grid - patch_jitter
    if patch_size <= 0:
        raise ValueError(
            f"Invalid jigsaw patch size. crop_size={crop_size}, "
            f"splits_per_side={splits_per_side}, patch_jitter={patch_jitter}"
        )
    return patch_size


def export_torchscript(model, args, export_dir, device):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model_to_export = model.module if isinstance(model, nn.DataParallel) else model
    model_to_export.eval()

    if args.task == "jigsaw":
        patch_count = args.splits_per_side * args.splits_per_side
        patch_size = compute_patch_size(args.crop_size, args.splits_per_side, args.patch_jitter)
        dummy = torch.randn(1, patch_count, 3, patch_size, patch_size, device=device)
    else:
        shape = args.serving_input_shape or (None, 64, 64, 3)
        if len(shape) != 4:
            raise ValueError(f"Expected serving_input_shape with 4 dims, got {shape}")
        _, h, w, c = shape
        if h is None or w is None or c is None:
            h, w, c = 64, 64, 3
        dummy = torch.randn(1, c, h, w, device=device)

    traced = torch.jit.trace(model_to_export, dummy)
    out_path = export_dir / "model.ts"
    traced.save(str(out_path))
    return str(out_path)


def get_optimizer(args, model):
    base_lr = args.lr * (args.batch_size / args.lr_scale_batch_size)

    if (args.optimizer or "sgd").lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=base_lr,
            weight_decay=args.weight_decay if args.weight_decay is not None else 1e-4,
        )

    return optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay if args.weight_decay is not None else 1e-4,
    )


def load_permutations(path):
    if path is None:
        raise ValueError(
            "Jigsaw requires --permutations_path pointing to permutations_100_max.bin "
            "(or another compatible permutation bank)."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Permutation file not found: {path}")

    data = np.fromfile(str(path), dtype=np.int32)
    if data.size < 2:
        raise ValueError(f"Invalid permutation file: {path}")

    num_perms = int(data[0])
    perm_len = int(data[1])
    expected = 2 + num_perms * perm_len

    if data.size != expected:
        raise ValueError(
            f"Invalid permutation file size. Expected {expected} int32 values, got {data.size}."
        )

    perms = data[2:].reshape(num_perms, perm_len)
    perms = perms - 1  # Google file is 1-indexed
    return torch.tensor(perms, dtype=torch.long)

import models.jigsaw_head as jighead

# class JigsawHead(nn.Module):
#     """
#     Faithful-enough PyTorch head for the Google jigsaw path:
#     concat permuted patch embeddings along channels, then classify permutation.
#     """

#     def __init__(self, in_channels, num_classes, hidden_dim=4096, dropout=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(hidden_dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
#         self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         x = x.mean(dim=(2, 3))
#         return x

import models.linearJigsaw_head as linjighead

# class JigsawHeadLinear(nn.Module):
#     def __init__(self, in_dim, num_classes):
#         super().__init__()

#         self.lin1 = nn.Linear(in_dim, in_dim)
#         self.bn1 = nn.BatchNorm1d(in_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.lin2 = nn.Linear(in_dim, num_classes)

#     def forward(self, x):
#         x = self.lin1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.lin2(x)
#         return x 
    
import models.linearJigsaw_head_Deeper as linjigDeep

def permute_and_concat_batch_patches(patch_embeddings, perms):
    """
    patch_embeddings: [B, P, C, H, W]
    perms: [M, P]

    Returns:
        [B*M, P*C, H, W]
    """
    if patch_embeddings.ndim != 5:
        raise ValueError(f"Expected patch embeddings with shape [B,P,C,H,W], got {tuple(patch_embeddings.shape)}")
    if perms.ndim != 2:
        raise ValueError(f"Expected perms with shape [M,P], got {tuple(perms.shape)}")

    bsz, patch_count, channels, height, width = patch_embeddings.shape
    subset_size, perm_len = perms.shape

    if patch_count != perm_len:
        raise ValueError(
            f"Patch count / permutation length mismatch: patches={patch_count}, perm_len={perm_len}"
        )

    expanded = patch_embeddings.unsqueeze(1).expand(
        bsz, subset_size, patch_count, channels, height, width
    )
    gather_index = perms.view(1, subset_size, perm_len, 1, 1, 1).expand(
        bsz, subset_size, perm_len, channels, height, width
    )

    permuted = torch.gather(expanded, dim=2, index=gather_index)
    concat = permuted.reshape(bsz * subset_size, patch_count * channels, height, width)
    return concat


def permute_and_concat_linear_features(patch_embeddings, perms):
    """
    patch_embeddings: [B, P, C]
    perms: [M, P]

    Returns:
        [B*M, P*C]
    """
    if patch_embeddings.ndim != 3:
        raise ValueError(f"Expected [B,P,C], got {tuple(patch_embeddings.shape)}")
    if perms.ndim != 2:
        raise ValueError(f"Expected [M,P], got {tuple(perms.shape)}")

    bsz, patch_count, channels = patch_embeddings.shape
    subset_size, perm_len = perms.shape

    if patch_count != perm_len:
        raise ValueError(
            f"Patch count / permutation length mismatch: patches={patch_count}, perm_len={perm_len}"
        )

    expanded = patch_embeddings.unsqueeze(1).expand(
        bsz, subset_size, patch_count, channels
    )
    gather_index = perms.view(1, subset_size, perm_len, 1).expand(
        bsz, subset_size, perm_len, channels
    )

    permuted = torch.gather(expanded, dim=2, index=gather_index)
    concat = permuted.reshape(bsz * subset_size, patch_count * channels)
    return concat


class JigsawModel(nn.Module):
    def __init__(self, backbone, embed_dim, permutations, perm_subset_size, linHead, linheadDeep_flag : bool):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("permutations", permutations)
        self.perm_subset_size = perm_subset_size
        self.linHead = linHead
        self.linheadDeep_flag = linheadDeep_flag
        total_perm_count, perm_len = permutations.shape
        subset_size = min(perm_subset_size, total_perm_count)
        self.perm_len = perm_len
        self.embed_dim = embed_dim

        fixed_perm_indices = torch.arange(subset_size, dtype=torch.long)
        fixed_perms = permutations[fixed_perm_indices]

        self.register_buffer("fixed_perm_indices", fixed_perm_indices)
        self.register_buffer("fixed_permutations", fixed_perms)

        dummy = torch.randn(perm_len, 3, 64, 64)
        with torch.no_grad():
            feats = self.backbone(dummy)

        if feats.ndim != 4:
            raise ValueError(
                f"Expected backbone to return [P,C,H,W] spatial embeddings for jigsaw head setup, got {tuple(feats.shape)}"
            )

        feat_c = feats.shape[1]

        if self.linHead:
            if self.linheadDeep_flag:
                self.head = linjigDeep.JigsawHeadLinear(
                    in_dim = feat_c * perm_len, 
                    num_classes=subset_size,
                )
            else:
                self.head = linjighead.JigsawHeadLinear(
                in_dim=feat_c * perm_len,
                num_classes=subset_size,
                )
        else:
            self.head = jighead.JigsawHead(
                in_channels=feat_c * perm_len,
                num_classes=subset_size,
            )

    def _select_permutation_subset(self, training, device):
        total = self.permutations.shape[0]
        subset = min(self.perm_subset_size, total)

        if training:
            idx = torch.randperm(total, device=device)[:subset]
        else:
            idx = torch.arange(subset, device=device)

        perms = self.permutations[idx].to(device)
        return idx, perms

    def forward(self, x):
        """
        x: [B, P, C, H, W]

        Returns:
            {
              "logits": [B*M, num_classes],
              "labels": [B*M],
              "perm_indices": [M]
            }
        """
        if x.ndim != 5:
            raise ValueError(f"JigsawModel expected input [B,P,C,H,W], got {tuple(x.shape)}")

        bsz, patch_count, channels, height, width = x.shape

        if patch_count != self.perm_len:
            raise ValueError(
                f"Expected {self.perm_len} patches per image from permutation bank, got {patch_count}"
            )

        flat = x.reshape(bsz * patch_count, channels, height, width)

        feats = self.backbone(flat)
        if feats.ndim != 4:
            raise ValueError(
                f"Expected backbone to return [B*P,C,H,W] spatial embeddings for jigsaw, got {tuple(feats.shape)}"
            )

        _, feat_c, feat_h, feat_w = feats.shape

        perm_indices = self.fixed_perm_indices.to(x.device)
        selected_perms = self.fixed_permutations.to(x.device)

        if self.linHead:
            # [B*P, C, H, W] -> [B*P, C]
            feats = feats.mean(dim=(2, 3))
            # [B*P, C] -> [B, P, C]
            feats = feats.reshape(bsz, patch_count, feat_c)
            # [B, P, C] + [M, P] -> [B*M, P*C]
            concat_feats = permute_and_concat_linear_features(feats, selected_perms)
        else:
            # [B*P, C, H, W] -> [B, P, C, H, W]
            feats = feats.reshape(bsz, patch_count, feat_c, feat_h, feat_w)
            # [B, P, C, H, W] + [M, P] -> [B*M, P*C, H, W]
            concat_feats = permute_and_concat_batch_patches(feats, selected_perms)

        logits = self.head(concat_feats)

        m = selected_perms.shape[0]
        labels = torch.arange(m, device=x.device).repeat(bsz)

        return {
            "logits": logits,
            "labels": labels,
            "perm_indices": perm_indices,
        }

def build_model(args):
    def _build_from_model_fn(model_fn, extra_kwargs=None):
        """
        Build the model from functools.partial returned by get_net(...),
        but filter kwargs against the actual ResNet constructor, not the
        thin resnet50(...) wrapper.
        """
        merged_kwargs = {}
        merged_kwargs.update(model_fn.keywords or {})
        if extra_kwargs:
            merged_kwargs.update(extra_kwargs)

        target_ctor = model_fn.func

        if getattr(target_ctor, "__name__", "") == "resnet50":
            from models.pytorch_resnet import ResNet
            accepted = set(inspect.signature(ResNet.__init__).parameters.keys())
            accepted.discard("self")
        else:
            accepted = set(inspect.signature(target_ctor).parameters.keys())

        filtered_kwargs = {k: v for k, v in merged_kwargs.items() if k in accepted}
        return target_ctor(*model_fn.args, **filtered_kwargs)

    if args.task == "jigsaw":
        permutations = load_permutations(args.permutations_path)

        # For jigsaw, backbone should emit spatial embeddings, not pooled logits.
        model_fn = get_net(args, num_classes=args.embed_dim)
        backbone = _build_from_model_fn(model_fn, extra_kwargs={"global_pool": False})

        model = JigsawModel(
            backbone=backbone,
            embed_dim=args.embed_dim,
            permutations=permutations,
            perm_subset_size=args.perm_subset_size,
            linHead=args.linearJigsaw_head, 
            linheadDeep_flag=args.deepLinear_head
        )
        return model

    if args.task in ("relative_patch_location", "rotation", "exemplar"):
        num_classes = args.embed_dim if args.embed_dim is not None else 1000
    else:
        num_classes = datasets.get_num_classes(build_data_cfg(args))
        if args.task.lower() == "downstream":
            if args.load_model is None:
                raise ValueError("--load_model not present: must enter a model path to run downstream mode")
            elif args.layer_extractor is None:
                raise ValueError("--layer_extractor not present: must enter a layer to register forward hook")
            else:
                model_fn = get_net(args, num_classes=args.embed_dim if args.embed_dim is not None else 1000)
                model = _build_from_model_fn(model_fn, extra_kwargs={"global_pool": False})

                modelPath = Path(str(args.load_model))
                payload = torch.load(modelPath, map_location="cpu")
                if "model_state_dict" not in payload:
                    raise ValueError("Checkpoint missing 'model_state_dict'")
                model.load_state_dict(payload["model_state_dict"], strict=False)
                for param in model.parameters():
                    param.requires_grad = False
                def add_forward_hook():
                    def hook(module, inputs, output):
                        model._extracted_features = output
                    return hook

                target_module = dict(model.named_modules()).get(args.layer_extractor)
                if target_module is None:
                    raise ValueError(f"Layer '{args.layer_extractor}' not found in model")

                target_module.register_forward_hook(add_forward_hook())

                dummy = torch.randn(1, 3, 64, 64)
                with torch.no_grad():
                    _ = model(dummy)

                if not hasattr(model, "_extracted_features"):
                    raise ValueError(
                        f"Forward hook on layer '{args.layer_extractor}' did not capture any features"
                    )

                hooked_feats = model._extracted_features

                from models.downstream_head import DownstreamHead
                from models.downstream_head_linear import DownstreamHeadLinear

                if hooked_feats.ndim == 4:
                    feat_dim = hooked_feats.shape[1]
                    head = DownstreamHead(in_channels=feat_dim, num_classes=num_classes)
                    for param in head.parameters():
                        param.requires_grad = True
                elif hooked_feats.ndim == 2:
                    feat_dim = hooked_feats.shape[1]
                    head = DownstreamHeadLinear(in_dim=feat_dim, num_classes=num_classes)
                    for param in head.parameters():
                        param.requires_grad = True
                else:
                    raise ValueError(
                        f"Unsupported hooked feature shape {tuple(hooked_feats.shape)}. "
                        "Expected [B,C,H,W] or [B,D]."
                    )

                class DownstreamModel(nn.Module):
                    def __init__(self, backbone, head):
                        super().__init__()
                        self.backbone = backbone
                        self.head = head

                    def forward(self, x):
                        self.backbone.eval()
                        with torch.no_grad():
                            _ = self.backbone(x)
                        feats = self.backbone._extracted_features
                        return self.head(feats)

                return DownstreamModel(model, head)

    model_fn = get_net(args, num_classes=num_classes)
    model = _build_from_model_fn(model_fn)
    return model
def infer_targets(batch, args, device):
    images = batch["image"]

    if args.task == "jigsaw":
        return images, None

    if args.task not in ("rotation",):
        if "label" not in batch:
            raise ValueError(
                "Batch does not contain 'label'. Current training loop expects "
                "dataset batches shaped like {'image': ..., 'label': ...}."
            )
        labels = batch["label"].long().to(device)
        return images, labels

    if images.ndim == 5:
        b, r, c, h, w = images.shape
        if r != 4:
            raise ValueError(f"Expected 4 rotations, got shape {tuple(images.shape)}")
        labels = torch.arange(4, device=device).repeat(b)
        images = images.view(b * r, c, h, w)
        return images, labels

    if "label" not in batch:
        raise ValueError("Rotation task needs either stacked rotations or labels.")
    labels = batch["label"].long().to(device)
    return images, labels


def flatten_patch_batch_if_needed(images, labels=None):
    if images.ndim == 5:
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)

        if labels is not None:
            if labels.ndim == 2 and labels.shape == (b, n):
                labels = labels.reshape(-1)
            elif labels.ndim == 1 and labels.numel() == b:
                pass

    return images, labels


def compute_outputs_and_loss(model, batch, device, args, criterion):
    batch = move_batch_to_device(batch, device)
    images, labels = infer_targets(batch, args, device)

    if args.task == "jigsaw":
        outputs = model(images)
        logits = outputs["logits"]
        labels = outputs["labels"].long()
        loss = criterion(logits, labels)
        return logits, labels, loss

    images, labels = flatten_patch_batch_if_needed(images, labels)
    logits = model(images)

    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Label / logits batch mismatch. labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
        )

    loss = criterion(logits, labels)
    return logits, labels, loss


@torch.no_grad()
def evaluate(model, data_loader, device, args, checkpoint_path=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in data_loader:
        logits, labels, loss = compute_outputs_and_loss(model, batch, device, args, criterion)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return {
        "checkpoint": checkpoint_path,
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "num_examples": total_count,
    }


def train(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)

    base_lr = args.lr * (args.batch_size / args.lr_scale_batch_size)

    train_count = datasets.get_count(args.train_split or "train", build_data_cfg(args))
    updates_per_epoch = max(train_count // args.batch_size, 1)
    num_steps = int(math.ceil(args.epochs * updates_per_epoch))

    print(f"train_examples: {train_count}")
    print(f"updates_per_epoch: {updates_per_epoch}")
    print(f"train_steps: {num_steps}")

    global_step = 0
    best_metric = None

    save_every_secs = args.save_checkpoints_secs if args.save_checkpoints_secs is not None else 600
    last_save_time = time.time()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.workdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    done = False
    while not done:
        for batch in train_loader:
            epoch_float = global_step / max(updates_per_epoch, 1)
            maybe_adjust_learning_rate(optimizer, base_lr, epoch_float, args)

            model.train()
            optimizer.zero_grad()

            logits, labels, loss = compute_outputs_and_loss(model, batch, device, args, criterion)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean().item()

                print(
                    f"step={global_step}/{num_steps} "
                    f"epoch={epoch_float:.3f} "
                    f"lr={current_lr(optimizer):.6f} "
                    f"loss={loss.item():.4f} "
                    f"acc={acc:.4f}"
                )

            now = time.time()
            if now - last_save_time >= save_every_secs:
                save_checkpoint(args.workdir, global_step, epoch_float, model, optimizer, best_metric)
                last_save_time = now

            if global_step >= num_steps:
                done = True
                break

    final_ckpt = save_checkpoint(
        args.workdir,
        global_step,
        args.epochs,
        model,
        optimizer,
        best_metric,
        name="checkpoint_last.pt",
    )

    if val_loader is not None:
        result = evaluate(model, val_loader, device, args, checkpoint_path=final_ckpt)
        print("final_eval:", result)

    Path(os.path.join(args.workdir, "TRAINING_IS_DONE")).touch()

    if args.task != "jigsaw":
        export_dir = os.path.join(args.workdir, "export", "torchscript")
        exported = export_torchscript(model, args, export_dir, device)
        print(f"Exported TorchScript model to: {exported}")
    else:
        print("Skipping TorchScript export for jigsaw.")
        
def run_eval(model, val_loader, device, args):
    checkpoints = list_checkpoints(args.workdir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.workdir}")

    results = []
    for checkpoint in checkpoints:
        load_checkpoint(model, optimizer=None, checkpoint_path=str(checkpoint), device=device)
        result = evaluate(model, val_loader, device, args, checkpoint_path=str(checkpoint))
        print("eval:", result)
        results.append(result)

        if args.task != "jigsaw":
            export_dir = os.path.join(args.workdir, "export", "torchscript")
            export_torchscript(model, args, export_dir, device) 

        if os.path.exists(os.path.join(args.workdir, "TRAINING_IS_DONE")):
            break

    latest = checkpoints[-1]
    load_checkpoint(model, optimizer=None, checkpoint_path=str(latest), device=device)
    result = evaluate(model, val_loader, device, args, checkpoint_path=str(latest))
    print("latest_eval:", result)
    return result


def build_loaders(args):
    cfg = build_data_cfg(args)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    train_loader = None
    if not args.run_eval:
        train_loader = datasets.get_data(
            params={
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "pin_memory": not args.no_pin_memory,
            },
            split_name=args.train_split or "train",
            is_training=True,
            shuffle=True,
            num_epochs=int(math.ceil(args.epochs)),
            drop_remainder=True,
            cfg=cfg,
        )

    val_loader = datasets.get_data(
        params={
            "batch_size": eval_batch_size,
            "num_workers": args.num_workers,
            "pin_memory": not args.no_pin_memory,
        },
        split_name=args.val_split or "val",
        is_training=False,
        shuffle=False,
        num_epochs=1,
        drop_remainder=False,
        cfg=cfg,
    )

    return train_loader, val_loader


def train_and_eval(args):
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    device = get_device(num_gpus=args.num_gpus, use_tpu=args.use_tpu)
    print(f"workdir: {args.workdir}")
    print(f"device: {device}")

    set_random_seed(args.random_seed)

    model = build_model(args).to(device)

    if torch.cuda.is_available() and args.num_gpus > 1:
        available = torch.cuda.device_count()
        use_count = min(args.num_gpus, available)
        model = nn.DataParallel(model, device_ids=list(range(use_count)))

    train_loader, val_loader = build_loaders(args)

    if args.run_eval:
        return run_eval(model, val_loader, device, args)

    train(model, train_loader, val_loader, device, args)
    print("I'm done with my work, ciao!")


def get_parser():
    parser = argparse.ArgumentParser()

    # General run setup flags.
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--use_tpu", type=str2bool, default=False)
    parser.add_argument("--run_eval", type=str2bool, default=False)
    parser.add_argument("--tpu_worker_name", type=str, default="tpu_worker")

    # Detailed experiment flags.
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--keep_checkpoint_every_n_hours", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--save_checkpoints_secs", type=int, default=600)
    parser.add_argument("--serving_input_key", type=str, default="image")
    parser.add_argument("--serving_input_shape", type=parse_shape, default=(None, 64, 64, 3))
    parser.add_argument("--signature", type=str, default=None)

    parser.add_argument("--task", type=str, required=True, help="Enter 'downstream' exactly like this for downstream classification")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    # Pretext-task flags.
    parser.add_argument("--embed_dim", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--num_of_inception_patches", type=int, default=None)
    parser.add_argument("--patch_jitter", type=int, default=0)
    parser.add_argument("--perm_subset_size", type=int, default=8)
    parser.add_argument("--splits_per_side", type=int, default=3)
    parser.add_argument("--permutations_path", type=str, default=None)

    # Evaluation flags.
    parser.add_argument("--eval_model", type=str, default=None)
    parser.add_argument("--hub_module", type=str, default=None)
    parser.add_argument("--pool_mode", type=str, default=None)
    parser.add_argument("--combine_patches", type=str, default=None)

    # Model flags.

    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--filters_factor", type=int, default=4)
    parser.add_argument("--last_relu", type=str2bool, default=True)
    parser.add_argument("--mode", type=str, default="v2")
    parser.add_argument("--linearJigsaw_head", type=str2bool, default=False)
    parser.add_argument("--deepLinear_head", type=str2bool, default=False)

    #downstream flags 
    parser.add_argument("--load_model", type=str, required=False, help="Enter relative path to the model")
    parser.add_argument("--layer_extractor", type=str, required=False, help="blockX.X.convX or head.convX or classifier or head.linX")
    parser.add_argument("--downstream_preprocessing", type=str, default=None)
    # Optimization flags.
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--decay_epochs", type=parse_int_tuple, default=None)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr_scale_batch_size", type=float, required=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Pre-processing flags.
    parser.add_argument("--crop_size", type=parse_int_or_pair, default=255)
    parser.add_argument("--grayscale_probability", type=float, default=1.0)
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--randomize_resize_method", type=str2bool, default=False)
    parser.add_argument("--resize_size", type=parse_int_or_pair, default=(292, 292))
    parser.add_argument("--smaller_size", type=int, default=256)

    # PyTorch extras.
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()