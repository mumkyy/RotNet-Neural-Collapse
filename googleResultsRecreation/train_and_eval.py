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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(num_gpus=1, use_tpu=False):
    del use_tpu
    if torch.cuda.is_available() and num_gpus > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def build_data_cfg(args):
    return SimpleNamespace(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        preprocessing=args.preprocessing,
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


def build_model(args):
    # For jigsaw-style configs, embed_dim is the classifier output size.
    if args.task in ("jigsaw", "relative_patch_location", "rotation", "exemplar"):
        num_classes = args.embed_dim if args.embed_dim is not None else 1000
    else:
        num_classes = datasets.get_num_classes(build_data_cfg(args))

    model_fn = get_net(args, num_classes=num_classes)

    # Compatibility layer:
    # models/utils.py may pass kwargs that models/pytorch_resnet.py ignores.
    sig = inspect.signature(model_fn.func)
    accepted = set(sig.parameters.keys())

    merged_kwargs = {}
    merged_kwargs.update(model_fn.keywords or {})

    filtered_kwargs = {k: v for k, v in merged_kwargs.items() if k in accepted}
    model = model_fn.func(*model_fn.args, **filtered_kwargs)
    return model


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


def get_milestones(args):
    if args.decay_epochs is None:
        return []
    return list(args.decay_epochs)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


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


def list_checkpoints(workdir):
    ckpt_dir = Path(workdir)
    if not ckpt_dir.exists():
        return []
    return sorted(
        [p for p in ckpt_dir.glob("checkpoint_step_*.pt")],
        key=lambda p: p.stat().st_mtime,
    )


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


def export_torchscript(model, args, export_dir, device):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    shape = args.serving_input_shape or (None, 64, 64, 3)
    if len(shape) != 4:
        raise ValueError(f"Expected serving_input_shape with 4 dims, got {shape}")

    _, h, w, c = shape
    if h is None or w is None or c is None:
        h, w, c = 64, 64, 3

    model_to_export = model.module if isinstance(model, nn.DataParallel) else model
    model_to_export.eval()

    dummy = torch.randn(1, c, h, w, device=device)
    traced = torch.jit.trace(model_to_export, dummy)
    out_path = export_dir / "model.ts"
    traced.save(str(out_path))
    return str(out_path)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def infer_targets(batch, args, device):
    images = batch["image"]

    # Generic supervised path
    if args.task not in ("rotation",):
        if "label" not in batch:
            raise ValueError(
                "Batch does not contain 'label'. Current training loop expects "
                "dataset batches shaped like {'image': ..., 'label': ...}."
            )
        labels = batch["label"].long().to(device)
        return images, labels

    # Rotation path
    if images.ndim == 5:
        # [B, 4, C, H, W] -> [B*4, C, H, W], labels 0..3 repeated B times
        b, r, c, h, w = images.shape
        if r != 4:
            raise ValueError(f"Expected 4 rotations, got shape {tuple(images.shape)}")
        labels = torch.arange(4, device=device).repeat(b)
        images = images.view(b * 4, c, h, w)
        return images, labels

    if "label" not in batch:
        raise ValueError("Rotation task needs either stacked rotations or labels.")
    labels = batch["label"].long().to(device)
    return images, labels


def flatten_patch_batch_if_needed(images, labels=None):
    # If preprocessing produced [B, N, C, H, W], flatten to [B*N, C, H, W].
    # This is useful for patch-wise classifiers or debugging, but only if labels match.
    if images.ndim == 5:
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)

        if labels is not None:
            if labels.ndim == 2 and labels.shape == (b, n):
                labels = labels.reshape(-1)
            elif labels.ndim == 1 and labels.numel() == b:
                # per-image label; leave as-is
                pass

    return images, labels


@torch.no_grad()
def evaluate(model, data_loader, device, args, checkpoint_path=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        images, labels = infer_targets(batch, args, device)
        images, labels = flatten_patch_batch_if_needed(images, labels)

        logits = model(images)

        # If logits are per patch but labels are per image, that mismatch should fail loudly.
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Label / logits batch mismatch during eval. "
                f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
            )

        loss = criterion(logits, labels)
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

            batch = move_batch_to_device(batch, device)
            images, labels = infer_targets(batch, args, device)
            images, labels = flatten_patch_batch_if_needed(images, labels)

            model.train()
            optimizer.zero_grad()

            logits = model(images)

            if labels.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"Label / logits batch mismatch during train. "
                    f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}. "
                    f"If you are using crop_patches for jigsaw, you still need "
                    f"task-specific permutation targets/head logic."
                )

            loss = criterion(logits, labels)
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

    export_dir = os.path.join(args.workdir, "export", "torchscript")
    exported = export_torchscript(model, args, export_dir, device)
    print(f"Exported TorchScript model to: {exported}")


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

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    # Pretext-task flags.
    parser.add_argument("--embed_dim", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--num_of_inception_patches", type=int, default=None)
    parser.add_argument("--patch_jitter", type=int, default=0)
    parser.add_argument("--perm_subset_size", type=int, default=8)
    parser.add_argument("--splits_per_side", type=int, default=3)

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