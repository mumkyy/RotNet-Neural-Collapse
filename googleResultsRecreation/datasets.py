#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from preprocess import get_preprocess_fn


def _existing_dir(path):
    path = Path(path).expanduser()
    return path if path.exists() and path.is_dir() else None


def _resolve_dataset_root(dataset_name, dataset_dir):
    """
    Resolve the dataset root directory.

    For your cluster layout, valid roots for imagenette include:
      /project/amr239/gma35/RotNet-Neural-Collapse/RNC/datasets/Imagenette/imagenette2-160
      /project/amr239/gma35/RotNet-Neural-Collapse/RNC/datasets/Imagenette
      /project/amr239/gma35/RotNet-Neural-Collapse/RNC/datasets

    and we normalize them to the directory that directly contains train/ and val/.
    """
    dataset_dir = Path(os.path.expanduser(str(dataset_dir)))

    candidates = []

    if dataset_name.lower() == "imagenette":
        candidates.extend([
            dataset_dir,
            dataset_dir / "imagenette2-160",
            dataset_dir / "Imagenette" / "imagenette2-160",
            dataset_dir / "Imagenette",
            dataset_dir / "imagenette",
        ])
    else:
        candidates.append(dataset_dir)

    for candidate in candidates:
        candidate = Path(candidate)
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate

    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve dataset root for dataset='{dataset_name}' from dataset_dir='{dataset_dir}'. "
        f"Tried:\n{tried}\n"
        f"Need a directory containing train/ and val/."
    )


def _resolve_split_dir(dataset_name, dataset_dir, split_name):
    root = _resolve_dataset_root(dataset_name, dataset_dir)
    split_dir = root / split_name
    if split_dir.is_dir():
        return split_dir

    raise FileNotFoundError(
        f"Could not find split '{split_name}' under dataset root '{root}'."
    )


class WrappedImageFolder(ImageFolder):
    """
    Returns dict batches compatible with your current training code:

        {
            "image": tensor,
            "label": torch.long scalar
        }

    If preprocessing includes crop_patches, then image will already be shaped
    like [P, C, H, W] for a single sample, and DataLoader will batch it into
    [B, P, C, H, W], which is what your jigsaw path expects.
    """

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform, loader=default_loader)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class DatasetImagenette(object):
    NUM_CLASSES = 10

    def __init__(
        self,
        split_name,
        preprocess_fn,
        dataset_dir,
        num_epochs,
        shuffle,
        random_seed=None,
        drop_remainder=True,
    ):
        del num_epochs
        del random_seed

        self.dataset_name = "imagenette"
        self.split_name = split_name
        self.preprocess_fn = preprocess_fn
        self.dataset_root = _resolve_dataset_root(self.dataset_name, dataset_dir)
        self.split_dir = _resolve_split_dir(self.dataset_name, dataset_dir, split_name)
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

        self.dataset = WrappedImageFolder(
            root=str(self.split_dir),
            transform=self.preprocess_fn,
        )

        self.COUNTS = {
            "train": len(WrappedImageFolder(
                root=str(_resolve_split_dir(self.dataset_name, dataset_dir, "train")),
                transform=None,
            )),
            "val": len(WrappedImageFolder(
                root=str(_resolve_split_dir(self.dataset_name, dataset_dir, "val")),
                transform=None,
            )),
        }

        self.NUM_CLASSES = len(self.dataset.classes)

    def input_fn(self, params):
        batch_size = params["batch_size"]
        num_workers = params.get("num_workers", 0)
        pin_memory = params.get("pin_memory", False)

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=self.drop_remainder,
        )


class DatasetImagenet(object):
    """
    Generic ImageFolder-style ImageNet support.
    Expects dataset_dir to resolve to a directory containing train/ and val/.
    """

    NUM_CLASSES = 1000

    def __init__(
        self,
        split_name,
        preprocess_fn,
        dataset_dir,
        num_epochs,
        shuffle,
        random_seed=None,
        drop_remainder=True,
    ):
        del num_epochs
        del random_seed

        self.dataset_name = "imagenet"
        self.split_name = split_name
        self.preprocess_fn = preprocess_fn
        self.dataset_root = _resolve_dataset_root(self.dataset_name, dataset_dir)
        self.split_dir = _resolve_split_dir(self.dataset_name, dataset_dir, split_name)
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

        self.dataset = WrappedImageFolder(
            root=str(self.split_dir),
            transform=self.preprocess_fn,
        )

        self.NUM_CLASSES = len(self.dataset.classes)

    def input_fn(self, params):
        batch_size = params["batch_size"]
        num_workers = params.get("num_workers", 0)
        pin_memory = params.get("pin_memory", False)

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=self.drop_remainder,
        )


DATASET_MAP = {
    "imagenette": DatasetImagenette,
    "imagenet": DatasetImagenet,
}


def get_data(
    params,
    split_name,
    is_training,
    shuffle=True,
    num_epochs=None,
    drop_remainder=False,
    cfg=None,
):
    """
    PyTorch equivalent of the TF get_data(...).

    Returns:
        torch.utils.data.DataLoader
    """
    if cfg is None:
        raise ValueError("get_data(...) requires cfg in the PyTorch version.")

    dataset_name = str(cfg.dataset).lower()
    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unsupported dataset: {cfg.dataset}. "
            f"Supported datasets: {list(DATASET_MAP.keys())}"
        )

    preprocess_fn = get_preprocess_fn(
        fn_names=cfg.preprocessing,
        is_training=is_training,
        resize_size=cfg.resize_size,
        crop_size=cfg.crop_size,
        grayscale_probability=cfg.grayscale_probability,
        splits_per_side=cfg.splits_per_side,
        patch_jitter=cfg.patch_jitter,
        smaller_size=cfg.smaller_size,
    )

    dataset_cls = DATASET_MAP[dataset_name]
    dataset_obj = dataset_cls(
        split_name=split_name,
        preprocess_fn=preprocess_fn,
        dataset_dir=cfg.dataset_dir,
        num_epochs=num_epochs,
        shuffle=shuffle,
        random_seed=getattr(cfg, "random_seed", None),
        drop_remainder=drop_remainder,
    )

    return dataset_obj.input_fn(params)


def get_count(split_name, cfg=None):
    if cfg is None:
        raise ValueError("get_count(...) requires cfg in the PyTorch version.")

    dataset_name = str(cfg.dataset).lower()
    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unsupported dataset: {cfg.dataset}. "
            f"Supported datasets: {list(DATASET_MAP.keys())}"
        )

    if dataset_name == "imagenette":
        root = _resolve_split_dir("imagenette", cfg.dataset_dir, split_name)
        return len(ImageFolder(root=str(root)))

    if dataset_name == "imagenet":
        root = _resolve_split_dir("imagenet", cfg.dataset_dir, split_name)
        return len(ImageFolder(root=str(root)))

    raise ValueError(f"Unsupported dataset: {cfg.dataset}")


def get_num_classes(cfg=None):
    if cfg is None:
        raise ValueError("get_num_classes(...) requires cfg in the PyTorch version.")

    dataset_name = str(cfg.dataset).lower()
    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unsupported dataset: {cfg.dataset}. "
            f"Supported datasets: {list(DATASET_MAP.keys())}"
        )

    if dataset_name == "imagenette":
        root = _resolve_split_dir("imagenette", cfg.dataset_dir, "train")
        return len(ImageFolder(root=str(root)).classes)

    if dataset_name == "imagenet":
        root = _resolve_split_dir("imagenet", cfg.dataset_dir, "train")
        return len(ImageFolder(root=str(root)).classes)

    raise ValueError(f"Unsupported dataset: {cfg.dataset}")