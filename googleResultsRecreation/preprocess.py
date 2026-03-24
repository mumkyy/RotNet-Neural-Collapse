#!/usr/bin/env python3

"""Preprocessing methods (PyTorch version)."""

import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

import patch_model_preprocess as pp_lib


def get_resize_preprocess(size):
    return transforms.Resize(size)


def get_resize_small(smaller_size):
    return transforms.Resize(smaller_size)


def get_crop(is_training, crop_size):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    if is_training:
        return transforms.RandomCrop(crop_size)
    return transforms.CenterCrop(crop_size)


def get_random_flip_lr(is_training):
    if is_training:
        return transforms.RandomHorizontalFlip()
    return transforms.Lambda(lambda x: x)


def get_to_gray_preprocess(prob):
    def _to_gray(img):
        if random.random() < prob:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        return img
    return transforms.Lambda(_to_gray)


def get_value_range_preprocess(vmin=-1, vmax=1):
    def _scale(img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)  # [0,1]
        return vmin + img * (vmax - vmin)
    return transforms.Lambda(_scale)


def get_standardization_preprocess():
    def _standardize(img):
        # Single image: [C,H,W]
        if img.ndim == 3:
            mean = img.mean()
            std = img.std()
            if std <= 0:
                std = torch.tensor(1.0, device=img.device, dtype=img.dtype)
            return (img - mean) / std

        # Stack of patches: [N,C,H,W]
        if img.ndim == 4:
            mean = img.mean(dim=(1, 2, 3), keepdim=True)
            std = img.std(dim=(1, 2, 3), keepdim=True)
            std = torch.where(std > 0, std, torch.ones_like(std))
            return (img - mean) / std

        raise ValueError(f"Unsupported tensor shape for standardization: {tuple(img.shape)}")

    return transforms.Lambda(_standardize)


def get_rotate_preprocess():
    def _rotate(img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)

        imgs = [
            img,
            torch.rot90(img, k=1, dims=(1, 2)),
            torch.rot90(img, k=2, dims=(1, 2)),
            torch.rot90(img, k=3, dims=(1, 2)),
        ]
        return torch.stack(imgs, dim=0)
    return transforms.Lambda(_rotate)


def get_preprocess_fn(
    fn_names,
    is_training,
    resize_size=(292, 292),
    crop_size=255,
    grayscale_probability=1.0,
    splits_per_side=3,
    patch_jitter=0,
    smaller_size=256,
):
    """
    Builds a torchvision Compose from a comma-separated preprocessing spec.

    Args:
        fn_names: e.g. "resize,to_gray,crop,crop_patches,standardization"
        is_training: bool
        resize_size: tuple[int,int] or int
        crop_size: tuple[int,int] or int
        grayscale_probability: float
        splits_per_side: int
        patch_jitter: int
        smaller_size: int
    """
    if not fn_names:
        fn_names = "plain_preprocess"

    def expand(fn_name):
        fn_name = fn_name.strip()

        if fn_name == "plain_preprocess":
            yield transforms.Lambda(lambda x: x)

        elif fn_name == "resize":
            yield get_resize_preprocess(resize_size)

        elif fn_name == "resize_small":
            yield get_resize_small(smaller_size)

        elif fn_name == "crop":
            yield get_crop(is_training, crop_size)

        elif fn_name == "central_crop":
            yield get_crop(False, crop_size)

        elif fn_name == "flip_lr":
            yield get_random_flip_lr(is_training)

        elif fn_name == "to_gray":
            yield get_to_gray_preprocess(grayscale_probability)

        elif fn_name == "0_to_1":
            yield transforms.ToTensor()

        elif fn_name == "-1_to_1":
            yield get_value_range_preprocess(-1, 1)

        elif fn_name == "standardization":
            # Ensure tensor first
            yield transforms.Lambda(lambda x: x if torch.is_tensor(x) else TF.to_tensor(x))
            yield get_standardization_preprocess()

        elif fn_name == "rotate":
            # Ensure tensor first
            yield transforms.Lambda(lambda x: x if torch.is_tensor(x) else TF.to_tensor(x))
            yield get_rotate_preprocess()

        elif fn_name == "crop_patches":
            # Ensure tensor first because patch extractor expects [C,H,W]
            yield transforms.Lambda(lambda x: x if torch.is_tensor(x) else TF.to_tensor(x))
            yield pp_lib.get_crop_patches_fn(
                is_training,
                split_per_side=splits_per_side,
                patch_jitter=patch_jitter,
            )

        else:
            raise ValueError(f"Unsupported preprocessing: {fn_name}")

    transforms_list = []
    for fn_name in fn_names.split(","):
        for t in expand(fn_name):
            transforms_list.append(t)

    return transforms.Compose(transforms_list)