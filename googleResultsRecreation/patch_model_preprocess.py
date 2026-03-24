from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch


def crop(image, is_training, crop_size):
    """
    Args:
        image: torch.Tensor of shape [C, H, W]
        is_training: bool
        crop_size: [h, w]

    Returns:
        Cropped image tensor of shape [C, h, w]
    """
    h, w = crop_size[0], crop_size[1]
    _, H, W = image.shape

    if h > H or w > W:
        raise ValueError(
            f"Crop size {(h, w)} is larger than image size {(H, W)}"
        )

    if is_training:
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
    else:
        top = (H - h) // 2
        left = (W - w) // 2

    return image[:, top:top + h, left:left + w]


def image_to_patches(image, is_training, split_per_side, patch_jitter=0):
    """
    Crops split_per_side x split_per_side patches from input image.

    Args:
        image: input image tensor with shape [C, H, W]
        is_training: bool
        split_per_side: number of splits per side
        patch_jitter: jitter of each patch from each grid

    Returns:
        Patches tensor with shape [patch_count, C, h_patch, w_patch]
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected image with shape [C, H, W], got {tuple(image.shape)}"
        )

    c, h, w = image.shape

    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter

    if h_patch <= 0 or w_patch <= 0:
        raise ValueError(
            f"patch_jitter={patch_jitter} is too large for grid size "
            f"({h_grid}, {w_grid})"
        )

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            top = i * h_grid
            left = j * w_grid

            # grid cell
            p = image[:, top:top + h_grid, left:left + w_grid]

            # crop a smaller tile from the grid cell to break edge continuity
            if h_patch < h_grid or w_patch < w_grid:
                p = crop(p, is_training, [h_patch, w_patch])

            patches.append(p)

    return torch.stack(patches, dim=0)


def get_crop_patches_fn(is_training, split_per_side, patch_jitter=0):
    """
    Returns a transform function that crops split_per_side x split_per_side patches.

    Input:
        image tensor [C, H, W]

    Output:
        patches tensor [patch_count, C, h_patch, w_patch]
    """
    def _crop_patches_pp(image):
        return image_to_patches(
            image,
            is_training=is_training,
            split_per_side=split_per_side,
            patch_jitter=patch_jitter,
        )

    return _crop_patches_pp 