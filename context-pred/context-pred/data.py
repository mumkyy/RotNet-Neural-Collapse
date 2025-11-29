import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from torchvision import datasets
import os
from enum import Enum

class Modes(Enum):
  SUPERVISED = 0
  EIGHT = 1
  QUAD = 2

class ContextPred(Dataset):
  def __init__(self, patch_dim, base_ds, transform=None, gap=None, chromatic=True, jitter=True):
    self.patch_dim, self.gap, self.chromatic, self.jitter = patch_dim, gap, chromatic, jitter
    self.transform = transform
    self.base_ds = base_ds
    self.patch_loc_arr = [(1, 1), (1, 2), (1, 3),(2, 1),(2, 3),(3, 1), (3, 2), (3, 3)]
    self.num_locs = len(self.patch_loc_arr)
  
  def get_patch_from_grid(self, image, loc_index):
    image = np.array(image)
    patch_dim = self.patch_dim
    H, W = image.shape[0], image.shape[1]

    gap = self.gap if self.gap is not None else 0

    #Same ratio from the paper: 96 patch size -> 7 pixel jitter
    if self.jitter:
      jitter = max(1, (patch_dim * 7) // 96)  
    else:
      jitter = 0
    jx = np.random.randint(-jitter,jitter+1)
    jy =  np.random.randint(-jitter,jitter+1)

    offset_x, offset_y = image.shape[0] - (patch_dim*3 + gap*2), image.shape[1] - (patch_dim*3 + gap*2)
    start_grid_x, start_grid_y = offset_x//2, offset_y//2
    loc = loc_index
    tempx, tempy = self.patch_loc_arr[loc]
    
    patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1) + jx
    patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1) + jy

    #clamping in case jitter goes over the boundaries of the image
    patch_x_pt = max(0, min(patch_x_pt, H - patch_dim))
    patch_y_pt = max(0, min(patch_y_pt, W - patch_dim))
  
    random_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]

    patch_x_pt = start_grid_x + patch_dim * (2-1) + gap * (2-1)
    patch_y_pt = start_grid_y + patch_dim * (2-1) + gap * (2-1)
    uniform_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]
    
    random_patch_label = loc
    
    return uniform_patch, random_patch, random_patch_label

  def __len__(self):
    return len(self.base_ds) * 8
  
  def __getitem__(self, index):
    img_index = index // 8
    loc_index = index % self.num_locs

    img, _ = self.base_ds[img_index]
    uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(img, self.patch_dim, loc_index)

    #convert patches to float
    uniform_patch = uniform_patch.astype(np.float32) / 255.0
    random_patch = random_patch.astype(np.float32) / 255.0

    if self.chromatic:
      #randomly choose color channels to choose and drop
      c_keep = np.random.randint(0,3)
      c_drop1, c_drop2 = {0,1,2} - {c_keep}

      # Drop color channels and replace with gaussian noise(std ~1/100 of the std of the remaining channel)
      u_noise_std = 0.01 * np.std(uniform_patch[:, :, c_keep])
      r_noise_std = 0.01 * np.std(random_patch[:, :, c_keep])

      uniform_patch[:, :, c_drop1] = np.random.normal(0.5, u_noise_std, uniform_patch.shape[:2])
      uniform_patch[:, :, c_drop2] = np.random.normal(0.5, u_noise_std, uniform_patch.shape[:2])
      random_patch[:, :, c_drop1] = np.random.normal(0.5, r_noise_std, random_patch.shape[:2])
      random_patch[:, :, c_drop2] = np.random.normal(0.5, r_noise_std , random_patch.shape[:2])

    random_patch_label = np.array(random_patch_label).astype(np.int64)

    if self.transform:
      uniform_patch = self.transform(uniform_patch).clone()
      random_patch  = self.transform(random_patch).clone()


    return uniform_patch, random_patch, random_patch_label


class QuadContextPred(Dataset):
  def __init__(self, patch_dim, base_ds, transform=None, gap=None, chromatic=True, jitter=True ):
    self.patch_dim, self.gap, self.chromatic, self.jitter  = patch_dim, gap, chromatic, jitter
    self.transform  = transform
    self.base_ds    = base_ds

    self.num_pairs = 6

  def __len__(self):
    return len(self.base_ds) * self.num_pairs

  def _crop_quadrant(self, image, q_idx):

    H, W = image.shape[0], image.shape[1]
    patch_dim = self.patch_dim
    gap = self.gap if self.gap is not None else 0

    if self.jitter:
      jitter = max(1, (patch_dim * 7) // 96)   # paper ratio
    else:
      jitter = 0

    half_h = H // 2
    half_w = W // 2

    # raw quadrant bounds
    if q_idx == 0:       # TL
      r0, r1 = 0,       half_h
      c0, c1 = 0,       half_w
    elif q_idx == 1:     # TR
      r0, r1 = 0,       half_h
      c0, c1 = half_w,  W
    elif q_idx == 2:     # BL
      r0, r1 = half_h,  H
      c0, c1 = 0,       half_w
    else:                # BR (3)
      r0, r1 = half_h,  H
      c0, c1 = half_w,  W

    r0 = r0 + gap // 2
    r1 = r1 - gap // 2
    c0 = c0 + gap // 2
    c1 = c1 - gap // 2

    # Make sure patch_dim fits; if not, clamp
    quad_h = max(1, r1 - r0)
    quad_w = max(1, c1 - c0)
    ph = min(patch_dim, quad_h)
    pw = min(patch_dim, quad_w)

    # Center inside the (possibly shrunk) quadrant
    base_r = r0 + (quad_h - ph) // 2
    base_c = c0 + (quad_w - pw) // 2

    # Jitter around that center
    jx = np.random.randint(-jitter, jitter + 1)
    jy = np.random.randint(-jitter, jitter + 1)

    r = np.clip(base_r + jx, r0, r1 - ph)
    c = np.clip(base_c + jy, c0, c1 - pw)

    patch = image[r:r + ph, c:c + pw]
    return patch

  def get_quadrants(self, image, loc_index):
    image = np.array(image)

    if loc_index == 0:
      # top vs bottom: TL–BL
      patch1 = self._crop_quadrant(image, 0)  # TL
      patch2 = self._crop_quadrant(image, 2)  # BL
      label = 0
    elif loc_index == 1:
      # top vs bottom: TR–BR
      patch1 = self._crop_quadrant(image, 1)  # TR
      patch2 = self._crop_quadrant(image, 3)  # BR
      label = 0
    elif loc_index == 2:
      # left vs right: TL–TR
      patch1 = self._crop_quadrant(image, 0)  # TL
      patch2 = self._crop_quadrant(image, 1)  # TR
      label = 1
    elif loc_index == 3:
      # left vs right: BL–BR
      patch1 = self._crop_quadrant(image, 2)  # BL
      patch2 = self._crop_quadrant(image, 3)  # BR
      label = 1
    elif loc_index == 4:
      # right diagonal: TL–BR
      patch1 = self._crop_quadrant(image, 0)  # TL
      patch2 = self._crop_quadrant(image, 3)  # BR
      label = 2
    else:
      # left diagonal: TR–BL
      patch1 = self._crop_quadrant(image, 1)  # TR
      patch2 = self._crop_quadrant(image, 2)  # BL
      label = 3

    return patch1, patch2, label

  def __getitem__(self, index):
    img_index  = index // self.num_pairs
    loc_index  = index %  self.num_pairs

    img, _ = self.base_ds[img_index]
    patch1, patch2, label = self.get_quadrants(img, loc_index)

    # convert patches to float
    patch1 = patch1.astype(np.float32) / 255.0
    patch2 = patch2.astype(np.float32) / 255.0

    if self.chromatic:
      # randomly choose color channels to keep/drop (same strategy as ContextPred)
      c_keep = np.random.randint(0, 3)
      c_drop1, c_drop2 = {0, 1, 2} - {c_keep}

      # Drop color channels and replace with Gaussian noise
      p1_noise_std = 0.01 * np.std(patch1[:, :, c_keep])
      p2_noise_std = 0.01 * np.std(patch2[:, :, c_keep])

      patch1[:, :, c_drop1] = np.random.normal(0.5, p1_noise_std, patch1.shape[:2])
      patch1[:, :, c_drop2] = np.random.normal(0.5, p1_noise_std, patch1.shape[:2])
      patch2[:, :, c_drop1] = np.random.normal(0.5, p2_noise_std, patch2.shape[:2])
      patch2[:, :, c_drop2] = np.random.normal(0.5, p2_noise_std, patch2.shape[:2])

    label = np.array(label).astype(np.int64)

    if self.transform:
        patch1 = self.transform(patch1).clone()
        patch2 = self.transform(patch2).clone()

    return patch1, patch2, label


def get_loaders(mode,patch_dim,batch_size,num_workers,root,gap=None,chromatic=None,jitter=None):

  tf = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

  btf = transforms.Resize((160, 160))
  
  supervised = (mode == Modes.SUPERVISED)

  if supervised:
    tf.insert(0,btf)

  tf = transforms.Compose(tf)

  possible_dirs = [
      os.path.join(root, "imagenette2-160"),
      os.path.join(root, "imagenette2")  
  ]

  already_there = any(os.path.isdir(d) for d in possible_dirs)
  download_flag = not already_there
  
  train_data = datasets.Imagenette(
    root=root,
    split="train",
    download=download_flag,
    transform=(tf if supervised else btf),    
    size="160px"
  )

  val_data = datasets.Imagenette(
      root=root,
      split="val",
      download=False,
      transform=(tf if supervised else btf),
      size="160px"
  )

  if mode == Modes.SUPERVISED:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_data, batch_size=batch_size,shuffle=False, num_workers=num_workers)
  
  else:
    if mode == Modes.EIGHT:
      train_set = ContextPred(patch_dim=patch_dim,base_ds=train_data,transform=tf,gap=gap,chromatic=chromatic,jitter=jitter)
      val_set = ContextPred(patch_dim=patch_dim,base_ds=val_data,transform=tf,gap=gap,chromatic=chromatic,jitter=jitter)

    elif mode == Modes.QUAD:
      train_set = QuadContextPred(patch_dim=patch_dim,base_ds=train_data,transform=tf,gap=gap,chromatic=chromatic,jitter=jitter)
      val_set = QuadContextPred(patch_dim=patch_dim,base_ds=val_data,transform=tf,gap=gap,chromatic=chromatic,jitter=jitter)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

  return train_loader, val_loader