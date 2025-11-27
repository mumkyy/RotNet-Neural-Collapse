import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from torchvision import datasets
import os



#########################################
# This class generates patches for training
#########################################

class MyDataset(Dataset):
  def __init__(self, patch_dim, gap, base_ds, transform=None):
    self.patch_dim, self.gap = patch_dim, gap
    self.transform = transform
    self.base_ds = base_ds
    self.patch_loc_arr = [(1, 1), (1, 2), (1, 3),(2, 1),(2, 3),(3, 1), (3, 2), (3, 3)]
    self.num_locs = len(self.patch_loc_arr)
  
  def get_patch_from_grid(self, image, patch_dim, gap, loc_index):
    image = np.array(image)
    H, W = image.shape[0], image.shape[1]

    #Same ratio from the paper: 96 patch size -> 7 pixel jitter
    jitter = max(1,(patch_dim * 7 )// 96)
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
    uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(img, self.patch_dim, self.gap, loc_index)

    #convert patches to float
    uniform_patch = uniform_patch.astype(np.float32) / 255.0
    random_patch = random_patch.astype(np.float32) / 255.0

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


def getLoaders(patch_dim,gap,batch_size,num_workers,root,supervised):

  tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
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
    transform=(tf if supervised else None),    
    size="160px"
  )

  val_data = datasets.Imagenette(
      root=root,
      split="val",
      download=False,
      transform=(tf if supervised else None),
      size="160px"
  )

  if supervised:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_data, batch_size=batch_size,shuffle=False, num_workers=num_workers)
  else:
    train_set = MyDataset(patch_dim=patch_dim,gap=gap,base_ds=train_data,transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_set = MyDataset(patch_dim=patch_dim,gap=gap,base_ds=val_data,transform=tf)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

  return train_loader, val_loader