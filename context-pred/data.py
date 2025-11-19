import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import v2
from torchvision import datasets



#########################################
# This class generates patches for training
#########################################

class MyDataset(Dataset):
  def __init__(self, patch_dim, gap, base_ds, transform=None):
    self.patch_dim, self.gap = patch_dim, gap
    self.transform = transform
    self.base_ds = base_ds
  
  def get_patch_from_grid(self, image, patch_dim, gap):
    image = np.array(image)

    offset_x, offset_y = image.shape[0] - (patch_dim*3 + gap*2), image.shape[1] - (patch_dim*3 + gap*2)
    start_grid_x, start_grid_y = np.random.randint(0, offset_x), np.random.randint(0, offset_y)
    patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
    loc = np.random.randint(len(patch_loc_arr))
    tempx, tempy = patch_loc_arr[loc]
    
    patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1)
    patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1)
    random_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]

    patch_x_pt = start_grid_x + patch_dim * (2-1) + gap * (2-1)
    patch_y_pt = start_grid_y + patch_dim * (2-1) + gap * (2-1)
    uniform_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]
    
    random_patch_label = loc
    
    return uniform_patch, random_patch, random_patch_label

  def __len__(self):
    return len(self.base_ds)
  
  def __getitem__(self, index):
    img, _ = self.base_ds[index]
    uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(img, self.patch_dim, self.gap)

    # Dropped color channels 2 and 3 and replaced with gaussian noise(std ~1/100 of the std of the remaining channel)
    uniform_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), (uniform_patch.shape[0],uniform_patch.shape[1]))
    uniform_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), (uniform_patch.shape[0],uniform_patch.shape[1]))
    random_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), (random_patch.shape[0],random_patch.shape[1]))
    random_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), (random_patch.shape[0],random_patch.shape[1]))

    random_patch_label = np.array(random_patch_label).astype(np.int64)
        
    if self.transform:
      uniform_patch = self.transform(uniform_patch)
      random_patch = self.transform(random_patch)

    return uniform_patch, random_patch, random_patch_label


def getLoaders(patch_dim,gap,batch_size,num_workers):
  train_data = datasets.Imagenette(
    root="./data",
    split="train",
    download=True,
    transform=None,    
    size="full"
  )

  val_data = datasets.Imagenette(
      root="./data",
      split="val",
      download=True,
      transform=None,
      size="full"
  )

  tf = v2.Compose([v2.ToTensor(),v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

  train_set = MyDataset(patch_dim=patch_dim,gap=gap,base_ds=train_data,transform=tf)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)


  val_set = MyDataset(patch_dim=patch_dim,gap=gap,base_ds=val_data,transform=tf)
  val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

  return train_loader, val_loader