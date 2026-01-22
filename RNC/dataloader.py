from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './datasets/cifar10'
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'
_PLACES205_DATASET_DIR = './datasets/Places205'
_IMAGENETTE_DATASET_DIR = './datasets/Imagenette'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(int(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None, pretext_mode='rotation', sigmas=None, 
                 kernel_sizes=None, patch_jitter=0,color_distort=False,
                 color_dist_strength=1.0, fixed_perms=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.pretext_mode = pretext_mode
        self.sigmas = sigmas
        self.kernel_sizes = kernel_sizes
        self.patch_jitter = patch_jitter
        self.color_distort = color_distort
        self.color_dist_strength = color_dist_strength

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the 
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        from maxHamming import generate_maximal_hamming_distance_set
        #per discussion today
        #performance at 7 was quite bad with MSE - will try 4
        if fixed_perms is not None:
            self.N_jigsaw = len(fixed_perms)
            self.jigsaw_perms = fixed_perms
        else:
            # Fallback (or for initial generation)
            self.N_jigsaw = 4
            P_1based = generate_maximal_hamming_distance_set(self.N_jigsaw, K=4)
            self.jigsaw_perms = [tuple(x-1 for x in p) for p in P_1based]

        if self.dataset_name=='imagenet':
            assert(self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.split!='train':
                transforms_list = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x).copy(),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)
        elif self.dataset_name=='places205':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            if self.split!='train':
                transforms_list = [
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x).copy(),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
                else:
                    transforms_list = [
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
            self.transform = transforms.Compose(transforms_list)
            self.data = Places205(root=_PLACES205_DATASET_DIR, split=self.split,
                transform=self.transform)
        elif self.dataset_name=='cifar10':
            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x).copy())
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR_DATASET_DIR, train=self.split=='train',
                download=True, transform=self.transform)
        
        elif self.dataset_name == 'imagenette':
            dn = 'Imagenette'
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.split != 'train':
                transforms_list = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x).copy(),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x).copy(),
                    ]
            self.data = torchvision.datasets.Imagenette(_IMAGENETTE_DATASET_DIR,
                                                         split="train",
                                                         download=False,
                                                         transform=self.transform)
        
        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))
        
        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)

    
    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
        if self.dataset_name=='cifar10':
            labels = self.data.targets        # works for both train and test objects
            data   = self.data.data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            if self.split=='test':
                self.data.test_labels = labels
                self.data.test_data = data
            else: 
                self.data.train_labels = labels
                self.data.train_data = data

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items(): 
                assert(len(v)==num_imgs_per_cat)

        elif self.dataset_name=='imagenet':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(self.dataset_name))
        elif self.dataset_name=='places205':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(self.dataset_name))
        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img.copy()
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def add_gaussian_noise(img, sigma):
    #Apply zero-mean Gaussian noise with std `sigma` to an HxWxC numpy image
    x = img.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, float(sigma), size=x.shape).astype(np.float32)
    x = np.clip(x + noise, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8).copy()

def apply_gaussian_blur(img, sigma=1.0, kernel_size=5):
    #Return img blurred with std `sigma` (expects HxWxC uint8)
    img_pil = Image.fromarray(img)
    blurred = blurred_img = F.gaussian_blur(img_pil, kernel_size=kernel_size, sigma=sigma)
    return np.array(blurred_img).copy()


def color_distortion_PIL(img_pil : Image.Image, 
                         strength: float = 1.0, 
                         p_jitter: float = 0.8, 
                         p_gray : float = 0.2) -> Image.Image: 
    
    cj = transforms.ColorJitter(
        brightness=0.8*strength, 
        contrast=0.8*strength, 
        saturation=0.8*strength, 
        hue=0.2*strength
    )
    if random.random() < p_jitter: 
        img_pil = cj(img_pil)

    if random.random() < p_gray: 
        img_pil = transforms.functional.to_grayscale(img_pil, num_output_channels=3)
    return img_pil
#TODO test the implementation of the jigsaw

from typing import List, Tuple, Optional
def four_way_jigsaw(img , perms: List[Tuple[int, int, int, int]] , patch_jitter : int, label: Optional[int] = None) -> Tuple[np.ndarray, int] : 
    img_H, img_W, img_C  = img.shape
    #we want to sub divide into quadrants : cifar 10 32x32 so each quadrant will be 16x16 = HxW
    assert img_H % 2 == 0 and img_W % 2 == 0
    q_H, q_W = img_H // 2 , img_W // 2

    #apply jitter and color distort to jigsaw 12/22/25
    #jitter in effect will be the displacement of quadrants (i.e. instead of the full context of the image take some subset and make 'shifted' quadrants)

    j = int(max(0, patch_jitter))
    j = min(j, q_H - 1, q_W - 1) 
    if j > 0: 
        img_pad = np.pad(img, ((j,j), (j,j), (0,0)), mode ="reflect")
    else : 
        img_pad = img

    base_coords = [
        (0,0), #TL
        (0,q_W), #TR
        (q_H,0), # BL 
        (q_H, q_W), #BR
    ]
#application of jitter to the image
#take the max (i.e. non negative) jitter value between the desired jitter, quadrant H/W, image H / W that remains in the absence of each quadrant
    patches = [] 
    for (y0,x0) in base_coords: 
        dy = random.randint(-j, j) if j > 0 else 0
        dx = random.randint(-j, j) if j > 0 else 0

        y = y0 + dy + j
        x = x0 + dx + j

        patch = img_pad[y:y+q_H, x:x+q_W, :]

        assert patch.shape[0] == q_H and patch.shape[1] == q_W, patch.shape

        patches.append(patch)

    if label is None:
        label = random.randrange(len(perms))
    else:
        label = int(label) % len(perms)
    perm = perms[label]

    top = np.concatenate([patches[perm[0]], patches[perm[1]]], axis = 1)
    
    bot = np.concatenate([patches[perm[2]], patches[perm[3]]], axis = 1)

    jig = np.concatenate([top, bot], axis = 0)
    return jig.copy() , label

#IGNORE THIS IMPLEMENTED IN CONTEXT-PRED

# #TODO: figure out how to concatenate output tensors and how should be fed into model
# #THIS IS THE LOGIC FOR THE QUADRANTS PRETEXT TASK 
# def get_quadrants(img):
#     # img: (C, H, W)
#     C, H, W = img.shape
#     assert H % 2 == 0 and W % 2 == 0, "H and W must be even."
#     mid_H, mid_W = H // 2, W // 2
#     TL = img[:, :mid_H, :mid_W]
#     TR = img[:, :mid_H,  mid_W:]
#     BL = img[:,  mid_H:, :mid_W]
#     BR = img[:,  mid_H:,  mid_W:]
#     return TL, TR, BL, BR

# ORIENT_MAP = {
#     0: ("TL", "TR"),  # side-to-side top
#     1: ("BL", "BR"),  # side-to-side bottom
#     2: ("TL", "BL"),  # left column
#     3: ("TR", "BR"),  # right column
#     4: ("TR", "BL"),  # diag 1
#     5: ("TL", "BR"),  # diag 2
# }
# def sample_pair_by_orientation(img, orientation):
    
#     TL, TR, BL, BR = get_quadrants(img)
#     name2patch = {"TL": TL, "TR": TR, "BL": BL, "BR": BR}
#     a_name, b_name = ORIENT_MAP[orientation]
#     return name2patch[a_name], name2patch[b_name]


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        self.pretext_mode = getattr(self.dataset, 'pretext_mode', 'rotation')

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                mode = getattr(self.dataset, 'pretext_mode', self.pretext_mode)

                if mode == 'gaussian_noise':
                    sigmas = list(getattr(self.dataset, 'sigmas', [1e-3, 1e-2, 1e-1, 1.0]))
                    noisy_imgs = [self.transform(add_gaussian_noise(img0, s)) for s in sigmas]
                    labels = torch.arange(len(noisy_imgs), dtype=torch.long)
                    return torch.stack(noisy_imgs, dim=0), labels
            
                if mode == 'gaussian_blur':
                    kernel_sizes = list(getattr(self.dataset, 'kernel_sizes', [3, 5, 7, 9]))
                    blurred = [self.transform(apply_gaussian_blur(img0, kernel_size=k)) for k in kernel_sizes]
                    labels = torch.arange(len(blurred), dtype=torch.long)
                    return torch.stack(blurred, 0), labels


                # if mode == 'shuffle' :
                #     #    Returns:
                #     #    pair: Tensor of shape (2, C, H/2, W/2)  [patchA, patchB]
                #     #    label: LongTensor scalar in {0..5} 
                #     orientation = random.randint(0, 5)
                #     pA, pB = sample_pair_by_orientation(img0, orientation)

                #     if transform is not None:
                #         pA = transform(pA)
                #         pB = transform(pB)

                #     # 4) stack for model input
                #     pair = torch.stack([pA, pB], dim=0)  # (2, C, H/2, W/2)
                #     label = torch.tensor(orientation, dtype=torch.long)

                #     return pair, label



                if mode == 'jigsaw':
                    perms = getattr(self.dataset, "jigsaw_perms", None)
                    if perms is None : 
                        raise RuntimeError("permutations attribute of Jigsaw did not get initialized in Genericdataset")
                    
                    patch_jitter = int(getattr(self.dataset, "patch_jitter", 0))
                    cd_strength = float(getattr(self.dataset, "color_dist_strength", 1.0))
                    cd_enable = bool (getattr(self.dataset, "color_distort", False))
                    if getattr(self.dataset, "split", "").lower() == "test":
                        label = idx % len(perms)
                    else:
                        label = None  # lets four_way_jigsaw pick random
                    if cd_enable: 
                        img_pil = Image.fromarray(img0)
                        img_pil = color_distortion_PIL(img_pil, strength=cd_strength)
                        img0 = np.asarray(img_pil).copy()

                    
                    jigsaw_image, label = four_way_jigsaw(
                        img0, perms, patch_jitter=patch_jitter, label=label
                    )

                    jigsaw_tensor = self.transform(jigsaw_image).unsqueeze(0)
                    label_tensor = torch.LongTensor([label])

                    return jigsaw_tensor, label_tensor


                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('imagenet','train', random_sized_crop=True)
    dataloader = DataLoader(dataset, batch_size=8, unsupervised=True)

    for b in dataloader(0):
        data, label = b
        break

    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4,4,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
