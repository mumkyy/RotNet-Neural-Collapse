from common import * 

device = "cuda" if torch.cuda.is_available() else "cpu"

# rotate examples class
class RotateExamples(Dataset):
  def __init__(self, base_dataset):
    self.base_dataset = base_dataset
    self.rotations = [0, 90, 180, 270]

  def __len__(self):
    return len(self.base_dataset) * 4

  def __getitem__(self, idx):
    org_idx = idx // 4
    rot_idx = idx % 4

    img, _ = self.base_dataset[org_idx]
    angle = self.rotations[rot_idx]
    if int(angle) != 0:
      # Legacy transforms.functional expects standard Tensors or PIL Images
      rot_img = transforms.functional.rotate(img, angle)
    else:
      rot_img = img

    return rot_img, rot_idx
  
# dataset loading and configuration
class PreProcess():
    def __init__(self, dataset_name, global_bs):
        self.dataset_name = dataset_name

        if dataset_name.lower() == "cifar10":
            # CIFAR-10 Legacy Pipeline
            transform = transforms.Compose([
                # Fuses v2.ToImage() and v2.ToDtype(scale=True) into one operation
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            trainloader = Dataloader(train_set, batch_size=global_bs, shuffle=True, num_workers=2)

            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = Dataloader(test_set, batch_size=global_bs, shuffle=False, num_workers=2)

            # Rotated Examples
            rotated_train_set = RotateExamples(train_set)
            rotated_train_loader = Dataloader(rotated_train_set, batch_size=global_bs, shuffle=True, num_workers=2)
            rotated_test_set = RotateExamples(test_set)
            rotated_test_loader = Dataloader(rotated_test_set, batch_size=global_bs, shuffle=False, num_workers=2)

            self.rotated_train_loader = rotated_train_loader
            self.rotated_test_loader = rotated_test_loader 
            self.trainloader = trainloader
            self.testloader = testloader
            self.image_size = 32
        elif self.dataset_name.lower() == "imagenette":
            # Imagenette Legacy Pipeline
            TARGET_RESOLUTION = 224
            train_transforms = transforms.Compose([
                transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),
                # Fuses v2.ToImage() and v2.ToDtype(scale=True) into one operation
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            val_transforms = train_transforms
            try:
                train_dataset = datasets.Imagenette(
                    root="./data",
                    split="train",
                    size="320px",
                    download=True,
                    transform=train_transforms
                )

                val_dataset = datasets.Imagenette(
                    root="./data",
                    split="val",
                    size="320px",
                    download=True,
                    transform=val_transforms
                )
                self.image_size = TARGET_RESOLUTION
            except: 
                TARGET_RESOLUTION = 160
                train_transforms = transforms.Compose([
                    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),
                    # Fuses v2.ToImage() and v2.ToDtype(scale=True) into one operation
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                val_transforms = train_transforms
                train_dataset = datasets.ImageFolder ("../datasets/Imagenette/imagenette2-160/train", transform=train_transforms)
               
                val_dataset = datasets.ImageFolder ("../datasets/Imagenette/imagenette2-160/val", transform=val_transforms)

                self.image_size = TARGET_RESOLUTION 
            trainloader = Dataloader(train_dataset, shuffle=True, batch_size=global_bs, num_workers=2)
            testloader = Dataloader(val_dataset, shuffle=False, batch_size=global_bs, num_workers=2)

            rotated_imagenette_train_set = RotateExamples(train_dataset)
            rotated_imagenette_test_set = RotateExamples(val_dataset)

            rotated_train_loader = Dataloader(rotated_imagenette_train_set, shuffle=True, batch_size=global_bs, num_workers=2)
            rotated_test_loader = Dataloader(rotated_imagenette_test_set, shuffle=False, batch_size=global_bs, num_workers=2)

            self.rotated_train_loader = rotated_train_loader
            self.rotated_test_loader = rotated_test_loader 
            self.trainloader = trainloader
            self.testloader = testloader
        
        else:
           supported = ["imagenette", "cifar10"]
           raise ValueError(f"{dataset_name} not supported must be one of {supported}")
