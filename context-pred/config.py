class Config():
  ROOT = 'tiny-imagenet-200'
  TRAIN_PATH = 'tiny-imagenet-200/train'
  VAL_PATH = 'tiny-imagenet-200/val'
  TEST_PATH = 'tiny-imagenet-200/test'
  subset_data = 1000
  patch_dim = 15
  gap = 3
  batch_size = 64
  num_epochs = 65
  lr = 0.0005
  num_workers = 12