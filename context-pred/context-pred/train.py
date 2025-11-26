import torch
import torch.optim as optim
import torch.nn as nn
from model import AlexNetwork
import numpy as np
from data import getLoaders
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
import importlib


torch.manual_seed(108)
torch.cuda.manual_seed_all(108)
np.random.seed(108)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config",type=str)
  parser.add_argument("--num-workers",type=int,default=12)
  return parser.parse_args()

def main():
  args = parse_args()

  cfg_mod = importlib.import_module(f"configs.{args.config}")
  cfg = cfg_mod.config

  train_opt = cfg["data_train_opt"]
  net_cfg   = cfg["networks"]["model"]

  root = train_opt["dataset_root"]

  batch_size  = train_opt["batch_size"]
  patch_dim   = train_opt["patch_dim"]
  gap         = train_opt["gap"]
  num_epochs  = cfg["max_num_epochs"]
  num_workers = args.num_workers

  loss_cfg = cfg.get("criterions", {}).get("loss", {})
  ctype = loss_cfg.get("ctype", "CrossEntropyLoss")
  loss_opt = loss_cfg.get("opt") or {}

  optim_cfg = net_cfg["optim_params"]
  opt_type = optim_cfg.get("optim_type", "sgd").lower()


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = AlexNetwork().to(device)

  #############################################
  # Initialized Optimizer, criterion, scheduler
  #############################################

  if opt_type == "sgd":
    optimizer = optim.SGD(
        model.parameters(),
        lr=optim_cfg["lr"],
        momentum=optim_cfg.get("momentum", 0.9),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        nesterov=optim_cfg.get("nesterov", False),
    )
  elif opt_type == "adam":
      optimizer = optim.Adam(
          model.parameters(),
          lr=optim_cfg["lr"],
          weight_decay=optim_cfg.get("weight_decay", 0.0),
      )
  else:
      raise ValueError(f"Unsupported optim_type: {opt_type}")

  if ctype == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss(**loss_opt)
    loss_mode = "ce"
  elif ctype == "MSELoss":
    criterion = nn.MSELoss(**loss_opt)
    loss_mode = "mse"
  else:
    raise ValueError(f"Unsupported loss type: {ctype}")
  
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3)

  trainloader, valloader = getLoaders(patch_dim,gap,batch_size,num_workers,root)

  os.makedirs("checkpoints", exist_ok=True)

  ############################
  # Training/Validation Engine
  ############################

  global_trn_loss = []
  global_val_loss = []
  # previous_val_loss = 100

  for epoch in range(num_epochs):
      train_running_loss = []
      val_running_loss = []
      start_time = time.time()
      model.train()

      #train
      for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
          uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
          optimizer.zero_grad()
          output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
          if loss_mode == "ce":
              loss = criterion(output, random_patch_label)
          else:
              # MSE: one-hot targets
              target = torch.zeros_like(output)
              target.scatter_(1, random_patch_label.unsqueeze(1), 1.0)
              loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          
          train_running_loss.append(loss.item())
        
      correct = 0
      total = 0

      #val
      model.eval()
      with torch.no_grad():
        for idx, data in tqdm(enumerate(valloader), total=len(valloader)):
          uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
          output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
          if loss_mode == "ce":
              loss = criterion(output, random_patch_label)
          else:
              # MSE: one-hot targets
              target = torch.zeros_like(output)
              target.scatter_(1, random_patch_label.unsqueeze(1), 1.0)
              loss = criterion(output, target)
          val_running_loss.append(loss.item())
        
          _, predicted = output.max(1)
          total += random_patch_label.size(0)
          correct += (predicted == random_patch_label).sum().item()
        print('Val Progress --- total:{}, correct:{}'.format(total, correct))
        print(f'Val Accuracy: {100 * correct / total:.2f}%')

      global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
      global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

      scheduler.step(global_val_loss[-1])

      print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
          epoch + 1, num_epochs, global_trn_loss[-1], global_val_loss[-1],
          (time.time() - start_time) / 60))
      
      if (epoch + 1) % 20 == 0:
        MODEL_SAVE_PATH = f'checkpoints/{args.config}.pt'
        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'global_trnloss': global_trn_loss,
                'global_valloss': global_val_loss,
            },
            MODEL_SAVE_PATH,
        )

  plt.plot(range(len(global_trn_loss)), global_trn_loss, label='TRN Loss')
  plt.plot(range(len(global_val_loss)), global_val_loss, label='VAL Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training/Validation Loss plot')
  plt.legend()
  plt.show()

if __name__ == "__main__":
   main()