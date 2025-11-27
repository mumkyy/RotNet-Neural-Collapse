import torch
import torch.optim as optim
import torch.nn as nn
from model import AlexNetwork, AlexClassifier
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

  arch = net_cfg.get("arch", "AlexNetwork")
  is_pretext = (arch == "AlexNetwork")

  root = train_opt["dataset_root"]

  batch_size  = train_opt["batch_size"]
  num_epochs  = cfg["max_num_epochs"]
  num_workers = args.num_workers

  if is_pretext:
    patch_dim = train_opt["patch_dim"]
    gap       = train_opt["gap"]
  else:
    patch_dim = None
    gap       = None

  loss_cfg = cfg.get("criterions", {}).get("loss", {})
  ctype = loss_cfg.get("ctype", "CrossEntropyLoss")
  loss_opt = loss_cfg.get("opt") or {}

  optim_cfg = net_cfg["optim_params"]
  opt_type = optim_cfg.get("optim_type", "sgd").lower()


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if is_pretext:
    # context-pred backbone
    model = AlexNetwork(**net_cfg.get("opt", {})).to(device)
  else:
    # downstream classifier
    model = AlexClassifier(**net_cfg.get("opt", {})).to(device)

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

  supervised = not is_pretext
  trainloader, valloader = getLoaders(patch_dim, gap, batch_size, num_workers, root, supervised)

  checkpoint_dir = f"checkpoints/{args.config}"
  os.makedirs(checkpoint_dir, exist_ok=True)
  ############################
  # Training/Validation Engine
  ############################

  global_trn_loss = []
  global_val_loss = []
  global_val_acc  = []

  for epoch in range(num_epochs):
      train_running_loss = []
      val_running_loss = []
      start_time = time.time()
      model.train()

      #train
      for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
          optimizer.zero_grad()

          if is_pretext:
                # (uniform_patch, random_patch, label)
            uniform_patch, random_patch, labels = data
            uniform_patch = uniform_patch.to(device)
            random_patch  = random_patch.to(device)
            labels        = labels.to(device)
            output, _, _  = model(uniform_patch, random_patch)
          else:
              # (image, label)
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

          if loss_mode == "ce":
                loss = criterion(output, labels)
          else:
              # MSE: one-hot targets
              target = torch.zeros_like(output)
              target.scatter_(1, labels.unsqueeze(1), 1.0)
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
          if is_pretext:
            uniform_patch, random_patch, labels = data
            uniform_patch = uniform_patch.to(device)
            random_patch  = random_patch.to(device)
            labels        = labels.to(device)
            output, _, _  = model(uniform_patch, random_patch)
          else:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

          if loss_mode == "ce":
            loss = criterion(output, labels)
          else:
            target = torch.zeros_like(output)
            target.scatter_(1, labels.unsqueeze(1), 1.0)
            loss = criterion(output, target)

          val_running_loss.append(loss.item())
        
          _, predicted = output.max(1)
          total   += labels.size(0)
          correct += (predicted == labels).sum().item()

        print('Val Progress --- total:{}, correct:{}'.format(total, correct))
        print(f'Val Accuracy: {100 * correct / total:.2f}%')

      val_acc = 100 * correct / total
      avg_train_loss = sum(train_running_loss) / len(train_running_loss)
      avg_val_loss = sum(val_running_loss) / len(val_running_loss)

      global_trn_loss.append(avg_train_loss)
      global_val_loss.append(avg_val_loss)
      global_val_acc.append(val_acc)

      scheduler.step(avg_val_loss)

      print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
          epoch + 1, num_epochs, avg_train_loss, avg_val_loss,
          (time.time() - start_time) / 60))
      
      model_save_path = f'{checkpoint_dir}/{epoch+1:03d}.pt'
      torch.save(
          {
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'accuracy': val_acc,
              'loss': avg_val_loss,
              'global_trnloss': global_trn_loss,
              'global_valloss': global_val_loss,
              'global_val_acc': global_val_acc,
          },
          model_save_path,
      )
      with open(f'{checkpoint_dir}/metrics.txt', 'a') as f:
        f.write(f'{epoch+1},{global_trn_loss[-1]},{global_val_loss[-1]},{val_acc}\n')

  plot_path = f'{checkpoint_dir}/training_plots.png'
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.plot(range(1, len(global_trn_loss)+1), global_trn_loss, label='Train Loss', marker='o')
  plt.plot(range(1, len(global_val_loss)+1), global_val_loss, label='Val Loss', marker='x')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training / Validation Loss')
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(range(1, len(global_val_acc)+1), global_val_acc, label='Val Acc', marker='o', color='green')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Validation Accuracy')
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_path, dpi=150, bbox_inches='tight')
  print(f'Plots saved to {plot_path}')

if __name__ == "__main__":
   main()