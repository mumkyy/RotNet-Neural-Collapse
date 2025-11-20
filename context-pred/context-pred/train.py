import torch
import torch.optim as optim
import torch.nn as nn
from model import AlexNetwork
from config import Config
import numpy as np
from data import getLoaders
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


torch.manual_seed(108)
torch.cuda.manual_seed_all(108)
np.random.seed(108)

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNetwork().to(device)

#############################################
# Initialized Optimizer, criterion, scheduler
#############################################

optimizer = optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3)

trainloader, valloader = getLoaders(Config.patch_dim,Config.gap,Config.batch_size,Config.num_workers)


os.makedirs("checkpoints", exist_ok=True)

############################
# Training/Validation Engine
############################

global_trn_loss = []
global_val_loss = []
# previous_val_loss = 100

for epoch in range(Config.num_epochs):
    train_running_loss = []
    val_running_loss = []
    start_time = time.time()
    model.train()

    #train
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
        loss = criterion(output, random_patch_label)
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
        loss = criterion(output, random_patch_label)
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
        epoch + 1, Config.num_epochs, global_trn_loss[-1], global_val_loss[-1],
        (time.time() - start_time) / 60))
    
    if (epoch + 1) % 20 == 0:
      MODEL_SAVE_PATH = f'checkpoints/model_{Config.batch_size}_{Config.num_epochs}_{Config.lr}_{Config.patch_dim}_{Config.gap}.pt'
      torch.save(
          {
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
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