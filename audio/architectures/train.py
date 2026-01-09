import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import importlib.util
import sys

# Dynamic import helper
def load_model_from_file(def_file, opt):
    spec = importlib.util.spec_from_file_location("model_module", def_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Force input channels to 1 because we are using Spectrograms
    opt['num_inchannels'] = 1 
    return module.create_model(opt)

def train_loop(config):
    # 1. Setup Data
    from dataloader import create_dataloader
    train_loader = create_dataloader(config['data_train_opt'])
    test_loader = create_dataloader(config['data_test_opt'])

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. Setup Model
    model_cfg = config['networks']['model']
    model = load_model_from_file(model_cfg['def_file'], model_cfg['opt'])
    model = model.to(device)

    # 4. Setup Optimizer
    optim_params = model_cfg['optim_params']
    optimizer = optim.SGD(
        model.parameters(),
        lr=optim_params['lr'],
        momentum=optim_params['momentum'],
        weight_decay=optim_params['weight_decay'],
        nesterov=optim_params['nesterov']
    )

    # 5. Setup Loss
    criterion_cfg = config['criterions']['loss']
    if criterion_cfg['ctype'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 6. Training Loop
    epochs = config['max_num_epochs']
    lut_lr = optim_params['LUT_lr'] # List of tuples [(epoch, lr), ...]

    print("Starting Training...")
    
    for epoch in range(1, epochs + 1):
        # --- LR Scheduler (LUT) ---
        # Check if current epoch matches a LUT step
        for step_epoch, new_lr in lut_lr:
            if epoch == step_epoch:
                print(f"Adjusting LR to {new_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # --- Loss Calculation ---
            if isinstance(criterion, nn.MSELoss):
                # For MSE, we need One-Hot targets: [Batch, 2]
                # target is currently [Batch] indices (0 or 1)
                target_onehot = F.one_hot(target, num_classes=2).float()
                loss = criterion(output, target_onehot)
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            # --- Metrics ---
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({'Loss': f"{running_loss/(total/data.size(0)):.4f}", 'Acc': f"{100.*correct/total:.2f}%"})

        # (Optional) Test loop can go here