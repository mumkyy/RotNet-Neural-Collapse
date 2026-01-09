# architectures/train.py
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataloader import create_dataloader


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_from_file(def_file: str, opt: Dict) -> nn.Module:
    """
    def_file: path to python file defining create_model(opt) -> nn.Module
    opt: network opt dict
    """
    def_file = str(def_file)
    if not Path(def_file).exists():
        raise FileNotFoundError(f"Model def_file not found: {def_file}")

    spec = importlib.util.spec_from_file_location("model_module", def_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "create_model"):
        raise AttributeError(f"{def_file} must define create_model(opt).")

    # Spectrograms are single-channel (C=1)
    opt = dict(opt)
    opt["num_inchannels"] = int(opt.get("num_inchannels", 1))

    model = module.create_model(opt)
    return model


def build_optimizer(model: nn.Module, optim_params: Dict) -> optim.Optimizer:
    optim_type = optim_params.get("optim_type", "sgd").lower()
    lr = float(optim_params.get("lr", 0.1))

    if optim_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_params.get("momentum", 0.9)),
            weight_decay=float(optim_params.get("weight_decay", 5e-4)),
            nesterov=bool(optim_params.get("nesterov", True)),
        )
    elif optim_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=float(optim_params.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported optim_type: {optim_type}")


def build_criterion(criterion_cfg: Dict) -> nn.Module:
    ctype = criterion_cfg.get("ctype", "CrossEntropyLoss")
    if ctype == "MSELoss":
        return nn.MSELoss()
    if ctype == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss ctype: {ctype}")


def apply_lut_lr(optimizer: optim.Optimizer, lut_lr, epoch: int) -> None:
    if not lut_lr:
        return
    for step_epoch, new_lr in lut_lr:
        if epoch == int(step_epoch):
            for pg in optimizer.param_groups:
                pg["lr"] = float(new_lr)
            print(f"[LR] epoch={epoch} -> lr={new_lr}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    batches = 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(data)  # logits (B, 2)
        if isinstance(criterion, nn.MSELoss):
            target_onehot = F.one_hot(target, num_classes=output.size(1)).float()
            loss = criterion(output, target_onehot)
        else:
            loss = criterion(output, target)

        loss_sum += float(loss.item())
        batches += 1

        pred = output.argmax(dim=1)
        total += int(target.size(0))
        correct += int((pred == target).sum().item())

    avg_loss = loss_sum / max(1, batches)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def train_loop(config: Dict, no_cuda: bool = False) -> None:
    set_seed(int(config.get("seed", 42)))

    # Data
    train_loader = create_dataloader(config["data_train_opt"])
    test_loader = create_dataloader(config["data_test_opt"])

    # Device
    use_cuda = torch.cuda.is_available() and (not no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # Model
    model_cfg = config["networks"]["model"]
    model = load_model_from_file(model_cfg["def_file"], model_cfg["opt"]).to(device)

    # Optimizer + LR LUT
    optim_params = model_cfg.get("optim_params", {})
    optimizer = build_optimizer(model, optim_params)
    lut_lr = optim_params.get("LUT_lr", [])

    # Loss
    criterion = build_criterion(config["criterions"]["loss"]).to(device)

    epochs = int(config.get("max_num_epochs", 200))
    log_every = int(config.get("log_every", 1))

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        apply_lut_lr(optimizer, lut_lr, epoch)

        model.train()
        running_loss = 0.0
        batches = 0
        total = 0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for data, target in pbar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)  # logits (B, 2)

            if isinstance(criterion, nn.MSELoss):
                target_onehot = F.one_hot(target, num_classes=output.size(1)).float()
                loss = criterion(output, target_onehot)
            else:
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

            pred = output.argmax(dim=1)
            total += int(target.size(0))
            correct += int((pred == target).sum().item())

            pbar.set_postfix(
                loss=f"{running_loss/max(1,batches):.4f}",
                acc=f"{100.0*correct/max(1,total):.2f}%",
                lr=f"{optimizer.param_groups[0]['lr']:.5f}",
            )

        if epoch % log_every == 0:
            train_loss = running_loss / max(1, batches)
            train_acc = 100.0 * correct / max(1, total)

            test_loss, test_acc = evaluate(model, test_loader, device, criterion)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
            )
