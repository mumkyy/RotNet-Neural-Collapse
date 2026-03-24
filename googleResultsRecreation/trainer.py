"""Base trainer class (PyTorch version)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import datasets


def get_lr(global_step, base_lr, steps_per_epoch,
           decay_epochs, lr_decay_factor, warmup_epochs):
    """
    Matches the TensorFlow schedule:

    - linear warmup for warmup_epochs
    - then piecewise constant decay at decay_epochs
    """
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")

    warmup_steps = warmup_epochs * steps_per_epoch

    if warmup_epochs > 0 and global_step < warmup_steps:
        return global_step * (base_lr / float(warmup_steps))

    lr = base_lr
    for e in decay_epochs:
        if global_step >= e * steps_per_epoch:
            lr *= lr_decay_factor
    return lr


class Trainer(object):
    """Base trainer class."""

    def __init__(self, args, update_batchnorm_params=True):
        self.args = args
        self.update_batchnorm_params = update_batchnorm_params

        split = getattr(args, "train_split", "train")
        data_cfg = getattr(args, "_data_cfg", None)
        num_samples = datasets.get_count(split, cfg=data_cfg)
        steps_per_epoch = num_samples // args.batch_size
        if steps_per_epoch <= 0:
            steps_per_epoch = 1

        self.steps_per_epoch = steps_per_epoch
        self.global_step = 0

        # Same semantics as TF version
        lr_factor = 1.0
        if getattr(args, "lr_scale_batch_size", 0):
            lr_factor = args.batch_size / float(args.lr_scale_batch_size)

        decay_epochs = getattr(args, "decay_epochs", None)
        if not decay_epochs:
            decay_epochs = [args.epochs]

        self.base_lr = args.lr * lr_factor
        self.decay_epochs = list(decay_epochs)
        self.lr_decay_factor = getattr(args, "lr_decay_factor", 0.1)
        self.warmup_epochs = getattr(args, "warmup_epochs", 0)

    def get_current_lr(self):
        return get_lr(
            global_step=self.global_step,
            base_lr=self.base_lr,
            steps_per_epoch=self.steps_per_epoch,
            decay_epochs=self.decay_epochs,
            lr_decay_factor=self.lr_decay_factor,
            warmup_epochs=self.warmup_epochs,
        )

    def build_optimizer(self, model):
        lr = self.get_current_lr()
        optimizer_name = getattr(self.args, "optimizer", "sgd").lower()

        if optimizer_name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.0,  # handled manually below if add_reg_loss=True
            )
        elif optimizer_name == "adam":
            return optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=0.0,  # handled manually below if add_reg_loss=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def update_optimizer_lr(self, optimizer):
        lr = self.get_current_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _l2_regularization_loss(self, model, var_list=None):
        weight_decay = getattr(self.args, "weight_decay", 0.0)
        if not weight_decay:
            return None

        params = var_list
        if params is None:
            params = [p for p in model.parameters() if p.requires_grad]

        reg = None
        for p in params:
            if p.requires_grad:
                term = torch.sum(p.pow(2))
                reg = term if reg is None else reg + term

        if reg is None:
            return None
        return weight_decay * reg

    def get_train_step(self, model, loss,
                       optimizer,
                       var_list=None,
                       add_reg_loss=True):
        """
        PyTorch replacement for TF get_train_op.

        Runs:
        - optional L2 reg loss
        - backward
        - optimizer step
        - global step increment

        Returns:
            total_loss, current_lr
        """
        self.update_optimizer_lr(optimizer)

        total_loss = loss
        if add_reg_loss:
            reg_loss = self._l2_regularization_loss(model, var_list=var_list)
            if reg_loss is not None:
                total_loss = total_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        self.global_step += 1
        return total_loss, self.get_current_lr()


def make_estimator(mode, model=None, loss=None, eval_metrics=None,
                   predictions=None, optimizer=None, trainer=None):
    """
    Lightweight PyTorch replacement for the TF EstimatorSpec helper.

    Returns a plain dict describing the step result.
    """
    if mode == "predict":
        if predictions is None:
            raise ValueError("Need to pass `predictions` for predict mode.")
        return {
            "mode": mode,
            "predictions": predictions,
        }

    if mode == "eval":
        return {
            "mode": mode,
            "loss": loss,
            "eval_metrics": eval_metrics,
        }

    if mode == "train":
        if loss is None:
            raise ValueError("Need to pass `loss` for train mode.")
        if trainer is None:
            raise ValueError("Need to pass `trainer` for train mode.")
        if model is None:
            raise ValueError("Need to pass `model` for train mode.")
        if optimizer is None:
            raise ValueError("Need to pass `optimizer` for train mode.")

        total_loss, lr = trainer.get_train_step(
            model=model,
            loss=loss,
            optimizer=optimizer,
        )
        return {
            "mode": mode,
            "loss": total_loss,
            "lr": lr,
        }

    raise ValueError(f"Unsupported mode {mode}")