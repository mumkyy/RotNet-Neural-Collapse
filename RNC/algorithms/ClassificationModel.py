from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time
import torch.nn.functional as F

from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.feats = {}
        if 'nc_reg' in opt:
            for layer in opt['nc_reg']['layers']:
                idx = self.networks['model'].all_feat_names.index(layer)
                block = self.networks['model']._feature_blocks[idx]
                block.register_forward_hook(lambda m,i,o, name=layer:
                                            self.feats.__setitem__(name, o.view(o.size(0),-1)))


    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients
            self.optimizers['model'].zero_grad()
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = dataX
        labels_var = labels
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        #********************************************************

        C = pred_var.size(1)
    
        #*************** COMPUTE LOSSES *************************
        crit = self.criterions['loss']
        if isinstance(crit, nn.MSELoss):
            # one-hot encode targets for MSE
            y_oh = F.one_hot(labels_var, num_classes=C).float().to(pred_var.device)
            loss_total = crit(pred_var, y_oh)
        else:
            loss_total = crit(pred_var, labels_var)

        # --- NC1 penalty (safe, simplest) ---
        if do_train and ('nc_reg' in self.opt):
            # optional warmup: enable after N epochs
            warm = self.opt['nc_reg'].get('warmup_epochs', 0)
            if self.curr_epoch >= warm:
                for layer in self.opt['nc_reg']['layers']:
                    z = self.feats.get(layer, None)
                    if z is None:       # hook hasn’t fired for some reason
                        continue
                    z = z.float()        # AMP stability

                    mu = z.mean(0)
                    Sw = z.new_tensor(0.0)
                    Sb = z.new_tensor(0.0)
                    for c in range(C):
                        mask = (labels_var == c)
                        if mask.sum() < 2:
                            continue
                        zc = z[mask]
                        mu_c = zc.mean(0)
                        Sw = Sw + ((zc - mu_c)**2).sum()
                        Sb = Sb + zc.size(0) * ((mu_c - mu)**2).sum()

                    nc1 = torch.clamp(Sw / (Sb.detach() + 1e-6), min=1e-3)
                    penalty = -torch.log(nc1 + 1e-6)
                    w = self.opt['nc_reg']['weights'][layer]
                    loss_total = loss_total + w * penalty


        record = {}
        # precision still computed on raw logits
        record['prec1'] = accuracy(pred_var.data, labels_var, topk=(1,))[0].item()
        record['loss']  = loss_total.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
