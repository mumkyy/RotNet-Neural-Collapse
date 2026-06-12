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
            model = self.networks['model']
            for layer in opt['nc_reg']['layers']:
                feat_module = model.get_feature_module(layer)
                #NO FLATTEN HERE    
                feat_module.register_forward_hook(
                    lambda m, i, o, name=layer: self.feats.__setitem__(name, o)
                )

        '''        
        NC-3 Regularizer :
        use in config under [opt][nc3_reg] it should be a dictionary of the following shape 
        {
          last_layer: 'name of the penultimate layer to the classifier where the features will be extracted from', 
          classifier: 'name of the classifier where the weights will come from' ,
          lambdaNC3 : float 
        }

        example 

        {
            "last_layer": "lin2", 
            "classifier": "classifier", 
            lambdaNC3: 1e-4
        }

        the value of lambdaNC3 which will be the scalar multiple of the log term produced in the 
        formulation of the penalty : total_loss += (-log(NC-3 + epsilon) * lambdaNC3) 
        '''

        if 'nc3_reg' in opt:
            self.nc3Feats = {}
            self.seenExamples = 0

            model = self.networks['model']

            self.penult_layer = opt['nc3_reg']['last_layer']
            classifier_layer = opt['nc3_reg']['classifier']

            try:
                penult_feats = model.get_feature_module(self.penult_layer)
                classifier_module = model.get_feature_module(classifier_layer)
            except KeyError as e:
                raise ValueError(
                    f"Invalid nc3_reg feature key: {e}. "
                    f"Passed last_layer={self.penult_layer}, classifier={classifier_layer}. "
                    f"Available keys: {getattr(model, 'all_feat_names', 'unknown')}"
                )

            self.W = classifier_module.weight

            penult_feats.register_forward_hook(
                lambda module, input, output, name=self.penult_layer:
                    self.nc3Feats.__setitem__(name, output)
            )

            self.runningSum = None

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
            if self.opt.get('mse_on_probs', False):
                pred_for_loss = torch.sigmoid(pred_var)
            else:
                pred_for_loss = pred_var

            loss_total = crit(pred_for_loss, y_oh)
        else:
            loss_total = crit(pred_var, labels_var)
        loss_cls = loss_total
        nc1_penalty_total = 0.0
        nc1_value_total = 0.0
        nc1_sw_total = 0.0
        nc1_sb_total = 0.0
        nc1_layers = 0

        if do_train and ('nc3_reg' in self.opt): 
            
            eps = 1e-6 
            nc3 = 0.0

            pen_z = self.nc3Feats[self.penult_layer]

            if pen_z.dim() == 4:
                pen_z = pen_z.mean(dim=(2,3))
            B, D = pen_z.shape
            C = pred_var.size(1)

            if self.runningSum is None:
                self.runningSum = pen_z.new_zeros(1, D)

            # if self.muCRunningSums is None:
            #     self.muCRunningSums = pen_z.new_zeros(C, D)

            with torch.no_grad():
                self.runningSum += pen_z.detach().sum(dim=0, keepdim=True) # This should sum up each representation in the batch and then add to the running sum 
                self.seenExamples += B 
                muG = self.runningSum / self.seenExamples 

            


            nc3counts = torch.bincount(labels_var, minlength=C).float().to(pen_z.device)
            batch_sums = pen_z.new_zeros(C, D)
            batch_sums.index_add_(0, labels_var, pen_z)
            # self.muCRunningSums += batch_sums
            batch_muC = batch_sums / nc3counts.clamp_min(1.0).unsqueeze(1)  # (C, D)

            M_dot = batch_muC - muG 
            
            if self.W.size(0) == C and self.W.size(1) == D:
                WT = self.W 
            else:
                WT = (self.W).T 
            
            Wnorm = self.W.norm(dim=1, keepdim=True, p=2).clamp_min(eps)

            M_dotnorm = M_dot.norm(dim=1, keepdim=True, p=2).clamp_min(eps)

            

            left = WT / Wnorm
            right = M_dot / M_dotnorm

            present = (nc3counts > 0)
            left = left[present]
            right = right[present]

            diff = left - right

            nc3 = torch.norm(diff, ord='fro')

            loss_total += (self.opt['nc3_reg']['lambdaNC3'] * torch.log(nc3 + eps) * -1) 

        # --- NC1 penalty ---
        #no warmup
        if do_train and ('nc_reg' in self.opt):
            eps = 1e-6
            for layer in self.opt['nc_reg']['layers']:
                z = self.feats.get(layer, None)
                if z is None:       # hook hasn’t fired for some reason
                    continue

                if z.dim() == 4:
                    z = z.mean(dim=(2, 3))

                B, D = z.shape

                # Per-class counts + means
                counts = torch.bincount(labels_var, minlength=C).float()
                sums = z.new_zeros(C, D)
                sums.index_add_(0, labels_var, z)
                means = sums / counts.clamp_min(1.0).unsqueeze(1)  # (C, D)

                # trace(Sw) = average within-class squared distance
                mu_y = means.index_select(0, labels_var)           # (B, D)
                trace_Sw = ((z - mu_y) ** 2).sum() / float(B)

                # trace(Sb): sum_c n_c ||mu_c - mu||^2 where mu is global mean of samples
                mu = z.mean(0)                                                          # (D,)
                diff = means - mu.unsqueeze(0)                                          # (C,D)
                trace_Sb = (counts.unsqueeze(1) * (diff ** 2)).sum() / float(B)         # scalar

                if self.opt['nc_reg'].get('detach_sb', False):
                    nc1 = trace_Sw / (trace_Sb.detach() + eps)
                else:
                    nc1 = trace_Sw / (trace_Sb + eps)

                if self.opt['nc_reg'].get('inverse',False):
                    penalty = 1.0 / (nc1 + eps)
                else:
                    penalty = -torch.log(nc1 + eps)

                w = self.opt['nc_reg']['weights'][layer]
                loss_total = loss_total + w * penalty
                nc1_penalty_total += (w * penalty).item()
                nc1_value_total += nc1.item()
                nc1_sw_total += trace_Sw.item()
                nc1_sb_total += trace_Sb.item()
                nc1_layers += 1


        record = {}
        # precision still computed on raw logits
        record['prec1'] = accuracy(pred_var.detach(), labels_var, topk=(1,))[0].item()
        record['loss']  = loss_total.item()
        record['loss_cls'] = loss_cls.item()
        if nc1_layers > 0:
            record['nc1'] = nc1_value_total / nc1_layers
            record['nc1_penalty'] = nc1_penalty_total
            record['nc1_sw'] = nc1_sw_total / nc1_layers
            record['nc1_sb'] = nc1_sb_total / nc1_layers
            record['nc1_layers'] = nc1_layers
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
