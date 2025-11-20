"""Define a generic class for training and testing learning algorithms."""
from __future__ import print_function
import os
import os.path
import importlib.util
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

import utils
import datetime
import logging

from pdb import set_trace as breakpoint


class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()

        self.logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}

        self.keep_best_model_metric_name = opt['best_metric'] if ('best_metric' in opt) else None

    def set_experiment_dir(self, directory_path):
        self.exp_dir = directory_path
        os.makedirs(self.exp_dir, exist_ok=True)

        self.vis_dir = os.path.join(directory_path, 'visuals')
        os.makedirs(self.vis_dir, exist_ok=True)

        self.preds_dir = os.path.join(directory_path, 'preds')
        os.makedirs(self.preds_dir, exist_ok=True)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)


    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info('Set network %s' % key)
            def_file = val['def_file']
            net_opt = val['opt']
            self.optim_params[key] = val.get('optim_params', None)
            pretrained_path = val.get('pretrained', None)
            self.networks[key] = self.init_network(def_file, net_opt, pretrained_path, key)

    def init_network(self, net_def_file, net_opt, pretrained_path, key):
        self.logger.info('==> Initialize network %s from file %s with opts: %s' % (key, net_def_file, net_opt))
        if not os.path.isfile(net_def_file):
            raise ValueError('Non existing file: {0}'.format(net_def_file))

        spec = importlib.util.spec_from_file_location("custom_model", net_def_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        network = module.create_model(net_opt)

        if pretrained_path:
            self.load_pretrained(network, pretrained_path)

        return network

    def load_pretrained(self, network, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % pretrained_path)
        assert os.path.isfile(pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        if pretrained_model['network'].keys() == network.state_dict().keys():
            network.load_state_dict(pretrained_model['network'])
        else:
            self.logger.info('==> WARNING: network parameters in pre-trained file do not strictly match')
            for pname, param in network.named_parameters():
                if pname in pretrained_model['network']:
                    self.logger.info('==> Copying parameter %s' % pname)
                    param.data.copy_(pretrained_model['network'][pname])

    def init_all_optimizers(self):
        self.optimizers = {}
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams:
                self.optimizers[key] = self.init_optimizer(self.networks[key], oparams, key)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for network: %s' % (optim_type, optim_opts, key))

        if optim_type == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate, betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            return torch.optim.SGD(parameters, lr=learning_rate,
                                   momentum=optim_opts['momentum'],
                                   nesterov=optim_opts.get('nesterov', False),
                                   weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val.get('opt', None)
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def load_to_gpu(self):
        for key in self.networks:
            self.networks[key] = self.networks[key].cuda()
        for key in self.criterions:
            self.criterions[key] = self.criterions[key].cuda()
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key in self.networks:
            if self.optimizers[key] is None:
                continue
            self.save_network(key, epoch, suffix)
            self.save_optimizer(key, epoch, suffix)

    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info('Load checkpoint of epoch %d' % epoch)
        for key in self.networks:
            if self.optim_params[key] is None:
                continue
            self.load_network(key, epoch, suffix)

        if train:
            self.init_all_optimizers()
            for key in self.networks:
                if self.optim_params[key] is None:
                    continue
                self.load_optimizer(key, epoch, suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=''):
        for key in self.networks:
            if self.optimizers[key] is None:
                continue
            filename_net = self._get_net_checkpoint_filename(key, epoch) + suffix
            if os.path.isfile(filename_net):
                os.remove(filename_net)
            filename_optim = self._get_optim_checkpoint_filename(key, epoch) + suffix
            if os.path.isfile(filename_optim):
                os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix=''):
        filename = self._get_net_checkpoint_filename(net_key, epoch) + suffix
        state = {'epoch': epoch, 'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        filename = self._get_optim_checkpoint_filename(net_key, epoch) + suffix
        state = {'epoch': epoch, 'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def load_network(self, net_key, epoch, suffix=''):
        filename = self._get_net_checkpoint_filename(net_key, epoch) + suffix
        assert os.path.isfile(filename)
        checkpoint = torch.load(filename)
        self.networks[net_key].load_state_dict(checkpoint['network'])

    def load_optimizer(self, net_key, epoch, suffix=''):
        filename = self._get_optim_checkpoint_filename(net_key, epoch) + suffix
        assert os.path.isfile(filename)
        checkpoint = torch.load(filename)
        self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, f"{net_key}_net_epoch{epoch}")

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, f"{net_key}_optim_epoch{epoch}")

    def solve(self, data_loader_train, data_loader_test):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if not self.optimizers:
            self.init_all_optimizers()

        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch + 1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.logger.info('==> Training stats: %s' % train_stats)

            self.save_checkpoint(self.curr_epoch + 1)
            #if start_epoch != self.curr_epoch:
                #self.delete_checkpoint(self.curr_epoch)

            if data_loader_test is not None:
                eval_stats = self.evaluate(data_loader_test)
                self.logger.info('==> Evaluation stats: %s' % eval_stats)
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)

        self.print_eval_stats_of_best_model()

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))
        self.dloader = data_loader
        self.dataset_train = data_loader.dataset

        for key, network in self.networks.items():
            network.train() if self.optimizers[key] else network.eval()

        disp_step = self.opt.get('disp_step', 50)
        train_stats = utils.DAverageMeter()
        self.bnumber = len(data_loader())
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx + 1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch + 1, idx + 1, len(data_loader), train_stats.average()))

        return train_stats.average()

    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        eval_stats = utils.DAverageMeter()
        self.bnumber = len(dloader())
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())
        return eval_stats.average()

    def adjust_learning_rates(self, epoch):
        for key, oparams in self.optim_params.items():
            if not oparams or 'LUT_lr' not in oparams:
                continue
            LUT = oparams['LUT_lr']
            lr = next((lr for max_epoch, lr in LUT if max_epoch > epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if metric_name not in eval_stats:
                raise ValueError(f'The provided metric {metric_name} is not computed.')
            metric_val = eval_stats[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(self.curr_epoch + 1, suffix='.best')
                #if self.best_epoch is not None:
                #    self.delete_checkpoint(self.best_epoch + 1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s' % (metric_name, self.best_epoch + 1, self.best_stats))

    def train_step(self, batch):
        pass

    def evaluation_step(self, batch):
        pass

    def allocate_tensors(self):
        self.tensors = {}
