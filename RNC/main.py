from __future__ import print_function
import argparse
import os
import importlib.util

import algorithms as alg
from dataloader import DataLoader, GenericDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',         type=str, required=True, default='',  help='config file with parameters of the experiment')
    parser.add_argument('--evaluate',    default=False, action='store_true')
    parser.add_argument('--checkpoint',  type=int,      default=0,     help='checkpoint (epoch id) that will be loaded')
    parser.add_argument('--num_workers', type=int,      default=4,     help='number of data loading workers')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    parser.add_argument('--disp_step',   type=int,      default=50,    help='display step during training')
    args_opt = parser.parse_args()

    exp_config_file = os.path.join('.', 'config', args_opt.exp + '.py')
    config_root_arr = args_opt.exp.split('/')
    cfg_ROOT = config_root_arr[-1]
    exp_directory = os.path.join('.', 'experiments', cfg_ROOT)

    print('Launching experiment: %s' % exp_config_file)
    spec = importlib.util.spec_from_file_location("config", exp_config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    config['exp_dir'] = exp_directory
    print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
    print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

    data_train_opt = config['data_train_opt']
    data_test_opt = config['data_test_opt']
    num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None

    dataset_train = GenericDataset(
        dataset_name=data_train_opt['dataset_name'],
        split=data_train_opt['split'],
        random_sized_crop=data_train_opt['random_sized_crop'],
        num_imgs_per_cat=num_imgs_per_cat)
    
    dataset_test = GenericDataset(
        dataset_name=data_test_opt['dataset_name'],
        split=data_test_opt['split'],
        random_sized_crop=data_test_opt['random_sized_crop'])
    
    # Set pretext parameters only for unsupervised (backbone training)

    if data_train_opt['unsupervised']:
        dataset_train.pretext_mode = data_train_opt.get('pretext_mode', 'rotation')
        dataset_train.sigmas = data_train_opt.get('sigmas')
        dataset_train.kernel_sizes = data_train_opt.get('kernel_sizes')

        dataset_test.pretext_mode = data_test_opt.get('pretext_mode', 'rotation')
        dataset_test.sigmas = data_test_opt.get('sigmas')
        dataset_test.kernel_sizes = data_test_opt.get('kernel_sizes')

        # ---- NEW: jigsaw augmentation knobs ----
        dataset_train.patch_jitter = data_train_opt.get("patch_jitter", 0)
        dataset_train.color_distort = data_train_opt.get("color_distort", False)
        dataset_train.color_dist_strength = data_train_opt.get("color_dist_strength", 1.0)

        dataset_test.patch_jitter = data_test_opt.get("patch_jitter", 0)
        dataset_test.color_distort = data_test_opt.get("color_distort", False)
        dataset_test.color_dist_strength = data_test_opt.get("color_dist_strength", 1.0)
    
    # Auto-set classification head size for jigsaw
    if data_train_opt.get('pretext_mode', 'rotation') == 'jigsaw':
        shared_perms = dataset_train.jigsaw_perms
        dataset_test.jigsaw_perms = shared_perms
        config['networks']['model']['opt']['num_classes'] = len(dataset_train.jigsaw_perms)
        print("Jigsaw classes =", config['networks']['model']['opt']['num_classes'])
        print("Shared jigsaw perms =", shared_perms)



    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=data_train_opt['batch_size'],
        unsupervised=data_train_opt['unsupervised'],
        epoch_size=data_train_opt['epoch_size'],
        num_workers=args_opt.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=data_test_opt['batch_size'],
        unsupervised=data_test_opt['unsupervised'],
        epoch_size=data_test_opt['epoch_size'],
        num_workers=args_opt.num_workers,
        shuffle=False)

    config['disp_step'] = args_opt.disp_step
    #
    algorithm = getattr(alg, config['algorithm_type'])(config)
    if args_opt.cuda:
        algorithm.load_to_gpu()
    #passes checkpoint 200 to algorithim 
    if args_opt.checkpoint > 0:
        algorithm.load_checkpoint(args_opt.checkpoint, train=(not args_opt.evaluate))

    if not args_opt.evaluate:
        algorithm.solve(dloader_train, dloader_test)
    else:
        algorithm.evaluate(dloader_test)

if __name__ == '__main__':
    main()
