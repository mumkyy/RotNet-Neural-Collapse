from __future__ import print_function
import argparse
import os
import importlib.util

from train import train_loop as alg
from dataloader import DataLoader, GenericDataset

def parse_args(): 
    p = argparse.ArgumentParser(
        "Main will initialize the training loop"
    )
    p.add_argument(
        '--exp', required=True,
        help = 'config file with path'
    )
    p.add_argument(
        '--no_cuda', default=False, 
        help = 'disable cuda'
    )
    return p.parse_args()

def main(): 
    args = parse_args()
    cfg_file = 'config/' + args.exp + '.py'
    exp_dir = os.path.join('.','experiment',args.exp)

    spec = importlib.util.spec_from_file_location(cfg_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config 

    if config['exp_dir']: 
        exp_dir =  config['exp_dir']   
    

    print(f"launching experimen {args.exp}")

    data_train_opt = config[data_train_opt] 
    data_test_opt = config[data_test_opt]

    num_classes = config[num_classes]
    

if __name__ == '__main__': 
    main()
