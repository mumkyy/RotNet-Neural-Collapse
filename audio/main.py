import argparse
import os
import importlib.util
import sys

# Ensure architectures can be imported
sys.path.append(os.getcwd())

from architectures.train import train_loop

def parse_args(): 
    p = argparse.ArgumentParser(description="Main will initialize the training loop")
    p.add_argument('--exp', required=True, help='config file path (e.g., config/speechCommands/reverse/MSE/...)')
    return p.parse_args()

def main(): 
    args = parse_args()
    
    # Construct python module path
    # Assuming user passes: config/speechCommands/reverse/MSE/.../config_file
    cfg_path = args.exp
    if not cfg_path.endswith('.py'):
        cfg_path += '.py'
        
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Dynamic Config Loading
    spec = importlib.util.spec_from_file_location("config_module", cfg_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    config = config_module.config 
    
    print(f"Launching experiment: {cfg_path}")
    print(f"Model Stages: {config['networks']['model']['opt']['num_stages']}")
    print(f"Loss Function: {config['criterions']['loss']['ctype']}")

    train_loop(config)

if __name__ == '__main__': 
    main()