# main.py
import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Ensure repo root is importable (so "architectures.*" works when running from root)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from architectures.train import train_loop


def _load_config(cfg_path: str):
    cfg_path = Path(cfg_path)
    if cfg_path.suffix != ".py":
        cfg_path = cfg_path.with_suffix(".py")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    spec = importlib.util.spec_from_file_location("config_module", str(cfg_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "config"):
        raise AttributeError(f"Config file {cfg_path} must define a top-level variable named `config`.")
    return module.config, str(cfg_path)


def parse_args():
    p = argparse.ArgumentParser(description="Run audio pretext training from a config file.")
    p.add_argument(
        "--exp",
        required=True,
        help="Path to config .py (e.g. config/speechCommands/reverse/MSE/.../SPEECHCOMMANDS_Reverse_NINAudio_TwoClass.py)",
    )
    p.add_argument("--no_cuda", action="store_true", help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main():
    args = parse_args()
    config, cfg_path = _load_config(args.exp)

    print(f"Launching experiment: {cfg_path}")
    print(f"Max epochs: {config.get('max_num_epochs')}")
    print(f"Model file: {config['networks']['model']['def_file']}")
    print(f"Loss: {config['criterions']['loss']['ctype']}")

    train_loop(config, no_cuda=args.no_cuda)


if __name__ == "__main__":
    main()
