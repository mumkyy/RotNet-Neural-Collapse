#!/bin/bash -l
#SBATCH --job-name=nc3lw_one
#SBATCH --output=../out/%x.%j.out
#SBATCH --error=../out/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=amr239
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4000M

module load wulver
module load Miniforge3
conda activate rotnet_legacy

# All args forwarded to main.py, e.g.:
#   submit_one.sh --exp <path> --output_model_path <dir> [--pretrained_path <dir>] --num_workers 8 --cuda
python main.py "$@"
