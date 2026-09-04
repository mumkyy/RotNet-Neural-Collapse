#!/bin/bash -l
#SBATCH --job-name=meas_one
#SBATCH --output=../out/%x.%j.out
#SBATCH --error=../out/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=amr239
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4000M

module load wulver
module load Miniforge3
conda activate rotnet_legacy

# All args forwarded to the measurement entrypoint, e.g.:
#   submit_meas_one.sh --exp <path> --exp-dir <scratch dir> --arch-class <cls> \
#                      --dataset_name_arg <ds> --pretext-mode <mode> --num-classes <N> \
#                      --layers <keys> --out-root <dir> --nc4 --nc4-layerwise --pabs --nc2
python -m measurements.measurementsFix "$@"
