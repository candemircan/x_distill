#!/bin/bash

#SBATCH -J extract_representations
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH -t 04:00:00
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4

source $HOME/.bashrc
conda activate x_distill
python $X_DISTILL/bin/extract_representations.py