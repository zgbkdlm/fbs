#!/bin/bash
#SBATCH -A Berzelius-2024-58
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH -o train-sb.log
#SBATCH -t 01-00:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nvidia-smi
python -u sb_imgs/train.py --dataset="mnist" --sde="lin" --batch_size=64 --nsteps=32 --schedule="cos" --nepochs=10 --grad_clip --vmap_loss --nn_dim=64 --nsbs=20
