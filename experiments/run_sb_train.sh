#!/bin/bash
#SBATCH -A Berzelius-2024-58
#SBATCH --gpus=1
#SBATCH -o train.log
#SBATCH -t 01-18:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nvidia-smi
python -u sb_imgs/train.py --dataset=$1 --sde=$2 --batch_size=128 --nsteps=32 --schedule="cos" --T=0.5 --nepochs=500 --grad_clip
