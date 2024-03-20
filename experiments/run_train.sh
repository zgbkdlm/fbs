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
python -u train.py --dataset=$1 --sde=$2 --batch_size=256 --nsteps=256 --schedule="cos" --nepochs=3000 --grad_clip --save_mem
