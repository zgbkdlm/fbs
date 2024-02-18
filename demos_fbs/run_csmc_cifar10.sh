#!/bin/bash
#SBATCH -A Berzelius-2024-28
#SBATCH --gpus=1
#SBATCH -o cifar10.log
#SBATCH -t 11:00:00
#SBATCH -C "fat"

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_fbs

nvidia-smi
python -u csmc_cifar10.py --train --task="supr" --batch_size=64 --nsteps=64 --grad_clip --test_nsteps=1000 --test_epoch=29 --test_ema --test_seed=567 --nparticles=100 --ngibbs=500