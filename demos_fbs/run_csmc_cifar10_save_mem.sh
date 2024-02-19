#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o cifar10-sm.log
#SBATCH -t 05:00:00
#SBATCH -C "fat"

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_fbs

nvidia-smi
python -u csmc_cifar10.py --train --task="supr" --batch_size=256 --nsteps=256 --nepochs=100 --grad_clip --lr=1e-3 --test_nsteps=500 --test_epoch=99 --test_ema --test_seed=76543 --nparticles=200 --ngibbs=500 --doob
