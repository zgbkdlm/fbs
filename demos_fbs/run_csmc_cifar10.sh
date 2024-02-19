#!/bin/bash
#SBATCH -A Berzelius-2023-194
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
python -u csmc_cifar10.py --train --task="supr" --batch_size=128 --nsteps=128 --grad_clip --test_nsteps=500 --nepochs=40 --test_epoch=39 --test_ema --test_seed=76543 --nparticles=200 --ngibbs=500 --doob
