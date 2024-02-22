#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o celeba-sm.log
#SBATCH -t 11:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_fbs

nvidia-smi
python -u csmc_celeba.py --train --task="supr" --batch_size=128 --nsteps=128 --nepochs=1200 --grad_clip --save_mem --test_nsteps=500 --test_epoch=1199 --test_ema --test_seed=6 --nparticles=200 --ngibbs=500 --doob
