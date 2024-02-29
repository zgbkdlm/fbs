#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o celeba-sm.log
#SBATCH -t 01-12:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_fbs

nvidia-smi
python -u csmc_celeba.py --train --task=$1 --batch_size=128 --nsteps=128 --nepochs=3100 --grad_clip --save_mem --test_nsteps=500 --test_epoch=3099 --test_ema --test_seed=555 --nparticles=100 --doob
