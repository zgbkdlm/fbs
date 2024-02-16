#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o mnist.log
#SBATCH -t 10:30:00
#SBATCH -C "fat"

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_fbs

nvidia-smi
python csmc_mnist.py --train --nn="mlp"
