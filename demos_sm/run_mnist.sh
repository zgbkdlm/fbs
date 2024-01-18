#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o mnist.log
#SBATCH -t 01:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_sm

nvidia-smi
python -u mnist.py --train --nn="unet" --schedule="cos" --batch_size=32
