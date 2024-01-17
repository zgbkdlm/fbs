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
python mnist.py --train --nn="conv" --schedule="cos"
python mnist.py --train --nn="conv" --schedule="exp"
python mnist.py --train --nn="conv" --schedule="const"
