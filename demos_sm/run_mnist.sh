#!/bin/bash
#SBATCH -A Berzelius-2023-146
#SBATCH --gpus=1
#SBATCH -o mnist.log
#SBATCH -t 01:00:00

nn=$1
batch_size=$2
nsteps=$3
lr=$4
sde=$5

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_sm

nvidia-smi
python -u mnist.py --train --schedule="cos" --nn=$nn --batch_size=$batch_size --nsteps=$nsteps --lr=$lr --sde=$sde
