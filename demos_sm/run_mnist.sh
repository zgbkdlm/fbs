#!/bin/bash
#SBATCH -A Berzelius-2023-146
#SBATCH --gpus=1
#SBATCH -o mnist.log
#SBATCH -t 02:00:00
#SBATCH -C "fat"

nn=$1
batch_size=$2
nsteps=$3
lr=$4
sde=$5
loss_type=$6
grad_clip=$7


source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_sm

nvidia-smi

if "$grad_clip"
then
  python -u mnist.py --train --schedule="const" --nn=$nn --batch_size=$batch_size --nsteps=$nsteps --lr=$lr --sde=$sde --loss_type=$loss_type --grad_clip
else
  python -u mnist.py --train --schedule="const" --nn=$nn --batch_size=$batch_size --nsteps=$nsteps --lr=$lr --sde=$sde --loss_type=$loss_type
fi
