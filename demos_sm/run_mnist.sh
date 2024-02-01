#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o mnist.log
#SBATCH -t 03:00:00
#SBATCH -C "fat"

upsampling=$1
sde=$2
batch_size=$3
nsteps=$4
lr=$5
schedule=$6
loss_type=$7
grad_clip=$8


source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/fbs
source ./venv/bin/activate

cd demos_sm

nvidia-smi

if "$grad_clip"
then
  python -u mnist.py --train --upsampling=$upsampling --sde=$sde --batch_size=$batch_size --nsteps=$nsteps --lr=$lr --schedule=$schedule --loss_type=$loss_type --grad_clip
else
  python -u mnist.py --train --upsampling=$upsampling --sde=$sde --batch_size=$batch_size --nsteps=$nsteps --lr=$lr --schedule=$schedule --loss_type=$loss_type
fi
