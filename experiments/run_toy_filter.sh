#!/bin/bash
#SBATCH -A Berzelius-2024-58
#SBATCH --gpus=1
#SBATCH -o toy-filter.log
#SBATCH -t 02:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nvidia-smi

for i in $(seq 0 99);
do
    python toy/gp_filter.py --id=$i --d=100 --nsamples=10000 --nparticles=10
done
