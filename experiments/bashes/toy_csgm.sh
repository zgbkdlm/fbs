#!/bin/bash
#SBATCH -A Berzelius-2024-58
#SBATCH --gpus=1
#SBATCH -o toy-csgm.log
#SBATCH -t 01-10:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.01

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

sde=$1

PARALLEL_MAX=20
SEQUENTIAL_MAX=5

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
    for (( j=0;j<PARALLEL_MAX;j++ ))
    do
        k=$(( i*PARALLEL_MAX+j ))
        python toy/gp_csgm.py --id=$k --d=100 --nsamples=10000 --sde=$sde &
    done
    wait
done
