#!/bin/bash
#SBATCH -A Berzelius-2024-58
#SBATCH --gpus=1
#SBATCH -o toy-gibbs.log
#SBATCH -t 02:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nparticles=$1
sde=$2

PARALLEL_MAX=10
SEQUENTIAL_MAX=10

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
    for (( j=0;j<PARALLEL_MAX;j++ ))
    do
        k=$(( i*PARALLEL_MAX+j ))
        python toy/gp_gibbs.py --id=$k --d=100 --nsamples=10000 --nparticles=$nparticles --sde=$sde --explicit_backward &
    done
    wait
done
