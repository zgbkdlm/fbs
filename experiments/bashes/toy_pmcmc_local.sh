#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05

nparticles=$1
sde=$2
delta=$3

PARALLEL_MAX=10
SEQUENTIAL_MAX=10

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
    for (( j=0;j<PARALLEL_MAX;j++ ))
    do
        k=$(( i*PARALLEL_MAX+j ))
        python toy/gp_pmcmc.py --id=$k --d=100 --nsamples=10000 --nparticles=$nparticles --sde=$sde --delta=$delta &
    done
    wait
done
