#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.01

nparticles=$1

PARALLEL_MAX=20
SEQUENTIAL_MAX=5

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
    for (( j=0;j<PARALLEL_MAX;j++ ))
    do
        k=$(( i*PARALLEL_MAX+j ))
        python toy/gp_filter.py --id=$k --d=100 --nsamples=10000 --nparticles=$nparticles &
    done
    wait
done
