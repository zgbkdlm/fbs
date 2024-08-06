#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.02

nparticles=$1

PRUNS=10

for obs_var in 0.01 0.1 1 10; do
    for (( i=0;i<PRUNS;i++ )); do
        python toy/gp_linear_gibbs.py --id=$i --d=100 --obs_var=$obs_var --nsamples=10000 --nparticles=$nparticles --explicit_backward &
    done
    wait
done
