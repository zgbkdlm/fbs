#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false

nparticles=$1

for i in $(seq 0 99);
do
    python toy/gp_filter.py --id=$i --d=100 --nsamples=10000 --nparticles=$nparticles &
done
wait
