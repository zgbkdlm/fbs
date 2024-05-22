#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

for (( i=0;i<10;i++ ))
do
    python sb_imgs/supr.py --sb_step=9 --method='filter' --y0_id=$i --test_nsteps=64 --nsamples=100 --nparticles=100 &
    python sb_imgs/supr.py --sb_step=9 --method='gibbs' --y0_id=$i --test_nsteps=64 --nsamples=100 --nparticles=100 &
    wait
done
