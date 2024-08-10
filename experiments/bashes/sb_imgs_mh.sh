#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

nparticles=$1
use_mh=$2

for (( i=0;i<100;i++ ))
do
    if $use_mh; then
      python sb_imgs/supr_with_acc.py --sb_step=9 --y0_id=$i --test_nsteps=128 --nsamples=100 --nparticles=$nparticles --use_mh
    else
      python sb_imgs/supr_with_acc.py --sb_step=9 --y0_id=$i --test_nsteps=128 --nsamples=100 --nparticles=$nparticles
    fi
done
