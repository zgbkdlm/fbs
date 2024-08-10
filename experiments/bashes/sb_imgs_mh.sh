#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

nparticles=$1
use_mh=$2

if $use_mh; then
  python sb_imgs/supr_with_acc.py --sb_step=9 --ny0s=1000 --test_nsteps=128 --nsamples=100 --nparticles=$nparticles --use_mh
else
  python sb_imgs/supr_with_acc.py --sb_step=9 --ny0s=1000 --test_nsteps=128 --nsamples=100 --nparticles=$nparticles
fi
