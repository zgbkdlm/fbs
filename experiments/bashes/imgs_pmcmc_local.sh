#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

dataset=$1
nparticles=$1

if [[ "$dataset" == "mnist" ]]
then
    rect_size=15
elif [[ "$dataset" == "celeba-64" ]]
then
    rect_size=40
elif [[ "$dataset" == "celeba-128" ]]
then
    rect_size=70
else
    echo "Invalid dataset"
    exit 1
fi

python imgs/inpainting.py --dataset="$dataset" --rect_size=$rect_size --sde="lin" --method="pmcmc-0.005" --test_nsteps=1000 --test_epoch=2999 --test_ema --test_seed=996 --ny0s=10 --nparticles=$nparticles --nsamples=100 &
python imgs/supr.py --dataset="$dataset" --rate=4 --sde="lin" --method="pmcmc-0.005" --test_nsteps=1000 --test_epoch=2999 --test_ema --test_seed=996 --ny0s=10 --nparticles=$nparticles --nsamples=100 &
wait