#!/bin/bash

if [[ "$1" == "filter" ]]
then
    for i in $(seq 0 49);
    do
        python toy/gp_filter.py --id=$i --d=100 --nsamples=10000 --nparticles=10
    done
elif [[ "$1" == "gibbs" ]]
then
    for i in $(seq 0 49);
    do
        python toy/gp_gibbs.py --id=$i --d=100 --nsamples=10000 --nparticles=10 --explicit_backward
    done
elif [[ "$1" == "pmcmc" ]]
then
    for i in $(seq 0 49);
    do
        python toy/gp_pmcmc.py --id=$i --d=100 --nsamples=10000 --nparticles=10 --delta=0.01
    done
elif [[ "$1" == "twisted" ]]
then
    for i in $(seq 0 49);
    do
        python toy/gp_twisted.py --id=$i --d=100 --nsamples=10000 --nparticles=10
    done
else
    echo "What do you mean?"
fi
