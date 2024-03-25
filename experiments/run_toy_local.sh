#!/bin/bash

for i in $(seq 0 99);
do
    python toy/gp_filter.py --id=$i --d=100 --nsamples=10000 --nparticles=10
done

for i in $(seq 0 99);
do
    python toy/gp_gibbs.py --id=$i --d=100 --nsamples=10000 --nparticles=10 --explicit_backward
done

for i in $(seq 0 99);
do
    python toy/gp_pmcmc.py --id=$i --d=100 --nsamples=10000 --nparticles=10 --delta=0.01
done
