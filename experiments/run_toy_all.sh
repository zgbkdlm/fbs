#!/bin/bash

nparticles=$1
sde=$2

sbatch bashes/toy_filter.sh $nparticles $sde
sbatch bashes/toy_gibbs.sh $nparticles $sde
sbatch bashes/toy_pmcmc.sh $nparticles $sde 0.005
sbatch bashes/toy_twisted.sh $nparticles $sde
