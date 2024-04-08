#!/bin/bash

dataset=$1
nparticles=$1

slurm bashes/imgs_filter.sh $dataset $nparticles
slurm bashes/imgs_gibbs.sh $dataset $nparticles
slurm bashes/imgs_pmcmc.sh $dataset $nparticles
slurm bashes/imgs_twisted.sh $dataset $nparticles
