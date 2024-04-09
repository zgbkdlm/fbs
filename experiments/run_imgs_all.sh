#!/bin/bash

mkdir -p imgs/results_supr/arrs
mkdir -p imgs/results_supr/imgs

mkdir -p imgs/results_inpainting/arrs
mkdir -p imgs/results_inpainting/imgs

dataset=$1
nparticles=$1

slurm bashes/imgs_filter.sh $dataset $nparticles
slurm bashes/imgs_gibbs.sh $dataset $nparticles
slurm bashes/imgs_pmcmc.sh $dataset $nparticles
slurm bashes/imgs_twisted.sh $dataset $nparticles
slurm bashes/imgs_csgm.sh $dataset
