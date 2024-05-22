#!/bin/bash

mkdir -p imgs/results_supr/arrs
mkdir -p imgs/results_supr/imgs

mkdir -p imgs/results_inpainting/arrs
mkdir -p imgs/results_inpainting/imgs

dataset=$1
nparticles=$2

sbatch bashes/imgs_filter.sh $dataset $nparticles
sbatch bashes/imgs_gibbs.sh $dataset $nparticles
sbatch bashes/imgs_pmcmc.sh $dataset $nparticles
sbatch bashes/imgs_twisted.sh $dataset $nparticles
sbatch bashes/imgs_csgm.sh $dataset
