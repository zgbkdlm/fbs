#!/bin/bash

nparticles=$1
sde=$2
cluster=$3

if [[ "$cluster" == "dardel" ]]
then
    sbatch bashes/toy_filter_dardel.sh $nparticles $sde
    sbatch bashes/toy_gibbs_dardel.sh $nparticles $sde
    sbatch bashes/toy_pmcmc_dardel.sh $nparticles $sde 0.005
    sbatch bashes/toy_twisted_dardel.sh $nparticles $sde
    sbatch bashes/toy_csgm_dardel.sh $sde
else
    sbatch bashes/toy_filter.sh $nparticles $sde
    sbatch bashes/toy_gibbs.sh $nparticles $sde
    sbatch bashes/toy_pmcmc.sh $nparticles $sde 0.005
    sbatch bashes/toy_twisted.sh $nparticles $sde
    sbatch bashes/toy_csgm.sh $sde
fi
