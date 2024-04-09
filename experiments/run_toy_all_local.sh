#!/bin/bash

nparticles=$1
sde=$2

bash bashes/toy_filter_local.sh $nparticles $sde
bash bashes/toy_gibbs_local.sh $nparticles $sde
bash bashes/toy_pmcmc_local.sh $nparticles $sde 0.005
bash bashes/toy_twisted_local.sh $nparticles $sde
bash bashes/toy_csgm_local.sh $sde
