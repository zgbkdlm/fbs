#!/bin/bash

mkdir -p sb/results

sbatch bashes/sb_filter_dardel.sh 4
sbatch bashes/sb_filter_dardel.sh 8
sbatch bashes/sb_filter_dardel.sh 16
sbatch bashes/sb_filter_dardel.sh 32
sbatch bashes/sb_filter_dardel.sh 64
sbatch bashes/sb_gibbs_dardel.sh 4
sbatch bashes/sb_gibbs_dardel.sh 8
sbatch bashes/sb_gibbs_dardel.sh 16
sbatch bashes/sb_gibbs_dardel.sh 32
sbatch bashes/sb_gibbs_dardel.sh 64
