#!/bin/bash
#SBATCH -A naiss2024-22-535
#SBATCH -o toy-pmcmc.log
#SBATCH -p main
#SBATCH -n 30
#SBATCH --mem=8G
#SBATCH -a 0-99
#SBATCH -t 10:00:00

source ~/.bashrc

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nparticles=$1
sde=$2
delta=$3

python -u toy/gp_pmcmc.py --id=$SLURM_ARRAY_TASK_ID --d=100 --nsamples=10000 --nparticles=$nparticles --sde=$sde --delta=$delta | tee logs/toy-pmcmc.log
