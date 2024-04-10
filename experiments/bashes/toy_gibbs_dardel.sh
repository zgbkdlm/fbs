#!/bin/bash
#SBATCH -A naiss2024-22-535
#SBATCH -o toy-gibbs.log
#SBATCH -p shared
#SBATCH -n 20
#SBATCH --mem=16G
#SBATCH -a 0-99
#SBATCH -t 05:00:00

source ~/.bashrc

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nparticles=$1
sde=$2

python -u toy/gp_gibbs.py --id=$SLURM_ARRAY_TASK_ID --d=100 --nsamples=10000 --nparticles=$nparticles --sde=$sde --explicit_backward | tee -a logs/toy-gibbs.log
