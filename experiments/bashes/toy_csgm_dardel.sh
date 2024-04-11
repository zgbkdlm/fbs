#!/bin/bash
#SBATCH -A naiss2024-22-535
#SBATCH -o toy-csgm.log
#SBATCH -p main
#SBATCH -n 30
#SBATCH --mem=8G
#SBATCH -a 0-99
#SBATCH -t 05:00:00

source ~/.bashrc

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

sde=$1

python -u toy/gp_csgm.py --id=$SLURM_ARRAY_TASK_ID --d=100 --nsamples=10000 --sde=$sde | tee logs/toy-csgm.log
