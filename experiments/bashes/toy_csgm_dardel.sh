#!/bin/bash
#SBATCH -A naiss2024-22-535
#SBATCH -o toy-csgm.log
#SBATCH -p shared
#SBATCH -n 20
#SBATCH --mem=16G
#SBATCH -a 0-99
#SBATCH -t 02:00:00

source ~/.bashrc

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

sde=$1

python toy/gp_csgm.py --id=$SLURM_ARRAY_TASK_ID --d=100 --nsamples=10000 --sde=$sde
