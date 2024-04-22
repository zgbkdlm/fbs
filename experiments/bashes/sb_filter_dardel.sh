#!/bin/bash
#SBATCH -A naiss2024-22-535
#SBATCH -o sb-filter.log
#SBATCH -p main
#SBATCH -n 30
#SBATCH --mem=4G
#SBATCH -a 0-99
#SBATCH -t 00:30:00

source ~/.bashrc

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

nparticles=$1

python -u sb/filter.py --id=$SLURM_ARRAY_TASK_ID --d=10 --nsamples=10000 --nparticles=$nparticles --x0="proper" &
python -u sb/filter.py --id=$SLURM_ARRAY_TASK_ID --d=10 --nsamples=10000 --nparticles=$nparticles --x0="heuristic"
wait
