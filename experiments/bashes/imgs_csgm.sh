#!/bin/bash
#SBATCH -A Berzelius-2024-28
#SBATCH --gpus=1
#SBATCH -o imgs-csgm.log
#SBATCH -t 20:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.40

cd $WRKDIR/fbs
source ./venv/bin/activate

cd experiments

dataset=$1
start_from=${2:-0}

if [[ "$dataset" == "mnist" ]]
then
    rect_size=15
    sr_rate=4
elif [[ "$dataset" == "celeba-64" ]]
then
    rect_size=32
    sr_rate=2
elif [[ "$dataset" == "celeba-128" ]]
then
    rect_size=64
    sr_rate=2
else
    echo "Invalid dataset"
    exit 1
fi

python imgs/inpainting_csgm.py --dataset="$dataset" --rect_size=$rect_size --sde="lin" --test_nsteps=1000 --test_epoch=2999 --test_ema --test_seed=996 --ny0s=100 --start_from=$start_from --nsamples=100 &
python imgs/supr_csgm.py --dataset="$dataset" --rate=$sr_rate --sde="lin" --test_nsteps=1000 --test_epoch=2999 --test_ema --test_seed=996 --ny0s=100 --start_from=$start_from --nsamples=100 &
wait
