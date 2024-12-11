#!/bin/bash

#SBATCH --account=bcsi-delta-gpu
#SBATCH --array=0
#SBATCH --partition=gpuA100x4
#SBATCH --time=00-10:00:00
### NODE INFO ###
#SBATCH --nodes=1
### CPU/MEM/GPU options ###
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
### LOG INFO ###
#SBATCH --job-name=diffvax
#SBATCH --output=logs/slurm/cv/cv%A-%a.log
export MODEL_NAME="diffvax" 
export RUN_NAME=${diffvax}"_cv"
mkdir -p logs/
# ctrl d both of above to modify MODEL_NAME and RUN_NAME with job and log name

# module purge


python diffvax.py --epochs 10 --bs 4 --lr 1e-5 --alpha 4 --num_workers 4 --dataset_limit 100 --run_name "diffvax_dl_100"