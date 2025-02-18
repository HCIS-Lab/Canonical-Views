#!/bin/bash

if [ "$1" = "" ]; then
    echo "Job name cannot be empty"
    exit 1
fi

export JOB_NAME=$1_$(date '+%Y-%m-%d_%H:%M:%S')

#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00

# Ensure the output directory exists before Slurm writes logs
mkdir -p experiments

# Redirect stdout and stderr properly (this prevents Slurm from creating `slurm-<job_id>.out`)
exec > "experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2> "experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

module --ignore-cache load "conda"
source ~/.bashrc
conda activate hopper

# Run PyTorch script
srun python rl.py