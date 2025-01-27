#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=random  # Default job name, overridden by `-J`

# Ensure the output directory exists before Slurm writes logs
mkdir -p experiments

# Redirect stdout and stderr properly (this prevents Slurm from creating `slurm-<job_id>.out`)
exec > "experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2> "experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

module --ignore-cache load "conda"
source ~/.bashrc
conda activate hiervl

# Run PyTorch script
srun python rl.py