#!/bin/bash
#SBATCH --job-name=extraction
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Adjust based on your CPU requirements
#SBATCH --mem=256G  # Adjust based on your memory needs
#SBATCH --time=48:00:00  # Adjust max runtime
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

module --ignore-cache load "conda"
source ~/.bashrc
conda activate hiervl

# Run PyTorch script
srun python feature_extraction.py