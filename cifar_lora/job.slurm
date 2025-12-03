#!/bin/bash
#SBATCH --job-name=cl_lora
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd $SLURM_SUBMIT_DIR

module load anaconda3/2024.06/gcc-13.2.0

/gpfs/users/fernandeda/.conda/envs/cl_env/bin/python continual_clip.py