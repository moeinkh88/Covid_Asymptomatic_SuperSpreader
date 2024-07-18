#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --account=project_2007115
#SBATCH --partition=small
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=5G

module load julia
srun julia Fit_Orders_FDE.jl