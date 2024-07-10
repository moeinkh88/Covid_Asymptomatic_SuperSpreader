#!/bin/bash
#SBATCH --job-name=julia_serial
#SBATCH --account=project_2007115
#SBATCH --partition=small
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=10G

module load julia
srun julia Turing-ODE-CSC7Asym.jl
