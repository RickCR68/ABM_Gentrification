#!/bin/bash
#SBATCH --job-name=abm_gsa
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=gsa_output_%j.log

# 1. Clear loaded system defaults and load a stable python module environment
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# 2. Change directory to where the job was submitted from
cd $SLURM_SUBMIT_DIR

# 3. Activate your Python virtual environment where SALib and Mesa are installed
source ../.venv/bin/activate

# 4. Run the global sensitivity analysis script
python3 batch_gsa.py