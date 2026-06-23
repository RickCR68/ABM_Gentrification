#!/bin/bash
#SBATCH --job-name=gsa_split_512
#SBATCH --output=gsa_split_%j.log
#SBATCH --error=gsa_split_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=rome

# 1. Load required system modules (adjust if your setup requires specific Python modules)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# 2. Activate your virtual environment
source ~/ABM_Gentrification/.venv/bin/activate

# 3. Inform your script how many workers each part can use 
# (matching --cpus-per-task)
export SLURM_CPUS_PER_TASK=32

echo "Launching 4 GSA parts in parallel..."

# 4. Launch all 4 parts simultaneously into the background using '&'
GSA_PART=1 python batch_gsa_split.py > part_1_run.log 2>&1 &
GSA_PART=2 python batch_gsa_split.py > part_2_run.log 2>&1 &
GSA_PART=3 python batch_gsa_split.py > part_3_run.log 2>&1 &
GSA_PART=4 python batch_gsa_split.py > part_4_run.log 2>&1 &

# CRITICAL: Wait for all background processes to finish before ending the Slurm job
wait

echo "All 4 parts have completed running!"