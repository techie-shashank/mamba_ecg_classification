#!/bin/bash
#SBATCH --job-name=mamba_run                     # Job name
#SBATCH --output=outputs/logs/mamba_%j.out      # Standard output (%j = JobID)
#SBATCH --error=outputs/logs/mamba_%j.err       # Standard error
#SBATCH --gres=gpu:1                             # Request 8 GPU
#SBATCH --time=1-00:00:00                           # Max runtime

# Load modules
module load python
module load cuda/12.6

# Set environment variables
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
conda activate mamba_ts_env

# Change directory to project
cd ~/projects/mamba_ts_forecasting

# Run training
python scripts/quick_start.py
