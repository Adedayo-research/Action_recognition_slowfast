#!/bin/bash
#SBATCH --job-name=slowfast_soccernet
#SBATCH --partition=gpu            # CHANGE THIS to your university's GPU partition name (e.g., gpu, compute, gpu-a100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 cores for fast dataloading
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=32G                  # Request 32GB RAM
#SBATCH --time=48:00:00            # Max time 48 hours
#SBATCH --output=slurm_logs/training_%j.out
#SBATCH --error=slurm_logs/training_%j.err

echo "Starting SLURM job..."
echo "Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 1. Load university modules (CHANGE THESE if your university uses different module names)
# module load python/3.10
# module load cuda/11.8

# 2. Activate Python environment
# Assume you have created a virtual environment named 'venv' in your project directory
source venv/bin/activate

# 3. Set the dataset path environment variable!
# CHANGE THIS to wherever you downloaded the 500-game dataset on the cluster's scratch or project drive.
export SOCCERNET_DIR="/path/to/your/cluster/SoccerNet_Dataset"

# 4. Create log directory
mkdir -p slurm_logs
mkdir -p experiment_2/checkpoints

# 5. Run the training sweep agent!
echo "Launching training script..."
python experiment_2/scripts/training_unfrozen_blocks4_5.py

echo "Job finished."
