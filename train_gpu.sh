#!/bin/bash
#SBATCH --job-name=rl-multi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=rivulet
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Training Script for Cluster
# This launches an independent training process, one per GPU

echo "=== Pong RL Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load CUDA
spack load cuda@13.0.0

# Activate Conda environment
source /opt/csg/anaconda/etc/profile.d/conda.sh
conda activate ai-pong

# Create log directories
mkdir -p logs

# Launch an independent training process, one per GPU
# Each process trains its own agent with different seeds
# Results can be compared or ensembled later

echo "Launching training on 1 GPU..."

python train.py --steps 1000000 --n_envs 8

echo "All training processes completed!"
echo "Models saved in ./models/"
