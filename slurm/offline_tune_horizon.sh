#!/usr/bin/env bash

#SBATCH --job-name=offline_tune_horizon
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=gpu-standard
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:30:00

echo "Starting Optuna worker ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "SLURM assigned GPU devices: ${CUDA_VISIBLE_DEVICES}"
echo "GPU device count: $(nvidia-smi --list-gpus | wc -l)"

echo "Sleeping for $((SLURM_ARRAY_TASK_ID * 5)) seconds to avoid optuna race condition"
sleep $((SLURM_ARRAY_TASK_ID * 5))

uv run python -m offline.tune_horizon
