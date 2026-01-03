#!/usr/bin/env bash

#SBATCH --job-name=offline_train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

uv run python -m offline.bc.train
