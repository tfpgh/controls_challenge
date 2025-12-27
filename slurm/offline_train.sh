#!/usr/bin/env bash

#SBATCH --job-name=offline_train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

uv run python -m offline.bc.train
