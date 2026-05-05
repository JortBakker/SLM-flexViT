#!/bin/bash
#SBATCH --job-name=flexgpt_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -p gpu_h100
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=logs/flexgpt_tiny_%j.out
#SBATCH --error=logs/flexgpt_tiny_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source "$HOME/FlexViT/venv/bin/activate"

mkdir -p "$HOME/FlexViT/logs"
cd "$HOME/FlexViT"

srun python3 run_experiment.py run flexgpt,wikitext2.tiny
