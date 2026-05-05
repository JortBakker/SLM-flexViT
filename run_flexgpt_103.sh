#!/bin/bash
#SBATCH --job-name=flexgpt_103
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -p gpu_h100
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=logs/flexgpt_103_%j.out
#SBATCH --error=logs/flexgpt_103_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source "$HOME/FlexViT/myenv/bin/activate"

mkdir -p "$HOME/FlexViT/logs"
cd "$HOME/FlexViT"

nvidia-smi

echo "Snellius Job Started"
echo | date
echo "Node name: $(hostname)"

srun python3 run_experiment.py run flexgpt,wikitext103.3levels

echo "Job Complete"
echo | date