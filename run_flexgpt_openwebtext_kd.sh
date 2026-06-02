#!/bin/bash
#SBATCH --job-name=flexgpt_openwebtext_kd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH -p gpu_h100
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=logs/flexgpt_openwebtext_kd_%j.out
#SBATCH --error=logs/flexgpt_openwebtext_kd_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source "$HOME/FlexViT/myenv/bin/activate"

mkdir -p "$HOME/FlexViT/logs"
cd "$HOME/FlexViT"

export HF_DATASETS_CACHE=$HOME/hf_cache
export HF_HOME=$HOME/hf_home

nvidia-smi

echo "Snellius Job Started"
echo | date
echo "Node name: $(hostname)"

srun python3 run_experiment.py run flexgpt,openwebtext.kd_from_gpt2

echo "Job Complete"
echo | date
