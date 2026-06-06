#!/bin/bash
#SBATCH --job-name=lm_eval_llama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH -p gpu_h100
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=logs/lm_eval_llama_%j.out
#SBATCH --error=logs/lm_eval_llama_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source "$HOME/FlexViT/myenv/bin/activate"

mkdir -p "$HOME/FlexViT/logs"
mkdir -p "$HOME/FlexViT/results"
cd "$HOME/FlexViT"

export HF_DATASETS_CACHE=/scratch-shared/$USER/hf_cache
export HF_HOME=/scratch-shared/$USER/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Evaluating JackFram/llama-160m baseline"

srun lm_eval --model hf \
    --model_args pretrained=JackFram/llama-160m \
    --tasks wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa,triviaqa \
    --device cuda:0 \
    --output_path results/llama_160m_baseline_eval.json

echo "Eval complete"
