#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH -p gpu_h100
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=logs/lm_eval_%j.out
#SBATCH --error=logs/lm_eval_%j.err

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

CONFIG=${1:-"flexgpt,wikitext103.kd_from_gpt2"}
TASKS=${2:-"lambada_openai"}
OUTPUT="results/$(echo $CONFIG | tr ',' '_' | tr '.' '_')_eval.json"

echo "Evaluating: $CONFIG"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT"

srun python3 eval_lm_harness.py \
    --config "$CONFIG" \
    --tasks "$TASKS" \
    --output "$OUTPUT"

echo "Eval complete"
