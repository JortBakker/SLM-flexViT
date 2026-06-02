#!/bin/bash
#SBATCH --job-name=preprocess_owt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=rome
#SBATCH --output=logs/preprocess_owt_%j.out
#SBATCH --error=logs/preprocess_owt_%j.err

export HF_DATASETS_CACHE=$HOME/hf_cache
export HF_HOME=$HOME/hf_home

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source "$HOME/FlexViT/myenv/bin/activate"
cd "$HOME/FlexViT"

echo "Preprocessing started"
echo | date

python -c "import utils; utils.load_openwebtext(num_workers=16)"

echo "Preprocessing complete"
echo | date
