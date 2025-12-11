#!/bin/bash
#SBATCH --job-name="droid_qwen7b"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=outputs/droid_qwen7b_%j.out
#SBATCH --error=outputs/droid_qwen7b_%j.err


mkdir -p outputs

echo "SLURM_JOBID=$SLURM_JOBID"
echo "Node list: $SLURM_JOB_NODELIST"


source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3

cd /vision/u/yinhang/pre_process/task2

python -u process_droid_caption_qwen.py
