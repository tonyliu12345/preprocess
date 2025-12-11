#!/bin/bash
#SBATCH --job-name="fs_depth_10k"
#SBATCH --account=vision            
#SBATCH --partition=svl          
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --exclude=svl12,svl13
#SBATCH --cpus-per-task=8           
#SBATCH --mem=64G                   
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/depth_%A_%a.out  
#SBATCH --error=logs/depth_%A_%a.err
#SBATCH --array=0-99%30             

# 1. 创建日志目录
mkdir -p logs

echo "SLURM_JOBID=$SLURM_JOBID"
echo "Node list: $SLURM_JOB_NODELIST"

# --- 路径配置 ---
TSV_PATH="/vision/u/yinhang/pre_process/task3/stereo_job_list_output/fs_stereo_jobs_10000.tsv"

# --- 自动计算分片 ---
# 总量 10000
# 份数 30 (对应 array 0-29)
TOTAL_DATA=10000
NUM_ARRAYS=30       # 必须和 --array=0-29 保持一致！
CHUNK_SIZE=$((TOTAL_DATA / NUM_ARRAYS))

# 计算当前 Job 的起止索引
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE))

# 修正最后一个 Job 的边界 (防止整除带来的尾数丢失)
if [ $SLURM_ARRAY_TASK_ID -eq 29 ]; then
    END_IDX=$TOTAL_DATA
fi

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing range: $START_IDX to $END_IDX"

# --- 运行 Python ---
python run_fs_depth_no_norm.py \
    --tsv "$TSV_PATH" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX"