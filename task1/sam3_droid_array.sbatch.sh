#!/bin/bash
#SBATCH --job-name="sam3_droid"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-31%32                # 只 32 个 job，每个占 1 GPU
#SBATCH --output=logs/sam3_droid_%A_%a.out
#SBATCH --error=logs/sam3_droid_%A_%a.err


echo "JOB $SLURM_ARRAY_JOB_ID / TASK $SLURM_ARRAY_TASK_ID"

# ===== 超参数 =====
CHUNK=20                 # 每次 python 跑多少 episodes（跟你现在脚本保持一致）
N_TASKS=32               # array 里一共有多少个 task（0-31 -> 32）
TOTAL_EPISODES=10000     # 想覆盖多少个 episode（可以改成 95658 跑满 droid_101）

# ===== 环境 =====
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3

cd /vision/u/yinhang/pre_process/task1

TASK_ID=${SLURM_ARRAY_TASK_ID}

# 这个 task 的第一个 offset
OFFSET=$(( TASK_ID * CHUNK ))

# 每次循环，offset 往前跳 N_TASKS * CHUNK
STEP=$(( N_TASKS * CHUNK ))

echo "Task ${TASK_ID}: start OFFSET=${OFFSET}, STEP=${STEP}, TOTAL_EPISODES=${TOTAL_EPISODES}"

while [ ${OFFSET} -lt ${TOTAL_EPISODES} ]; do
    echo "Task ${TASK_ID}: processing episodes [${OFFSET}, $((OFFSET + CHUNK - 1))]"
    python droid_sam3_multi.py \
        --offset ${OFFSET} \
        --count ${CHUNK}

    OFFSET=$(( OFFSET + STEP ))
done

echo "Task ${TASK_ID}: done."
