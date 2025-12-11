#!/bin/bash
#SBATCH --job-name="sam3_droid"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --exclude=svl12,svl13
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-31%32                # Only 32 jobs, each taking 1 GPU
#SBATCH --output=logs/sam3_droid_%A_%a.out
#SBATCH --error=logs/sam3_droid_%A_%a.err


echo "JOB $SLURM_ARRAY_JOB_ID / TASK $SLURM_ARRAY_TASK_ID"

# ===== Hyperparameters =====
CHUNK=20                 # How many episodes per python run (consistent with your current script)
N_TASKS=32               # Total number of tasks in the array (0-31 -> 32)
TOTAL_EPISODES=10000     # How many episodes to cover (can change to 95658 to run full droid_101)

# ===== Environment =====
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3

cd /vision/u/yinhang/pre_process/task1

TASK_ID=${SLURM_ARRAY_TASK_ID}

# The first offset for this task
OFFSET=$(( TASK_ID * CHUNK ))

# In each loop, offset jumps forward by N_TASKS * CHUNK
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