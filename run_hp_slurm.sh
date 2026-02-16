#!/bin/bash
#SBATCH --job-name=hp_collect
#SBATCH --output=logs/hp_%a.log
#SBATCH --error=logs/hp_%a.err
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ^^^^^^^^^^^^^^^^^^^^
# --array=0-N where N = number of tasks - 1 (from generate_task_list.py output)
# %50 = max 50 concurrent jobs (adjust to your cluster's fairshare policy)

# ============================================================
# HOW TO USE:
#
# Step 1: Generate the task list (if not already done):
#   python generate_task_list.py --output_dir ./hp_tuning_output --max_tasks 5000
#   # Or reuse the existing task_list.json from meta_learning_output:
#   cp ./meta_learning_output/task_list.json ./hp_tuning_output/
#
# Step 2: Submit the array job:
#   mkdir -p logs
#   sbatch run_hp_slurm.sh
#
# Step 3: After all jobs finish, merge results:
#   python merge_hp_results.py --output_dir ./hp_tuning_output
# ============================================================

# --- Environment setup ---
# ... (Keep your SBATCH directives at the top) ...

# --- Environment setup ---
source ~/.bashrc
conda activate lgbm_env

# ============================================================
# [CRITICAL] CACHE CONFIGURATION - PREVENT HOME DIR USAGE
# ============================================================
# 1. Define the safe work directory
SAFE_CACHE_ROOT="/work/inestp05/lightgbm_project"

# 2. Force OpenML to use this path (HPCollector.py reads this variable)
export OPENML_CACHE_DIR="${SAFE_CACHE_ROOT}/openml_cache"

# 3. Force generic caching (pip, matplotlib, etc.) to use this path
export XDG_CACHE_HOME="${SAFE_CACHE_ROOT}/.cache"

# 4. Create them immediately so Python doesn't complain
mkdir -p "$OPENML_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

echo "GLOBAL: Cache forced to $SAFE_CACHE_ROOT"
# ============================================================

# Set Threading Variables to match SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# ... (rest of your script) ...
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create directories
mkdir -p logs
mkdir -p hp_tuning_output/worker_csvs
mkdir -p hp_tuning_output/worker_checkpoints
mkdir -p hp_tuning_output/worker_logs

# --- Configuration ---
OUTPUT_DIR="/work/inestp05/lightgbm_project/Hyperparameter_MetaModel/hp_tuning_output"
N_FOLDS=5
N_REPEATS=3
N_HP_CONFIGS=25    # 25 LHS + 1 default = 26 configs per dataset
TIME_BUDGET=7200   # 120 minutes per dataset

echo "============================================"
echo "HP Tuning Worker $SLURM_ARRAY_TASK_ID starting"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time: $(date)"
echo "============================================"

# --- OFFSET CALCULATION ---
OFFSET=${OFFSET:-0}
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
echo "SLURM ID: $SLURM_ARRAY_TASK_ID + Offset: $OFFSET = Real Task ID: $REAL_ID"

python HPCollector.py \
    --slurm_id $REAL_ID \
    --output_dir $OUTPUT_DIR \
    --n_folds $N_FOLDS \
    --n_repeats $N_REPEATS \
    --n_hp_configs $N_HP_CONFIGS \
    --time_budget $TIME_BUDGET

EXIT_CODE=$?
echo "============================================"
echo "HP Worker $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
echo "Time: $(date)"
echo "============================================"
