#!/bin/bash
#SBATCH --job-name=meta_collect
#SBATCH --output=logs/meta_%a.log
#SBATCH --error=logs/meta_%a.err
#SBATCH --partition=cpu
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ^^^^^^^^^^^^^^^^^^^^
# --array=0-N where N = number of tasks - 1 (from generate_task_list.py output)
# %50 = max 50 concurrent jobs (adjust to your cluster's fairshare policy)

# ============================================================
# HOW TO USE:
#
# Step 1: Generate the task list (run ONCE, on login node):
#   python generate_task_list.py --output_dir ./meta_learning_output --max_tasks 5000
#   # This prints "Submit SLURM array with: --array=0-XXXX"
#   # Update the --array line above with the printed range
#
# Step 2: Submit the array job:
#   mkdir -p logs
#   sbatch run_slurm.sh
#
# Step 3: After all jobs finish, merge results:
#   python merge_results.py --output_dir ./meta_learning_output
# ============================================================

# --- Environment setup ---
source ~/.bashrc
conda activate lgbm_env



# Set Threading Variables to match SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Redirect all Python/System caches to WORK directory
export XDG_CACHE_HOME="/work/inestp05/lightgbm_project/.cache"
export XDG_CONFIG_HOME="/work/inestp05/lightgbm_project/.config"
export XDG_DATA_HOME="/work/inestp05/lightgbm_project/.local/share"

# 2. Force OpenML via Environment Variable (Stronger than Python config)
export OPENML_CACHE_DIR="/work/inestp05/lightgbm_project/openml_cache"

# 3. Joblib (used by sklearn/lightgbm) often uses /tmp or home
export JOBLIB_TEMP_FOLDER="/work/inestp05/lightgbm_project/tmp"

# Create all these paths immediately
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $XDG_DATA_HOME $OPENML_CACHE_DIR $JOBLIB_TEMP_FOLDER

# Create directories
mkdir -p logs
mkdir -p meta_learning_output/worker_csvs
mkdir -p meta_learning_output/worker_checkpoints
mkdir -p meta_learning_output/worker_logs

# --- Configuration ---
OUTPUT_DIR="/work/inestp05/lightgbm_project/FE_PP_MetaModel/meta_learning_output"
N_FOLDS=5
N_REPEATS=3
TIME_BUDGET=14400   # 

echo "============================================"
echo "Worker $SLURM_ARRAY_TASK_ID starting"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time: $(date)"
echo "============================================"

# --- OFFSET CALCULATION ---
# Use the OFFSET environment variable if provided, otherwise default to 0
OFFSET=${OFFSET:-0}
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
echo "SLURM ID: $SLURM_ARRAY_TASK_ID + Offset: $OFFSET = Real Task ID: $REAL_ID"
# --------------------------

python DataCollector_slurm.py \
    --slurm_id $REAL_ID \
    --output_dir $OUTPUT_DIR \
    --n_folds $N_FOLDS \
    --n_repeats $N_REPEATS \
    --time_budget $TIME_BUDGET

EXIT_CODE=$?
echo "============================================"
echo "Worker $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
echo "Time: $(date)"
echo "============================================"
