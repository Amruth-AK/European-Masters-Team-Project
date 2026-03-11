#!/bin/bash
#SBATCH --job-name=fe_collect
#SBATCH --output=logs/%x_%a.log
#SBATCH --error=logs/%x_%a.err
#SBATCH --partition=cpu
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# USAGE — Run all four collectors:

# Step 0: Confirm task count:
#   python count_classification_tasks.py --task_list /work/inestp05/Final/task_list.json
#   # Prints: "Classification tasks: 2000. Use --array=0-999 with Offsets"
#
# Step 1: Submit Collectors in two batches (due to MaxArraySize=1000)
#   mkdir -p logs
#
#   # --- NUMERICAL ---
#   sbatch --array=0-999%50 --job-name=num_0 --export=COLLECTOR=numerical,OUTPUT_DIR=/work/inestp05/Final/output_numerical,OFFSET=0 run_collector.sh
#   sbatch --array=0-999%50 --job-name=num_1000 --export=COLLECTOR=numerical,OUTPUT_DIR=/work/inestp05/Final/output_numerical,OFFSET=1000 run_collector.sh
#
#   # --- CATEGORICAL ---
#   sbatch --array=0-999%50 --job-name=cat_0 --export=COLLECTOR=categorical,OUTPUT_DIR=/work/inestp05/Final/output_categorical,OFFSET=0 run_collector.sh
#   sbatch --array=0-999%50 --job-name=cat_1000 --export=COLLECTOR=categorical,OUTPUT_DIR=/work/inestp05/Final/output_categorical,OFFSET=1000 run_collector.sh
#
#   # --- INTERACTION ---
#   sbatch --array=0-999%50 --job-name=int_0 --export=COLLECTOR=interaction,OUTPUT_DIR=/work/inestp05/Final/output_interactions,OFFSET=0 run_collector.sh
#   sbatch --array=0-999%50 --job-name=int_1000 --export=COLLECTOR=interaction,OUTPUT_DIR=/work/inestp05/Final/output_interactions,OFFSET=1000 run_collector.sh
#
#   # --- ROW ---
#   sbatch --array=0-999%50 --job-name=row_0 --export=COLLECTOR=row,OUTPUT_DIR=/work/inestp05/Final/output_row,OFFSET=0 run_collector.sh
#   sbatch --array=0-999%50 --job-name=row_1000 --export=COLLECTOR=row,OUTPUT_DIR=/work/inestp05/Final/output_row,OFFSET=1000 run_collector.sh
#
# Step 2: After all jobs finish, merge each:
#   python merge_collector_results.py --output_dir /work/inestp05/Final/output_numerical --prefix numerical_transforms
#   python merge_collector_results.py --output_dir /work/inestp05/Final/output_categorical --prefix categorical_transforms
#   python merge_collector_results.py --output_dir /work/inestp05/Final/output_interactions --prefix interaction_features
#   python merge_collector_results.py --output_dir /work/inestp05/Final/output_row --prefix row_features
# ============================================================

# --- Environment setup ---
source ~/.bashrc
conda activate lgbm_env

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=42

# Cache directories -> /work (not /home)
export XDG_CACHE_HOME="/work/inestp05/Final/.cache"
export XDG_CONFIG_HOME="/work/inestp05/Final/.config"
export XDG_DATA_HOME="/work/inestp05/Final/.local/share"
export OPENML_CACHE_DIR="/work/inestp05/Final/openml_cache"
export JOBLIB_TEMP_FOLDER="/work/inestp05/Final/tmp"
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $XDG_DATA_HOME $OPENML_CACHE_DIR $JOBLIB_TEMP_FOLDER

# --- Required env variables ---
# COLLECTOR: numerical | categorical | interaction | row
# OUTPUT_DIR: where to write results
COLLECTOR=${COLLECTOR:?ERROR: Set COLLECTOR env var (numerical/categorical/interaction/row)}
OUTPUT_DIR=${OUTPUT_DIR:?ERROR: Set OUTPUT_DIR env var}

TASK_LIST="/work/inestp05/Final/task_list.json"
N_FOLDS=5
N_REPEATS=3
TIME_BUDGET=7200

# Create output directories
mkdir -p logs
mkdir -p ${OUTPUT_DIR}/worker_csvs
mkdir -p ${OUTPUT_DIR}/worker_checkpoints

echo "============================================"
echo "Collector: $COLLECTOR"
echo "Worker:    $SLURM_ARRAY_TASK_ID"
echo "Output:    $OUTPUT_DIR"
echo "Host:      $(hostname)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Time:      $(date)"
echo "============================================"

# --- OFFSET CALCULATION ---
# Use the OFFSET environment variable if provided, otherwise default to 0
# Useful when SLURM max array index is limited (e.g., MaxArraySize=1000)
# and you need to process tasks 1000+
OFFSET=${OFFSET:-0}
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
echo "SLURM ID: $SLURM_ARRAY_TASK_ID + Offset: $OFFSET = Real Task ID: $REAL_ID"

# --- Run ---
cd /work/inestp05/Final

python collector_slurm.py \
    --slurm_id $REAL_ID \
    --collector $COLLECTOR \
    --output_dir $OUTPUT_DIR \
    --task_list $TASK_LIST \
    --n_folds $N_FOLDS \
    --n_repeats $N_REPEATS \
    --time_budget $TIME_BUDGET

EXIT_CODE=$?
echo "============================================"
echo "Worker $SLURM_ARRAY_TASK_ID (Real ID: $REAL_ID) [$COLLECTOR] finished with exit code $EXIT_CODE"
echo "Time: $(date)"
echo "============================================"