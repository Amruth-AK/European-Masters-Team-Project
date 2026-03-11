#!/bin/bash
# =============================================================================
# run_ablation.sh — Submit parallel formula ablation to SLURM
# =============================================================================
#
# USAGE
# -----
# Step 0: check how many cells are in the grid (prints N)
#   python ablate_slurm.py --print_grid
#
# Step 1: submit the array  (currently 84 cells = 4 types × 21 formulas)
#   mkdir -p logs
#   sbatch --array=0-83%20 run_ablation.sh
#
#   --array=0-83   one job per (type, formula) cell
#   %20            max 20 running concurrently (tune to your cluster policy)
#
# Step 2: after all jobs finish, merge and print the table
#   python ablate_slurm.py --merge --output_dir $ABLATION_DIR
#
# =============================================================================

#SBATCH --job-name=fe_ablate
#SBATCH --output=logs/%x_%a.log
#SBATCH --error=logs/%x_%a.err
#SBATCH --partition=cpu
#SBATCH --time=01:00:00          # 1h per cell is generous; most finish in ~10-20 min
#SBATCH --cpus-per-task=4        # LightGBM uses n_jobs=-1; 4 CPUs is plenty
#SBATCH --mem=16G

# --- Environment ---
source ~/.bashrc
conda activate lgbm_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=42

# --- Paths (edit these) ---
WORK_DIR="/work/inestp05/Final"
ABLATION_DIR="${WORK_DIR}/ablation_results"

NUMERICAL_CSV="${WORK_DIR}/output_numerical/numerical_transforms_merged.csv"
CATEGORICAL_CSV="${WORK_DIR}/output_categorical/categorical_transforms_merged.csv"
INTERACTION_CSV="${WORK_DIR}/output_interactions/interaction_features_merged.csv"
ROW_CSV="${WORK_DIR}/output_row/row_features_merged.csv"

mkdir -p logs "${ABLATION_DIR}/ablation_workers"

echo "============================================"
echo "Ablation cell:  $SLURM_ARRAY_TASK_ID"
echo "Host:           $(hostname)"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "Time:           $(date)"
echo "============================================"

cd "${WORK_DIR}"

python ablate_slurm.py \
    --cell_id          $SLURM_ARRAY_TASK_ID \
    --output_dir       "${ABLATION_DIR}" \
    --numerical_csv    "${NUMERICAL_CSV}" \
    --categorical_csv  "${CATEGORICAL_CSV}" \
    --interaction_csv  "${INTERACTION_CSV}" \
    --row_csv          "${ROW_CSV}" \
    --n_cv_splits      5 \
    --cls_strategy     composite_pos

EXIT_CODE=$?
echo "============================================"
echo "Cell $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE at $(date)"
echo "============================================"
exit $EXIT_CODE