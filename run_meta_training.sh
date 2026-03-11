#!/bin/bash
# =============================================================================
# run_meta_training.sh — Parallel SLURM jobs for meta-model training
# =============================================================================
#
# Submits 4 independent training jobs (numerical, categorical, interaction, row)
# in parallel, then a merge job that fires only after all 4 succeed.
#
# Formulas (from ablation results):
#   numerical   → cohens_d_w20
#   categorical → cohens_d_w25
#   interaction → cohens_d_w10
#   row         → cohens_d
#
# USAGE:
#   bash run_meta_training.sh                          # default params, no tuning
#   bash run_meta_training.sh --tune                   # with Optuna HP tuning
#   bash run_meta_training.sh --tune --n_trials 100    # tuning with 100 trials
#
# =============================================================================

set -e

WORK_DIR="/work/inestp05/Final"
OUTPUT_DIR="${WORK_DIR}/meta_models"
LOG_DIR="${WORK_DIR}/logs/meta_training"

NUMERICAL_CSV="${WORK_DIR}/output_numerical/numerical_transforms_merged.csv"
CATEGORICAL_CSV="${WORK_DIR}/output_categorical/categorical_transforms_merged.csv"
INTERACTION_CSV="${WORK_DIR}/output_interactions/interaction_features_merged.csv"
ROW_CSV="${WORK_DIR}/output_row/row_features_merged.csv"

# Pass-through any extra args (e.g. --tune --n_trials 100) to train_meta_models.py
EXTRA_ARGS="$@"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "============================================"
echo "Submitting meta-model training jobs"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: $EXTRA_ARGS"
echo "============================================"

# Common preamble used inside every --wrap (avoids repetition)
PREAMBLE="
    source ~/.bashrc
    conda activate lgbm_env
    export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
    export XDG_CACHE_HOME=${WORK_DIR}/.cache
    export JOBLIB_TEMP_FOLDER=${WORK_DIR}/tmp
    mkdir -p ${WORK_DIR}/.cache ${WORK_DIR}/tmp
    cd ${WORK_DIR}
"

# -----------------------------------------------------------------------------
# Numerical — formula: cohens_d_w20
# -----------------------------------------------------------------------------
NUM_JOB=$(sbatch --parsable \
    --job-name=meta_numerical \
    --output="${LOG_DIR}/numerical_%j.log" \
    --error="${LOG_DIR}/numerical_%j.err" \
    --partition=cpu \
    --time=06:00:00 \
    --cpus-per-task=16 \
    --mem=64G \
    --export=ALL \
    --wrap="${PREAMBLE}
        echo \"Starting numerical training on \$(hostname) at \$(date)\"
        python train_meta_models.py \
            --numerical_csv   ${NUMERICAL_CSV} \
            --categorical_csv ${CATEGORICAL_CSV} \
            --interaction_csv ${INTERACTION_CSV} \
            --row_csv         ${ROW_CSV} \
            --output_dir      ${OUTPUT_DIR} \
            --types           numerical \
            --formula         cohens_d_w20 \
            ${EXTRA_ARGS}
        echo \"Numerical done at \$(date)\"
    ")

# -----------------------------------------------------------------------------
# Categorical — formula: cohens_d_w25
# -----------------------------------------------------------------------------
CAT_JOB=$(sbatch --parsable \
    --job-name=meta_categorical \
    --output="${LOG_DIR}/categorical_%j.log" \
    --error="${LOG_DIR}/categorical_%j.err" \
    --partition=cpu \
    --time=02:00:00 \
    --cpus-per-task=16 \
    --mem=32G \
    --export=ALL \
    --wrap="${PREAMBLE}
        echo \"Starting categorical training on \$(hostname) at \$(date)\"
        python train_meta_models.py \
            --numerical_csv   ${NUMERICAL_CSV} \
            --categorical_csv ${CATEGORICAL_CSV} \
            --interaction_csv ${INTERACTION_CSV} \
            --row_csv         ${ROW_CSV} \
            --output_dir      ${OUTPUT_DIR} \
            --types           categorical \
            --formula         cohens_d_w25 \
            ${EXTRA_ARGS}
        echo \"Categorical done at \$(date)\"
    ")

# -----------------------------------------------------------------------------
# Interaction — formula: cohens_d_w10
# -----------------------------------------------------------------------------
INT_JOB=$(sbatch --parsable \
    --job-name=meta_interaction \
    --output="${LOG_DIR}/interaction_%j.log" \
    --error="${LOG_DIR}/interaction_%j.err" \
    --partition=cpu \
    --time=06:00:00 \
    --cpus-per-task=16 \
    --mem=64G \
    --export=ALL \
    --wrap="${PREAMBLE}
        echo \"Starting interaction training on \$(hostname) at \$(date)\"
        python train_meta_models.py \
            --numerical_csv   ${NUMERICAL_CSV} \
            --categorical_csv ${CATEGORICAL_CSV} \
            --interaction_csv ${INTERACTION_CSV} \
            --row_csv         ${ROW_CSV} \
            --output_dir      ${OUTPUT_DIR} \
            --types           interaction \
            --formula         cohens_d_w10 \
            ${EXTRA_ARGS}
        echo \"Interaction done at \$(date)\"
    ")

# -----------------------------------------------------------------------------
# Row — formula: cohens_d
# -----------------------------------------------------------------------------
ROW_JOB=$(sbatch --parsable \
    --job-name=meta_row \
    --output="${LOG_DIR}/row_%j.log" \
    --error="${LOG_DIR}/row_%j.err" \
    --partition=cpu \
    --time=01:00:00 \
    --cpus-per-task=16 \
    --mem=16G \
    --export=ALL \
    --wrap="${PREAMBLE}
        echo \"Starting row training on \$(hostname) at \$(date)\"
        python train_meta_models.py \
            --numerical_csv   ${NUMERICAL_CSV} \
            --categorical_csv ${CATEGORICAL_CSV} \
            --interaction_csv ${INTERACTION_CSV} \
            --row_csv         ${ROW_CSV} \
            --output_dir      ${OUTPUT_DIR} \
            --types           row \
            --formula         cohens_d \
            ${EXTRA_ARGS}
        echo \"Row done at \$(date)\"
    ")

echo "Submitted:"
echo "  numerical   job: $NUM_JOB  (formula: cohens_d_w20)"
echo "  categorical job: $CAT_JOB  (formula: cohens_d_w25)"
echo "  interaction job: $INT_JOB  (formula: cohens_d_w10)"
echo "  row         job: $ROW_JOB  (formula: cohens_d)"

# -----------------------------------------------------------------------------
# Merge — runs only after all four succeed (afterok)
# -----------------------------------------------------------------------------
MERGE_JOB=$(sbatch --parsable \
    --job-name=meta_merge \
    --output="${LOG_DIR}/merge_%j.log" \
    --error="${LOG_DIR}/merge_%j.err" \
    --partition=cpu \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=4G \
    --dependency=afterok:${NUM_JOB}:${CAT_JOB}:${INT_JOB}:${ROW_JOB} \
    --export=ALL \
    --wrap="
        source ~/.bashrc
        conda activate lgbm_env
        cd ${WORK_DIR}
        echo \"Merging training reports at \$(date)\"
        python - <<'PYEOF'
import json, os, glob
output_dir = '${OUTPUT_DIR}'
report_path = os.path.join(output_dir, 'training_report.json')
# Collect all partial type configs and merge into one report
merged = {}
for ctype in ['numerical', 'categorical', 'interaction', 'row']:
    cfg = os.path.join(output_dir, ctype, f'{ctype}_config.json')
    if os.path.exists(cfg):
        with open(cfg) as f:
            data = json.load(f)
        merged[ctype] = {
            'n_training_rows': data.get('n_training_rows'),
            'n_training_datasets': data.get('n_training_datasets'),
            'n_features': len(data.get('feature_names', [])),
            'method_vocab': data.get('method_vocab'),
            'cv_results': data.get('cv_results'),
            'tuning': data.get('tuning'),
            'formula': data.get('target_formula'),
            'cls_strategy': data.get('cls_strategy'),
        }
with open(report_path, 'w') as f:
    json.dump(merged, f, indent=2, default=str)
print(f'Merged report written to {report_path}')
for ctype, r in merged.items():
    cv = r.get('cv_results') or {}
    r2 = cv.get('regression', {}).get('r2', {}).get('mean', 'N/A')
    auc = cv.get('classification', {}).get('auc', {}).get('mean', 'N/A')
    tuned = ' [tuned]' if r.get('tuning') else ''
    formula = r.get('formula', 'N/A')
    print(f'  {ctype:>15}: {r[\"n_training_rows\"]} rows, '
          f'{r[\"n_training_datasets\"]} datasets, '
          f'R2={r2}, AUC={auc}, formula={formula}{tuned}')
PYEOF
        echo \"All done at \$(date)\"
    ")

echo "  merge       job: $MERGE_JOB (waits for all 4 — afterok)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/numerical_${NUM_JOB}.log"
echo "  tail -f ${LOG_DIR}/categorical_${CAT_JOB}.log"
echo "  tail -f ${LOG_DIR}/interaction_${INT_JOB}.log"
echo "  tail -f ${LOG_DIR}/row_${ROW_JOB}.log"
echo ""
echo "To rerun just one type after a failure:"
echo "  sbatch ... --wrap=\"python train_meta_models.py --types numerical --formula cohens_d_w20 ...\""