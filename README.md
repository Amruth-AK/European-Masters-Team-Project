# Feature Engineering Recommender — Meta-Learning Pipeline

A meta-learning system that recommends feature engineering transforms for tabular classification tasks. The pipeline collects empirical data on how transforms affect model performance across hundreds of OpenML datasets, trains meta-models on this data, and serves recommendations through a Streamlit app.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Step 1 — Task List Preparation](#4-step-1--task-list-preparation)
5. [Step 2 — Data Collection (SLURM)](#5-step-2--data-collection-slurm)
6. [Step 3 — Merging Collector Results](#6-step-3--merging-collector-results)
7. [Step 4 — Training Meta-Models](#7-step-4--training-meta-models)
8. [Step 5 — Running the Streamlit App](#8-step-5--running-the-streamlit-app)
9. [Project Structure](#9-project-structure)
10. [Running Without SLURM (Local Mode)](#10-running-without-slurm-local-mode)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

The system works in three phases:

**Phase A — Data Collection:** For each of ~2,900 OpenML classification datasets, the pipeline tests a catalog of feature transforms (numerical, categorical, and interaction) and records whether each transform improves LightGBM performance. Each experiment captures dataset-level meta-features (size, class balance, correlations, etc.), column-level meta-features (skewness, cardinality, entropy, etc.), and the outcome (ROC-AUC delta, statistical significance, calibrated effect size).

**Phase B — Meta-Model Training:** Three LightGBM meta-models (one per transform type) are trained on the collected data. Given a new dataset's meta-features and a candidate transform, they predict the expected performance improvement (calibrated delta) and whether the improvement is statistically significant.

**Phase C — Recommendation App:** A Streamlit app accepts a user-uploaded CSV, computes meta-features, queries the meta-models for all applicable transforms, and presents a ranked list of suggestions. Users can then train a baseline vs. enhanced LightGBM model and compare results on a held-out test set.

### Transform Catalog

| Type | Methods |
|------|---------|
| **Numerical** | `log_transform`, `sqrt_transform`, `polynomial_square`, `polynomial_cube`, `reciprocal_transform`, `quantile_binning`, `impute_median`, `missing_indicator` |
| **Categorical** | `frequency_encoding`, `target_encoding`, `onehot_encoding`, `hashing_encoding`, `missing_indicator` |
| **Interaction** | `product_interaction`, `division_interaction`, `addition_interaction`, `abs_diff_interaction`, `group_mean`, `group_std`, `cat_concat` |

---

## 2. Prerequisites

- **Python 3.9+** (tested with 3.10/3.11)
- **Conda** (recommended for environment management)
- **SLURM cluster** (for parallelized data collection; local fallback described in Section 10)
- **Internet access** (to download datasets from OpenML during collection)

### Python Dependencies

```
lightgbm
scikit-learn
pandas
numpy
scipy
openml
streamlit
```

---

## 3. Environment Setup

Create and activate a Conda environment:

```bash
conda create -n lgbm_env python=3.10 -y
conda activate lgbm_env

pip install lightgbm scikit-learn pandas numpy scipy openml streamlit
```

Set up your working directory (adjust paths to your system):

```bash
# Example: using /work/<username>/Final as the base directory
export PROJECT_DIR=/work/$USER/Final
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Copy all project files into this directory
cp /path/to/source/*.py .
cp /path/to/source/*.sh .
cp /path/to/source/task_list.json .
```

---

## 4. Step 1 — Task List Preparation

The file `task_list.json` is the registry of OpenML tasks to process. It is already provided with 2,916 tasks (classification and regression). Only classification tasks are used by the collectors.

To verify the number of classification tasks and generate the exact SLURM submit commands:

```bash
python count_classification_tasks.py --task_list $PROJECT_DIR/task_list.json
```

Example output:

```
Task list: /work/user/Final/task_list.json
  Total tasks:          2916
  Classification:       312
  Regression (skipped): 2604

Array range: --array=0-311

======================================================================
SUBMIT COMMANDS (copy & paste):
======================================================================

mkdir -p logs

sbatch --array=0-311%50 --job-name=num_collect \
  --export=COLLECTOR=numerical,OUTPUT_DIR=/work/user/Final/output_numerical \
  run_collector.sh

sbatch --array=0-311%50 --job-name=cat_collect \
  --export=COLLECTOR=categorical,OUTPUT_DIR=/work/user/Final/output_categorical \
  run_collector.sh

sbatch --array=0-311%50 --job-name=int_collect \
  --export=COLLECTOR=interaction,OUTPUT_DIR=/work/user/Final/output_interactions \
  run_collector.sh
```

The `%50` suffix limits concurrency to 50 simultaneous array jobs. Adjust as needed for your cluster.

> **Note:** If you have more than 999 classification tasks, the script automatically generates batched submit commands using the `OFFSET` mechanism to work around SLURM's `MaxArraySize` limit.

---

## 5. Step 2 — Data Collection (SLURM)

Data collection is the most time-intensive step. Each SLURM array job processes one OpenML dataset with one collector type (numerical, categorical, or interaction). Per dataset, the collector evaluates a LightGBM baseline via repeated 5-fold cross-validation, then tests every applicable transform on every applicable column/pair.

### 5.1 Configure the SLURM Script

Before submitting, review and edit `run_collector.sh` to match your cluster:

```bash
# In run_collector.sh, update these paths to match your setup:

# Line 44: Your conda environment activation
source ~/.bashrc
conda activate lgbm_env

# Line 52-57: Cache directories (must be on fast storage, not /home)
export XDG_CACHE_HOME="/work/$USER/Final/.cache"
export OPENML_CACHE_DIR="/work/$USER/Final/openml_cache"
# ... etc.

# Line 65: Path to your task list
TASK_LIST="/work/$USER/Final/task_list.json"

# Line 93: Working directory containing the Python files
cd /work/$USER/Final
```

Also review the SLURM resource requests at the top of the file:

```bash
#SBATCH --partition=cpu       # Adjust to your cluster's partition name
#SBATCH --time=04:00:00       # 4 hours per task (increase if needed)
#SBATCH --cpus-per-task=8     # LightGBM uses all available cores
#SBATCH --mem=32G             # 32 GB RAM per task
```

### 5.2 Submit the Jobs

Create the logs directory and submit all three collector types:

```bash
cd $PROJECT_DIR
mkdir -p logs

# Numerical transforms
sbatch --array=0-311%50 --job-name=num_collect \
  --export=COLLECTOR=numerical,OUTPUT_DIR=$PROJECT_DIR/output_numerical \
  run_collector.sh

# Categorical transforms
sbatch --array=0-311%50 --job-name=cat_collect \
  --export=COLLECTOR=categorical,OUTPUT_DIR=$PROJECT_DIR/output_categorical \
  run_collector.sh

# Interaction features
sbatch --array=0-311%50 --job-name=int_collect \
  --export=COLLECTOR=interaction,OUTPUT_DIR=$PROJECT_DIR/output_interactions \
  run_collector.sh
```

Replace `311` with your actual max index from `count_classification_tasks.py`.

### 5.3 Monitor Progress

```bash
# Check job status
squeue -u $USER

# Check logs for a specific worker
cat logs/num_collect_42.log

# Count completed checkpoints
ls output_numerical/worker_checkpoints/ | wc -l
ls output_categorical/worker_checkpoints/ | wc -l
ls output_interactions/worker_checkpoints/ | wc -l
```

Each worker writes to its own CSV file (e.g., `output_numerical/worker_csvs/numerical_transforms_worker_00042.csv`) and a checkpoint JSON, so there are no file-locking issues.

### 5.4 What Each Collector Produces

Each collector generates one CSV row per (dataset, column/pair, method) combination. The columns include:

- **Dataset meta-features** (13 fields): `n_rows`, `n_cols`, `n_classes`, `class_imbalance_ratio`, `avg_feature_corr`, `landmarking_score`, etc.
- **Column/pair meta-features** (16–20 fields, varies by type): `skewness`, `kurtosis_val`, `entropy`, `n_unique`, `mutual_info_score`, `pearson_corr`, etc.
- **Intervention details**: `column_name`, `method`
- **Outcome metrics**: `delta`, `delta_normalized`, `p_value`, `p_value_bonferroni`, `is_significant`, `is_significant_bonferroni`, `cohens_d`, `calibrated_delta`

The key target variable for meta-model training is `calibrated_delta` — the raw AUC improvement normalized by the dataset's noise floor (Cohen's d-like calibration).

---

## 6. Step 3 — Merging Collector Results

After all SLURM jobs have finished, merge the per-worker CSV files into a single file per collector type:

```bash
python merge_collector_results.py \
  --output_dir $PROJECT_DIR/output_numerical \
  --prefix numerical_transforms

python merge_collector_results.py \
  --output_dir $PROJECT_DIR/output_categorical \
  --prefix categorical_transforms

python merge_collector_results.py \
  --output_dir $PROJECT_DIR/output_interactions \
  --prefix interaction_features
```

This produces:

```
output_numerical/numerical_transforms_merged.csv
output_categorical/categorical_transforms_merged.csv
output_interactions/interaction_features_merged.csv
```

The merge script also prints a status summary (how many workers succeeded, failed, or were skipped) and removes duplicate rows.

---

## 7. Step 4 — Training Meta-Models

Train three meta-model pairs (regressor + classifier) from the merged CSVs:

```bash
python train_meta_models.py \
  --numerical_csv   $PROJECT_DIR/output_numerical/numerical_transforms_merged.csv \
  --categorical_csv $PROJECT_DIR/output_categorical/categorical_transforms_merged.csv \
  --interaction_csv $PROJECT_DIR/output_interactions/interaction_features_merged.csv \
  --output_dir      $PROJECT_DIR/meta_models \
  --n_cv_splits     5
```

### What This Does

For each collector type:

1. Loads the merged CSV and prepares features (dataset meta-features + column/pair meta-features + one-hot encoded method).
2. Evaluates generalization using **GroupKFold** cross-validation (grouped by `dataset_name`, so no data from the same dataset leaks between train and validation).
3. Trains a **LightGBM regressor** to predict `calibrated_delta` (expected improvement).
4. Trains a **LightGBM classifier** to predict `is_significant_bonferroni` (is the improvement real?).
5. Saves models and configuration.

### Output Structure

```
meta_models/
├── numerical/
│   ├── numerical_regressor.txt       # LightGBM booster (text format)
│   ├── numerical_classifier.txt
│   └── numerical_config.json         # Feature lists, method vocab, CV results
├── categorical/
│   ├── categorical_regressor.txt
│   ├── categorical_classifier.txt
│   └── categorical_config.json
├── interaction/
│   ├── interaction_regressor.txt
│   ├── interaction_classifier.txt
│   └── interaction_config.json
└── training_report.json              # Combined evaluation summary
```

The training script prints cross-validation metrics. Expected ranges (depending on data volume): regression R² of 0.1–0.4, classification AUC of 0.6–0.8. These numbers reflect genuine out-of-dataset generalization since GroupKFold ensures the model never sees transforms from the same dataset in both train and validation.

---

## 8. Step 5 — Running the Streamlit App

Launch the recommendation app:

```bash
streamlit run recommend_app.py -- --model_dir $PROJECT_DIR/meta_models
```

> **Note:** The `--` before `--model_dir` is required to separate Streamlit's arguments from the app's arguments. Alternatively, the model directory can be set interactively in the sidebar.

### App Workflow

1. **Upload Training CSV** — Upload a tabular classification dataset as a CSV file.
2. **Select Target Column** — Choose which column is the classification target.
3. **Analyze & Get Suggestions** — The app computes meta-features for the dataset and every column/pair, queries all three meta-models, and presents a ranked table of transform suggestions with predicted impact.
4. **Train Models** — Trains a baseline LightGBM (no transforms) and an enhanced LightGBM (with the top-K selected transforms applied). Reports validation metrics for both.
5. **Upload Test CSV & Compare** — Upload a held-out test set to get a final side-by-side comparison (ROC-AUC, F1, accuracy, etc.) of baseline vs. enhanced models.

### Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Meta-models directory | `./meta_models` | Path to the trained models from Step 4 |
| Max suggestions to apply | 10 | How many top-ranked transforms to apply |
| Min predicted delta | 0.0 | Only suggest transforms above this calibrated delta threshold |

---

## 9. Project Structure

```
.
├── task_list.json                     # Registry of 2,916 OpenML tasks
├── count_classification_tasks.py      # Counts tasks & prints sbatch commands
│
├── collector_utils.py                 # Shared evaluation engine (CV, stats, meta-features)
├── collect_numerical_transforms.py    # Numerical single-column collector
├── collect_categorical_transforms.py  # Categorical single-column collector
├── collect_interaction_features.py    # Two-column interaction collector
│
├── collector_slurm.py                 # Unified SLURM worker (dispatches to any collector)
├── run_collector.sh                   # SLURM batch script
│
├── merge_collector_results.py         # Merges per-worker CSVs after collection
├── train_meta_models.py              # Trains meta-models from merged CSVs
├── recommend_app.py                   # Streamlit recommendation app
└── README.md                          # This file
```

### Key Design Decisions

- **Evaluation Metric**: ROC-AUC (weighted OVR for multiclass).
- **Cross-Validation**: 5-fold × 3 repeats for data collection; GroupKFold for meta-model training.
- **Statistical Testing**: Paired t-test with post-hoc Bonferroni correction per dataset.
- **Effect Size Calibration**: `calibrated_delta = raw_delta / noise_floor_std`, where the noise floor is estimated by evaluating the baseline model with a permuted (random) column appended. This filters out "improvements" that are within noise.
- **Column Pruning**: ID-like columns, constant columns, and leaking columns (>0.95 correlation with target) are automatically removed before evaluation.
- **No Data Leakage**: All transforms are applied inside CV folds (fit on train, transform both train and val).

---

## 10. Running Without SLURM (Local Mode)

If you do not have access to a SLURM cluster, you can run the collectors sequentially on a single machine. This is practical for a small subset of tasks but will take a long time for the full task list.

### Option A — Use the Standalone Collectors Directly

Each collector can be run independently without SLURM:

```bash
# Numerical (processes ALL classification tasks sequentially)
python collect_numerical_transforms.py \
  --task_list ./task_list.json \
  --output_dir ./output_numerical \
  --n_folds 5 \
  --n_repeats 3 \
  --time_budget 7200

# Categorical
python collect_categorical_transforms.py \
  --task_list ./task_list.json \
  --output_dir ./output_categorical

# Interaction
python collect_interaction_features.py \
  --task_list ./task_list.json \
  --output_dir ./output_interactions
```

These modes write results to a single CSV file each and use their own checkpoint mechanism, so they can be interrupted and resumed.

### Option B — Use the SLURM Worker Script Manually

Process one task at a time by specifying a SLURM ID manually:

```bash
# Process task index 0 with the numerical collector
python collector_slurm.py --slurm_id 0 --collector numerical \
  --output_dir ./output_numerical --task_list ./task_list.json

# Process task index 1
python collector_slurm.py --slurm_id 1 --collector numerical \
  --output_dir ./output_numerical --task_list ./task_list.json

# ... etc.
```

After processing all tasks, merge as described in Step 3.

---

## 11. Troubleshooting

**OpenML download failures:** Some datasets may fail to download from OpenML due to network issues or dataset availability. These tasks are marked as "failed" in the checkpoints and skipped. Re-run the same SLURM job array after some time, and workers will skip already-completed tasks.

**Out of memory:** Some large datasets may exceed the 32 GB memory allocation. Increase `--mem` in the SLURM script, or reduce `--time_budget` to skip datasets that take too long. The collector automatically samples datasets exceeding 100M cells.

**OpenML cache directory:** The SLURM script caches downloaded datasets in `/work/<user>/Final/openml_cache`. If you run into disk quota issues, clean this directory or point it to a location with more space.

**Checkpoint resume:** All collectors support checkpointing. If a SLURM job is killed (timeout, OOM), simply resubmit the same array. Workers check for existing checkpoints and skip completed tasks.

**Meta-model training with partial data:** The meta-model training script handles missing collector types gracefully. If you only have numerical and categorical data (no interactions), just omit the `--interaction_csv` flag and the interaction meta-model will be skipped.

**Streamlit port conflicts:** If the default port 8501 is taken, use `streamlit run recommend_app.py --server.port 8502 -- --model_dir ./meta_models`.
