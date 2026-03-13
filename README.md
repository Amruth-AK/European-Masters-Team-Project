# Feature Engineering Recommender

A meta-learning-powered tool that predicts which feature engineering transforms will improve classification performance on tabular datasets. Instead of trial-and-error, the system uses LightGBM meta-models—trained on hundreds of OpenML datasets—to recommend transforms tailored to the characteristics of your specific data.

## How It Works

The recommender follows a meta-learning approach:

1. **Data collection**: For each of ~1,200 OpenML classification datasets and each candidate transform, the actual AUC delta was measured using repeated stratified k-fold cross-validation with paired t-tests and Bonferroni correction. Around ~50 dataset-level meta-features and ~15 column-level meta-features were computed alongside each measurement.

2. **Meta-model training**: Four independent LightGBM models (one per transform family) were trained to predict whether a given transform will improve AUC on an unseen dataset, using only the meta-features as input. Evaluation uses GroupKFold by dataset name to ensure no leakage.

3. **Inference**: Given a new dataset, the tool computes the same meta-features, feeds them to the meta-models, and returns a ranked list of recommended transforms with predicted AUC deltas.

## Transform Families

| Family | Transforms | Scope |
|---|---|---|
| **Numerical** | `log_transform`, `sqrt_transform`, `polynomial_square`, `polynomial_cube`, `reciprocal_transform`, `quantile_binning`, `impute_median`, `missing_indicator` | Per-column |
| **Categorical** | `frequency_encoding`, `target_encoding`, `onehot_encoding`, `hashing_encoding`, `missing_indicator` | Per-column |
| **Interaction** | `product_interaction`, `division_interaction`, `addition_interaction`, `abs_diff_interaction`, `group_mean`, `group_std`, `cat_concat` | Column pairs |
| **Row-level** | `row_numeric_stats` (mean/median/sum/std/min/max/range), `row_zero_stats`, `row_missing_stats` | Across all columns per row |

## Streamlit Application

The main interface is a Streamlit app (`recommend_app.py`) with a guided workflow:

1. **Upload** — Load a CSV and select the target column. The tool auto-detects feature types, missing values, and class imbalance.
2. **Analyze** — Computes meta-features and queries the meta-models. Returns ranked suggestions with predicted AUC deltas.
3. **Train** — Trains baseline (raw features) and enhanced (with selected transforms) LightGBM classifiers on a held-out validation split. Compares ROC-AUC, accuracy, F1, precision, recall, and log-loss.
4. **Test** — Upload a test CSV to evaluate both models side-by-side.
5. **Report** — Exports a full analysis report (HTML, Markdown, or PDF) with executive summary, data quality advisories, model comparison, and reproducibility metadata.

An optional LLM chat assistant (`chat_component.py`) lives in the sidebar, powered by the Google Gemini API, providing context-aware guidance at every step.

### Running the App

```bash
pip install streamlit lightgbm pandas numpy scikit-learn openml
streamlit run recommend_app.py
```

## Data Collection Pipeline

The collection pipeline runs on SLURM clusters and processes OpenML classification tasks in parallel.

### Collectors

Each collector evaluates one family of transforms across all tasks:

| Script | Output | Description |
|---|---|---|
| `collect_numerical_transforms.py` | `numerical_transforms.csv` | Per-column numerical transforms |
| `collect_categorical_transforms.py` | `categorical_transforms.csv` | Per-column categorical transforms |
| `collect_interaction_features.py` | `interaction_features.csv` | Pairwise column interactions |
| `collect_row_features.py` | `row_features.csv` | Row-level aggregation families |

Shared utilities live in `collector_utils.py` (CV evaluation, paired testing, meta-feature computation, delta calibration). The SLURM dispatcher is `collector_slurm.py`, configured via `run_collector.sh`.

### Running Collection

```bash
# Submit a SLURM array job for numerical transforms
COLLECTOR=numerical OUTPUT_DIR=/path/to/output sbatch --array=0-999 run_collector.sh

# After all jobs complete, merge worker CSVs
python merge_collector_results.py --output_dir /path/to/output --prefix numerical_transforms
```

## Meta-Model Training

`train_meta_models.py` trains four pairs of models (regressor + binary classifier) using the merged collection data. Key design choices:

- **GroupKFold** by dataset name prevents train/val leakage
- **Optuna** hyperparameter tuning (when available)
- **Calibrated delta** targets derived from Cohen's d with type-specific weighting
- **Composite positive** classification strategy

### Running Training

```bash
python train_meta_models.py \
    --numerical_csv   ./output_numerical/numerical_transforms_merged.csv \
    --categorical_csv ./output_categorical/categorical_transforms_merged.csv \
    --interaction_csv ./output_interactions/interaction_features_merged.csv \
    --row_csv         ./output_row/row_features_merged.csv \
    --output_dir      ./meta_models
```

Each type produces: `{type}_regressor.txt`, `{type}_classifier.txt`, and `{type}_config.json`.

### Training Statistics

| Type | Training Rows | Datasets | Features | Regression R² | Classification AUC |
|---|---|---|---|---|---|
| Numerical | 149,776 | 1,103 | 43 | 0.604 ± 0.103 | 0.673 ± 0.039 |
| Categorical | 19,999 | 423 | 38 | 0.660 ± 0.080 | 0.770 ± 0.032 |
| Interaction | 174,798 | 1,201 | 59 | 0.685 ± 0.055 | 0.629 ± 0.017 |
| Row | 2,037 | 1,104 | 33 | 0.280 ± 0.159 | 0.689 ± 0.033 |

## Ablation Study

`ablate_slurm.py` and `run_ablation.sh` run ablation experiments to measure the contribution of individual meta-feature groups. Results are stored in `ablation_results.json`.

## Project Structure

```
├── recommend_app.py                 # Main Streamlit application
├── transforms.py                    # Transform implementations (fit/apply)
├── ui_components.py                 # Streamlit UI helpers
├── app_constants.py                 # App configuration and constants
├── report_generator.py              # HTML/Markdown/PDF report generation
├── chat_component.py                # LLM chat assistant (Gemini)
│
├── collect_numerical_transforms.py  # Numerical transform collector
├── collect_categorical_transforms.py# Categorical transform collector
├── collect_interaction_features.py  # Interaction feature collector
├── collect_row_features.py          # Row-level feature collector
├── collector_utils.py               # Shared collection utilities
├── collector_slurm.py               # SLURM array job dispatcher
├── merge_collector_results.py       # Merge worker CSV outputs
├── count_classification_tasks.py    # Count available OpenML tasks
│
├── train_meta_models.py             # Meta-model training pipeline
├── ablate_slurm.py                  # Ablation study runner
│
├── run_collector.sh                 # SLURM job script for collection
├── run_meta_training.sh             # SLURM job script for training
├── run_ablation.sh                  # SLURM job script for ablation
│
├── training_report.json             # Meta-model evaluation results
└── ablation_results.json            # Ablation study results
```

## Requirements

- Python 3.10+
- LightGBM
- scikit-learn
- pandas, numpy
- Streamlit (for the app)
- OpenML (for data collection)
- Optuna (optional, for hyperparameter tuning)
- WeasyPrint (optional, for PDF export)
- google-generativeai (optional, for the chat assistant)


