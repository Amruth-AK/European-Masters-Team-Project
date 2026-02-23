"""
Add refinement dimension columns to an existing hp_tuning_db.csv.

Reads a CSV that has dataset meta (task_type, n_classes, class_imbalance_ratio, n_cols, ...),
computes classification_subtype, imbalance_level, high_dim_level per row using the same
logic as HPCollector.compute_refinement_dimensions, and writes a new CSV with the three
columns added (or overwritten). Use this on Adrian's or any hp_tuning_db that lacks
these columns before training the HP meta-model with refinement features.

Usage:
    python add_refinement_columns.py --input hp_tuning_db.csv --output hp_tuning_db_refined.csv
    python add_refinement_columns.py --input ./path/to/adrian_hp_tuning_db.csv --output ./hp_tuning_db_adrian_refined.csv
"""

import argparse
import os
import pandas as pd

# Same logic as HPCollector (no HPCollector import to avoid lightgbm etc.)
IMBALANCE_RATIO_BALANCED = 0.2
IMBALANCE_RATIO_MODERATE = 0.05
HIGH_DIM_LOW = 20
HIGH_DIM_HIGH = 100

CSV_SCHEMA = [
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'class_imbalance_ratio', 'n_classes',
    'target_std', 'target_skew', 'target_kurtosis', 'target_nunique_ratio',
    'landmarking_score', 'landmarking_score_norm',
    'avg_feature_corr', 'max_feature_corr', 'avg_target_corr', 'max_target_corr',
    'avg_numeric_sparsity', 'linearity_gap',
    'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
    'matrix_rank_ratio',
    'std_feature_importance', 'max_minus_min_importance',
    'pct_features_above_median_importance', 'avg_skewness', 'avg_kurtosis',
    'hp_num_leaves', 'hp_max_depth', 'hp_learning_rate', 'hp_n_estimators',
    'hp_min_child_samples', 'hp_subsample', 'hp_colsample_bytree',
    'hp_reg_alpha', 'hp_reg_lambda', 'hp_max_bin',
    'hp_config_name', 'hp_is_default',
    'hp_leaves_depth_ratio', 'hp_regularization_strength', 'hp_sample_ratio',
    'hp_lr_estimators_product',
    'metric_accuracy', 'metric_balanced_accuracy', 'metric_f1_weighted',
    'metric_auc', 'metric_log_loss', 'metric_auc_std',
    'metric_rmse', 'metric_mae', 'metric_r2', 'metric_mape', 'metric_rmse_std',
    'primary_score', 'primary_score_std', 'rank_in_dataset', 'delta_vs_default',
    'normalized_score', 'pct_of_best',
    'train_time_seconds', 'actual_n_estimators',
    'openml_task_id', 'dataset_name', 'dataset_id', 'task_type',
    'classification_subtype', 'imbalance_level', 'high_dim_level',
]


def _compute_refinement_dimensions(task_type: str, ds_meta: dict) -> dict:
    n_classes = ds_meta.get("n_classes", 2)
    class_imbalance_ratio = ds_meta.get("class_imbalance_ratio", 1.0)
    n_cols = ds_meta.get("n_cols", 0)
    if task_type == "regression":
        classification_subtype = "na"
        imbalance_level = "na"
    else:
        classification_subtype = "multiclass" if (n_classes is not None and n_classes > 2) else "binary"
        if class_imbalance_ratio is None or class_imbalance_ratio < 0:
            imbalance_level = "na"
        elif class_imbalance_ratio >= IMBALANCE_RATIO_BALANCED:
            imbalance_level = "balanced"
        elif class_imbalance_ratio >= IMBALANCE_RATIO_MODERATE:
            imbalance_level = "moderate"
        else:
            imbalance_level = "imbalanced"
    if n_cols is None or n_cols < HIGH_DIM_LOW:
        high_dim_level = "low"
    elif n_cols < HIGH_DIM_HIGH:
        high_dim_level = "medium"
    else:
        high_dim_level = "high"
    return {
        "classification_subtype": classification_subtype,
        "imbalance_level": imbalance_level,
        "high_dim_level": high_dim_level,
    }


def add_refinement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification_subtype, imbalance_level, high_dim_level to each row."""
    out = df.copy()
    task_col = "task_type"
    if task_col not in out.columns:
        raise ValueError(f"CSV must have column '{task_col}' for refinement dimensions.")
    required = ["n_classes", "class_imbalance_ratio", "n_cols"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"CSV missing required columns for refinement: {missing}")

    res = []
    for _, row in out.iterrows():
        task_type = str(row[task_col]).strip().lower() if pd.notna(row[task_col]) else "classification"
        if task_type not in ("classification", "regression"):
            task_type = "classification"
        def _num(v):
            x = row.get(v)
            return None if pd.isna(x) else x

        ds_meta = {
            "n_classes": _num("n_classes"),
            "class_imbalance_ratio": _num("class_imbalance_ratio"),
            "n_cols": _num("n_cols"),
        }
        dims = _compute_refinement_dimensions(task_type, ds_meta)
        res.append(dims)

    for key in ("classification_subtype", "imbalance_level", "high_dim_level"):
        out[key] = [r[key] for r in res]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Add refinement columns (classification_subtype, imbalance_level, high_dim_level) to hp_tuning_db.csv"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV path (e.g. Adrian's hp_tuning_db.csv)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path (e.g. hp_tuning_db_refined.csv)")
    parser.add_argument(
        "--schema-order",
        action="store_true",
        help="Reorder columns to match HPCollector.CSV_SCHEMA (only includes columns present in CSV)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    df = add_refinement_columns(df)
    print("  Added: classification_subtype, imbalance_level, high_dim_level")

    if args.schema_order:
        # Keep only columns that exist in both CSV and schema, in schema order
        order = [c for c in CSV_SCHEMA if c in df.columns]
        df = df[order]
        print(f"  Reordered columns to match CSV_SCHEMA ({len(order)} columns)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
