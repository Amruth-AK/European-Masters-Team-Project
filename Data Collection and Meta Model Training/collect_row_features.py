"""
collect_row_features.py — Row-Level Aggregation Feature Collector
=================================================================

Collects meta-learning data for row-level aggregation features.
Unlike column-level collectors, transforms here operate ACROSS columns per row,
adding new columns derived from all (or all numeric) columns.

KEY DESIGN: GROUP-BASED EVALUATION
  Methods are organized into *families* of closely related transforms.
  All features within a family are added simultaneously in a single CV
  evaluation pass, rather than one pass per individual method.
  This is statistically sound (features are correlated anyway) and
  ~3x faster than individual testing.

  Family 1 — row_numeric_stats:
      row_mean, row_median, row_sum, row_std, row_min, row_max, row_range
      Added together → one AUC delta for the whole family.
      Requires: ≥2 numeric columns.

  Family 2 — row_zero_stats:
      row_zero_count, row_zero_percentage
      Requires: ≥2 numeric cols AND at least one zero value in the data.

  Family 3 — row_missing_stats:
      row_missing_count, row_missing_percentage
      Requires: at least one NaN in the dataset.

Output schema: one CSV row per applicable *family* (not per individual method).
The `method` field holds the family name; `methods_in_group` lists
the individual methods that were added.

Classification tasks only. Evaluation via ROC-AUC with repeated k-fold CV.

Usage:
    python collect_row_features.py --task_list ./task_list.json --output_dir ./output_row
"""

import pandas as pd
import numpy as np
import os
import json
import time
import gc
import traceback
import argparse

from sklearn.preprocessing import LabelEncoder

import openml
import warnings
warnings.filterwarnings('ignore')

from collector_utils import (
    BASE_PARAMS,
    evaluate_model, evaluate_with_intervention,
    paired_ttest_bonferroni, compute_cohens_d, compute_calibrated_delta,
    compute_noise_floor, compute_delta, normalize_delta,
    get_dataset_meta, prune_columns,
    check_ceiling, write_csv, sanitize_string, ensure_numeric_target,
    filter_rare_classes, prepare_data_for_model, compute_scale_pos_weight,
)

import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score


# =============================================================================
# CSV SCHEMA — Row Features
# =============================================================================

SCHEMA_ROW = [
    # --- Dataset-level meta-features (shared with other collectors) ---
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',

    # --- Row-level dataset statistics (new, computed once per dataset) ---
    'n_numeric_cols_used',          # how many numeric cols fed into the transform
    'avg_numeric_mean',             # mean of column-wise means (numeric cols)
    'avg_numeric_std',              # mean of column-wise stds
    'avg_missing_pct',              # mean missing % across all columns
    'max_missing_pct',              # worst column missing %
    'avg_row_variance',             # mean row-wise variance across the dataset
    'pct_rows_with_any_missing',    # fraction of rows that have ≥1 NaN
    'pct_cells_zero',               # fraction of numeric cells that are exactly 0
    'pct_rows_with_any_zero',       # fraction of rows with ≥1 zero in numeric cols
    'numeric_col_corr_mean',        # mean pairwise |corr| among numeric cols
    'numeric_col_corr_max',         # max pairwise |corr| among numeric cols
    'avg_row_entropy',              # mean Shannon entropy across rows (numeric, discretised)
    'numeric_range_ratio',          # mean(col_max - col_min) / (global_std + 1e-8)

    # --- Group / Intervention ---
    'method',                       # family name: row_numeric_stats | row_zero_stats | row_missing_stats
    'methods_in_group',             # comma-separated individual methods added

    # --- Outcomes ---
    'delta', 'delta_normalized', 'absolute_score',
    't_statistic', 'p_value', 'p_value_bonferroni',
    'is_significant', 'is_significant_bonferroni',
    'cohens_d', 'null_std', 'calibrated_delta',

    # --- Identifiers ---
    'openml_task_id', 'dataset_name', 'dataset_id',
]


# =============================================================================
# ROW FEATURE FAMILIES
# =============================================================================

# Each family is evaluated as a unit: all methods are added simultaneously,
# then the combined model is evaluated once.
#
# 'requires' gates:
#   'numeric_2'  → ≥2 numeric columns
#   'has_zeros'  → numeric_2 AND at least one zero cell
#   'has_missing'→ at least one NaN anywhere in X
#
ROW_FAMILIES = {
    'row_numeric_stats': {
        'methods': [
            'row_mean', 'row_median', 'row_sum',
            'row_std', 'row_min', 'row_max', 'row_range',
        ],
        'requires': 'numeric_2',
        'description': 'Location and spread aggregations across all numeric columns per row',
    },
    'row_zero_stats': {
        'methods': ['row_zero_count', 'row_zero_percentage'],
        'requires': 'has_zeros',
        'description': 'Count and fraction of zero-valued numeric entries per row',
    },
    'row_missing_stats': {
        'methods': ['row_missing_count', 'row_missing_percentage'],
        'requires': 'has_missing',
        'description': 'Count and fraction of missing entries per row (all columns)',
    },
}


# =============================================================================
# INDIVIDUAL ROW TRANSFORMS
# =============================================================================

def _row_mean(X_num):
    return X_num.mean(axis=1)

def _row_median(X_num):
    return X_num.median(axis=1)

def _row_sum(X_num):
    return X_num.sum(axis=1)

def _row_std(X_num):
    return X_num.std(axis=1).fillna(0.0)

def _row_min(X_num):
    return X_num.min(axis=1)

def _row_max(X_num):
    return X_num.max(axis=1)

def _row_range(X_num):
    return X_num.max(axis=1) - X_num.min(axis=1)

def _row_zero_count(X_num):
    return (X_num == 0).sum(axis=1)

def _row_zero_percentage(X_num):
    n_cols = X_num.shape[1]
    return (X_num == 0).sum(axis=1) / max(n_cols, 1)

def _row_missing_count(X):
    return X.isnull().sum(axis=1)

def _row_missing_percentage(X):
    n_cols = X.shape[1]
    return X.isnull().sum(axis=1) / max(n_cols, 1)


INDIVIDUAL_TRANSFORMS = {
    'row_mean':               lambda X_num, X_all: _row_mean(X_num),
    'row_median':             lambda X_num, X_all: _row_median(X_num),
    'row_sum':                lambda X_num, X_all: _row_sum(X_num),
    'row_std':                lambda X_num, X_all: _row_std(X_num),
    'row_min':                lambda X_num, X_all: _row_min(X_num),
    'row_max':                lambda X_num, X_all: _row_max(X_num),
    'row_range':              lambda X_num, X_all: _row_range(X_num),
    'row_zero_count':         lambda X_num, X_all: _row_zero_count(X_num),
    'row_zero_percentage':    lambda X_num, X_all: _row_zero_percentage(X_num),
    'row_missing_count':      lambda X_num, X_all: _row_missing_count(X_all),
    'row_missing_percentage': lambda X_num, X_all: _row_missing_percentage(X_all),
}


def apply_row_family(X_train, X_val, y_train, family_name, numeric_cols):
    """
    Add all features from a family to X_train and X_val.
    Row transforms require no fit step — they are computed from the row itself,
    so there is zero leakage risk.

    Returns: (success, X_train_augmented, X_val_augmented)
    """
    try:
        methods = ROW_FAMILIES[family_name]['methods']

        for split_X in [X_train, X_val]:
            is_train = (split_X is X_train)
            target_X = X_train if is_train else X_val

            Xnum = X_train[numeric_cols].fillna(X_train[numeric_cols].median()) if is_train \
                   else X_val[numeric_cols].fillna(X_train[numeric_cols].median())
            Xall = target_X

            for method in methods:
                fn = INDIVIDUAL_TRANSFORMS[method]
                col_series = fn(Xnum, Xall)
                col_series.index = target_X.index
                target_X[f"__row_{method}"] = col_series

        return True, X_train, X_val

    except Exception as e:
        return False, X_train, X_val


# =============================================================================
# APPLICABILITY GATES
# =============================================================================

def check_family_applicable(family_name, numeric_cols, X):
    """Return (applicable: bool, reason: str)."""
    req = ROW_FAMILIES[family_name]['requires']

    if req == 'numeric_2':
        if len(numeric_cols) < 2:
            return False, f"only {len(numeric_cols)} numeric col(s)"
        return True, ''

    if req == 'has_zeros':
        if len(numeric_cols) < 2:
            return False, f"only {len(numeric_cols)} numeric col(s)"
        n_zeros = (X[numeric_cols] == 0).values.sum()
        if n_zeros == 0:
            return False, 'no zero cells'
        return True, ''

    if req == 'has_missing':
        n_missing = X.isnull().values.sum()
        if n_missing == 0:
            return False, 'no missing values'
        return True, ''

    return True, ''


# =============================================================================
# ROW META-FEATURES
# =============================================================================

def compute_row_meta(X, numeric_cols):
    """
    Compute row-level dataset statistics once before the family loop.
    These describe the structure of the numeric feature space and are
    meaningful predictors of whether row aggregations will help.
    """
    meta = {}
    meta['n_numeric_cols_used'] = len(numeric_cols)

    if len(numeric_cols) == 0:
        meta.update({
            'avg_numeric_mean': np.nan,
            'avg_numeric_std': np.nan,
            'avg_row_variance': np.nan,
            'pct_cells_zero': np.nan,
            'pct_rows_with_any_zero': np.nan,
            'numeric_col_corr_mean': np.nan,
            'numeric_col_corr_max': np.nan,
            'avg_row_entropy': np.nan,
            'numeric_range_ratio': np.nan,
        })
    else:
        X_num = X[numeric_cols]
        meta['avg_numeric_mean'] = float(X_num.mean().mean())
        meta['avg_numeric_std']  = float(X_num.std().mean())
        meta['avg_row_variance'] = float(X_num.var(axis=1).mean())

        # Zero stats
        total_cells = X_num.shape[0] * X_num.shape[1]
        n_zeros = float((X_num == 0).values.sum())
        meta['pct_cells_zero'] = n_zeros / max(total_cells, 1)
        meta['pct_rows_with_any_zero'] = float((X_num == 0).any(axis=1).mean())

        # Pairwise correlation among numeric cols
        if len(numeric_cols) >= 2:
            corr = X_num.corr().abs()
            np.fill_diagonal(corr.values, np.nan)
            meta['numeric_col_corr_mean'] = float(np.nanmean(corr.values))
            meta['numeric_col_corr_max']  = float(np.nanmax(corr.values))
        else:
            meta['numeric_col_corr_mean'] = np.nan
            meta['numeric_col_corr_max']  = np.nan

        # Row-level Shannon entropy (discretise into 10 bins per col)
        try:
            binned = pd.DataFrame(index=X_num.index)
            for c in numeric_cols:
                col = X_num[c].fillna(X_num[c].median())
                binned[c] = pd.qcut(col, q=10, labels=False, duplicates='drop').fillna(0)
            # Per-row value counts across columns (treat as discrete dist)
            def row_entropy(row):
                counts = row.value_counts()
                p = counts / counts.sum()
                return float(-(p * np.log2(p + 1e-10)).sum())
            meta['avg_row_entropy'] = float(binned.apply(row_entropy, axis=1).mean())
        except Exception:
            meta['avg_row_entropy'] = np.nan

        # Numeric range ratio: how spread out are the column ranges vs overall std
        col_ranges = X_num.max() - X_num.min()
        global_std = float(X_num.values.std()) + 1e-8
        meta['numeric_range_ratio'] = float(col_ranges.mean() / global_std)

    # Missing stats (across all columns, not just numeric)
    meta['avg_missing_pct']          = float(X.isnull().mean().mean())
    meta['max_missing_pct']          = float(X.isnull().mean().max())
    meta['pct_rows_with_any_missing'] = float(X.isnull().any(axis=1).mean())

    return meta


# =============================================================================
# CV LOOP FOR A ROW FAMILY
# =============================================================================

def evaluate_row_family(X, y, family_name, numeric_cols,
                         baseline_folds, params,
                         n_folds=5, n_repeats=3):
    """
    Run the full repeated k-fold evaluation for a single row family.
    Adds all family features inside each fold (no leakage).

    Returns: (mean_score, std_score, fold_scores) or (None, None, None)
    """
    y_numeric = ensure_numeric_target(y)
    X, y_numeric = filter_rare_classes(X, y_numeric, n_folds, n_repeats)
    n_classes = y_numeric.nunique()

    p = params.copy()
    if n_classes == 2:
        p['scale_pos_weight'] = compute_scale_pos_weight(y_numeric)

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    fold_scores = []

    for train_idx, val_idx in rkf.split(X):
        X_train = X.iloc[train_idx].copy()
        X_val   = X.iloc[val_idx].copy()
        y_train = y_numeric.iloc[train_idx]
        y_val   = y_numeric.iloc[val_idx]

        if y_val.nunique() < 2:
            continue

        # Apply row family features (no leakage: all ops are row-local)
        success, X_train, X_val = apply_row_family(
            X_train, X_val, y_train, family_name, numeric_cols
        )
        if not success:
            continue

        X_tr, X_vl = prepare_data_for_model(X_train, X_val)

        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_tr, y_train,
            eval_set=[(X_vl, y_val)],
            callbacks=[early_stopping(10, verbose=False)]
        )

        if n_classes == 2:
            y_pred = model.predict_proba(X_vl)[:, 1]
            score = roc_auc_score(y_val, y_pred)
        else:
            y_pred = model.predict_proba(X_vl)
            score = roc_auc_score(y_val, y_pred, multi_class='ovr', average='weighted')

        fold_scores.append(score)

    if not fold_scores:
        return None, None, None

    return float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1)), fold_scores


# =============================================================================
# MAIN COLLECTOR FUNCTION
# =============================================================================

def collect_row_features(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    n_folds: int = 5,
    n_repeats: int = 3,
    time_budget: int = 7200,
) -> pd.DataFrame:
    """
    Collect row-level aggregation feature meta-data for one dataset.

    For each applicable row feature family:
      1. Add all family features simultaneously inside CV folds.
      2. Record AUC delta vs. the no-augmentation baseline.
      3. Return one row per evaluated family.

    Args:
        X:            Feature matrix (pre-cleaned, no target column).
        y:            Target series (encoded as integer classes).
        dataset_name: Human-readable dataset name (for logging).
        n_folds:      Number of CV folds.
        n_repeats:    Number of CV repeats.
        time_budget:  Max wall-clock seconds for this dataset.

    Returns:
        pd.DataFrame with columns matching SCHEMA_ROW.
        Empty DataFrame if dataset is unusable.
    """
    start_time = time.time()
    print(f"\n[ROW] Dataset: {dataset_name} | Shape: {X.shape}")

    # -----------------------------------------------------------------
    # 0. Prune problematic columns (ID-like, constant, leaking)
    # -----------------------------------------------------------------
    X, dropped = prune_columns(X, y)
    if dropped:
        print(f"[ROW]   prune_columns: dropped {len(dropped)} cols: "
              f"{[c for c, _ in dropped[:5]]}"
              + (" ..." if len(dropped) > 5 else ""))

    # -----------------------------------------------------------------
    # 1. Identify numeric columns
    # -----------------------------------------------------------------
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[ROW]   Numeric cols: {len(numeric_cols)} / {X.shape[1]}")

    # Skip entirely if no numeric columns
    if len(numeric_cols) == 0:
        print(f"[ROW]   No numeric columns — skipping")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # 2. Dataset-level meta-features (shared schema fields)
    # -----------------------------------------------------------------
    ds_meta = get_dataset_meta(X, y)

    # -----------------------------------------------------------------
    # 3. Row-level meta-features (new, computed once)
    # -----------------------------------------------------------------
    row_meta = compute_row_meta(X, numeric_cols)

    # -----------------------------------------------------------------
    # 4. Baseline evaluation (no augmentation)
    # -----------------------------------------------------------------
    base_score, base_std, baseline_folds = evaluate_model(
        X, y, params=BASE_PARAMS, n_folds=n_folds, n_repeats=n_repeats
    )

    if base_score is None:
        print(f"[ROW]   Baseline failed — skipping")
        return pd.DataFrame()

    is_ceiling, near_ceiling, ceiling_reason = check_ceiling(base_score)
    if is_ceiling:
        print(f"[ROW]   Ceiling hit ({ceiling_reason}) — skipping")
        return pd.DataFrame()

    headroom = max(1.0 - base_score, 0.001)
    ds_meta['baseline_score']   = base_score
    ds_meta['baseline_std']     = base_std
    ds_meta['relative_headroom'] = headroom

    null_std, _ = compute_noise_floor(baseline_folds)

    print(f"[ROW]   Baseline AUC: {base_score:.4f}  (headroom: {headroom:.4f})")

    # -----------------------------------------------------------------
    # 5. Family loop
    # -----------------------------------------------------------------
    results = []
    n_tests = 0

    for family_name, family_cfg in ROW_FAMILIES.items():
        if time.time() - start_time > time_budget:
            print(f"[ROW]   Time budget exceeded — stopping early")
            break

        # --- Applicability gate ---
        applicable, reason = check_family_applicable(family_name, numeric_cols, X)
        if not applicable:
            print(f"[ROW]   {family_name}: SKIP ({reason})")
            continue

        print(f"[ROW]   {family_name}: evaluating {family_cfg['methods']} ...")

        # --- CV evaluation ---
        score, std, folds = evaluate_row_family(
            X, y, family_name, numeric_cols,
            baseline_folds, BASE_PARAMS,
            n_folds=n_folds, n_repeats=n_repeats,
        )

        if score is None:
            print(f"[ROW]   {family_name}: evaluation failed — skipping")
            continue

        n_tests += 1

        delta      = compute_delta(base_score, score)
        delta_norm = normalize_delta(delta, base_score)
        t_stat, p_val, p_bonf, sig, sig_bonf = paired_ttest_bonferroni(
            baseline_folds, folds, n_tests
        )
        cd = compute_cohens_d(baseline_folds, folds)
        cal_delta = compute_calibrated_delta(delta, null_std)

        sig_str = "✓" if sig else "✗"
        print(f"[ROW]     {family_name}: delta={delta_norm:+.2f}%  "
              f"p={p_val:.4f} {sig_str}  AUC={score:.4f}")

        row = {**ds_meta, **row_meta}
        row.update({
            'method':           family_name,
            'methods_in_group': ','.join(family_cfg['methods']),
            'delta':            delta,
            'delta_normalized': delta_norm,
            'absolute_score':   score,
            't_statistic':      t_stat,
            'p_value':          p_val,
            'p_value_bonferroni':        p_bonf,
            'is_significant':            sig,
            'is_significant_bonferroni': sig_bonf,
            'cohens_d':         cd,
            'null_std':         null_std,
            'calibrated_delta': cal_delta,
        })
        results.append(row)

    # -----------------------------------------------------------------
    # 6. Post-hoc Bonferroni correction with final n_tests count
    # -----------------------------------------------------------------
    df = pd.DataFrame(results)
    if not df.empty and n_tests > 0:
        df['p_value_bonferroni'] = (df['p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['is_significant_bonferroni'] = df['p_value_bonferroni'] < 0.05

    elapsed = time.time() - start_time
    print(f"[ROW]   Done: {len(df)} rows, {n_tests} families in {elapsed:.1f}s")

    return df


# =============================================================================
# OPENML PIPELINE RUNNER (standalone, non-SLURM mode)
# =============================================================================

def run_openml_collection(output_dir='./output_row',
                           task_list_file=None,
                           n_folds=5, n_repeats=3,
                           time_budget=7200):
    """Run row feature collection across OpenML datasets (local, sequential)."""
    os.makedirs(output_dir, exist_ok=True)

    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    with open(task_list_file, 'r') as f:
        task_list = json.load(f)

    task_list = [t for t in task_list if t.get('task_type') == 'classification']
    print(f"[ROW] Tasks: {len(task_list)} classification tasks")

    csv_file        = os.path.join(output_dir, 'row_features.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint_row.json')

    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed = set(json.load(f).get('processed', []))

    for i, entry in enumerate(task_list):
        task_id  = entry['task_id']
        ckpt_key = str(task_id)

        if ckpt_key in processed:
            continue

        try:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(task_list)}] Task {task_id}: {entry.get('dataset_name', '?')}")

            task    = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            target_col = dataset.default_target_attribute

            if not target_col:
                print(f"  No target — skipping")
                processed.add(ckpt_key)
                continue

            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')

            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

            df_result = collect_row_features(
                X, y, dataset_name=dataset.name,
                n_folds=n_folds, n_repeats=n_repeats,
                time_budget=time_budget,
            )

            if not df_result.empty:
                df_result['openml_task_id'] = task_id
                df_result['dataset_name']   = dataset.name
                df_result['dataset_id']     = dataset.dataset_id
                write_csv(df_result, csv_file, SCHEMA_ROW)

            processed.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed': list(processed)}, f)

            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            processed.add(ckpt_key)
            continue

    print(f"\n[ROW] Collection complete. Results: {csv_file}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Row Feature Data Collection")
    parser.add_argument('--output_dir',  default='./output_row')
    parser.add_argument('--task_list',   default=None)
    parser.add_argument('--n_folds',     type=int, default=5)
    parser.add_argument('--n_repeats',   type=int, default=3)
    parser.add_argument('--time_budget', type=int, default=7200)
    args = parser.parse_args()

    run_openml_collection(
        output_dir=args.output_dir,
        task_list_file=args.task_list,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        time_budget=args.time_budget,
    )
