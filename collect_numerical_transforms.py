"""
collect_numerical_transforms.py — Numerical Single-Column Transform Collector
==============================================================================

Collects meta-learning data for single-column numerical transformations.
For each numeric column in each dataset, tests a set of transforms (log, sqrt, 
square, binning, etc.) and records:
  - Dataset-level meta-features
  - Column-level meta-features (numeric-specific)
  - Intervention outcome (delta, p-value, effect size)

Classification tasks only. Evaluation via ROC-AUC with repeated k-fold CV.

Output: One CSV row per (dataset, column, transform) combination.

Usage:
    python collect_numerical_transforms.py --task_list ./task_list.json --output_dir ./output_numerical
"""

import pandas as pd
import numpy as np
import os
import json
import time
import gc
import traceback
import argparse

from scipy.stats import skew, kurtosis, shapiro, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

import openml
import warnings
warnings.filterwarnings('ignore')

from collector_utils import (
    BASE_PARAMS, INDIVIDUAL_PARAMS,
    evaluate_model, evaluate_with_intervention,
    paired_ttest_bonferroni, compute_cohens_d, compute_calibrated_delta,
    compute_noise_floor, compute_delta, normalize_delta,
    get_dataset_meta, prune_columns, get_baseline_importances,
    check_ceiling, write_csv, sanitize_string, ensure_numeric_target,
)


# =============================================================================
# CSV SCHEMA — Numerical Transforms
# =============================================================================

SCHEMA_NUMERICAL = [
    # --- Dataset-level (13 fields) ---
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',
    
    # --- Column-level: numeric-specific (18 fields) ---
    'null_pct', 'unique_ratio', 'outlier_ratio',
    'skewness', 'kurtosis_val', 'coeff_variation',
    'zeros_ratio', 'entropy',
    'is_binary', 'range_iqr_ratio',
    'baseline_feature_importance', 'importance_rank_pct',
    'spearman_corr_target', 'mutual_info_score',
    'shapiro_p_value',
    'bimodality_coefficient',
    'pct_negative', 'pct_in_0_1_range',
    
    # --- Intervention (3 fields) ---
    'column_name', 'method',
    
    # --- Full-model evaluation (8 fields) ---
    'delta', 'delta_normalized',
    'absolute_score',
    't_statistic', 'p_value', 'p_value_bonferroni',
    'is_significant', 'is_significant_bonferroni',
    
    # --- Individual evaluation (8 fields) ---
    'individual_baseline_score', 'individual_intervention_score',
    'individual_delta', 'individual_delta_normalized',
    'individual_p_value', 'individual_p_value_bonferroni',
    'individual_is_significant', 'individual_is_significant_bonferroni',
    
    # --- Effect sizes (4 fields) ---
    'cohens_d', 'individual_cohens_d',
    'calibrated_delta', 'individual_calibrated_delta',
    'null_std',
    
    # --- Identifiers ---
    'openml_task_id', 'dataset_name', 'dataset_id',
]


# =============================================================================
# NUMERICAL TRANSFORMS
# =============================================================================

NUMERICAL_METHODS = [
    'log_transform',
    'sqrt_transform',
    'polynomial_square',
    'polynomial_cube',
    'reciprocal_transform',
    'quantile_binning',
    'impute_median',
    'missing_indicator',
]


def apply_numerical_transform(X_train, X_val, y_train, col, col_second, method):
    """
    Apply a single numerical transform inside CV folds.
    Returns: (success, X_train, X_val)
    """
    try:
        if method == 'log_transform':
            fill = X_train[col].min()
            temp_tr = X_train[col].fillna(fill)
            temp_vl = X_val[col].fillna(fill)
            offset = abs(temp_tr.min()) + 1 if temp_tr.min() <= 0 else 0
            X_train[col] = np.log1p(temp_tr + offset)
            X_val[col] = np.log1p(temp_vl + offset)
        
        elif method == 'sqrt_transform':
            fill = X_train[col].median()
            temp_tr = X_train[col].fillna(fill)
            temp_vl = X_val[col].fillna(fill)
            offset = abs(temp_tr.min()) + 1 if temp_tr.min() < 0 else 0
            X_train[col] = np.sqrt(temp_tr + offset)
            X_val[col] = np.sqrt(temp_vl + offset)
        
        elif method == 'polynomial_square':
            med = X_train[col].median()
            X_train[f"{col}_sq"] = X_train[col].fillna(med) ** 2
            X_val[f"{col}_sq"] = X_val[col].fillna(med) ** 2
        
        elif method == 'polynomial_cube':
            med = X_train[col].median()
            X_train[f"{col}_cube"] = X_train[col].fillna(med) ** 3
            X_val[f"{col}_cube"] = X_val[col].fillna(med) ** 3
        
        elif method == 'reciprocal_transform':
            eps = 1e-5
            med = X_train[col].median()
            X_train[f"{col}_recip"] = 1.0 / (X_train[col].fillna(med).abs() + eps)
            X_val[f"{col}_recip"] = 1.0 / (X_val[col].fillna(med).abs() + eps)
        
        elif method == 'quantile_binning':
            _, bin_edges = pd.qcut(
                X_train[col].dropna(), q=5, retbins=True, duplicates='drop'
            )
            X_train[col] = pd.cut(X_train[col], bins=bin_edges, labels=False, include_lowest=True)
            X_val[col] = pd.cut(X_val[col], bins=bin_edges, labels=False, include_lowest=True)
        
        elif method == 'impute_median':
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)
        
        elif method == 'missing_indicator':
            X_train[f"{col}_is_na"] = X_train[col].isnull().astype(int)
            X_val[f"{col}_is_na"] = X_val[col].isnull().astype(int)
        
        return True, X_train, X_val
    
    except Exception as e:
        return False, X_train, X_val


# =============================================================================
# COLUMN-LEVEL META-FEATURES (NUMERIC)
# =============================================================================

def get_numeric_column_meta(series, y, importance, importance_rank_pct):
    """Compute meta-features for a single numeric column."""
    clean = series.dropna()
    y_numeric = ensure_numeric_target(y)
    
    meta = {}
    meta['null_pct'] = float(series.isnull().mean())
    meta['unique_ratio'] = float(series.nunique() / max(len(series), 1))
    meta['is_binary'] = int(series.nunique() <= 2)
    meta['baseline_feature_importance'] = float(importance)
    meta['importance_rank_pct'] = float(importance_rank_pct)
    
    if len(clean) < 5:
        # Not enough data for statistics
        meta.update({
            'outlier_ratio': 0.0, 'skewness': 0.0, 'kurtosis_val': 0.0,
            'coeff_variation': 0.0, 'zeros_ratio': 0.0, 'entropy': 0.0,
            'range_iqr_ratio': 1.0, 'spearman_corr_target': 0.0,
            'mutual_info_score': 0.0, 'shapiro_p_value': 0.5,
            'bimodality_coefficient': 0.0, 'pct_negative': 0.0,
            'pct_in_0_1_range': 0.0,
        })
        return meta
    
    # Outlier ratio (IQR method)
    Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:
        outliers = ((clean < Q1 - 1.5 * IQR) | (clean > Q3 + 1.5 * IQR)).mean()
        meta['outlier_ratio'] = float(outliers)
        meta['range_iqr_ratio'] = float((clean.max() - clean.min()) / IQR)
    else:
        meta['outlier_ratio'] = 0.0
        meta['range_iqr_ratio'] = 1.0
    
    # Distribution shape
    meta['skewness'] = float(skew(clean, nan_policy='omit'))
    meta['kurtosis_val'] = float(kurtosis(clean, nan_policy='omit'))
    
    std_val = clean.std()
    mean_val = clean.mean()
    meta['coeff_variation'] = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-10 else 0.0
    meta['zeros_ratio'] = float((clean == 0).mean())
    meta['pct_negative'] = float((clean < 0).mean())
    meta['pct_in_0_1_range'] = float(((clean >= 0) & (clean <= 1)).mean())
    
    # Entropy (via histogram)
    try:
        counts, _ = np.histogram(clean, bins=min(50, max(int(len(clean) ** 0.5), 5)))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        meta['entropy'] = float(-np.sum(probs * np.log2(probs)))
    except Exception:
        meta['entropy'] = 0.0
    
    # Normality test (Shapiro, subsample for speed)
    try:
        sample = clean.sample(min(5000, len(clean)), random_state=42)
        _, p_val = shapiro(sample)
        meta['shapiro_p_value'] = float(p_val)
    except Exception:
        meta['shapiro_p_value'] = 0.5
    
    # Bimodality coefficient: (skew^2 + 1) / (kurtosis + 3)
    sk = meta['skewness']
    kt = meta['kurtosis_val']
    meta['bimodality_coefficient'] = float((sk ** 2 + 1) / (kt + 3)) if (kt + 3) > 0 else 0.0
    
    # Correlation with target
    try:
        clean_idx = series.notna() & y_numeric.notna()
        if clean_idx.sum() > 10:
            sp, _ = spearmanr(series[clean_idx], y_numeric[clean_idx])
            meta['spearman_corr_target'] = float(abs(sp)) if not np.isnan(sp) else 0.0
        else:
            meta['spearman_corr_target'] = 0.0
    except Exception:
        meta['spearman_corr_target'] = 0.0
    
    # Mutual information with target
    try:
        filled = series.fillna(series.median()).to_frame()
        mi = mutual_info_classif(filled, y_numeric, random_state=42)[0]
        meta['mutual_info_score'] = float(mi)
    except Exception:
        meta['mutual_info_score'] = 0.0
    
    return meta


# =============================================================================
# METHOD APPLICABILITY GATES
# =============================================================================

def should_test_method(method, col_meta, series):
    """Check if a method makes sense for this column."""
    if method == 'impute_median' and col_meta['null_pct'] == 0:
        return False
    if method == 'missing_indicator' and col_meta['null_pct'] == 0:
        return False
    if method == 'sqrt_transform' and (series.min() < 0 if len(series.dropna()) > 0 else True):
        return False
    if method == 'log_transform' and col_meta['is_binary']:
        return False
    if method == 'quantile_binning' and col_meta['unique_ratio'] < 0.01:
        return False
    if method in ['polynomial_square', 'polynomial_cube'] and col_meta['is_binary']:
        return False
    if method == 'reciprocal_transform' and col_meta['is_binary']:
        return False
    return True


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

def collect_numerical_transforms(X, y, dataset_name="unknown",
                                  n_folds=5, n_repeats=3,
                                  time_budget=7200):
    """
    Collect meta-learning data for numerical single-column transforms.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        dataset_name: Name for logging
        n_folds, n_repeats: CV configuration
        time_budget: Max seconds for this dataset
    
    Returns:
        pd.DataFrame with rows matching SCHEMA_NUMERICAL
    """
    start_time = time.time()
    y = ensure_numeric_target(y)
    
    print(f"[NUM] Starting: {dataset_name} | Shape: {X.shape}")
    
    # --- Column pruning ---
    X, dropped = prune_columns(X, y)
    if dropped:
        print(f"[NUM]   Dropped {len(dropped)} columns: {[d[1] for d in dropped[:5]]}")
    
    if X.shape[1] == 0:
        print(f"[NUM]   No columns left after pruning")
        return pd.DataFrame()
    
    # --- Baseline evaluation ---
    print(f"[NUM]   Computing baseline...")
    base_score, base_std, baseline_folds, importances = evaluate_model(
        X, y, n_folds=n_folds, n_repeats=n_repeats, return_importance=True
    )
    
    if base_score is None:
        print(f"[NUM]   Baseline failed")
        return pd.DataFrame()
    
    print(f"[NUM]   Baseline AUC: {base_score:.5f} (std: {base_std:.5f})")
    
    # Ceiling check
    should_skip, _, reason = check_ceiling(base_score)
    if should_skip:
        print(f"[NUM]   Skipping: {reason}")
        return pd.DataFrame()
    
    # --- Dataset meta-features ---
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = base_score
    ds_meta['baseline_std'] = base_std
    ds_meta['relative_headroom'] = max(1.0 - base_score, 0.001)
    
    # Noise floor
    null_std, _ = compute_noise_floor(baseline_folds)
    
    # Feature importances
    imp = pd.Series(importances, index=X.columns)
    imp_ranks = imp.rank(ascending=False, pct=True)
    
    # --- Select numeric columns ---
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    print(f"[NUM]   Numeric columns: {len(numeric_cols)}")
    
    if not numeric_cols:
        print(f"[NUM]   No numeric columns found")
        return pd.DataFrame()
    
    # --- Test transforms ---
    results = []
    n_tests = 0
    
    for col_idx, col in enumerate(numeric_cols):
        if time.time() - start_time > time_budget:
            print(f"[NUM]   Time budget exceeded at column {col_idx}/{len(numeric_cols)}")
            break
        
        # Column meta-features
        col_meta = get_numeric_column_meta(
            X[col], y,
            importance=float(imp.get(col, 0)),
            importance_rank_pct=float(imp_ranks.get(col, 0.5))
        )
        
        # Individual baseline (just this column)
        indiv_base_score, indiv_base_std, indiv_base_folds = evaluate_model(
            X[[col]], y, params=INDIVIDUAL_PARAMS, n_folds=n_folds, n_repeats=n_repeats
        )
        
        for method in NUMERICAL_METHODS:
            if not should_test_method(method, col_meta, X[col]):
                continue
            
            if time.time() - start_time > time_budget:
                break
            
            # Full-model evaluation
            full_score, full_std, full_folds = evaluate_with_intervention(
                X, y, col_to_transform=col, method=method,
                apply_fn=apply_numerical_transform,
                n_folds=n_folds, n_repeats=n_repeats
            )
            
            if full_score is None:
                continue
            
            # Individual evaluation
            indiv_score, indiv_std, indiv_folds = evaluate_with_intervention(
                X[[col]], y, col_to_transform=col, method=method,
                apply_fn=apply_numerical_transform,
                params=INDIVIDUAL_PARAMS,
                n_folds=n_folds, n_repeats=n_repeats
            )
            
            n_tests += 1
            
            # Full-model stats
            delta = compute_delta(base_score, full_score)
            delta_norm = normalize_delta(delta, base_score)
            t_stat, p_val, p_bonf, sig, sig_bonf = paired_ttest_bonferroni(
                baseline_folds, full_folds, n_tests
            )
            
            # Individual stats
            indiv_delta = (
                compute_delta(indiv_base_score, indiv_score)
                if (indiv_score is not None and indiv_base_score is not None)
                else np.nan
            )
            indiv_delta_norm = normalize_delta(indiv_delta, indiv_base_score) if not np.isnan(indiv_delta) else np.nan
            if indiv_base_folds and indiv_folds:
                _, ip, ipb, isig, isigb = paired_ttest_bonferroni(
                    indiv_base_folds, indiv_folds, n_tests
                )
            else:
                ip, ipb, isig, isigb = np.nan, np.nan, False, False
            
            row = {**ds_meta, **col_meta}
            row.update({
                'column_name': sanitize_string(col),
                'method': method,
                'delta': delta,
                'delta_normalized': delta_norm,
                'absolute_score': full_score,
                't_statistic': t_stat,
                'p_value': p_val,
                'p_value_bonferroni': p_bonf,
                'is_significant': sig,
                'is_significant_bonferroni': sig_bonf,
                'individual_baseline_score': indiv_base_score,
                'individual_intervention_score': indiv_score,
                'individual_delta': indiv_delta,
                'individual_delta_normalized': indiv_delta_norm,
                'individual_p_value': ip,
                'individual_p_value_bonferroni': ipb,
                'individual_is_significant': isig,
                'individual_is_significant_bonferroni': isigb,
                'cohens_d': compute_cohens_d(baseline_folds, full_folds),
                'individual_cohens_d': compute_cohens_d(indiv_base_folds, indiv_folds) if indiv_folds else np.nan,
                'calibrated_delta': compute_calibrated_delta(delta, null_std),
                'individual_calibrated_delta': compute_calibrated_delta(indiv_delta, null_std) if not np.isnan(indiv_delta) else np.nan,
                'null_std': null_std,
            })
            
            results.append(row)
            
            sig_str = "✓" if sig else "✗"
            if not np.isnan(indiv_delta_norm):
                isig_str = "✓" if isig else "✗"
                indiv_str = f"indiv={indiv_delta_norm:+.2f}% p={ip:.4f} {isig_str}"
            else:
                indiv_str = "indiv=N/A"
            print(f"[NUM]     {col}.{method}: full={delta_norm:+.2f}% p={p_val:.4f} {sig_str} | {indiv_str}")
    
    # Post-hoc Bonferroni correction with final n_tests
    df = pd.DataFrame(results)
    if not df.empty and n_tests > 0:
        df['p_value_bonferroni'] = (df['p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['is_significant_bonferroni'] = df['p_value_bonferroni'] < 0.05
        df['individual_p_value_bonferroni'] = (df['individual_p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['individual_is_significant_bonferroni'] = df['individual_p_value_bonferroni'] < 0.05
    
    elapsed = time.time() - start_time
    print(f"[NUM]   Done: {len(df)} rows, {n_tests} tests in {elapsed:.1f}s")
    
    return df


# =============================================================================
# OPENML PIPELINE RUNNER
# =============================================================================

def run_openml_collection(output_dir='./output_numerical',
                           task_list_file=None,
                           n_folds=5, n_repeats=3,
                           time_budget=7200):
    """Run numerical transform collection across OpenML datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load task list
    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    with open(task_list_file, 'r') as f:
        task_list = json.load(f)
    
    # Filter to classification only
    task_list = [t for t in task_list if t.get('task_type') == 'classification']
    print(f"[NUM] Tasks: {len(task_list)} classification tasks")
    
    csv_file = os.path.join(output_dir, 'numerical_transforms.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint_numerical.json')
    
    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed = set(json.load(f).get('processed', []))
    
    for i, entry in enumerate(task_list):
        task_id = entry['task_id']
        ckpt_key = str(task_id)
        
        if ckpt_key in processed:
            continue
        
        try:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(task_list)}] Task {task_id}: {entry.get('dataset_name', '?')}")
            
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            target_col = dataset.default_target_attribute
            
            if not target_col:
                print(f"  No target — skipping")
                processed.add(ckpt_key)
                continue
            
            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            
            # Encode target
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
            
            df_result = collect_numerical_transforms(
                X, y, dataset_name=dataset.name,
                n_folds=n_folds, n_repeats=n_repeats,
                time_budget=time_budget
            )
            
            if not df_result.empty:
                df_result['openml_task_id'] = task_id
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                write_csv(df_result, csv_file, SCHEMA_NUMERICAL)
            
            processed.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed': list(processed)}, f)
            
            gc.collect()
        
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            processed.add(ckpt_key)
            continue
    
    print(f"\n[NUM] Collection complete. Results: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numerical Transform Data Collection")
    parser.add_argument('--output_dir', default='./output_numerical')
    parser.add_argument('--task_list', default=None)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--time_budget', type=int, default=7200)
    args = parser.parse_args()
    
    run_openml_collection(
        output_dir=args.output_dir,
        task_list_file=args.task_list,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        time_budget=args.time_budget,
    )