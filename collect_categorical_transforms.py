"""
collect_categorical_transforms.py — Categorical Single-Column Transform Collector
==================================================================================

Collects meta-learning data for single-column categorical transformations.
For each categorical column in each dataset, tests a set of encoding strategies
(frequency, target, one-hot, hashing) and records:
  - Dataset-level meta-features
  - Column-level meta-features (categorical-specific)
  - Intervention outcome (delta, p-value, effect size)

Classification tasks only. Evaluation via ROC-AUC with repeated k-fold CV.

Output: One CSV row per (dataset, column, transform) combination.

Usage:
    python collect_categorical_transforms.py --task_list ./task_list.json --output_dir ./output_categorical
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
# CSV SCHEMA — Categorical Transforms
# =============================================================================

SCHEMA_CATEGORICAL = [
    # --- Dataset-level (13 fields) ---
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',
    
    # --- Column-level: categorical-specific (16 fields) ---
    'null_pct', 'n_unique', 'unique_ratio',
    'entropy', 'normalized_entropy',
    'is_binary', 'is_low_cardinality', 'is_high_cardinality',
    'top_category_dominance', 'top3_category_concentration',
    'rare_category_pct',
    'conditional_entropy',
    'baseline_feature_importance', 'importance_rank_pct',
    'mutual_info_score', 'pps_score',
    
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
# CATEGORICAL TRANSFORMS
# =============================================================================

CATEGORICAL_METHODS = [
    'frequency_encoding',
    'target_encoding',
    'onehot_encoding',
    'hashing_encoding',
    'missing_indicator',
]

# Methods where individual evaluation is structurally invariant
# (injective re-encoding doesn't change LightGBM histogram splits on a single column)
INDIVIDUAL_INVARIANT_METHODS = {
    'frequency_encoding',
    'target_encoding',
}


def apply_categorical_transform(X_train, X_val, y_train, col, col_second, method):
    """
    Apply a single categorical transform inside CV folds.
    Returns: (success, X_train, X_val)
    """
    y_train_numeric = ensure_numeric_target(y_train)
    
    try:
        if method == 'frequency_encoding':
            str_train = X_train[col].astype(str)
            str_val = X_val[col].astype(str)
            freq = str_train.value_counts(normalize=True)
            X_train[col] = str_train.map(freq).fillna(0).astype(float)
            X_val[col] = str_val.map(freq).fillna(0).astype(float)
        
        elif method == 'target_encoding':
            str_train = X_train[col].astype(str)
            str_val = X_val[col].astype(str)
            global_mean = float(y_train_numeric.mean())
            agg = y_train_numeric.groupby(str_train).agg(['count', 'mean'])
            # Bayesian smoothing (m=10)
            smooth = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
            X_train[col] = str_train.map(smooth).fillna(global_mean).astype(float)
            X_val[col] = str_val.map(smooth).fillna(global_mean).astype(float)
        
        elif method == 'onehot_encoding':
            str_train = X_train[col].astype(str)
            str_val = X_val[col].astype(str)
            train_dummies = pd.get_dummies(str_train, prefix=col, drop_first=True)
            val_dummies = pd.get_dummies(str_val, prefix=col, drop_first=True)
            val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
            X_train = X_train.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            X_train = pd.concat([X_train, train_dummies], axis=1)
            X_val = pd.concat([X_val, val_dummies], axis=1)
        
        elif method == 'hashing_encoding':
            n_hash = 32
            X_train[col] = X_train[col].astype(str).apply(lambda x: hash(x) % n_hash)
            X_val[col] = X_val[col].astype(str).apply(lambda x: hash(x) % n_hash)
        
        elif method == 'missing_indicator':
            X_train[f"{col}_is_na"] = X_train[col].isnull().astype(int)
            X_val[f"{col}_is_na"] = X_val[col].isnull().astype(int)
        
        return True, X_train, X_val
    
    except Exception as e:
        return False, X_train, X_val


# =============================================================================
# COLUMN-LEVEL META-FEATURES (CATEGORICAL)
# =============================================================================

def get_categorical_column_meta(series, y, importance, importance_rank_pct):
    """Compute meta-features for a single categorical column."""
    y_numeric = ensure_numeric_target(y)
    str_series = series.astype(str).replace('nan', np.nan)
    
    meta = {}
    meta['null_pct'] = float(series.isnull().mean())
    
    n_unique = series.nunique(dropna=True)
    meta['n_unique'] = n_unique
    meta['unique_ratio'] = float(n_unique / max(len(series), 1))
    meta['is_binary'] = int(n_unique <= 2)
    meta['is_low_cardinality'] = int(n_unique <= 10)
    meta['is_high_cardinality'] = int(n_unique > 50)
    
    meta['baseline_feature_importance'] = float(importance)
    meta['importance_rank_pct'] = float(importance_rank_pct)
    
    # Category distribution
    vc = series.value_counts(normalize=True, dropna=True)
    meta['top_category_dominance'] = float(vc.iloc[0]) if len(vc) > 0 else 1.0
    meta['top3_category_concentration'] = float(vc.iloc[:3].sum()) if len(vc) > 0 else 1.0
    
    # Rare categories (appearing < 1% of the time)
    if len(vc) > 0:
        meta['rare_category_pct'] = float((vc < 0.01).mean())
    else:
        meta['rare_category_pct'] = 0.0
    
    # Entropy
    if len(vc) > 0:
        probs = vc.values
        probs = probs[probs > 0]
        meta['entropy'] = float(-np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(max(n_unique, 2))
        meta['normalized_entropy'] = float(meta['entropy'] / max_entropy) if max_entropy > 0 else 0.0
    else:
        meta['entropy'] = 0.0
        meta['normalized_entropy'] = 0.0
    
    # Conditional entropy H(Y|X) — how much uncertainty about Y remains after knowing X
    try:
        categories = series.dropna().unique()
        h_y_given_x = 0.0
        for cat in categories:
            mask = series == cat
            p_cat = mask.mean()
            y_cat = y_numeric[mask]
            if len(y_cat) > 0 and y_cat.nunique() > 1:
                vc_y = y_cat.value_counts(normalize=True)
                h_cat = -np.sum(vc_y * np.log2(vc_y.clip(lower=1e-10)))
                h_y_given_x += p_cat * h_cat
        meta['conditional_entropy'] = float(h_y_given_x)
    except Exception:
        meta['conditional_entropy'] = 0.0
    
    # Mutual information with target
    try:
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna('NaN'))
        mi = mutual_info_classif(
            encoded.reshape(-1, 1), y_numeric, random_state=42
        )[0]
        meta['mutual_info_score'] = float(mi)
    except Exception:
        meta['mutual_info_score'] = 0.0
    
    # Predictive Power Score (simplified: decision tree AUC)
    try:
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna('NaN'))
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(encoded.reshape(-1, 1), y_numeric)
        pps = float(dt.score(encoded.reshape(-1, 1), y_numeric))
        # Normalize to [0, 1] by subtracting random baseline
        random_baseline = 1.0 / max(y_numeric.nunique(), 2)
        meta['pps_score'] = max(0.0, (pps - random_baseline) / (1.0 - random_baseline))
    except Exception:
        meta['pps_score'] = 0.0
    
    return meta


# =============================================================================
# METHOD APPLICABILITY GATES
# =============================================================================

def should_test_method(method, col_meta):
    """Check if a method makes sense for this column."""
    n_unique = col_meta['n_unique']
    
    if method == 'missing_indicator' and col_meta['null_pct'] == 0:
        return False
    if method == 'onehot_encoding' and (n_unique < 2 or n_unique > 10):
        return False
    if method == 'hashing_encoding' and n_unique <= 10:
        return False  # Only useful for high cardinality
    return True


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

def collect_categorical_transforms(X, y, dataset_name="unknown",
                                    n_folds=5, n_repeats=3,
                                    time_budget=7200):
    """
    Collect meta-learning data for categorical single-column transforms.
    """
    start_time = time.time()
    y = ensure_numeric_target(y)
    
    print(f"[CAT] Starting: {dataset_name} | Shape: {X.shape}")
    
    # --- Column pruning ---
    X, dropped = prune_columns(X, y)
    if dropped:
        print(f"[CAT]   Dropped {len(dropped)} columns")
    
    if X.shape[1] == 0:
        return pd.DataFrame()
    
    # --- Baseline evaluation ---
    print(f"[CAT]   Computing baseline...")
    base_score, base_std, baseline_folds, importances = evaluate_model(
        X, y, n_folds=n_folds, n_repeats=n_repeats, return_importance=True
    )
    
    if base_score is None:
        print(f"[CAT]   Baseline failed")
        return pd.DataFrame()
    
    print(f"[CAT]   Baseline AUC: {base_score:.5f}")
    
    should_skip, _, reason = check_ceiling(base_score)
    if should_skip:
        print(f"[CAT]   Skipping: {reason}")
        return pd.DataFrame()
    
    # --- Dataset meta-features ---
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = base_score
    ds_meta['baseline_std'] = base_std
    ds_meta['relative_headroom'] = max(1.0 - base_score, 0.001)
    
    null_std, _ = compute_noise_floor(baseline_folds)
    
    imp = pd.Series(importances, index=X.columns)
    imp_ranks = imp.rank(ascending=False, pct=True)
    
    # --- Select categorical columns ---
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"[CAT]   Categorical columns: {len(cat_cols)}")
    
    if not cat_cols:
        print(f"[CAT]   No categorical columns found")
        return pd.DataFrame()
    
    # --- Test transforms ---
    results = []
    n_tests = 0
    
    for col_idx, col in enumerate(cat_cols):
        if time.time() - start_time > time_budget:
            print(f"[CAT]   Time budget exceeded at column {col_idx}/{len(cat_cols)}")
            break
        
        col_meta = get_categorical_column_meta(
            X[col], y,
            importance=float(imp.get(col, 0)),
            importance_rank_pct=float(imp_ranks.get(col, 0.5))
        )
        
        # Individual baseline
        indiv_base_score, indiv_base_std, indiv_base_folds = evaluate_model(
            X[[col]], y, params=INDIVIDUAL_PARAMS, n_folds=n_folds, n_repeats=n_repeats
        )
        
        for method in CATEGORICAL_METHODS:
            if not should_test_method(method, col_meta):
                continue
            
            if time.time() - start_time > time_budget:
                break
            
            # Full-model evaluation
            full_score, full_std, full_folds = evaluate_with_intervention(
                X, y, col_to_transform=col, method=method,
                apply_fn=apply_categorical_transform,
                n_folds=n_folds, n_repeats=n_repeats
            )
            
            if full_score is None:
                continue
            
            # Individual evaluation (skip for invariant methods)
            if method in INDIVIDUAL_INVARIANT_METHODS:
                indiv_score, indiv_folds = None, None
            else:
                indiv_score, _, indiv_folds = evaluate_with_intervention(
                    X[[col]], y, col_to_transform=col, method=method,
                    apply_fn=apply_categorical_transform,
                    params=INDIVIDUAL_PARAMS,
                    n_folds=n_folds, n_repeats=n_repeats
                )
            
            n_tests += 1
            
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
            elif method in INDIVIDUAL_INVARIANT_METHODS:
                indiv_str = "indiv=SKIP(invariant)"
            else:
                indiv_str = "indiv=N/A"
            print(f"[CAT]     {col}.{method}: full={delta_norm:+.2f}% p={p_val:.4f} {sig_str} | {indiv_str}")
    
    # Post-hoc Bonferroni correction
    df = pd.DataFrame(results)
    if not df.empty and n_tests > 0:
        df['p_value_bonferroni'] = (df['p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['is_significant_bonferroni'] = df['p_value_bonferroni'] < 0.05
        df['individual_p_value_bonferroni'] = (df['individual_p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['individual_is_significant_bonferroni'] = df['individual_p_value_bonferroni'] < 0.05
    
    elapsed = time.time() - start_time
    print(f"[CAT]   Done: {len(df)} rows, {n_tests} tests in {elapsed:.1f}s")
    
    return df


# =============================================================================
# OPENML PIPELINE RUNNER
# =============================================================================

def run_openml_collection(output_dir='./output_categorical',
                           task_list_file=None,
                           n_folds=5, n_repeats=3,
                           time_budget=7200):
    """Run categorical transform collection across OpenML datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    with open(task_list_file, 'r') as f:
        task_list = json.load(f)
    
    task_list = [t for t in task_list if t.get('task_type') == 'classification']
    print(f"[CAT] Tasks: {len(task_list)} classification tasks")
    
    csv_file = os.path.join(output_dir, 'categorical_transforms.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint_categorical.json')
    
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
                processed.add(ckpt_key)
                continue
            
            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
            
            df_result = collect_categorical_transforms(
                X, y, dataset_name=dataset.name,
                n_folds=n_folds, n_repeats=n_repeats,
                time_budget=time_budget
            )
            
            if not df_result.empty:
                df_result['openml_task_id'] = task_id
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                write_csv(df_result, csv_file, SCHEMA_CATEGORICAL)
            
            processed.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed': list(processed)}, f)
            
            gc.collect()
        
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            processed.add(ckpt_key)
            continue
    
    print(f"\n[CAT] Collection complete. Results: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorical Transform Data Collection")
    parser.add_argument('--output_dir', default='./output_categorical')
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