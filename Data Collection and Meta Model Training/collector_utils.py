"""
collector_utils.py — Shared Evaluation Engine
==============================================

Provides the core infrastructure for all three data collectors:
  - LightGBM model evaluation with repeated k-fold CV
  - Paired t-test with Bonferroni correction
  - Noise-floor calibration (Cohen's d, calibrated delta)
  - Data preparation (categorical encoding, NaN handling)
  - Dataset-level meta-feature extraction
  - Column pruning (ID detection, constant removal, leakage detection)

Classification only. Uses ROC-AUC as the evaluation metric.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis, shapiro, spearmanr, ttest_rel
import warnings
import csv
import os
import json
import time
import gc
import math
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')


# =============================================================================
# COLUMN NAME SANITISATION
# =============================================================================

import re

def sanitize_and_deduplicate_columns(df):
    """
    Remove JSON-unsafe and LightGBM-unsafe characters from column names,
    then resolve any duplicates introduced by sanitisation.
    Returns df with cleaned column names.
    """
    clean = [re.sub(r'[\[\]\{\}\":,\'\\/#@!$%^&*()\-+= ]', '_', str(c)) for c in df.columns]
    clean = [re.sub(r'_+', '_', c).strip('_') or 'col' for c in clean]
    seen = {}
    result = []
    for c in clean:
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    df.columns = result
    return df


# =============================================================================
# SPARSE COLUMN DENSIFICATION
# =============================================================================

def densify_sparse_columns(df):
    """Convert any SparseDtype columns to their dense equivalents."""
    for col in df.columns:
        if isinstance(df[col].dtype, pd.SparseDtype):
            df[col] = df[col].sparse.to_dense()
    return df


# =============================================================================
# RARE-CLASS FILTERING
# =============================================================================

def filter_rare_classes(X, y, n_folds, n_repeats):
    """Remove samples whose class has fewer than n_folds * n_repeats + 1 examples."""
    min_samples = n_folds * n_repeats + 1
    counts = y.value_counts()
    keep_classes = counts[counts >= min_samples].index
    mask = y.isin(keep_classes)
    if mask.sum() < len(y):
        n_dropped = len(y) - mask.sum()
        print(f"  filter_rare_classes: dropping {n_dropped} samples from rare-class instances")
    return X[mask], y[mask]


# =============================================================================
# IMBALANCE HANDLING
# =============================================================================

def compute_scale_pos_weight(y):
    """
    For binary classification, returns scale_pos_weight = n_neg / n_pos.
    Returns 1.0 for multiclass or balanced data.
    """
    counts = pd.Series(y).value_counts()
    if len(counts) != 2:
        return 1.0
    n_neg = counts.iloc[0]
    n_pos = counts.iloc[1]
    return float(n_neg / max(n_pos, 1))


# =============================================================================
# LightGBM DEFAULT PARAMETERS
# =============================================================================

BASE_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
}

INDIVIDUAL_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 15,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data_for_model(X_train, X_val):
    """Encode categoricals for LightGBM. Returns copies."""
    X_tr = X_train.copy()
    X_vl = X_val.copy()
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]):
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_vl[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_vl[col] = le.transform(X_vl[col].astype(str))
        X_tr[col] = X_tr[col].fillna(X_tr[col].median() if pd.api.types.is_numeric_dtype(X_tr[col]) else -999)
        X_vl[col] = X_vl[col].fillna(X_tr[col].median() if pd.api.types.is_numeric_dtype(X_tr[col]) else -999)
    return X_tr, X_vl


def ensure_numeric_target(y):
    """Convert target to numeric if needed."""
    if pd.api.types.is_numeric_dtype(y):
        return y
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)


# =============================================================================
# EVALUATION: REPEATED K-FOLD CV WITH LIGHTGBM
# =============================================================================

def evaluate_model(X, y, params=None, n_folds=5, n_repeats=3, return_importance=False):
    """
    Evaluate a LightGBM classifier via repeated k-fold CV.
    
    Returns:
        mean_score (float): Mean ROC-AUC across all folds
        std_score (float): Std of fold scores
        fold_scores (list): Individual fold AUC scores
        importances (np.array or None): Mean feature importances if requested
    """
    if params is None:
        params = BASE_PARAMS.copy()
    
    y_numeric = ensure_numeric_target(y)
    X, y_numeric = filter_rare_classes(X, y_numeric, n_folds, n_repeats)
    n_classes = y_numeric.nunique()

    params = params.copy()
    if n_classes == 2:
        params['scale_pos_weight'] = compute_scale_pos_weight(y_numeric)
    
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    fold_scores = []
    importance_acc = np.zeros(X.shape[1]) if return_importance else None
    n_folds_done = 0
    
    for train_idx, val_idx in rkf.split(X):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y_numeric.iloc[train_idx], y_numeric.iloc[val_idx]
        
        # Skip fold if validation has only one class
        if y_val.nunique() < 2:
            continue
        
        X_tr, X_vl = prepare_data_for_model(X_train, X_val)
        
        model = lgb.LGBMClassifier(**params)
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
        
        if return_importance:
            importance_acc += model.feature_importances_
        n_folds_done += 1
    
    if not fold_scores:
        return (None, None, None, None) if return_importance else (None, None, None)
    
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    
    if return_importance and n_folds_done > 0:
        importances = importance_acc / n_folds_done
        return mean_score, std_score, fold_scores, importances
    
    return mean_score, std_score, fold_scores


def evaluate_with_intervention(X, y, col_to_transform=None, col_second=None,
                                method=None, apply_fn=None, params=None,
                                n_folds=5, n_repeats=3, return_importance=False):
    """
    Evaluate model with an intervention applied inside CV folds (no leakage).
    
    Args:
        apply_fn: function(X_train, X_val, y_train, col, col_second, method) 
                  -> (success, X_train_modified, X_val_modified)
    """
    if params is None:
        params = BASE_PARAMS.copy()
    
    y_numeric = ensure_numeric_target(y)
    X, y_numeric = filter_rare_classes(X, y_numeric, n_folds, n_repeats)
    n_classes = y_numeric.nunique()

    params = params.copy()
    if n_classes == 2:
        params['scale_pos_weight'] = compute_scale_pos_weight(y_numeric)
    
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    fold_scores = []
    importance_acc = None
    n_folds_done = 0
    
    for train_idx, val_idx in rkf.split(X):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y_numeric.iloc[train_idx], y_numeric.iloc[val_idx]
        
        if y_val.nunique() < 2:
            continue
        
        # Apply intervention
        if apply_fn and method:
            success, X_train, X_val = apply_fn(
                X_train, X_val, y_train, col_to_transform, col_second, method
            )
            if not success:
                continue
        
        X_tr, X_vl = prepare_data_for_model(X_train, X_val)
        
        model = lgb.LGBMClassifier(**params)
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
        
        if return_importance:
            if importance_acc is None:
                importance_acc = np.zeros(len(X_tr.columns))
            # Handle shape changes from interventions that add/remove columns
            if len(model.feature_importances_) == len(importance_acc):
                importance_acc += model.feature_importances_
        n_folds_done += 1
    
    if not fold_scores:
        return (None, None, None, None) if return_importance else (None, None, None)
    
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    
    if return_importance and importance_acc is not None and n_folds_done > 0:
        return mean_score, std_score, fold_scores, importance_acc / n_folds_done
    
    return mean_score, std_score, fold_scores


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

def paired_ttest_bonferroni(baseline_folds, intervention_folds, n_tests):
    """
    Paired t-test between baseline and intervention fold scores.
    Returns: (t_stat, p_value, p_bonferroni, is_significant, is_significant_bonferroni)
    """
    if baseline_folds is None or intervention_folds is None:
        return np.nan, np.nan, np.nan, False, False
    
    bl = [s for s in baseline_folds if s is not None and not np.isnan(s)]
    iv = [s for s in intervention_folds if s is not None and not np.isnan(s)]
    
    min_len = min(len(bl), len(iv))
    if min_len < 3:
        return np.nan, np.nan, np.nan, False, False
    
    bl, iv = bl[:min_len], iv[:min_len]
    
    try:
        t_stat, p_value = ttest_rel(iv, bl)
        p_bonf = min(p_value * max(n_tests, 1), 1.0)
        return (float(t_stat), float(p_value), float(p_bonf),
                p_value < 0.05, p_bonf < 0.05)
    except Exception:
        return np.nan, np.nan, np.nan, False, False


def compute_cohens_d(baseline_folds, intervention_folds):
    """Cohen's d effect size from paired fold scores."""
    if baseline_folds is None or intervention_folds is None:
        return np.nan
    bl = [s for s in baseline_folds if s is not None and not np.isnan(s)]
    iv = [s for s in intervention_folds if s is not None and not np.isnan(s)]
    n = min(len(bl), len(iv))
    if n < 2:
        return np.nan
    diffs = [iv[i] - bl[i] for i in range(n)]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    if std_diff < 1e-10:
        return 0.0
    return float(mean_diff / std_diff)


def compute_calibrated_delta(delta, null_std):
    """Calibrated delta = delta / noise_floor_std."""
    if np.isnan(delta) or null_std < 1e-10:
        return np.nan
    return float(delta / null_std)


# =============================================================================
# NOISE FLOOR CALIBRATION
# =============================================================================

def compute_noise_floor(baseline_fold_scores):
    """
    Compute noise floor from baseline fold variance.
    Returns: (null_std, was_clamped)
    """
    if baseline_fold_scores is None or len(baseline_fold_scores) < 2:
        return 0.005, True
    
    valid = [s for s in baseline_fold_scores if s is not None and not np.isnan(s)]
    if len(valid) < 2:
        return 0.005, True
    
    raw_std = float(np.std(valid, ddof=1))
    clamped = raw_std < 0.001
    return max(raw_std, 0.001), clamped


# =============================================================================
# DATASET-LEVEL META-FEATURES (classification only)
# =============================================================================

def get_dataset_meta(X, y):
    """
    Compute dataset-level meta-features for classification tasks.
    """
    y_numeric = ensure_numeric_target(y)
    n_rows, n_cols = X.shape
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    
    meta = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_numeric_cols': len(numeric_cols),
        'n_cat_cols': len(cat_cols),
        'cat_ratio': len(cat_cols) / max(n_cols, 1),
        'missing_ratio': float(X.isnull().mean().mean()),
        'row_col_ratio': n_rows / max(n_cols, 1),
        'n_classes': int(y_numeric.nunique()),
    }
    
    # Class imbalance
    class_counts = y_numeric.value_counts()
    meta['class_imbalance_ratio'] = float(class_counts.max() / max(class_counts.min(), 1))
    
    # Correlation features (numeric only)
    if len(numeric_cols) >= 2:
        corr_matrix = X[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        meta['avg_feature_corr'] = float(corr_matrix.mean().mean())
        meta['max_feature_corr'] = float(corr_matrix.max().max())
        
        target_corrs = X[numeric_cols].corrwith(y_numeric).abs()
        meta['avg_target_corr'] = float(target_corrs.mean())
        meta['max_target_corr'] = float(target_corrs.max())
    else:
        meta['avg_feature_corr'] = 0.0
        meta['max_feature_corr'] = 0.0
        meta['avg_target_corr'] = 0.0
        meta['max_target_corr'] = 0.0
    
    # Landmarking: simple model performance
    try:
        X_enc = pd.DataFrame(index=X.index)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_enc[col] = X[col].fillna(X[col].median())
            else:
                le = LabelEncoder()
                X_enc[col] = le.fit_transform(X[col].astype(str).fillna('NaN'))
        
        from sklearn.model_selection import cross_val_score
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        scores = cross_val_score(dt, X_enc, y_numeric, cv=3, scoring='accuracy')
        meta['landmarking_score'] = float(scores.mean())
    except Exception:
        meta['landmarking_score'] = 0.5
    
    return meta


# =============================================================================
# COLUMN PRUNING AND CLEANUP
# =============================================================================

def is_likely_id_column(series, col_name):
    """Detect ID-like columns (monotonic, unique, named like 'id')."""
    name_lower = str(col_name).lower().strip()
    
    # Name-based detection
    id_patterns = ['id', 'index', 'row_num', 'record_id', 'pk', 'key']
    if name_lower in id_patterns or name_lower.endswith('_id'):
        return True
    
    # Monotonic integer check
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 10:
            if clean.is_monotonic_increasing or clean.is_monotonic_decreasing:
                if clean.nunique() == len(clean):
                    return True
    
    # Near-unique string column
    if not pd.api.types.is_numeric_dtype(series):
        if series.nunique() / max(len(series), 1) > 0.95:
            return True
    
    return False


def prune_columns(X, y):
    """
    Remove problematic columns:
    1. ID-like columns
    2. Constant / near-constant columns  
    3. Target leakage (correlation > 0.95 with target)
    
    Returns: (X_clean, dropped_log)
    """
    y_numeric = ensure_numeric_target(y)
    dropped = []
    keep = []
    
    for col in X.columns:
        # ID detection
        if is_likely_id_column(X[col], col):
            dropped.append((col, 'id_like'))
            continue
        
        # Constant
        if X[col].nunique(dropna=True) <= 1:
            dropped.append((col, 'constant'))
            continue
        
        # Near-constant (>99% same value)
        top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.99:
            dropped.append((col, f'near_constant_{top_freq:.3f}'))
            continue
        
        # Target leakage (numeric columns only)
        if pd.api.types.is_numeric_dtype(X[col]):
            try:
                clean = X[col].notna() & y_numeric.notna()
                if clean.sum() > 10:
                    corr = abs(X[col][clean].corr(y_numeric[clean]))
                    if corr > 0.95:
                        dropped.append((col, f'leakage_corr_{corr:.3f}'))
                        continue
            except Exception:
                pass
        
        keep.append(col)
    
    return X[keep], dropped


# =============================================================================
# FEATURE IMPORTANCE (BASELINE MODEL)
# =============================================================================

def get_baseline_importances(X, y, params=None):
    """
    Train a single LightGBM on full data to get feature importances.
    Returns: pd.Series of importances indexed by column name.
    """
    if params is None:
        params = BASE_PARAMS.copy()
    
    y_numeric = ensure_numeric_target(y)
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X_enc[col] = X[col].fillna(X[col].median())
        else:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X[col].astype(str).fillna('NaN'))
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_enc, y_numeric)
    return pd.Series(model.feature_importances_, index=X.columns)


# =============================================================================
# CSV WRITING UTILITIES
# =============================================================================

def write_csv(df_result, csv_file, schema):
    """Write results to CSV with strict schema enforcement."""
    for c in schema:
        if c not in df_result.columns:
            df_result[c] = np.nan
    df_out = df_result[schema]
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    df_out.to_csv(csv_file, mode='a', header=write_header, index=False)


def sanitize_string(val):
    """Clean string for CSV safety."""
    if val is None:
        return ''
    s = str(val)
    s = s.replace(',', '_').replace('"', '').replace("'", '').replace('\n', ' ')
    return s.strip()


# =============================================================================
# DELTA COMPUTATION
# =============================================================================

def compute_delta(base_score, new_score):
    """Raw delta: new - base."""
    if base_score is None or new_score is None:
        return np.nan
    return float(new_score - base_score)


def normalize_delta(delta, base_score):
    """Normalize delta as percentage of headroom."""
    if np.isnan(delta) or base_score is None:
        return np.nan
    headroom = max(1.0 - base_score, 0.001)
    return float(delta / headroom * 100)


# =============================================================================
# CEILING CHECK
# =============================================================================

def check_ceiling(baseline_score, threshold=0.99):
    """Check if baseline is too high for meaningful improvement."""
    if baseline_score is None:
        return True, False, "baseline_failed"   # is_ceiling=True, near_ceiling=False
    if baseline_score >= threshold:
        return True, True, f"ceiling_{baseline_score:.4f}"
    return False, baseline_score >= 0.95, ""