"""
Auto Feature Engineering - Streamlit Application (auto_fe_app.py)

Demo-ready interface:
1. Upload training dataset -> get FE/PP recommendations (with checkboxes)
2. Runs BASELINE (raw data) and AUTO-FE (enhanced) side-by-side
3. Upload test dataset -> evaluates BOTH models -> comparison table

Requirements:
    pip install streamlit pandas numpy lightgbm scikit-learn scipy

Usage:
    streamlit run auto_fe_app.py -- --model-dir ./meta_model
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
import math
import warnings
import tempfile
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, mean_squared_error, accuracy_score,
                             f1_score, mean_absolute_error, r2_score,
                             precision_score, recall_score, log_loss)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import skew, kurtosis, spearmanr
import lightgbm as lgb
import re

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    pass


# =============================================================================
# TEMPORAL COMPONENT DETECTION
# =============================================================================

TEMPORAL_PATTERNS = [
    (r'(?i)(?:^|[_\W])(?:month|mon(?:th)?|mo)(?:$|[_\W\d])',
     lambda mn, mx: 1 <= mn and mx <= 12, 'month', 12),
    (r'(?i)(?:^|[_\W])(?:dayofweek|day_of_week|dow|weekday|wday|day_of_wk)(?:$|[_\W\d])',
     lambda mn, mx: (0 <= mn and mx <= 6) or (1 <= mn and mx <= 7), 'day_of_week', 7),
    (r'(?i)(?:^|[_\W])(?:hour|hr|hh)(?:$|[_\W\d])',
     lambda mn, mx: 0 <= mn and mx <= 23, 'hour', 24),
    (r'(?i)(?:^|[_\W])(?:minute)(?:$|[_\W\d])',
     lambda mn, mx: 0 <= mn and mx <= 59, 'minute', 60),
    (r'(?i)(?:^|[_\W])(?:second)(?:$|[_\W\d])',
     lambda mn, mx: 0 <= mn and mx <= 59, 'second', 60),
    (r'(?i)(?:^|[_\W])(?:day(?:ofmonth|_of_month)?)(?:$|[_\W\d])',
     lambda mn, mx: 1 <= mn and mx <= 31, 'day_of_month', 31),
    (r'(?i)(?:^|[_\W])(?:quarter|qtr|q)(?:$|[_\W\d])',
     lambda mn, mx: 1 <= mn and mx <= 4, 'quarter', 4),
    (r'(?i)(?:^|[_\W])(?:week(?:ofyear|_of_year)?|wk|isoweek)(?:$|[_\W\d])',
     lambda mn, mx: 1 <= mn and mx <= 53, 'week', 53),
]


def detect_temporal_component(series, col_name):
    # If not already numeric, try coercing (handles "1","2","3" strings)
    if not pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(series, errors='coerce')
        # Accept only if most values converted successfully (>90%)
        if coerced.notna().sum() < 0.9 * series.notna().sum():
            return None, None
        series = coerced
    clean = series.dropna()
    if len(clean) < 10:
        return None, None
    try:
        if not (clean % 1 == 0).all():
            return None, None
    except:
        return None, None
    mn, mx = float(clean.min()), float(clean.max())
    for pattern, range_check, temporal_type, period in TEMPORAL_PATTERNS:
        if re.search(pattern, col_name):
            if range_check(mn, mx):
                return temporal_type, period
    return None, None


# =============================================================================
# TRANSFORM STACKING RESOLUTION
# =============================================================================

TRANSFORM_CATEGORIES = {
    'imputation':       (0, False),
    'encoding':         (1, True),
    'transformation':   (2, True),
    'feature_creation': (3, False),
}

METHOD_TO_CATEGORY = {
    'impute_median': 'imputation', 'missing_indicator': 'imputation',
    'target_encoding': 'encoding', 'frequency_encoding': 'encoding',
    'onehot_encoding': 'encoding', 'hashing_encoding': 'encoding',
    'log_transform': 'transformation', 'sqrt_transform': 'transformation',
    'quantile_binning': 'transformation',
    'polynomial_square': 'feature_creation',
    'product_interaction': 'feature_creation', 'division_interaction': 'feature_creation',
    'addition_interaction': 'feature_creation', 'subtraction_interaction': 'feature_creation',
    'abs_diff_interaction': 'feature_creation',
    'three_way_interaction': 'feature_creation', 'three_way_addition': 'feature_creation',
    'three_way_ratio': 'feature_creation',
    'group_mean': 'feature_creation', 'group_std': 'feature_creation',
    'cat_concat': 'feature_creation', 'text_stats': 'feature_creation',
    'row_stats': 'feature_creation',
    'date_extract_basic': 'encoding', 'date_cyclical_month': 'encoding',
    'date_cyclical_dow': 'encoding', 'date_elapsed_days': 'encoding',
    'cyclical_encode': 'feature_creation',
}


def resolve_transform_stack(recommendations):
    single_col_recs = [r for r in recommendations if not r.get('is_interaction', False)]
    interaction_recs = [r for r in recommendations if r.get('is_interaction', False)]
    col_groups = {}
    for rec in single_col_recs:
        col = rec['column']
        if col not in col_groups: col_groups[col] = []
        col_groups[col].append(rec)
    resolved = []
    for col, recs in col_groups.items():
        by_category = {}
        for rec in recs:
            cat = METHOD_TO_CATEGORY.get(rec['method'], 'feature_creation')
            if cat not in by_category: by_category[cat] = []
            by_category[cat].append(rec)
        col_stack = []
        for cat_name, (priority, exclusive) in sorted(TRANSFORM_CATEGORIES.items(), key=lambda x: x[1][0]):
            if cat_name not in by_category: continue
            cat_recs = by_category[cat_name]
            if exclusive and len(cat_recs) > 1:
                col_stack.append(max(cat_recs, key=lambda r: r.get('predicted_score', 0)))
            else:
                col_stack.extend(cat_recs)
        col_stack.sort(key=lambda r: TRANSFORM_CATEGORIES.get(
            METHOD_TO_CATEGORY.get(r['method'], 'feature_creation'), (99, False))[0])
        resolved.extend(col_stack)
    resolved.extend(interaction_recs)
    return resolved


# =============================================================================
# FORMULA-BASED LIGHTGBM HYPERPARAMETER SELECTION
# =============================================================================

def compute_lgbm_params(n_rows, n_cols, task_type='classification',
                        n_classes=2, missing_ratio=0.0, cat_ratio=0.0,
                        avg_feature_corr=0.0, class_imbalance_ratio=1.0):
    """
    Compute LightGBM hyperparameters from dataset characteristics.
    
    Philosophy: start from battle-tested defaults (leaves=31, lr=0.05, depth=6)
    and make small, justified adjustments.  LightGBM + early stopping is already
    robust — aggressive formula-based tuning tends to hurt more than help.
    
    Adjustments are only made when dataset characteristics clearly warrant it.
    n_estimators is set high and early stopping handles the rest.
    
    Returns: (params_dict, explanation_dict)
    """
    explanation = {}
    
    # ── num_leaves: primary capacity knob ──
    # Default 31 is good for most datasets.  Only increase for genuinely large
    # datasets where there's enough data to support complex trees.
    if n_rows < 1500:
        num_leaves = 20
        explanation['num_leaves'] = f"{num_leaves} (small dataset, prevent overfitting)"
    elif n_rows < 5000:
        num_leaves = 31
        explanation['num_leaves'] = f"{num_leaves} (default, sufficient data)"
    elif n_rows < 20000:
        num_leaves = 31
        # Slight increase only if we have many features to learn from
        if n_cols > 30:
            num_leaves = 40
        explanation['num_leaves'] = f"{num_leaves} (medium dataset{', many features' if n_cols > 30 else ''})"
    elif n_rows < 100000:
        num_leaves = 50
        explanation['num_leaves'] = f"{num_leaves} (large dataset, more capacity)"
    else:
        num_leaves = 63
        explanation['num_leaves'] = f"{num_leaves} (very large dataset)"
    
    # ── max_depth: always capped, never unlimited ──
    if num_leaves <= 25:
        max_depth = 5
    elif num_leaves <= 40:
        max_depth = 6
    elif num_leaves <= 63:
        max_depth = 7
    else:
        max_depth = 8
    explanation['max_depth'] = f"{max_depth} (capped for {num_leaves} leaves)"
    
    # ── n_estimators: set high, let early stopping decide ──
    n_estimators = 1500
    explanation['n_estimators'] = f"{n_estimators} (early stopping at patience=50 picks actual count)"
    
    # ── learning_rate: 0.05 is the sweet spot for most scenarios ──
    learning_rate = 0.05
    if n_rows < 1500:
        learning_rate = 0.08  # Small data: slightly faster convergence, fewer trees
    explanation['learning_rate'] = f"{learning_rate}"
    
    # ── min_child_samples: prevent overfitting on small groups ──
    if n_rows < 1500:
        min_child_samples = 20
    elif n_rows < 5000:
        min_child_samples = 10
    elif n_rows < 50000:
        min_child_samples = 15
    else:
        min_child_samples = 20
    explanation['min_child_samples'] = f"{min_child_samples} ({min_child_samples/n_rows*100:.1f}% of rows)"
    
    # ── subsample: minor adjustment for large datasets ──
    if n_rows > 50000:
        subsample = 0.7
    else:
        subsample = 0.8
    explanation['subsample'] = f"{subsample}"
    
    # ── colsample_bytree: reduce for wide datasets ──
    if n_cols > 200:
        colsample = 0.5
    elif n_cols > 50:
        colsample = 0.6
    else:
        colsample = 0.8
    explanation['colsample_bytree'] = f"{colsample} ({n_cols} features)"
    
    # ── regularization: keep close to defaults, slight bump for noisy data ──
    reg_alpha = 0.05
    reg_lambda = 0.5
    if missing_ratio > 0.15 or cat_ratio > 0.5:
        reg_alpha = 0.1
        reg_lambda = 1.0
        reg_reason = f"elevated (missing={missing_ratio:.0%}, cat={cat_ratio:.0%})"
    elif n_rows < 2000 and n_cols > 20:
        reg_alpha = 0.1
        reg_lambda = 1.5
        reg_reason = "elevated (small data, many features)"
    else:
        reg_reason = "default"
    explanation['reg_alpha'] = f"{reg_alpha} — L1, {reg_reason}"
    explanation['reg_lambda'] = f"{reg_lambda} — L2, {reg_reason}"
    
    params = {
        'n_estimators': n_estimators, 'learning_rate': learning_rate,
        'num_leaves': num_leaves, 'max_depth': max_depth,
        'min_child_samples': min_child_samples, 'subsample': subsample,
        'colsample_bytree': colsample, 'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
    }
    
    # Task-specific
    if task_type == 'classification':
        if n_classes > 2:
            params['objective'] = 'multiclass'
            params['num_class'] = int(n_classes)
            params['metric'] = 'multi_logloss'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'auc'
        if class_imbalance_ratio < 0.3:
            params['is_unbalance'] = True
            explanation['is_unbalance'] = f"True (minority ratio: {class_imbalance_ratio:.2f})"
    else:
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
    
    return params, explanation


# =============================================================================
# FEATURE COMPUTER (mirrors MetaDataCollector metadata)
# =============================================================================

class FeatureComputer:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
    
    def compute_dataset_meta(self, X, y):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        y_num = self._ensure_numeric(y)
        meta = {
            'n_rows': X.shape[0], 'n_cols': X.shape[1],
            'cat_ratio': len(X.select_dtypes(include=['object', 'category', 'bool']).columns) / max(X.shape[1], 1),
            'missing_ratio': X.isnull().sum().sum() / max(X.shape[0] * X.shape[1], 1),
            'row_col_ratio': X.shape[0] / max(X.shape[1], 1),
        }
        if self.task_type == 'classification':
            vc = y.value_counts()
            meta['class_imbalance_ratio'] = vc.min() / vc.max() if len(vc) > 1 else 1.0
            meta['n_classes'] = len(vc)
            meta['target_std'] = np.nan
            meta['target_skew'] = np.nan
        else:
            meta['class_imbalance_ratio'] = -1.0
            meta['n_classes'] = -1
            meta['target_std'] = float(y_num.std())
            meta['target_skew'] = float(skew(y_num.dropna()))
        meta['n_cols_before_selection'] = X.shape[1]
        meta['n_cols_selected'] = X.shape[1]
        meta['landmarking_score'] = self._landmarking(X, y)
        meta['landmarking_score_norm'] = (meta['landmarking_score'] - 0.5) * 2 if self.task_type == 'classification' else meta['landmarking_score']
        if len(numeric_cols) > 1:
            sample_cols = numeric_cols[:200]
            corr = X[sample_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            meta['avg_feature_corr'] = float(upper.stack().mean())
            meta['max_feature_corr'] = float(upper.stack().max())
        else:
            meta['avg_feature_corr'] = 0.0
            meta['max_feature_corr'] = 0.0
        if len(numeric_cols) > 0:
            t_corrs = [abs(X[c].corr(y_num)) for c in numeric_cols if not X[c].isnull().all()]
            t_corrs = [c for c in t_corrs if not np.isnan(c)]
            meta['avg_target_corr'] = float(np.mean(t_corrs)) if t_corrs else 0.0
            meta['max_target_corr'] = float(np.max(t_corrs)) if t_corrs else 0.0
        else:
            meta['avg_target_corr'] = 0.0
            meta['max_target_corr'] = 0.0
        if len(numeric_cols) > 0:
            meta['avg_numeric_sparsity'] = float(np.mean([(X[c] == 0).mean() for c in numeric_cols]))
        else:
            meta['avg_numeric_sparsity'] = 0.0
        meta['linearity_gap'] = 0.0
        meta['corr_graph_components'] = 0
        meta['corr_graph_clustering'] = 0.0
        meta['corr_graph_density'] = 0.0
        meta['matrix_rank_ratio'] = 0.0
        meta['baseline_score'] = 0.0
        meta['baseline_std'] = 0.0
        return meta
    
    def compute_column_meta(self, col_series, y):
        is_num = pd.api.types.is_numeric_dtype(col_series)
        clean = col_series.dropna()
        if is_num and len(clean) > 0:
            clean = clean.astype(float)
        y_num = self._ensure_numeric(y)
        outlier_ratio = 0.0
        iqr_val = 0.0
        if is_num and len(clean) > 0:
            q1, q3 = np.percentile(clean, [25, 75])
            iqr_val = q3 - q1
            if iqr_val > 0:
                outliers = clean[(clean < q1 - 1.5 * iqr_val) | (clean > q3 + 1.5 * iqr_val)]
                outlier_ratio = len(outliers) / len(clean)
        meta = {
            'null_pct': float(col_series.isnull().mean()),
            'unique_ratio': col_series.nunique() / max(len(col_series), 1),
            'is_numeric': int(is_num),
            'outlier_ratio': outlier_ratio,
            'entropy': float(self._entropy(clean)) if not is_num else 0.0,
            'baseline_feature_importance': 0.0,
            'is_temporal_component': 0,
            'temporal_period': 0,
        }
        if is_num:
            meta.update({
                'skewness': float(skew(clean)) if len(clean) > 2 else 0.0,
                'kurtosis': float(kurtosis(clean)) if len(clean) > 2 else 0.0,
                'coeff_variation': float(clean.std() / clean.mean()) if len(clean) > 0 and clean.mean() != 0 else 0.0,
                'zeros_ratio': float((clean == 0).mean()) if len(clean) > 0 else 0.0,
                'shapiro_p_value': np.nan, 'has_multiple_modes': 0,
                'bimodality_proxy_heuristic': 0,
                'range_iqr_ratio': float((clean.max() - clean.min()) / iqr_val) if iqr_val > 0 else 0.0,
                'dominant_quartile_pct': 0.25,
                'pct_in_0_1_range': float(((clean >= 0) & (clean <= 1)).mean()) if len(clean) > 0 else 0.0,
                'hartigan_dip_pval': 1.0, 'is_multimodal': 0,
                'top_category_dominance': np.nan, 'normalized_entropy': np.nan,
                'is_binary': 0, 'is_low_cardinality': 0, 'is_high_cardinality': 0,
                'top3_category_concentration': np.nan, 'rare_category_pct': np.nan,
                'conditional_entropy': np.nan,
            })
            try:
                meta['spearman_corr_target'] = float(spearmanr(clean, y_num.loc[clean.index]).correlation)
            except:
                meta['spearman_corr_target'] = np.nan
        else:
            nunique = col_series.nunique()
            meta.update({
                'skewness': np.nan, 'kurtosis': np.nan, 'coeff_variation': np.nan,
                'zeros_ratio': np.nan, 'shapiro_p_value': np.nan,
                'has_multiple_modes': 0, 'bimodality_proxy_heuristic': 0,
                'range_iqr_ratio': np.nan, 'dominant_quartile_pct': np.nan,
                'pct_in_0_1_range': np.nan, 'spearman_corr_target': np.nan,
                'hartigan_dip_pval': np.nan, 'is_multimodal': 0,
                'top_category_dominance': float(clean.value_counts(normalize=True).iloc[0]) if len(clean) > 0 else 0.0,
                'normalized_entropy': float(meta['entropy'] / np.log2(nunique)) if nunique > 1 else 0.0,
                'is_binary': int(nunique == 2),
                'is_low_cardinality': int(2 < nunique < 10),
                'is_high_cardinality': int(nunique >= 50 and nunique < len(col_series) * 0.5),
                'top3_category_concentration': float(clean.value_counts(normalize=True).head(3).sum()) if len(clean) > 0 else 0.0,
                'rare_category_pct': 0.0, 'conditional_entropy': 0.0,
            })
        meta['pps_score'] = self._pps(col_series, y_num)
        try:
            if is_num:
                mi_data = col_series.fillna(col_series.median()).to_frame()
            else:
                le = LabelEncoder()
                mi_data = pd.DataFrame(le.fit_transform(col_series.astype(str).fillna('NaN')),
                                      index=col_series.index, columns=['col'])
            if self.task_type == 'classification':
                meta['mutual_information_score'] = float(mutual_info_classif(mi_data, y_num, random_state=42)[0])
            else:
                meta['mutual_information_score'] = float(mutual_info_regression(mi_data, y_num, random_state=42)[0])
        except:
            meta['mutual_information_score'] = 0.0
        return meta
    
    def _ensure_numeric(self, y):
        if pd.api.types.is_numeric_dtype(y): return y
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    
    def _entropy(self, series):
        if len(series) == 0: return 0.0
        p = series.value_counts(normalize=True)
        return float(-(p * np.log2(p)).sum())
    
    def _pps(self, feature, target):
        try:
            if pd.api.types.is_numeric_dtype(feature):
                X = feature.fillna(feature.median()).values.reshape(-1, 1)
            else:
                le = LabelEncoder()
                X = le.fit_transform(feature.astype(str).fillna('NaN')).reshape(-1, 1)
            valid = feature.notna()
            X, y_v = X[valid], target[valid]
            if len(X) < 10: return 0.0
            if self.task_type == 'classification':
                m = DecisionTreeClassifier(max_depth=3, random_state=42); m.fit(X, y_v)
                proba = m.predict_proba(X)
                return float((roc_auc_score(y_v, proba[:, 1]) - 0.5) * 2) if proba.shape[1] == 2 else 0.0
            else:
                m = DecisionTreeRegressor(max_depth=3, random_state=42); m.fit(X, y_v)
                mse = mean_squared_error(y_v, m.predict(X))
                base = np.var(y_v)
                return float(max(0, 1 - mse / base)) if base > 0 else 0.0
        except:
            return 0.0
    
    def _landmarking(self, X, y):
        X_p = X.copy()
        for col in X_p.columns:
            if pd.api.types.is_numeric_dtype(X_p[col]):
                X_p[col] = X_p[col].fillna(X_p[col].median())
            else:
                X_p[col] = X_p[col].astype('category').cat.codes
        try:
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for tr, vl in kf.split(X_p):
                if self.task_type == 'classification':
                    m = DecisionTreeClassifier(max_depth=1, random_state=42)
                    m.fit(X_p.iloc[tr], y.iloc[tr])
                    try: scores.append(roc_auc_score(y.iloc[vl], m.predict_proba(X_p.iloc[vl])[:, 1]))
                    except: scores.append(0.5)
                else:
                    m = DecisionTreeRegressor(max_depth=1, random_state=42)
                    m.fit(X_p.iloc[tr], y.iloc[tr])
                    scores.append(mean_squared_error(y.iloc[vl], m.predict(X_p.iloc[vl])))
            return float(np.mean(scores))
        except:
            return 0.5 if self.task_type == 'classification' else 0.0


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

class RecommendationEngine:
    NUMERIC_METHODS = ['log_transform', 'sqrt_transform', 'quantile_binning',
                       'polynomial_square', 'impute_median', 'missing_indicator']
    CATEGORICAL_METHODS = ['frequency_encoding', 'target_encoding', 'onehot_encoding',
                          'hashing_encoding', 'missing_indicator']
    TEMPORAL_METHODS = ['cyclical_encode', 'impute_median', 'missing_indicator']
    
    def __init__(self, preparator, trainer):
        self.preparator = preparator
        self.trainer = trainer
    
    def recommend(self, X, y, task_type='classification', top_k=30, score_threshold=52.0):
        fc = FeatureComputer(task_type=task_type)
        ds_meta = fc.compute_dataset_meta(X, y)
        baseline = self._quick_baseline(X, y, task_type, fc)
        ds_meta['baseline_score'] = baseline['score']
        ds_meta['baseline_std'] = baseline['std']
        
        if task_type in self.preparator.task_type_encoder.classes_:
            task_encoded = int(self.preparator.task_type_encoder.transform([task_type])[0])
        else:
            task_encoded = 0
        
        y_num = fc._ensure_numeric(y)
        mi_scores = {}
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    d = X[col].fillna(X[col].median()).to_frame()
                else:
                    le = LabelEncoder()
                    d = pd.DataFrame(le.fit_transform(X[col].astype(str).fillna('NaN')),
                                    index=X[col].index, columns=[col])
                if task_type == 'classification':
                    mi_scores[col] = float(mutual_info_classif(d, y_num, random_state=42)[0])
                else:
                    mi_scores[col] = float(mutual_info_regression(d, y_num, random_state=42)[0])
            except:
                mi_scores[col] = 0.0
        
        candidates = []
        
        # Single-column candidates
        for col in X.columns:
            col_meta = fc.compute_column_meta(X[col], y)
            is_num = pd.api.types.is_numeric_dtype(X[col])
            temporal_type, temporal_period = detect_temporal_component(X[col], col)
            col_meta['is_temporal_component'] = int(temporal_type is not None)
            col_meta['temporal_period'] = temporal_period if temporal_period is not None else 0
            
            if temporal_type is not None:
                methods = self.TEMPORAL_METHODS.copy()
            elif is_num:
                methods = self.NUMERIC_METHODS.copy()
            else:
                nunique = X[col].nunique()
                methods = self.CATEGORICAL_METHODS.copy()
                if nunique <= 10:
                    methods = [m for m in methods if m != 'hashing_encoding']
                if not (2 < nunique <= 10):
                    methods = [m for m in methods if m != 'onehot_encoding']
                # Text-like columns: high cardinality with long average values
                if nunique > 20 and X[col].astype(str).str.len().mean() > 5:
                    methods.append('text_stats')
            
            if col_meta['null_pct'] == 0:
                methods = [m for m in methods if m not in ['impute_median', 'missing_indicator']]
            
            for method in methods:
                if method not in self.preparator.method_encoder.classes_:
                    continue
                method_encoded = int(self.preparator.method_encoder.transform([method])[0])
                features = {}
                features.update(ds_meta)
                features.update(col_meta)
                features['method_encoded'] = method_encoded
                features['task_type_encoded'] = task_encoded
                features['is_interaction'] = 0
                features['near_ceiling_flag'] = 0
                candidates.append({
                    'column': col, 'method': method,
                    'is_interaction': False, 'features': features,
                })
        
        # Interaction candidates (exclude temporal from arithmetic)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        arithmetic_nums = [c for c in numeric_cols if detect_temporal_component(X[c], c)[0] is None]
        top_nums = sorted(arithmetic_nums, key=lambda c: mi_scores.get(c, 0), reverse=True)[:5]
        top_cats = sorted(cat_cols, key=lambda c: mi_scores.get(c, 0), reverse=True)[:5]
        
        for i, col_a in enumerate(top_nums):
            for col_b in top_nums[i+1:]:
                pair_metrics = self._compute_pair_metrics(X, col_a, col_b)
                for method in ['product_interaction', 'division_interaction',
                               'addition_interaction', 'subtraction_interaction', 'abs_diff_interaction']:
                    if method in ('addition_interaction', 'subtraction_interaction', 'abs_diff_interaction'):
                        if pair_metrics.get('interaction_scale_ratio', 0) > 10:
                            continue
                    if method in self.preparator.method_encoder.classes_:
                        col_meta_a = fc.compute_column_meta(X[col_a], y)
                        features = {}
                        features.update(ds_meta)
                        features.update(col_meta_a)
                        features['method_encoded'] = int(self.preparator.method_encoder.transform([method])[0])
                        features['task_type_encoded'] = task_encoded
                        features['is_interaction'] = 1
                        features['near_ceiling_flag'] = 0
                        features.update(pair_metrics)
                        name_map = {
                            'product_interaction': f"{col_a}_x_{col_b}",
                            'division_interaction': f"{col_a}_div_{col_b}",
                            'addition_interaction': f"{col_a}_plus_{col_b}",
                            'subtraction_interaction': f"{col_a}_minus_{col_b}",
                            'abs_diff_interaction': f"{col_a}_absdiff_{col_b}",
                        }
                        candidates.append({
                            'column': name_map[method], 'method': method,
                            'col_a': col_a, 'col_b': col_b,
                            'is_interaction': True, 'features': features,
                        })
        
        # 3-way interactions
        for i, col_a in enumerate(top_nums[:3]):
            for j, col_b in enumerate(top_nums[i+1:4], start=i+1):
                for col_c in top_nums[j+1:4]:
                    pair_metrics_ab = self._compute_pair_metrics(X, col_a, col_b)
                    for method in ['three_way_interaction', 'three_way_addition', 'three_way_ratio']:
                        if method == 'three_way_addition':
                            pair_ac = self._compute_pair_metrics(X, col_a, col_c)
                            pair_bc = self._compute_pair_metrics(X, col_b, col_c)
                            if any(m.get('interaction_scale_ratio', 0) > 10
                                   for m in [pair_metrics_ab, pair_ac, pair_bc]):
                                continue
                        if method in self.preparator.method_encoder.classes_:
                            col_meta_a = fc.compute_column_meta(X[col_a], y)
                            features = {}
                            features.update(ds_meta)
                            features.update(col_meta_a)
                            features['method_encoded'] = int(self.preparator.method_encoder.transform([method])[0])
                            features['task_type_encoded'] = task_encoded
                            features['is_interaction'] = 1
                            features['near_ceiling_flag'] = 0
                            features.update(pair_metrics_ab)
                            name_map = {
                                'three_way_interaction': f"{col_a}_x_{col_b}_x_{col_c}",
                                'three_way_addition': f"{col_a}_plus_{col_b}_plus_{col_c}",
                                'three_way_ratio': f"{col_a}_x_{col_b}_div_{col_c}",
                            }
                            candidates.append({
                                'column': name_map[method], 'method': method,
                                'col_a': col_a, 'col_b': col_b, 'col_c': col_c,
                                'is_interaction': True, 'features': features,
                            })
        
        # Group-by interactions
        for cat in top_cats[:3]:
            for num in top_nums[:3]:
                if 'group_mean' in self.preparator.method_encoder.classes_:
                    col_meta_cat = fc.compute_column_meta(X[cat], y)
                    pair_metrics = self._compute_pair_metrics(X, cat, num)
                    features = {}
                    features.update(ds_meta)
                    features.update(col_meta_cat)
                    features['method_encoded'] = int(self.preparator.method_encoder.transform(['group_mean'])[0])
                    features['task_type_encoded'] = task_encoded
                    features['is_interaction'] = 1
                    features['near_ceiling_flag'] = 0
                    features.update(pair_metrics)
                    candidates.append({
                        'column': f"group_mean_{num}_by_{cat}",
                        'method': 'group_mean', 'col_a': cat, 'col_b': num,
                        'is_interaction': True, 'features': features,
                    })
        
        # Row-statistics candidate (global, not tied to a single column)
        if 'row_stats' in self.preparator.method_encoder.classes_:
            row_stats_features = {}
            row_stats_features.update(ds_meta)
            # Column-level features are meaningless for row_stats — fill with zeros/NaN
            # (the preparator's fill_values will median-impute these, same as during training)
            for col_feat in self.preparator.feature_columns:
                if col_feat not in row_stats_features:
                    row_stats_features[col_feat] = np.nan
            row_stats_features['method_encoded'] = int(self.preparator.method_encoder.transform(['row_stats'])[0])
            row_stats_features['task_type_encoded'] = task_encoded
            row_stats_features['is_interaction'] = 0
            row_stats_features['near_ceiling_flag'] = 0
            candidates.append({
                'column': 'GLOBAL_ROW_STATS', 'method': 'row_stats',
                'is_interaction': False, 'features': row_stats_features,
            })
        
        if not candidates:
            return [], baseline, ds_meta
        
        # Predict
        feature_rows = []
        for c in candidates:
            row = {fc_name: c['features'].get(fc_name, np.nan) for fc_name in self.preparator.feature_columns}
            feature_rows.append(row)
        X_cand = pd.DataFrame(feature_rows, columns=self.preparator.feature_columns)
        for col in X_cand.columns:
            X_cand[col] = pd.to_numeric(X_cand[col], errors='coerce')
        X_cand = X_cand.replace([np.inf, -np.inf], np.nan).fillna(self.preparator.fill_values).fillna(0)
        scores = self.trainer.predict(X_cand)
        for i, c in enumerate(candidates):
            c['predicted_score'] = float(scores[i])
        
        filtered = [c for c in candidates if c['predicted_score'] >= score_threshold]
        filtered.sort(key=lambda x: x['predicted_score'], reverse=True)
        resolved = resolve_transform_stack(filtered[:top_k * 2])
        return resolved[:top_k], baseline, ds_meta
    
    def _quick_baseline(self, X, y, task_type, fc):
        X_p = X.copy()
        for col in X_p.columns:
            if pd.api.types.is_numeric_dtype(X_p[col]):
                X_p[col] = X_p[col].fillna(X_p[col].median())
            else:
                X_p[col] = X_p[col].astype('category').cat.codes
        y_num = fc._ensure_numeric(y)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, vl in kf.split(X_p):
            if task_type == 'classification':
                m = lgb.LGBMClassifier(n_estimators=100, verbosity=-1, random_state=42)
                m.fit(X_p.iloc[tr], y_num.iloc[tr])
                probs = m.predict_proba(X_p.iloc[vl])
                if probs.shape[1] == 2:
                    scores.append(roc_auc_score(y_num.iloc[vl], probs[:, 1]))
                else:
                    scores.append(roc_auc_score(y_num.iloc[vl], probs, multi_class='ovr'))
            else:
                m = lgb.LGBMRegressor(n_estimators=100, verbosity=-1, random_state=42)
                m.fit(X_p.iloc[tr], y_num.iloc[tr])
                scores.append(mean_squared_error(y_num.iloc[vl], m.predict(X_p.iloc[vl])))
        return {'score': float(np.mean(scores)), 'std': float(np.std(scores))}
    
    def _compute_pair_metrics(self, X, col_a, col_b):
        metrics = {'pairwise_corr_ab': np.nan, 'pairwise_spearman_ab': np.nan,
                    'pairwise_mi_ab': np.nan, 'interaction_scale_ratio': np.nan}
        a_is_num = pd.api.types.is_numeric_dtype(X[col_a])
        b_is_num = pd.api.types.is_numeric_dtype(X[col_b])
        try:
            if a_is_num and b_is_num:
                clean_idx = X[col_a].notna() & X[col_b].notna()
                a_clean, b_clean = X[col_a][clean_idx], X[col_b][clean_idx]
                if len(a_clean) > 10:
                    metrics['pairwise_corr_ab'] = abs(a_clean.corr(b_clean))
                    sp_corr, _ = spearmanr(a_clean, b_clean)
                    metrics['pairwise_spearman_ab'] = abs(sp_corr) if not np.isnan(sp_corr) else np.nan
                    std_a, std_b = a_clean.std(), b_clean.std()
                    if min(std_a, std_b) > 1e-10:
                        metrics['interaction_scale_ratio'] = max(std_a, std_b) / min(std_a, std_b)
            if a_is_num:
                enc_a = X[col_a].fillna(X[col_a].median()).values.reshape(-1, 1)
            else:
                le_a = LabelEncoder()
                enc_a = le_a.fit_transform(X[col_a].astype(str).fillna('NaN')).reshape(-1, 1)
            if b_is_num:
                target_b = X[col_b].fillna(X[col_b].median())
            else:
                le_b = LabelEncoder()
                target_b = pd.Series(le_b.fit_transform(X[col_b].astype(str).fillna('NaN')), index=X[col_b].index)
            mi_val = mutual_info_regression(enc_a, target_b, random_state=42)[0]
            metrics['pairwise_mi_ab'] = float(mi_val)
        except:
            pass
        return metrics


# =============================================================================
# TRANSFORM APPLICATION (leak-safe)
# =============================================================================

class TransformApplicator:
    def __init__(self):
        self.fitted_transforms = []
    
    def fit_transform(self, X, y_train, recommendations):
        X = X.copy()
        self.fitted_transforms = []
        y_num = pd.to_numeric(y_train, errors='coerce') if y_train is not None else None
        for rec in recommendations:
            method = rec['method']
            # For interactions, 'column' is the output name — use col_a as source
            if rec.get('is_interaction'):
                col = rec.get('col_a')
            else:
                col = rec.get('column')
            col_b = rec.get('col_b')
            col_c = rec.get('col_c')
            try:
                spec = self._fit_single(X, y_num, method, col, col_b, col_c)
                if spec:
                    self._apply_single(X, spec)
                    self.fitted_transforms.append(spec)
            except:
                pass
        return X
    
    def transform(self, X):
        X = X.copy()
        for spec in self.fitted_transforms:
            try:
                self._apply_single(X, spec)
            except:
                pass
        return X
    
    def _fit_single(self, X, y_num, method, col, col_b=None, col_c=None):
        if method != 'row_stats':
            if col and col not in X.columns: return None
        if col_b and col_b not in X.columns: return None
        if col_c and col_c not in X.columns: return None
        spec = {'method': method, 'column': col, 'col_b': col_b, 'col_c': col_c}
        
        if method == 'log_transform':
            fill_val = float(X[col].min()) if not X[col].isnull().all() else 0.0
            spec['fill_val'] = fill_val
            spec['offset'] = float(abs(X[col].fillna(fill_val).min()) + 1) if X[col].fillna(fill_val).min() <= 0 else 0.0
        elif method == 'sqrt_transform':
            fill_val = float(X[col].min()) if not X[col].isnull().all() else 0.0
            spec['fill_val'] = fill_val
            spec['offset'] = float(abs(X[col].fillna(fill_val).min())) if X[col].fillna(fill_val).min() < 0 else 0.0
        elif method == 'impute_median':
            # Coerce string-encoded temporal columns to numeric
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
            spec['median_val'] = float(X[col].median())
        elif method == 'missing_indicator':
            spec['new_col'] = f"{col}_is_na"
        elif method == 'frequency_encoding':
            spec['freq_map'] = X[col].astype(str).value_counts(normalize=True).to_dict()
        elif method == 'target_encoding':
            if y_num is None: return None
            col_str = X[col].astype(str)
            global_mean = float(y_num.mean())
            agg = y_num.groupby(col_str).agg(['count', 'mean'])
            smooth = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
            spec['encoding_map'] = smooth.to_dict()
            spec['global_mean'] = global_mean
        elif method == 'onehot_encoding':
            if X[col].nunique() > 10: return None
            spec['categories'] = X[col].astype(str).unique().tolist()
        elif method == 'quantile_binning':
            _, edges = pd.qcut(X[col].dropna(), q=5, retbins=True, duplicates='drop')
            spec['bin_edges'] = edges.tolist()
            spec['n_bins'] = 5
        elif method == 'polynomial_square':
            spec['median_val'] = float(X[col].median())
            spec['new_col'] = f"{col}_squared"
        elif method == 'hashing_encoding':
            spec['n_bins'] = 100
        elif method == 'cyclical_encode':
            temporal_type, period = detect_temporal_component(X[col], col)
            if period is None: return None
            # Coerce string-encoded temporal columns to numeric
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
            spec['period'] = period
            spec['temporal_type'] = temporal_type
            spec['median_val'] = float(X[col].median())
            spec['new_col_sin'] = f"{col}_sin"
            spec['new_col_cos'] = f"{col}_cos"
        elif method in ('product_interaction', 'division_interaction', 'addition_interaction',
                        'subtraction_interaction', 'abs_diff_interaction'):
            spec['median_a'] = float(X[col].median())
            spec['median_b'] = float(X[col_b].median())
            name_map = {'product_interaction': f"{col}_x_{col_b}",
                        'division_interaction': f"{col}_div_{col_b}",
                        'addition_interaction': f"{col}_plus_{col_b}",
                        'subtraction_interaction': f"{col}_minus_{col_b}",
                        'abs_diff_interaction': f"{col}_absdiff_{col_b}"}
            spec['new_col'] = name_map[method]
        elif method in ('three_way_interaction', 'three_way_addition', 'three_way_ratio'):
            if not col_c or col_c not in X.columns: return None
            spec['median_a'] = float(X[col].median())
            spec['median_b'] = float(X[col_b].median())
            spec['median_c'] = float(X[col_c].median())
            spec['col_c'] = col_c
            name_map = {'three_way_interaction': f"{col}_x_{col_b}_x_{col_c}",
                        'three_way_addition': f"{col}_plus_{col_b}_plus_{col_c}",
                        'three_way_ratio': f"{col}_x_{col_b}_div_{col_c}"}
            spec['new_col'] = name_map[method]
        elif method == 'group_mean':
            if y_num is None and col_b is None: return None
            cat_str = X[col].astype(str)
            if col_b:
                spec['grp_map'] = X[col_b].groupby(cat_str).mean().to_dict()
                spec['fill_val'] = float(X[col_b].mean())
            else:
                spec['grp_map'] = y_num.groupby(cat_str).mean().to_dict()
                spec['fill_val'] = float(y_num.mean())
            spec['new_col'] = f"group_mean_{col_b or 'target'}_by_{col}"
        elif method == 'text_stats':
            spec['new_col_len'] = f"{col}_len"
            spec['new_col_wc'] = f"{col}_word_count"
            spec['new_col_dc'] = f"{col}_digit_count"
        elif method == 'row_stats':
            # Global row statistics — computed across all numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return None
            spec['numeric_cols'] = numeric_cols
            # Store per-column medians for imputation during transform
            spec['col_medians'] = {c: float(X[c].median()) for c in numeric_cols}
        else:
            return None
        return spec
    
    def _apply_single(self, X, spec):
        method = spec['method']
        col = spec.get('column')
        if method == 'log_transform':
            X[col] = np.log1p(X[col].fillna(spec['fill_val']) + spec['offset'])
        elif method == 'sqrt_transform':
            X[col] = np.sqrt(X[col].fillna(spec['fill_val']) + spec['offset'])
        elif method == 'impute_median':
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(spec['median_val'])
        elif method == 'missing_indicator':
            X[spec['new_col']] = X[col].isnull().astype(int)
        elif method == 'frequency_encoding':
            X[col] = X[col].astype(str).map(spec['freq_map']).fillna(0).astype(float)
        elif method == 'target_encoding':
            X[col] = X[col].astype(str).map(spec['encoding_map']).fillna(spec['global_mean']).astype(float)
        elif method == 'onehot_encoding':
            dummies = pd.get_dummies(X[col].astype(str), prefix=col, drop_first=True)
            expected = [f"{col}_{cat}" for cat in spec['categories'][1:]]
            for ec in expected:
                if ec not in dummies.columns: dummies[ec] = 0
            dummies = dummies.reindex(columns=[c for c in expected], fill_value=0)
            X.drop(columns=[col], inplace=True)
            for c in dummies.columns:
                X[c] = dummies[c].values
        elif method == 'quantile_binning':
            X[col] = pd.cut(X[col], bins=spec['bin_edges'], labels=False, include_lowest=True)
            X[col] = X[col].fillna(spec['n_bins'] // 2).astype(float)
        elif method == 'polynomial_square':
            X[spec['new_col']] = (X[col].fillna(spec['median_val'])) ** 2
        elif method == 'hashing_encoding':
            X[col] = X[col].astype(str).apply(lambda x: hash(x) % spec['n_bins'])
        elif method == 'cyclical_encode':
            # Coerce string-encoded temporal columns to numeric
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
            vals = X[col].fillna(spec['median_val'])
            X[spec['new_col_sin']] = np.sin(2 * np.pi * vals / spec['period'])
            X[spec['new_col_cos']] = np.cos(2 * np.pi * vals / spec['period'])
        elif method == 'product_interaction':
            X[spec['new_col']] = X[col].fillna(spec['median_a']) * X[spec['col_b']].fillna(spec['median_b'])
        elif method == 'division_interaction':
            X[spec['new_col']] = X[col].fillna(spec['median_a']) / (X[spec['col_b']].fillna(spec['median_b']) + 1e-5)
        elif method == 'addition_interaction':
            X[spec['new_col']] = X[col].fillna(spec['median_a']) + X[spec['col_b']].fillna(spec['median_b'])
        elif method == 'subtraction_interaction':
            X[spec['new_col']] = X[col].fillna(spec['median_a']) - X[spec['col_b']].fillna(spec['median_b'])
        elif method == 'abs_diff_interaction':
            X[spec['new_col']] = (X[col].fillna(spec['median_a']) - X[spec['col_b']].fillna(spec['median_b'])).abs()
        elif method == 'three_way_interaction':
            X[spec['new_col']] = (X[col].fillna(spec['median_a']) * X[spec['col_b']].fillna(spec['median_b']) * X[spec['col_c']].fillna(spec['median_c']))
        elif method == 'three_way_addition':
            X[spec['new_col']] = (X[col].fillna(spec['median_a']) + X[spec['col_b']].fillna(spec['median_b']) + X[spec['col_c']].fillna(spec['median_c']))
        elif method == 'three_way_ratio':
            X[spec['new_col']] = ((X[col].fillna(spec['median_a']) * X[spec['col_b']].fillna(spec['median_b'])) / (X[spec['col_c']].fillna(spec['median_c']) + 1e-5))
        elif method == 'group_mean':
            X[spec['new_col']] = X[col].astype(str).map(spec['grp_map']).fillna(spec['fill_val']).astype(float)
        elif method == 'text_stats':
            s = X[col].fillna("").astype(str)
            X[spec['new_col_len']] = s.str.len()
            X[spec['new_col_wc']] = s.apply(lambda x: len(x.split()))
            X[spec['new_col_dc']] = s.str.count(r'\d')
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'row_stats':
            numeric_cols = [c for c in spec['numeric_cols'] if c in X.columns]
            if len(numeric_cols) < 2:
                return
            # Impute with stored medians
            num_df = X[numeric_cols].copy()
            for c in numeric_cols:
                num_df[c] = num_df[c].fillna(spec['col_medians'].get(c, 0))
            X['row_mean'] = num_df.mean(axis=1)
            X['row_std'] = num_df.std(axis=1).fillna(0)
            X['row_sum'] = num_df.sum(axis=1)
            X['row_min'] = num_df.min(axis=1)
            X['row_max'] = num_df.max(axis=1)
            X['row_range'] = X['row_max'] - X['row_min']
            X['row_zeros_count'] = (num_df == 0).sum(axis=1)
            X['row_missing_ratio'] = X[numeric_cols].isnull().mean(axis=1)


def final_encode(X):
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype('int64') / 10**9
            X[col] = X[col].fillna(-1)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype('category').cat.codes
    return X.fillna(0)


# =============================================================================
# TRAINING & EVALUATION HELPERS
# =============================================================================

def train_lgbm_cv(X, y, params, task_type, n_splits=5):
    """Train LightGBM with CV. Returns (final_model, cv_scores, metric_name)."""
    if task_type == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = kf.split(X)
    
    cv_scores = []
    for tr_idx, vl_idx in split_iter:
        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]
        if task_type == 'classification':
            m = lgb.LGBMClassifier(**params)
        else:
            m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        if task_type == 'classification':
            probs = m.predict_proba(X_vl, num_iteration=m.best_iteration_)
            if probs.shape[1] == 2:
                cv_scores.append(roc_auc_score(y_vl, probs[:, 1]))
            else:
                cv_scores.append(roc_auc_score(y_vl, probs, multi_class='ovr'))
        else:
            cv_scores.append(mean_squared_error(y_vl, m.predict(X_vl, num_iteration=m.best_iteration_)))
    
    if task_type == 'classification':
        final_model = lgb.LGBMClassifier(**params)
    else:
        final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)
    metric_name = "AUC" if task_type == 'classification' else "MSE"
    return final_model, cv_scores, metric_name


def evaluate_on_test(model, X_test, y_test, task_type):
    """Evaluate model on test data. Returns metrics dict."""
    metrics = {}
    if task_type == 'classification':
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        if probs.shape[1] == 2:
            metrics['AUC'] = roc_auc_score(y_test, probs[:, 1])
        else:
            try: metrics['AUC'] = roc_auc_score(y_test, probs, multi_class='ovr')
            except: metrics['AUC'] = float('nan')
        metrics['Accuracy'] = accuracy_score(y_test, preds)
        metrics['F1 (macro)'] = f1_score(y_test, preds, average='macro', zero_division=0)
        metrics['Precision (macro)'] = precision_score(y_test, preds, average='macro', zero_division=0)
        metrics['Recall (macro)'] = recall_score(y_test, preds, average='macro', zero_division=0)
        try: metrics['Log Loss'] = log_loss(y_test, probs)
        except: metrics['Log Loss'] = float('nan')
    else:
        preds = model.predict(X_test)
        metrics['MSE'] = mean_squared_error(y_test, preds)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_test, preds)
        metrics['R2'] = r2_score(y_test, preds)
    return metrics


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="Auto Feature Engineering", layout="wide", page_icon="⚙️")
    st.title("⚙️ Auto Feature Engineering Pipeline")
    st.caption("Upload data → Get recommendations → Compare Baseline vs. Enhanced model")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        model_dir = st.text_input("Meta-model directory", value="./meta_model")
        task_type = st.selectbox("Task type", ['classification', 'regression'])
        score_threshold = st.slider("Score threshold", 50.0, 70.0, 52.0, 0.5,
                                    help="Minimum meta-model score to recommend a transform")
        top_k = st.slider("Max recommendations", 5, 50, 20)
        st.divider()
        st.caption("Meta-model must be trained first via `train_meta_model.py`")
    
    # --- Load meta-model ---
    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        st.error(f"Meta-model not found in `{model_dir}`. Run `train_meta_model.py` first.")
        st.code(f"python train_meta_model.py --meta-db ./meta_learning_output/meta_learning_db.csv --output-dir {model_dir}")
        return
    
    if 'engine' not in st.session_state:
        from train_meta_model import MetaDBPreparator as TrainPreparator, MetaModelTrainer as TrainTrainer
        prep = TrainPreparator()
        prep.load(os.path.join(model_dir, 'preparator.pkl'))
        trainer = TrainTrainer()
        trainer.load(os.path.join(model_dir, 'meta_model.pkl'))
        st.session_state['engine'] = RecommendationEngine(prep, trainer)
        st.session_state['preparator'] = prep
        st.session_state['trainer'] = trainer
        if trainer.cv_scores:
            st.sidebar.success(f"Meta-model loaded (CV MAE: {trainer.cv_scores['mae_mean']:.2f})")
    
    engine = st.session_state['engine']
    
    # =====================================================================
    # STEP 1: UPLOAD
    # =====================================================================
    st.header("📂 Step 1: Upload Training Data")
    train_file = st.file_uploader("Drop your training CSV here", type=['csv'], key='train')
    if train_file is None:
        st.info("Upload a CSV file to get started.")
        return
    
    df_train = pd.read_csv(train_file)
    c1, c2 = st.columns(2)
    c1.write(f"**Shape:** {df_train.shape[0]:,} rows × {df_train.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df_train.head(20), use_container_width=True)
    target_col = c2.selectbox("Target column", df_train.columns.tolist())
    
    # =====================================================================
    # STEP 2: GENERATE RECOMMENDATIONS
    # =====================================================================
    if st.button("🔍 Analyze Dataset & Generate Recommendations", type="primary"):
        y_train = df_train[target_col]
        X_train = df_train.drop(columns=[target_col])
        label_enc = None
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_train):
            label_enc = LabelEncoder()
            y_train = pd.Series(label_enc.fit_transform(y_train.astype(str)),
                               index=y_train.index, name=y_train.name)
        
        with st.spinner("Computing dataset features and generating recommendations..."):
            recs, baseline, ds_meta = engine.recommend(
                X_train, y_train, task_type=task_type,
                top_k=top_k, score_threshold=score_threshold)
        
        n_classes = int(y_train.nunique()) if task_type == 'classification' else 2
        suggested_params, hp_explanation = compute_lgbm_params(
            n_rows=X_train.shape[0], n_cols=X_train.shape[1],
            task_type=task_type, n_classes=n_classes,
            missing_ratio=ds_meta.get('missing_ratio', 0),
            cat_ratio=ds_meta.get('cat_ratio', 0),
            avg_feature_corr=ds_meta.get('avg_feature_corr', 0),
            class_imbalance_ratio=ds_meta.get('class_imbalance_ratio', 1.0))
        
        st.session_state.update({
            'X_train': X_train, 'y_train': y_train, 'label_enc': label_enc,
            'recommendations': recs, 'baseline': baseline, 'ds_meta': ds_meta,
            'target_col': target_col, 'suggested_params': suggested_params,
            'hp_explanation': hp_explanation, 'n_classes': n_classes,
        })
        # Reset downstream
        for key in ['baseline_model', 'enhanced_model', 'baseline_cv', 'enhanced_cv',
                     'enhanced_applicator', 'baseline_train_cols', 'enhanced_train_cols',
                     'test_metrics_baseline', 'test_metrics_enhanced']:
            st.session_state.pop(key, None)
    
    if 'recommendations' not in st.session_state:
        return
    
    recs = st.session_state['recommendations']
    baseline = st.session_state['baseline']
    
    st.divider()
    
    # =====================================================================
    # STEP 3: REVIEW RECOMMENDATIONS (checkboxes)
    # =====================================================================
    st.header("📋 Step 2: Review Recommendations")
    quick_metric = "AUC" if task_type == 'classification' else "MSE"
    st.write(f"**Quick baseline ({quick_metric}):** {baseline['score']:.5f} ± {baseline['std']:.5f}")
    
    if not recs:
        st.warning("No recommendations above threshold. Try lowering the score threshold.")
    else:
        st.write(f"**{len(recs)} recommendations found.** Deselect any you want to skip:")
        
        if 'rec_selection' not in st.session_state or len(st.session_state.get('rec_selection', [])) != len(recs):
            st.session_state['rec_selection'] = [True] * len(recs)
        
        # Header row
        hdr = st.columns([0.5, 1, 2.5, 3, 1.5])
        hdr[0].write("**Use**")
        hdr[1].write("**Score**")
        hdr[2].write("**Method**")
        hdr[3].write("**Column**")
        hdr[4].write("**Type**")
        
        for i, rec in enumerate(recs):
            cols = st.columns([0.5, 1, 2.5, 3, 1.5])
            st.session_state['rec_selection'][i] = cols[0].checkbox(
                f"r{i}", value=st.session_state['rec_selection'][i],
                label_visibility="collapsed", key=f"rec_check_{i}")
            cols[1].write(f"**{rec['predicted_score']:.1f}**")
            cols[2].write(rec['method'])
            # Show descriptive column info
            if rec['method'] == 'row_stats':
                cols[3].write("All numeric columns (8 row-level features)")
                cols[4].write("🌐 Global")
            elif rec.get('is_interaction'):
                col_a = rec.get('col_a', '')
                col_b = rec.get('col_b', '')
                if rec['method'] == 'group_mean':
                    cols[3].write(f"mean(`{col_b}`) by `{col_a}`")
                else:
                    cols[3].write(rec['column'])
                cols[4].write("🔗 Inter.")
            else:
                cols[3].write(rec['column'])
                cols[4].write("📊 Single")
    
    # =====================================================================
    # STEP 3b: MANUAL TRANSFORMS (user domain knowledge)
    # =====================================================================
    st.divider()
    with st.expander("➕ Add Manual Transforms (optional — use domain knowledge)", expanded=False):
        st.caption("Apply any transformation manually, even ones the meta-model doesn't recommend.")
        
        if 'manual_transforms' not in st.session_state:
            st.session_state['manual_transforms'] = []
        
        X_train = st.session_state['X_train']
        all_cols = X_train.columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in all_cols if c not in numeric_cols]
        
        # --- Type-safe method lists ---
        NUMERIC_ONLY_METHODS = ['log_transform', 'sqrt_transform', 'quantile_binning',
                                'polynomial_square', 'impute_median']
        CAT_ONLY_METHODS = ['frequency_encoding', 'target_encoding',
                            'onehot_encoding', 'hashing_encoding', 'text_stats']
        EITHER_TYPE_METHODS = ['missing_indicator', 'cyclical_encode']
        NUMERIC_INTERACTION_METHODS = ['product_interaction', 'division_interaction',
                                       'addition_interaction', 'subtraction_interaction',
                                       'abs_diff_interaction']
        GROUP_METHODS = ['group_mean']
        
        transform_type = st.radio("Transform type",
                                  ["Single-column", "Interaction (2 columns)", "Global (row statistics)"],
                                  horizontal=True, key="manual_type")
        
        if transform_type == "Single-column":
            mc1, mc2 = st.columns(2)
            m_col = mc1.selectbox("Column", all_cols, key="manual_col")
            
            # Build allowed methods based on column type
            col_is_numeric = m_col in numeric_cols
            if col_is_numeric:
                allowed_methods = NUMERIC_ONLY_METHODS + EITHER_TYPE_METHODS
            else:
                allowed_methods = CAT_ONLY_METHODS + EITHER_TYPE_METHODS
                # Filter by cardinality for cat methods
                nunique = X_train[m_col].nunique()
                if nunique <= 10:
                    allowed_methods = [m for m in allowed_methods if m != 'hashing_encoding']
                if not (2 < nunique <= 10):
                    allowed_methods = [m for m in allowed_methods if m != 'onehot_encoding']
            
            m_method = mc2.selectbox("Method", allowed_methods, key="manual_method")
            
            # --- Preview ---
            preview_col = st.empty()
            sample = X_train[m_col].dropna().head(5)
            type_label = "numeric" if col_is_numeric else "categorical"
            preview_text = f"**Preview:** `{m_method}` on `{m_col}` ({type_label}, {X_train[m_col].nunique()} unique)"
            if m_method == 'log_transform':
                preview_text += f"\n\n  Sample values: {sample.tolist()} → log1p(x + offset)"
            elif m_method == 'polynomial_square':
                preview_text += f"\n\n  Creates: `{m_col}_squared` = x²"
            elif m_method == 'cyclical_encode':
                temporal_type, period = detect_temporal_component(X_train[m_col], m_col)
                if temporal_type:
                    preview_text += f"\n\n  Detected: **{temporal_type}** (period={period}) → sin/cos encoding"
                else:
                    preview_text += "\n\n  ⚠️ No temporal pattern detected — cyclical encoding may not be useful"
            elif m_method == 'missing_indicator':
                miss_pct = X_train[m_col].isnull().mean() * 100
                preview_text += f"\n\n  Creates: `{m_col}_is_na` (missing: {miss_pct:.1f}%)"
            elif m_method in ('frequency_encoding', 'target_encoding'):
                preview_text += f"\n\n  Replaces categories with {'frequency ratios' if m_method == 'frequency_encoding' else 'smoothed target means'}"
            preview_col.info(preview_text)
            
            if st.button("➕ Add transform", key="add_manual"):
                new_rec = {
                    'column': m_col, 'method': m_method,
                    'is_interaction': False, 'predicted_score': -1,
                    'manual': True,
                }
                st.session_state['manual_transforms'].append(new_rec)
                st.rerun()
        
        elif transform_type == "Interaction (2 columns)":
            # Choose interaction sub-type first to constrain column selection
            interaction_kind = st.radio("Interaction kind",
                                         ["Numeric × Numeric (arithmetic)", "Categorical → Numeric (group-by)"],
                                         horizontal=True, key="int_kind")
            
            if interaction_kind == "Numeric × Numeric (arithmetic)":
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for arithmetic interactions.")
                else:
                    mc1, mc2, mc3 = st.columns(3)
                    m_col_a = mc1.selectbox("Column A (numeric)", numeric_cols, key="manual_col_a")
                    remaining = [c for c in numeric_cols if c != m_col_a]
                    m_col_b = mc2.selectbox("Column B (numeric)", remaining, key="manual_col_b")
                    m_method = mc3.selectbox("Operation", NUMERIC_INTERACTION_METHODS, key="manual_int_method")
                    
                    # Preview
                    op_symbols = {'product_interaction': '×', 'division_interaction': '÷',
                                  'addition_interaction': '+', 'subtraction_interaction': '−',
                                  'abs_diff_interaction': '|A−B|'}
                    name_map = {
                        'product_interaction': f"{m_col_a}_x_{m_col_b}",
                        'division_interaction': f"{m_col_a}_div_{m_col_b}",
                        'addition_interaction': f"{m_col_a}_plus_{m_col_b}",
                        'subtraction_interaction': f"{m_col_a}_minus_{m_col_b}",
                        'abs_diff_interaction': f"{m_col_a}_absdiff_{m_col_b}",
                    }
                    st.info(f"**Preview:** `{name_map[m_method]}` = `{m_col_a}` {op_symbols[m_method]} `{m_col_b}`")
                    
                    if st.button("➕ Add interaction", key="add_manual_int"):
                        new_rec = {
                            'column': name_map[m_method],
                            'method': m_method,
                            'col_a': m_col_a, 'col_b': m_col_b,
                            'is_interaction': True, 'predicted_score': -1,
                            'manual': True,
                        }
                        st.session_state['manual_transforms'].append(new_rec)
                        st.rerun()
            
            else:  # Group-by
                if not cat_cols or not numeric_cols:
                    st.warning("Need at least 1 categorical and 1 numeric column for group-by.")
                else:
                    mc1, mc2, mc3 = st.columns(3)
                    m_cat = mc1.selectbox("Group-by column (categorical)", cat_cols, key="manual_grp_cat")
                    m_num = mc2.selectbox("Aggregated column (numeric)", numeric_cols, key="manual_grp_num")
                    m_agg = mc3.selectbox("Aggregation", ['group_mean'], key="manual_grp_agg")
                    
                    n_groups = X_train[m_cat].nunique()
                    st.info(f"**Preview:** `group_mean_{m_num}_by_{m_cat}` = mean of `{m_num}` within each `{m_cat}` group ({n_groups} groups)")
                    
                    if st.button("➕ Add group-by", key="add_manual_grp"):
                        new_rec = {
                            'column': f"group_mean_{m_num}_by_{m_cat}",
                            'method': m_agg,
                            'col_a': m_cat, 'col_b': m_num,
                            'is_interaction': True, 'predicted_score': -1,
                            'manual': True,
                        }
                        st.session_state['manual_transforms'].append(new_rec)
                        st.rerun()
        
        else:  # Global (row statistics)
            n_num = len(numeric_cols)
            if n_num < 2:
                st.warning("Need at least 2 numeric columns for row statistics.")
            else:
                st.info(f"**Preview:** Adds 8 row-level features across {n_num} numeric columns:\n\n"
                        f"  `row_mean`, `row_std`, `row_sum`, `row_min`, `row_max`, `row_range`, "
                        f"`row_zeros_count`, `row_missing_ratio`")
                # Check if already added
                already_has = any(mt['method'] == 'row_stats' for mt in st.session_state['manual_transforms'])
                if already_has:
                    st.caption("✅ Row statistics already added.")
                elif st.button("➕ Add row statistics", key="add_manual_rowstats"):
                    new_rec = {
                        'column': 'GLOBAL_ROW_STATS', 'method': 'row_stats',
                        'is_interaction': False, 'predicted_score': -1,
                        'manual': True,
                    }
                    st.session_state['manual_transforms'].append(new_rec)
                    st.rerun()
        
        # Show current manual transforms
        if st.session_state['manual_transforms']:
            st.write(f"**Manual transforms ({len(st.session_state['manual_transforms'])}):**")
            for idx, mt in enumerate(st.session_state['manual_transforms']):
                mc_disp, mc_del = st.columns([5, 1])
                # Descriptive label
                if mt['method'] == 'row_stats':
                    mc_disp.write(f"• 🌐 `row_stats` — 8 global row-level features")
                elif mt.get('is_interaction'):
                    col_a = mt.get('col_a', '?')
                    col_b = mt.get('col_b', '?')
                    if mt['method'] == 'group_mean':
                        mc_disp.write(f"• 🔗 `group_mean` — mean of `{col_b}` grouped by `{col_a}`")
                    else:
                        mc_disp.write(f"• 🔗 `{mt['method']}` — `{col_a}` ↔ `{col_b}`")
                else:
                    mc_disp.write(f"• 📊 `{mt['method']}` on `{mt['column']}`")
                if mc_del.button("🗑️", key=f"del_manual_{idx}"):
                    st.session_state['manual_transforms'].pop(idx)
                    st.rerun()
        else:
            st.caption("No manual transforms added yet.")
    
    # =====================================================================
    # STEP 4: HP CONFIGURATION
    # =====================================================================
    st.divider()
    st.header("🎛️ Step 3: LightGBM Hyperparameters")
    
    suggested_params = st.session_state['suggested_params']
    hp_explanation = st.session_state['hp_explanation']
    
    hp_mode = st.radio("Hyperparameter mode",
                       ["🤖 Use suggested (data-driven)", "✏️ Custom override"],
                       horizontal=True)
    
    if hp_mode == "🤖 Use suggested (data-driven)":
        st.write("Parameters computed from dataset characteristics:")
        exp_data = []
        for key in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                     'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
            exp_data.append({'Parameter': key, 'Value': suggested_params[key],
                             'Reasoning': hp_explanation.get(key, '')})
        st.dataframe(pd.DataFrame(exp_data), use_container_width=True, hide_index=True)
        active_params = suggested_params.copy()
    else:
        st.write("Adjust hyperparameters manually (suggested values as defaults):")
        c1, c2, c3, c4 = st.columns(4)
        custom_n_est = c1.number_input("n_estimators", 50, 5000, suggested_params['n_estimators'], 50)
        custom_lr = c2.number_input("learning_rate", 0.001, 0.5, suggested_params['learning_rate'], 0.005, format="%.4f")
        custom_leaves = c3.number_input("num_leaves", 7, 255, suggested_params['num_leaves'], 4)
        custom_depth = c4.number_input("max_depth", -1, 15, suggested_params['max_depth'], 1)
        c5, c6, c7, c8 = st.columns(4)
        custom_min_child = c5.number_input("min_child_samples", 1, 200, suggested_params['min_child_samples'], 5)
        custom_subsample = c6.number_input("subsample", 0.3, 1.0, suggested_params['subsample'], 0.05)
        custom_colsample = c7.number_input("colsample_bytree", 0.3, 1.0, suggested_params['colsample_bytree'], 0.05)
        custom_reg_alpha = c8.number_input("reg_alpha", 0.0, 10.0, suggested_params['reg_alpha'], 0.05)
        custom_reg_lambda = st.number_input("reg_lambda", 0.0, 10.0, suggested_params['reg_lambda'], 0.05)
        active_params = suggested_params.copy()
        active_params.update({
            'n_estimators': custom_n_est, 'learning_rate': custom_lr,
            'num_leaves': custom_leaves, 'max_depth': custom_depth,
            'min_child_samples': custom_min_child, 'subsample': custom_subsample,
            'colsample_bytree': custom_colsample, 'reg_alpha': custom_reg_alpha,
            'reg_lambda': custom_reg_lambda})
    
    # =====================================================================
    # STEP 5: TRAIN BOTH MODELS
    # =====================================================================
    st.divider()
    st.header("🚀 Step 4: Train & Compare")
    
    if st.button("🚀 Train Baseline + Enhanced Models", type="primary"):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        n_classes = st.session_state['n_classes']
        selected_recs = [r for r, sel in zip(recs, st.session_state.get('rec_selection', [True]*len(recs))) if sel]
        # Append manual transforms from user domain knowledge
        manual_transforms = st.session_state.get('manual_transforms', [])
        if manual_transforms:
            selected_recs = selected_recs + manual_transforms
        
        # --- BASELINE: raw data, standard LightGBM ---
        progress = st.progress(0, text="Training baseline model (raw data, default HPs)...")
        X_base = final_encode(X_train.copy())
        base_params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
                        'random_state': 42, 'verbosity': -1, 'n_jobs': -1}
        if task_type == 'classification':
            if n_classes > 2:
                base_params.update({'objective': 'multiclass', 'num_class': n_classes, 'metric': 'multi_logloss'})
            else:
                base_params.update({'objective': 'binary', 'metric': 'auc'})
        else:
            base_params.update({'objective': 'regression', 'metric': 'rmse'})
        
        baseline_model, baseline_cv, metric_name = train_lgbm_cv(X_base, y_train, base_params, task_type)
        st.session_state['baseline_model'] = baseline_model
        st.session_state['baseline_cv'] = baseline_cv
        st.session_state['baseline_train_cols'] = X_base.columns.tolist()
        st.session_state['baseline_params'] = base_params
        
        # --- ENHANCED: auto-FE + smart HPs ---
        progress.progress(50, text="Training enhanced model (auto-FE + smart HPs)...")
        applicator = TransformApplicator()
        X_transformed = applicator.fit_transform(X_train, y_train, selected_recs)
        X_enhanced = final_encode(X_transformed)
        enhanced_model, enhanced_cv, _ = train_lgbm_cv(X_enhanced, y_train, active_params, task_type)
        
        st.session_state['enhanced_model'] = enhanced_model
        st.session_state['enhanced_cv'] = enhanced_cv
        st.session_state['enhanced_applicator'] = applicator
        st.session_state['enhanced_train_cols'] = X_enhanced.columns.tolist()
        st.session_state['enhanced_params'] = active_params
        st.session_state['selected_recs'] = selected_recs
        progress.progress(100, text="Done!")
    
    # --- Show CV comparison ---
    if 'baseline_cv' in st.session_state and 'enhanced_cv' in st.session_state:
        baseline_cv = st.session_state['baseline_cv']
        enhanced_cv = st.session_state['enhanced_cv']
        metric_name = "AUC" if task_type == 'classification' else "MSE"
        base_mean, base_std = np.mean(baseline_cv), np.std(baseline_cv)
        enh_mean, enh_std = np.mean(enhanced_cv), np.std(enhanced_cv)
        
        if task_type == 'classification':
            delta = enh_mean - base_mean
            improved = delta > 0
        else:
            delta = base_mean - enh_mean
            improved = delta > 0
        
        st.subheader(f"Cross-Validation Results ({metric_name})")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔵 Baseline", f"{base_mean:.5f} ± {base_std:.5f}")
        c2.metric("🟢 Enhanced", f"{enh_mean:.5f} ± {enh_std:.5f}",
                  delta=f"{enh_mean - base_mean:+.5f}")
        pct = abs(delta / base_mean * 100) if base_mean != 0 else 0
        c3.metric("Improvement", f"{pct:.2f}%",
                  delta="✅ Better" if improved else "⚠️ No improvement")
        
        selected_recs = st.session_state.get('selected_recs', [])
        with st.expander(f"📋 {len(selected_recs)} transforms applied + HP comparison"):
            tc1, tc2 = st.columns(2)
            with tc1:
                st.write("**Applied Transforms:**")
                for r in selected_recs:
                    if r['method'] == 'row_stats':
                        st.write(f"- 🌐 `row_stats` (8 row-level features)")
                    elif r.get('is_interaction') and r['method'] == 'group_mean':
                        st.write(f"- 🔗 `group_mean` — mean(`{r.get('col_b')}`) by `{r.get('col_a')}`")
                    elif r.get('is_interaction'):
                        st.write(f"- 🔗 `{r['method']}` → `{r['column']}`")
                    else:
                        st.write(f"- `{r['method']}` on `{r['column']}`")
            with tc2:
                st.write("**Baseline HPs:** LightGBM defaults (n_est=500, lr=0.05, leaves=31)")
                st.write("**Enhanced HPs:**")
                ep = st.session_state.get('enhanced_params', {})
                for k in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                           'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree']:
                    if k in ep:
                        st.write(f"- `{k}`: {ep[k]}")
        
        # --- Download preprocessed training data ---
        with st.expander("📥 Download Preprocessed Training Data"):
            dl1, dl2 = st.columns(2)
            # Baseline preprocessed
            X_base_dl = final_encode(st.session_state['X_train'].copy())
            base_train_csv = BytesIO()
            X_base_dl.to_csv(base_train_csv, index=False)
            dl1.download_button("📥 Baseline Train (preprocessed)",
                               base_train_csv.getvalue(),
                               "train_baseline_preprocessed.csv", "text/csv",
                               key="dl_base_train")
            # Enhanced preprocessed
            applicator = st.session_state.get('enhanced_applicator')
            if applicator:
                X_enh_dl = applicator.transform(st.session_state['X_train'])
                X_enh_dl = final_encode(X_enh_dl)
                enh_train_csv = BytesIO()
                X_enh_dl.to_csv(enh_train_csv, index=False)
                dl2.download_button("📥 Enhanced Train (preprocessed)",
                                   enh_train_csv.getvalue(),
                                   "train_enhanced_preprocessed.csv", "text/csv",
                                   key="dl_enh_train")
    
    # =====================================================================
    # STEP 6: TEST SET EVALUATION
    # =====================================================================
    st.divider()
    st.header("🧪 Step 5: Test Set Evaluation")
    
    if 'baseline_model' not in st.session_state:
        st.info("⬆️ Train both models first (Step 4)")
        return
    
    test_file = st.file_uploader("Drop your test CSV here", type=['csv'], key='test')
    if test_file is None:
        st.info("Upload a test CSV to evaluate both models side-by-side.")
        return
    
    df_test = pd.read_csv(test_file)
    st.write(f"**Test shape:** {df_test.shape[0]:,} rows × {df_test.shape[1]} columns")
    target_col = st.session_state.get('target_col')
    has_target = target_col in df_test.columns
    if not has_target:
        st.warning(f"Target column '{target_col}' not found. Predictions only.")
    
    if st.button("🔮 Evaluate Both Models on Test Set", type="primary"):
        label_enc = st.session_state.get('label_enc')
        if has_target:
            y_test = df_test[target_col]
            X_test_raw = df_test.drop(columns=[target_col])
            if label_enc is not None:
                y_test = pd.Series(label_enc.transform(y_test.astype(str)),
                                  index=y_test.index, name=y_test.name)
        else:
            X_test_raw = df_test.copy()
            y_test = None
        
        # Baseline test
        X_test_base = final_encode(X_test_raw.copy())
        base_cols = st.session_state['baseline_train_cols']
        for c in base_cols:
            if c not in X_test_base.columns: X_test_base[c] = 0
        X_test_base = X_test_base[base_cols]
        
        # Enhanced test
        applicator = st.session_state['enhanced_applicator']
        X_test_transformed = applicator.transform(X_test_raw)
        X_test_enh = final_encode(X_test_transformed)
        enh_cols = st.session_state['enhanced_train_cols']
        for c in enh_cols:
            if c not in X_test_enh.columns: X_test_enh[c] = 0
        X_test_enh = X_test_enh[enh_cols]
        
        baseline_model = st.session_state['baseline_model']
        enhanced_model = st.session_state['enhanced_model']
        
        if has_target and y_test is not None:
            metrics_base = evaluate_on_test(baseline_model, X_test_base, y_test, task_type)
            metrics_enh = evaluate_on_test(enhanced_model, X_test_enh, y_test, task_type)
            st.session_state['test_metrics_baseline'] = metrics_base
            st.session_state['test_metrics_enhanced'] = metrics_enh
            
            st.subheader("📊 Test Set Comparison")
            higher_better = {'AUC', 'Accuracy', 'F1 (macro)', 'Precision (macro)', 'Recall (macro)', 'R2'}
            results = {'Metric': [], '🔵 Baseline': [], '🟢 Enhanced': [], 'Δ': [], '': []}
            for metric in metrics_base:
                vb, ve = metrics_base[metric], metrics_enh[metric]
                if np.isnan(vb) or np.isnan(ve): continue
                diff = ve - vb
                is_hb = metric in higher_better
                ok = (diff > 0) == is_hb
                results['Metric'].append(metric)
                results['🔵 Baseline'].append(f"{vb:.5f}")
                results['🟢 Enhanced'].append(f"{ve:.5f}")
                results['Δ'].append(f"{diff:+.5f}")
                results[''].append("✅" if ok else ("➖" if diff == 0 else "⚠️"))
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            primary = 'AUC' if task_type == 'classification' else 'MSE'
            if primary in metrics_base and primary in metrics_enh:
                diff = metrics_enh[primary] - metrics_base[primary]
                st.success(f"**{primary}: {metrics_base[primary]:.5f} → {metrics_enh[primary]:.5f} ({diff:+.5f})**")
            
            # Download preprocessed test data
            with st.expander("📥 Download Preprocessed Test Data"):
                tdl1, tdl2 = st.columns(2)
                base_test_csv = BytesIO()
                X_test_base.to_csv(base_test_csv, index=False)
                tdl1.download_button("📥 Baseline Test (preprocessed)",
                                    base_test_csv.getvalue(),
                                    "test_baseline_preprocessed.csv", "text/csv",
                                    key="dl_base_test")
                enh_test_csv = BytesIO()
                X_test_enh.to_csv(enh_test_csv, index=False)
                tdl2.download_button("📥 Enhanced Test (preprocessed)",
                                    enh_test_csv.getvalue(),
                                    "test_enhanced_preprocessed.csv", "text/csv",
                                    key="dl_enh_test")
        else:
            base_preds = baseline_model.predict(X_test_base)
            enh_preds = enhanced_model.predict(X_test_enh)
            pred_df = pd.DataFrame({'baseline_pred': base_preds, 'enhanced_pred': enh_preds})
            st.dataframe(pred_df.head(20), use_container_width=True)
            csv_buf = BytesIO()
            pred_df.to_csv(csv_buf, index=False)
            st.download_button("📥 Download Predictions", csv_buf.getvalue(), "predictions.csv", "text/csv")
    
    # --- Save pipeline ---
    if 'test_metrics_baseline' in st.session_state:
        st.divider()
        if st.button("💾 Save Full Pipeline"):
            save_path = os.path.join(".", "fitted_pipeline.pkl")
            state = {
                'enhanced_applicator': st.session_state.get('enhanced_applicator'),
                'enhanced_model': st.session_state.get('enhanced_model'),
                'baseline_model': st.session_state.get('baseline_model'),
                'enhanced_params': st.session_state.get('enhanced_params'),
                'baseline_params': st.session_state.get('baseline_params'),
                'label_enc': st.session_state.get('label_enc'),
                'task_type': task_type,
                'enhanced_train_cols': st.session_state.get('enhanced_train_cols'),
                'baseline_train_cols': st.session_state.get('baseline_train_cols'),
                'selected_recs': st.session_state.get('selected_recs'),
                'test_metrics_baseline': st.session_state.get('test_metrics_baseline'),
                'test_metrics_enhanced': st.session_state.get('test_metrics_enhanced'),
            }
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            st.success(f"Pipeline saved to `{save_path}`")


if __name__ == "__main__":
    main()