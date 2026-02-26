"""
recommend_app.py — Feature Engineering Recommendation Tool
============================================================

Streamlit application that:
1. Accepts a training CSV + target column
2. Analyzes the dataset via trained meta-models
3. Suggests preprocessing transforms ranked by predicted impact
4. Trains two LightGBM models (baseline vs. enhanced)
5. Accepts a test CSV and compares both models

Self-contained — all meta-feature computation and transform logic is inline.

Usage:
    streamlit run recommend_app.py -- --model_dir ./meta_models
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import argparse
import re
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, log_loss, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew, kurtosis, shapiro, spearmanr
from scipy.stats import entropy as sp_entropy


# =============================================================================
# CONSTANTS
# =============================================================================

SENTINEL = -10.0

DATASET_FEATURES = [
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',
]

NUMERICAL_COLUMN_FEATURES = [
    'null_pct', 'unique_ratio', 'outlier_ratio',
    'skewness', 'kurtosis_val', 'coeff_variation',
    'zeros_ratio', 'entropy',
    'is_binary', 'range_iqr_ratio',
    'baseline_feature_importance', 'importance_rank_pct',
    'spearman_corr_target', 'mutual_info_score',
    'shapiro_p_value',
    'bimodality_coefficient',
    'pct_negative', 'pct_in_0_1_range',
]

CATEGORICAL_COLUMN_FEATURES = [
    'null_pct', 'n_unique', 'unique_ratio',
    'entropy', 'normalized_entropy',
    'is_binary', 'is_low_cardinality', 'is_high_cardinality',
    'top_category_dominance', 'top3_category_concentration',
    'rare_category_pct',
    'conditional_entropy',
    'baseline_feature_importance', 'importance_rank_pct',
    'mutual_info_score', 'pps_score',
]

INTERACTION_PAIR_FEATURES = [
    'n_numerical_cols',
    'pearson_corr', 'spearman_corr',
    'mutual_info_pair', 'mic_score', 'scale_ratio',
    'sum_importance', 'max_importance', 'min_importance',
    'sum_null_pct', 'max_null_pct',
    'sum_unique_ratio', 'abs_diff_unique_ratio',
    'sum_entropy', 'abs_diff_entropy',
    'sum_target_corr', 'abs_diff_target_corr',
    'sum_mi_target', 'abs_diff_mi_target',
    'both_binary',
]

NUMERICAL_METHODS = [
    'log_transform', 'sqrt_transform', 'polynomial_square',
    'polynomial_cube', 'reciprocal_transform', 'quantile_binning',
    'impute_median', 'missing_indicator',
]

CATEGORICAL_METHODS = [
    'frequency_encoding', 'target_encoding', 'onehot_encoding',
    'hashing_encoding', 'missing_indicator',
]

INTERACTION_METHODS_NUM_NUM = [
    'product_interaction', 'division_interaction',
    'addition_interaction', 'abs_diff_interaction',
]

INTERACTION_METHODS_CAT_NUM = ['group_mean', 'group_std']

INTERACTION_METHODS_CAT_CAT = ['cat_concat']

BASE_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
}

METHOD_DESCRIPTIONS = {
    'log_transform': 'Log transform — reduces right skew',
    'sqrt_transform': 'Square root transform — mild skew reduction',
    'polynomial_square': 'Add squared feature — captures U-shaped effects',
    'polynomial_cube': 'Add cubed feature — captures asymmetric nonlinearity',
    'reciprocal_transform': 'Add reciprocal (1/x) — captures diminishing returns',
    'quantile_binning': 'Quantile binning — discretizes into 5 bins',
    'impute_median': 'Impute missing with median',
    'missing_indicator': 'Add binary missing indicator column',
    'frequency_encoding': 'Replace categories with their frequency',
    'target_encoding': 'Replace categories with smoothed target mean',
    'onehot_encoding': 'One-hot encode (drop first)',
    'hashing_encoding': 'Hash encode into 32 buckets',
    'product_interaction': 'Multiply two columns (A × B)',
    'division_interaction': 'Divide columns (A / |B|)',
    'addition_interaction': 'Add columns (A + B)',
    'abs_diff_interaction': 'Absolute difference |A − B|',
    'group_mean': 'Group-by mean (numeric grouped by category)',
    'group_std': 'Group-by std (numeric grouped by category)',
    'cat_concat': 'Concatenate two categories → new feature',
}


# =============================================================================
# META-FEATURE COMPUTATION (self-contained from collectors)
# =============================================================================

def ensure_numeric_target(y):
    if pd.api.types.is_numeric_dtype(y):
        return y
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)


def sanitize_feature_names(df):
    df.columns = [re.sub(r'[\[\]\{\}":,]', '_', str(c)) for c in df.columns]
    if df.columns.duplicated().any():
        cols = list(df.columns)
        seen = {}
        for i, c in enumerate(cols):
            if c in seen:
                seen[c] += 1
                cols[i] = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
        df.columns = cols
    return df


def prepare_data_for_model(X_train, X_val):
    X_tr, X_vl = X_train.copy(), X_val.copy()
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]):
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_vl[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_vl[col] = le.transform(X_vl[col].astype(str))
        med = X_tr[col].median() if pd.api.types.is_numeric_dtype(X_tr[col]) else -999
        X_tr[col] = X_tr[col].fillna(med)
        X_vl[col] = X_vl[col].fillna(med)
    return X_tr, X_vl


def get_baseline_importances(X, y):
    y_numeric = ensure_numeric_target(y)
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X_enc[col] = X[col].fillna(X[col].median())
        else:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X[col].astype(str).fillna('NaN'))
    params = BASE_PARAMS.copy()
    params['n_estimators'] = 100
    model = lgb.LGBMClassifier(**params)
    model.fit(X_enc, y_numeric)
    return pd.Series(model.feature_importances_, index=X.columns)


def get_dataset_meta(X, y):
    y_numeric = ensure_numeric_target(y)
    n_rows, n_cols = X.shape
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    meta = {
        'n_rows': n_rows, 'n_cols': n_cols,
        'n_numeric_cols': len(numeric_cols), 'n_cat_cols': len(cat_cols),
        'cat_ratio': len(cat_cols) / max(n_cols, 1),
        'missing_ratio': float(X.isnull().mean().mean()),
        'row_col_ratio': n_rows / max(n_cols, 1),
        'n_classes': int(y_numeric.nunique()),
    }
    class_counts = y_numeric.value_counts()
    meta['class_imbalance_ratio'] = float(class_counts.max() / max(class_counts.min(), 1))

    if len(numeric_cols) >= 2:
        corr_matrix = X[numeric_cols].corr().abs().values.copy()
        np.fill_diagonal(corr_matrix, 0)
        meta['avg_feature_corr'] = float(corr_matrix.mean())
        meta['max_feature_corr'] = float(corr_matrix.max())
        target_corrs = X[numeric_cols].corrwith(y_numeric).abs()
        meta['avg_target_corr'] = float(target_corrs.mean())
        meta['max_target_corr'] = float(target_corrs.max())
    else:
        meta.update({k: 0.0 for k in ['avg_feature_corr', 'max_feature_corr',
                                        'avg_target_corr', 'max_target_corr']})
    try:
        X_enc = pd.DataFrame(index=X.index)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_enc[col] = X[col].fillna(X[col].median())
            else:
                le = LabelEncoder()
                X_enc[col] = le.fit_transform(X[col].astype(str).fillna('NaN'))
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X_enc, y_numeric)
        meta['landmarking_score'] = float(dt.score(X_enc, y_numeric))
    except Exception:
        meta['landmarking_score'] = 0.5

    return meta


def get_numeric_column_meta(series, y, importance, importance_rank_pct):
    clean = series.dropna()
    y_numeric = ensure_numeric_target(y)
    meta = {
        'null_pct': float(series.isnull().mean()),
        'unique_ratio': float(series.nunique() / max(len(series), 1)),
        'is_binary': int(series.nunique() <= 2),
        'baseline_feature_importance': float(importance),
        'importance_rank_pct': float(importance_rank_pct),
    }
    if len(clean) < 5:
        meta.update({'outlier_ratio': 0.0, 'skewness': 0.0, 'kurtosis_val': 0.0,
                      'coeff_variation': 0.0, 'zeros_ratio': 0.0, 'entropy': 0.0,
                      'range_iqr_ratio': 1.0, 'spearman_corr_target': 0.0,
                      'mutual_info_score': 0.0, 'shapiro_p_value': 0.5,
                      'bimodality_coefficient': 0.0, 'pct_negative': 0.0,
                      'pct_in_0_1_range': 0.0})
        return meta
    Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:
        meta['outlier_ratio'] = float(((clean < Q1 - 1.5*IQR) | (clean > Q3 + 1.5*IQR)).mean())
        meta['range_iqr_ratio'] = float((clean.max() - clean.min()) / IQR)
    else:
        meta['outlier_ratio'] = 0.0; meta['range_iqr_ratio'] = 1.0
    meta['skewness'] = float(skew(clean, nan_policy='omit'))
    meta['kurtosis_val'] = float(kurtosis(clean, nan_policy='omit'))
    std_val, mean_val = clean.std(), clean.mean()
    meta['coeff_variation'] = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-10 else 0.0
    meta['zeros_ratio'] = float((clean == 0).mean())
    meta['pct_negative'] = float((clean < 0).mean())
    meta['pct_in_0_1_range'] = float(((clean >= 0) & (clean <= 1)).mean())
    try:
        counts, _ = np.histogram(clean, bins=min(50, max(int(len(clean)**0.5), 5)))
        probs = counts / counts.sum(); probs = probs[probs > 0]
        meta['entropy'] = float(-np.sum(probs * np.log2(probs)))
    except Exception:
        meta['entropy'] = 0.0
    try:
        sample = clean.sample(min(5000, len(clean)), random_state=42)
        _, p = shapiro(sample); meta['shapiro_p_value'] = float(p)
    except Exception:
        meta['shapiro_p_value'] = 0.5
    sk, kt = meta['skewness'], meta['kurtosis_val']
    meta['bimodality_coefficient'] = float((sk**2 + 1) / (kt + 3)) if (kt + 3) > 0 else 0.0
    try:
        ci = series.notna() & y_numeric.notna()
        if ci.sum() > 10:
            sp, _ = spearmanr(series[ci], y_numeric[ci])
            meta['spearman_corr_target'] = float(abs(sp)) if not np.isnan(sp) else 0.0
        else: meta['spearman_corr_target'] = 0.0
    except Exception:
        meta['spearman_corr_target'] = 0.0
    try:
        filled = series.fillna(series.median()).to_frame()
        mi = mutual_info_classif(filled, y_numeric, random_state=42)[0]
        meta['mutual_info_score'] = float(mi)
    except Exception:
        meta['mutual_info_score'] = 0.0
    return meta


def get_categorical_column_meta(series, y, importance, importance_rank_pct):
    y_numeric = ensure_numeric_target(y)
    meta = {'null_pct': float(series.isnull().mean())}
    n_unique = series.nunique(dropna=True)
    meta['n_unique'] = n_unique
    meta['unique_ratio'] = float(n_unique / max(len(series), 1))
    meta['is_binary'] = int(n_unique <= 2)
    meta['is_low_cardinality'] = int(n_unique <= 10)
    meta['is_high_cardinality'] = int(n_unique > 50)
    meta['baseline_feature_importance'] = float(importance)
    meta['importance_rank_pct'] = float(importance_rank_pct)
    vc = series.value_counts(normalize=True, dropna=True)
    meta['top_category_dominance'] = float(vc.iloc[0]) if len(vc) > 0 else 1.0
    meta['top3_category_concentration'] = float(vc.iloc[:3].sum()) if len(vc) > 0 else 1.0
    meta['rare_category_pct'] = float((vc < 0.01).mean()) if len(vc) > 0 else 0.0
    if len(vc) > 0:
        probs = vc.values; probs = probs[probs > 0]
        meta['entropy'] = float(-np.sum(probs * np.log2(probs)))
        max_ent = np.log2(max(n_unique, 2))
        meta['normalized_entropy'] = float(meta['entropy'] / max_ent) if max_ent > 0 else 0.0
    else:
        meta['entropy'] = 0.0; meta['normalized_entropy'] = 0.0
    try:
        categories = series.dropna().unique()
        h_y_x = 0.0
        for cat in categories:
            mask = series == cat; p_cat = mask.mean()
            y_cat = y_numeric[mask]
            if len(y_cat) > 0 and y_cat.nunique() > 1:
                vc_y = y_cat.value_counts(normalize=True)
                h_y_x += p_cat * float(-np.sum(vc_y * np.log2(vc_y.clip(lower=1e-10))))
        meta['conditional_entropy'] = float(h_y_x)
    except Exception:
        meta['conditional_entropy'] = 0.0
    try:
        le = LabelEncoder()
        enc = le.fit_transform(series.astype(str).fillna('NaN'))
        mi = mutual_info_classif(enc.reshape(-1, 1), y_numeric, random_state=42)[0]
        meta['mutual_info_score'] = float(mi)
    except Exception:
        meta['mutual_info_score'] = 0.0
    try:
        le = LabelEncoder()
        enc = le.fit_transform(series.astype(str).fillna('NaN'))
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(enc.reshape(-1, 1), y_numeric)
        pps = float(dt.score(enc.reshape(-1, 1), y_numeric))
        rb = 1.0 / max(y_numeric.nunique(), 2)
        meta['pps_score'] = max(0.0, (pps - rb) / (1.0 - rb))
    except Exception:
        meta['pps_score'] = 0.0
    return meta


def _encode_for_mi(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.median()).values
    le = LabelEncoder()
    return le.fit_transform(series.astype(str).fillna('NaN'))


def _column_entropy(series):
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) < 5: return 0.0
        try:
            counts, _ = np.histogram(clean, bins=min(50, max(int(len(clean)**0.5), 5)))
            probs = counts / counts.sum(); probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))
        except Exception: return 0.0
    else:
        vc = series.value_counts(normalize=True, dropna=True)
        if len(vc) == 0: return 0.0
        probs = vc.values; probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


def get_pair_meta_features(col_a, col_b, y, imp_a, imp_b):
    y_numeric = ensure_numeric_target(y)
    a_num = pd.api.types.is_numeric_dtype(col_a)
    b_num = pd.api.types.is_numeric_dtype(col_b)
    n_numerical = int(a_num) + int(b_num)
    meta = {'n_numerical_cols': n_numerical}

    if n_numerical == 2:
        try:
            cl = col_a.notna() & col_b.notna()
            meta['pearson_corr'] = float(abs(col_a[cl].corr(col_b[cl]))) if cl.sum() > 10 else 0.0
        except Exception: meta['pearson_corr'] = 0.0
        try:
            cl = col_a.notna() & col_b.notna()
            sp, _ = spearmanr(col_a[cl], col_b[cl]) if cl.sum() > 10 else (0, 0)
            meta['spearman_corr'] = float(abs(sp)) if not np.isnan(sp) else 0.0
        except Exception: meta['spearman_corr'] = 0.0
    else:
        meta['pearson_corr'] = SENTINEL; meta['spearman_corr'] = SENTINEL

    try:
        ea = _encode_for_mi(col_a); eb = _encode_for_mi(col_b)
        mi = mutual_info_classif(ea.reshape(-1, 1),
              (eb * 100).astype(int) if b_num else eb, random_state=42)[0]
        meta['mutual_info_pair'] = float(mi)
    except Exception: meta['mutual_info_pair'] = 0.0

    try:
        n = len(col_a); nb = min(20, max(5, int(n**0.4)))
        if a_num:
            ab = pd.qcut(col_a.fillna(col_a.median()), q=nb, labels=False, duplicates='drop').values
        else:
            ab = LabelEncoder().fit_transform(col_a.astype(str).fillna('NaN'))
        if b_num:
            bb = pd.qcut(col_b.fillna(col_b.median()), q=nb, labels=False, duplicates='drop').values
        else:
            bb = LabelEncoder().fit_transform(col_b.astype(str).fillna('NaN'))
        mi_b = mutual_info_classif(ab.reshape(-1, 1), bb, discrete_features=True, random_state=42)[0]
        _, cb = np.unique(bb, return_counts=True)
        hb = sp_entropy(cb / cb.sum(), base=2)
        meta['mic_score'] = min(float(mi_b / max(hb, 1e-10)), 1.0) if hb > 0 else 0.0
    except Exception: meta['mic_score'] = 0.0

    if n_numerical == 2:
        try:
            sa, sb = col_a.std(), col_b.std()
            meta['scale_ratio'] = float(max(sa, sb) / min(sa, sb)) if min(sa, sb) > 1e-10 else 0.0
        except Exception: meta['scale_ratio'] = 0.0
    else:
        meta['scale_ratio'] = SENTINEL

    meta['sum_importance'] = float(imp_a + imp_b)
    meta['max_importance'] = float(max(imp_a, imp_b))
    meta['min_importance'] = float(min(imp_a, imp_b))
    na, nb_ = float(col_a.isnull().mean()), float(col_b.isnull().mean())
    meta['sum_null_pct'] = na + nb_; meta['max_null_pct'] = max(na, nb_)
    ua = float(col_a.nunique() / max(len(col_a), 1))
    ub = float(col_b.nunique() / max(len(col_b), 1))
    meta['sum_unique_ratio'] = ua + ub; meta['abs_diff_unique_ratio'] = abs(ua - ub)
    ea_, eb_ = _column_entropy(col_a), _column_entropy(col_b)
    meta['sum_entropy'] = ea_ + eb_; meta['abs_diff_entropy'] = abs(ea_ - eb_)

    if n_numerical == 2:
        def _tc(s):
            try:
                cl = s.notna() & y_numeric.notna()
                if cl.sum() > 10:
                    sp, _ = spearmanr(s[cl], y_numeric[cl])
                    return float(abs(sp)) if not np.isnan(sp) else 0.0
                return 0.0
            except Exception: return 0.0
        ta, tb = _tc(col_a), _tc(col_b)
        meta['sum_target_corr'] = ta + tb; meta['abs_diff_target_corr'] = abs(ta - tb)
    else:
        meta['sum_target_corr'] = SENTINEL; meta['abs_diff_target_corr'] = SENTINEL

    def _mi_t(s):
        try:
            e = _encode_for_mi(s)
            return float(mutual_info_classif(e.reshape(-1, 1), y_numeric, random_state=42)[0])
        except Exception: return 0.0
    ma, mb = _mi_t(col_a), _mi_t(col_b)
    meta['sum_mi_target'] = ma + mb; meta['abs_diff_mi_target'] = abs(ma - mb)
    meta['both_binary'] = int(col_a.nunique() <= 2 and col_b.nunique() <= 2)
    return meta


# =============================================================================
# METHOD APPLICABILITY GATES
# =============================================================================

def should_test_numerical(method, col_meta, series):
    if method == 'impute_median' and col_meta['null_pct'] == 0: return False
    if method == 'missing_indicator' and col_meta['null_pct'] == 0: return False
    if method == 'sqrt_transform':
        clean = series.dropna()
        if len(clean) > 0 and clean.min() < 0: return False
    if method == 'log_transform' and col_meta['is_binary']: return False
    if method == 'quantile_binning' and col_meta['unique_ratio'] < 0.01: return False
    if method in ('polynomial_square', 'polynomial_cube', 'reciprocal_transform'):
        if col_meta['is_binary']: return False
    return True


def should_test_categorical(method, col_meta):
    nu = col_meta['n_unique']
    if method == 'missing_indicator' and col_meta['null_pct'] == 0: return False
    if method == 'onehot_encoding' and (nu < 2 or nu > 10): return False
    if method == 'hashing_encoding' and nu <= 10: return False
    return True


# =============================================================================
# META-MODEL LOADING & PREDICTION
# =============================================================================

@st.cache_resource
def load_meta_models(model_dir):
    """Load all available meta-models from disk."""
    models = {}
    for ctype in ['numerical', 'categorical', 'interaction']:
        type_dir = os.path.join(model_dir, ctype)
        config_path = os.path.join(type_dir, f'{ctype}_config.json')
        reg_path = os.path.join(type_dir, f'{ctype}_regressor.txt')

        if not os.path.exists(config_path) or not os.path.exists(reg_path):
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        booster = lgb.Booster(model_file=reg_path)

        models[ctype] = {
            'booster': booster,
            'config': config,
            'feature_names': config['feature_names'],
            'method_vocab': config['method_vocab'],
        }
    return models


def build_feature_vector(meta_dict, method, config):
    """Build a single feature vector matching training schema."""
    feature_names = config['feature_names']
    method_vocab = config['method_vocab']

    row = {}
    # Meta-features
    for f in feature_names:
        if f.startswith('method_'):
            row[f] = 0
        else:
            row[f] = meta_dict.get(f, -999)

    # One-hot method
    method_col = f'method_{method}'
    if method_col in row:
        row[method_col] = 1

    return pd.DataFrame([row])[feature_names].fillna(-999)


def generate_suggestions(X, y, meta_models, baseline_score, baseline_std, progress_cb=None):
    """
    Run meta-models on all applicable (column, method) combinations.
    Returns a list of suggestion dicts sorted by predicted delta descending.
    """
    y_numeric = ensure_numeric_target(y)
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = baseline_score
    ds_meta['baseline_std'] = baseline_std
    ds_meta['relative_headroom'] = max(1.0 - baseline_score, 0.001)

    importances = get_baseline_importances(X, y)
    imp_ranks = importances.rank(ascending=False, pct=True)

    suggestions = []
    total_steps = 0
    done_steps = 0

    # Count work
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if 'numerical' in meta_models: total_steps += len(numeric_cols) * len(NUMERICAL_METHODS)
    if 'categorical' in meta_models: total_steps += len(cat_cols) * len(CATEGORICAL_METHODS)
    if 'interaction' in meta_models: total_steps += 50  # rough estimate
    total_steps = max(total_steps, 1)

    # --- Numerical suggestions ---
    if 'numerical' in meta_models and numeric_cols:
        mm = meta_models['numerical']
        for col in numeric_cols:
            col_meta = get_numeric_column_meta(
                X[col], y,
                importance=float(importances.get(col, 0)),
                importance_rank_pct=float(imp_ranks.get(col, 0.5))
            )
            combined_meta = {**ds_meta, **col_meta}

            for method in NUMERICAL_METHODS:
                done_steps += 1
                if progress_cb: progress_cb(done_steps / total_steps)
                if not should_test_numerical(method, col_meta, X[col]):
                    continue
                if method not in mm['method_vocab']:
                    continue

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]
                suggestions.append({
                    'type': 'numerical',
                    'column': col,
                    'column_b': None,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                })

    # --- Categorical suggestions ---
    if 'categorical' in meta_models and cat_cols:
        mm = meta_models['categorical']
        for col in cat_cols:
            col_meta = get_categorical_column_meta(
                X[col], y,
                importance=float(importances.get(col, 0)),
                importance_rank_pct=float(imp_ranks.get(col, 0.5))
            )
            combined_meta = {**ds_meta, **col_meta}

            for method in CATEGORICAL_METHODS:
                done_steps += 1
                if progress_cb: progress_cb(done_steps / total_steps)
                if not should_test_categorical(method, col_meta):
                    continue
                if method not in mm['method_vocab']:
                    continue

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]
                suggestions.append({
                    'type': 'categorical',
                    'column': col,
                    'column_b': None,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                })

    # --- Interaction suggestions ---
    if 'interaction' in meta_models and X.shape[1] >= 2:
        mm = meta_models['interaction']
        imp_dict = importances.to_dict()

        sorted_num = sorted(numeric_cols, key=lambda c: imp_dict.get(c, 0), reverse=True)[:8]
        sorted_cat = sorted(cat_cols, key=lambda c: imp_dict.get(c, 0), reverse=True)[:6]
        tested = set()

        pairs = []
        # num+num
        for i, a in enumerate(sorted_num):
            for b in sorted_num[i+1:]:
                if len(pairs) >= 30: break
                pairs.append((a, b, INTERACTION_METHODS_NUM_NUM))
        # cat+num
        cnt = 0
        for cat_c in sorted_cat:
            for num_c in sorted_num:
                if cnt >= 20: break
                pairs.append((cat_c, num_c, INTERACTION_METHODS_CAT_NUM))
                cnt += 1
        # cat+cat
        cnt = 0
        for i, a in enumerate(sorted_cat):
            for b in sorted_cat[i+1:]:
                if cnt >= 10: break
                if X[a].nunique() > len(X)*0.5 or X[b].nunique() > len(X)*0.5:
                    continue
                pairs.append((a, b, INTERACTION_METHODS_CAT_CAT))
                cnt += 1

        for col_a, col_b, methods in pairs:
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in tested: continue
            tested.add(pair_key)

            pair_meta = get_pair_meta_features(
                X[col_a], X[col_b], y,
                imp_a=float(imp_dict.get(col_a, 0)),
                imp_b=float(imp_dict.get(col_b, 0))
            )
            combined_meta = {**ds_meta, **pair_meta}

            for method in methods:
                done_steps += 1
                if progress_cb: progress_cb(min(done_steps / total_steps, 1.0))
                if method not in mm['method_vocab']:
                    continue

                # Applicability checks
                a_num = pd.api.types.is_numeric_dtype(X[col_a])
                b_num = pd.api.types.is_numeric_dtype(X[col_b])
                if method == 'division_interaction':
                    if X[col_a].nunique() <= 2 or X[col_b].nunique() <= 2: continue
                if method in ('group_mean', 'group_std'):
                    actual_cat = col_a if not a_num else col_b
                    actual_num = col_b if not a_num else col_a
                    if a_num == b_num: continue
                else:
                    actual_cat = col_a
                    actual_num = col_b

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]

                display_a, display_b = sorted([col_a, col_b])
                suggestions.append({
                    'type': 'interaction',
                    'column': col_a,
                    'column_b': col_b,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                })

    suggestions.sort(key=lambda s: s['predicted_delta'], reverse=True)
    return suggestions


# =============================================================================
# TRANSFORM FIT / APPLY PIPELINE
# =============================================================================

def fit_and_apply_suggestions(X_train, y_train, suggestions):
    """
    Apply selected suggestions to training data.
    Returns: (X_enhanced, fitted_params_list)
    fitted_params_list stores everything needed to replay on test data.
    """
    X_enh = X_train.copy()
    y_num = ensure_numeric_target(y_train)
    fitted = []

    for sug in suggestions:
        method = sug['method']
        col = sug['column']
        col_b = sug.get('column_b')
        params = {'method': method, 'type': sug['type'], 'column': col, 'column_b': col_b}

        try:
            if sug['type'] == 'numerical':
                if method == 'log_transform':
                    fill = float(X_train[col].min())
                    temp = X_train[col].fillna(fill)
                    offset = float(abs(temp.min()) + 1) if temp.min() <= 0 else 0.0
                    params.update({'fill': fill, 'offset': offset})
                    X_enh[col] = np.log1p(X_enh[col].fillna(fill) + offset)

                elif method == 'sqrt_transform':
                    fill = float(X_train[col].median())
                    temp = X_train[col].fillna(fill)
                    offset = float(abs(temp.min()) + 1) if temp.min() < 0 else 0.0
                    params.update({'fill': fill, 'offset': offset})
                    X_enh[col] = np.sqrt(X_enh[col].fillna(fill) + offset)

                elif method == 'polynomial_square':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_sq'] = X_enh[col].fillna(med) ** 2

                elif method == 'polynomial_cube':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_cube'] = X_enh[col].fillna(med) ** 3

                elif method == 'reciprocal_transform':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_recip'] = 1.0 / (X_enh[col].fillna(med).abs() + 1e-5)

                elif method == 'quantile_binning':
                    _, bin_edges = pd.qcut(X_train[col].dropna(), q=5, retbins=True, duplicates='drop')
                    params['bin_edges'] = bin_edges.tolist()
                    X_enh[col] = pd.cut(X_enh[col], bins=bin_edges, labels=False, include_lowest=True)

                elif method == 'impute_median':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[col] = X_enh[col].fillna(med)

                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif sug['type'] == 'categorical':
                if method == 'frequency_encoding':
                    freq = X_train[col].astype(str).value_counts(normalize=True).to_dict()
                    params['freq_map'] = freq
                    X_enh[col] = X_enh[col].astype(str).map(freq).fillna(0).astype(float)

                elif method == 'target_encoding':
                    str_col = X_train[col].astype(str)
                    gm = float(y_num.mean())
                    agg = y_num.groupby(str_col).agg(['count', 'mean'])
                    smooth = ((agg['count'] * agg['mean'] + 10 * gm) / (agg['count'] + 10)).to_dict()
                    params.update({'smooth_map': smooth, 'global_mean': gm})
                    X_enh[col] = X_enh[col].astype(str).map(smooth).fillna(gm).astype(float)

                elif method == 'onehot_encoding':
                    train_dummies = pd.get_dummies(X_train[col].astype(str), prefix=col, drop_first=True)
                    params['dummy_columns'] = train_dummies.columns.tolist()
                    enh_dummies = pd.get_dummies(X_enh[col].astype(str), prefix=col, drop_first=True)
                    enh_dummies = enh_dummies.reindex(columns=train_dummies.columns, fill_value=0)
                    X_enh = X_enh.drop(columns=[col])
                    X_enh = pd.concat([X_enh.reset_index(drop=True), enh_dummies.reset_index(drop=True)], axis=1)

                elif method == 'hashing_encoding':
                    X_enh[col] = X_enh[col].astype(str).apply(lambda x: hash(x) % 32)

                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif sug['type'] == 'interaction':
                a, b = col, col_b
                if method == 'product_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_x_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) * X_enh[b].fillna(mb)

                elif method == 'division_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_div_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) / (X_enh[b].fillna(mb).abs() + 1e-5)

                elif method == 'addition_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_plus_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) + X_enh[b].fillna(mb)

                elif method == 'abs_diff_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_absdiff_{b}'
                    X_enh[nc] = (X_enh[a].fillna(ma) - X_enh[b].fillna(mb)).abs()

                elif method == 'group_mean':
                    a_num = pd.api.types.is_numeric_dtype(X_train[a])
                    cat_c, num_c = (b, a) if a_num else (a, b)
                    grp = X_train[num_c].groupby(X_train[cat_c].astype(str)).mean().to_dict()
                    fv = float(X_train[num_c].mean())
                    params.update({'grp_map': grp, 'fill_val': fv, 'cat_col': cat_c, 'num_col': num_c})
                    nc = f'grpmean_{num_c}_by_{cat_c}'
                    X_enh[nc] = X_enh[cat_c].astype(str).map(grp).fillna(fv).astype(float)

                elif method == 'group_std':
                    a_num = pd.api.types.is_numeric_dtype(X_train[a])
                    cat_c, num_c = (b, a) if a_num else (a, b)
                    grp = X_train[num_c].groupby(X_train[cat_c].astype(str)).std().to_dict()
                    fv = float(X_train[num_c].std())
                    params.update({'grp_map': grp, 'fill_val': fv, 'cat_col': cat_c, 'num_col': num_c})
                    nc = f'grpstd_{num_c}_by_{cat_c}'
                    X_enh[nc] = X_enh[cat_c].astype(str).map(grp).fillna(fv).astype(float)

                elif method == 'cat_concat':
                    nc = f'{a}_concat_{b}'
                    combined = X_enh[a].astype(str) + '_' + X_enh[b].astype(str)
                    le = LabelEncoder(); le.fit(combined)
                    params['le_classes'] = le.classes_.tolist()
                    X_enh[nc] = le.transform(combined)

            fitted.append(params)

        except Exception as e:
            st.warning(f"Failed to apply {method} on {col}: {e}")
            continue

    return X_enh, fitted


def apply_fitted_to_test(X_test, fitted_params_list):
    """Apply pre-fitted transforms to test data."""
    X_enh = X_test.copy()

    for params in fitted_params_list:
        method = params['method']
        col = params['column']
        col_b = params.get('column_b')

        try:
            if params['type'] == 'numerical':
                if method == 'log_transform':
                    X_enh[col] = np.log1p(X_enh[col].fillna(params['fill']) + params['offset'])
                elif method == 'sqrt_transform':
                    X_enh[col] = np.sqrt(X_enh[col].fillna(params['fill']) + params['offset'])
                elif method == 'polynomial_square':
                    X_enh[f'{col}_sq'] = X_enh[col].fillna(params['median']) ** 2
                elif method == 'polynomial_cube':
                    X_enh[f'{col}_cube'] = X_enh[col].fillna(params['median']) ** 3
                elif method == 'reciprocal_transform':
                    X_enh[f'{col}_recip'] = 1.0 / (X_enh[col].fillna(params['median']).abs() + 1e-5)
                elif method == 'quantile_binning':
                    X_enh[col] = pd.cut(X_enh[col], bins=params['bin_edges'],
                                         labels=False, include_lowest=True)
                elif method == 'impute_median':
                    X_enh[col] = X_enh[col].fillna(params['median'])
                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif params['type'] == 'categorical':
                if method == 'frequency_encoding':
                    X_enh[col] = X_enh[col].astype(str).map(params['freq_map']).fillna(0).astype(float)
                elif method == 'target_encoding':
                    X_enh[col] = X_enh[col].astype(str).map(params['smooth_map']).fillna(params['global_mean']).astype(float)
                elif method == 'onehot_encoding':
                    dummies = pd.get_dummies(X_enh[col].astype(str), prefix=col, drop_first=True)
                    dummies = dummies.reindex(columns=params['dummy_columns'], fill_value=0)
                    X_enh = X_enh.drop(columns=[col])
                    X_enh = pd.concat([X_enh.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
                elif method == 'hashing_encoding':
                    X_enh[col] = X_enh[col].astype(str).apply(lambda x: hash(x) % 32)
                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif params['type'] == 'interaction':
                a, b = col, col_b
                if method == 'product_interaction':
                    X_enh[f'{a}_x_{b}'] = X_enh[a].fillna(params['med_a']) * X_enh[b].fillna(params['med_b'])
                elif method == 'division_interaction':
                    X_enh[f'{a}_div_{b}'] = X_enh[a].fillna(params['med_a']) / (X_enh[b].fillna(params['med_b']).abs() + 1e-5)
                elif method == 'addition_interaction':
                    X_enh[f'{a}_plus_{b}'] = X_enh[a].fillna(params['med_a']) + X_enh[b].fillna(params['med_b'])
                elif method == 'abs_diff_interaction':
                    X_enh[f'{a}_absdiff_{b}'] = (X_enh[a].fillna(params['med_a']) - X_enh[b].fillna(params['med_b'])).abs()
                elif method == 'group_mean':
                    cc, nc_ = params['cat_col'], params['num_col']
                    X_enh[f'grpmean_{nc_}_by_{cc}'] = X_enh[cc].astype(str).map(params['grp_map']).fillna(params['fill_val']).astype(float)
                elif method == 'group_std':
                    cc, nc_ = params['cat_col'], params['num_col']
                    X_enh[f'grpstd_{nc_}_by_{cc}'] = X_enh[cc].astype(str).map(params['grp_map']).fillna(params['fill_val']).astype(float)
                elif method == 'cat_concat':
                    nc = f'{a}_concat_{b}'
                    combined = X_enh[a].astype(str) + '_' + X_enh[b].astype(str)
                    known = set(params['le_classes'])
                    X_enh[nc] = combined.apply(lambda x: params['le_classes'].index(x) if x in known else -1)

        except Exception:
            continue

    return X_enh


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_lgbm_model(X_train, y_train, X_val, y_val, n_classes):
    """Train a LightGBM classifier with early stopping on validation set."""
    X_tr, X_vl = prepare_data_for_model(X_train, X_val)

    params = BASE_PARAMS.copy()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model, X_tr.columns.tolist()


def evaluate_on_set(model, X, y, train_columns, n_classes):
    """Evaluate model on a dataset. Returns metrics dict."""
    # Encode to match training
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            le = LabelEncoder()
            le.fit(X_enc[col].astype(str))
            X_enc[col] = le.transform(X_enc[col].astype(str))
        X_enc[col] = X_enc[col].fillna(X_enc[col].median() if pd.api.types.is_numeric_dtype(X_enc[col]) else -999)

    # Align columns
    for c in train_columns:
        if c not in X_enc.columns:
            X_enc[c] = 0
    X_enc = X_enc[train_columns]

    y_pred = model.predict(X_enc)
    y_pred_proba = model.predict_proba(X_enc)

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y, y_pred))

    try:
        if n_classes == 2:
            metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba[:, 1]))
        else:
            metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted'))
    except Exception:
        metrics['roc_auc'] = None

    try:
        avg = 'binary' if n_classes == 2 else 'weighted'
        metrics['f1'] = float(f1_score(y, y_pred, average=avg, zero_division=0))
        metrics['precision'] = float(precision_score(y, y_pred, average=avg, zero_division=0))
        metrics['recall'] = float(recall_score(y, y_pred, average=avg, zero_division=0))
    except Exception:
        metrics['f1'] = None; metrics['precision'] = None; metrics['recall'] = None

    try:
        metrics['log_loss'] = float(log_loss(y, y_pred_proba))
    except Exception:
        metrics['log_loss'] = None

    return metrics


# =============================================================================
# DEDUPLICATE SUGGESTIONS
# =============================================================================

def deduplicate_suggestions(suggestions):
    """
    For single-column transforms, keep only the best method per column.
    For interactions, keep only the best method per pair.
    """
    best_single = {}  # col -> best suggestion
    best_interaction = {}  # (col_a, col_b) -> best suggestion

    for s in suggestions:
        if s['type'] in ('numerical', 'categorical'):
            key = (s['type'], s['column'])
            if key not in best_single or s['predicted_delta'] > best_single[key]['predicted_delta']:
                best_single[key] = s
        else:
            pair = tuple(sorted([s['column'], s['column_b']]))
            key = (pair, s['method'])  # keep best per pair+method
            if key not in best_interaction or s['predicted_delta'] > best_interaction[key]['predicted_delta']:
                best_interaction[key] = s

    deduped = list(best_single.values()) + list(best_interaction.values())
    deduped.sort(key=lambda s: s['predicted_delta'], reverse=True)
    return deduped


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="Feature Engineering Recommender", layout="wide")
    st.title("🔧 Feature Engineering Recommender")
    st.caption("Upload a dataset → get transform suggestions from meta-models → compare baseline vs. enhanced model")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        model_dir = st.text_input("Meta-models directory", value="./meta_models")
        top_k = st.slider("Max suggestions to apply", 1, 30, 10)
        delta_threshold = st.number_input("Min predicted delta (calibrated)", value=0.0, step=0.1,
                                           help="Only suggest transforms above this threshold")
        st.divider()
        st.markdown("**Workflow**")
        st.markdown("1. Upload training CSV\n2. Select target column\n3. Analyze & get suggestions\n4. Train models\n5. Upload test CSV & compare")

    # Load meta-models
    if os.path.isdir(model_dir):
        meta_models = load_meta_models(model_dir)
        if meta_models:
            st.sidebar.success(f"Loaded: {', '.join(meta_models.keys())}")
        else:
            st.sidebar.error("No models found in directory")
            meta_models = {}
    else:
        st.sidebar.warning(f"Directory not found: {model_dir}")
        meta_models = {}

    # --- Session state init ---
    for key in ['train_df', 'target_col', 'X_train', 'y_train', 'suggestions',
                'selected_indices', 'baseline_model', 'enhanced_model',
                'baseline_train_cols', 'enhanced_train_cols', 'fitted_params',
                'n_classes', 'label_encoder', 'baseline_val_metrics', 'enhanced_val_metrics']:
        if key not in st.session_state:
            st.session_state[key] = None

    # =========================================================================
    # STEP 1: UPLOAD TRAINING DATA
    # =========================================================================
    st.header("① Upload Training Data")

    uploaded_train = st.file_uploader("Upload training CSV", type=['csv'], key='train_upload')

    if uploaded_train is not None:
        df = pd.read_csv(uploaded_train)
        df = sanitize_feature_names(df)
        st.session_state.train_df = df

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            st.dataframe(df.head(5), use_container_width=True, height=200)

        # Target selection
        target_col = st.selectbox("Select target column", options=df.columns.tolist())
        st.session_state.target_col = target_col

        y_raw = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode target
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        st.session_state.label_encoder = le
        n_classes = y.nunique()
        st.session_state.n_classes = n_classes

        st.write(f"**Task**: {'Binary' if n_classes == 2 else 'Multiclass'} classification "
                 f"({n_classes} classes)")

        class_dist = y_raw.value_counts()
        st.bar_chart(class_dist, height=150)

        st.session_state.X_train = X
        st.session_state.y_train = y

    # =========================================================================
    # STEP 2: ANALYZE & SUGGEST
    # =========================================================================
    if st.session_state.X_train is not None:
        st.divider()
        st.header("② Analyze & Get Suggestions")

        if not meta_models:
            st.warning("No meta-models loaded. Please check the model directory.")
        else:
            if st.button("🔍 Analyze Dataset", type="primary"):
                X = st.session_state.X_train
                y = st.session_state.y_train

                with st.spinner("Computing baseline score..."):
                    # Quick baseline via internal split
                    X_tr, X_vl, y_tr, y_vl = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    try:
                        X_tr_enc, X_vl_enc = prepare_data_for_model(X_tr, X_vl)
                        quick_model = lgb.LGBMClassifier(**{**BASE_PARAMS, 'n_estimators': 100})
                        quick_model.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)],
                                        callbacks=[lgb.early_stopping(10, verbose=False)])
                        n_classes = st.session_state.n_classes
                        if n_classes == 2:
                            baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc)[:, 1])
                        else:
                            baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc),
                                                            multi_class='ovr', average='weighted')
                        baseline_std = 0.01  # rough estimate
                    except Exception as e:
                        st.error(f"Baseline failed: {e}")
                        baseline_score = 0.5
                        baseline_std = 0.05

                st.info(f"Quick baseline ROC-AUC: **{baseline_score:.4f}**")

                progress_bar = st.progress(0, text="Generating suggestions...")

                def update_progress(pct):
                    progress_bar.progress(min(pct, 1.0), text=f"Analyzing... {pct*100:.0f}%")

                suggestions = generate_suggestions(
                    X, y, meta_models,
                    baseline_score=baseline_score,
                    baseline_std=baseline_std,
                    progress_cb=update_progress,
                )
                progress_bar.empty()

                # Filter and deduplicate
                suggestions = [s for s in suggestions if s['predicted_delta'] >= delta_threshold]
                suggestions = deduplicate_suggestions(suggestions)

                st.session_state.suggestions = suggestions
                st.success(f"Generated {len(suggestions)} suggestions")

        # Display suggestions
        if st.session_state.suggestions:
            suggestions = st.session_state.suggestions

            display_data = []
            for i, s in enumerate(suggestions[:50]):  # show top 50
                row = {
                    '#': i + 1,
                    'Type': s['type'].capitalize(),
                    'Column(s)': s['column'] + (f" × {s['column_b']}" if s['column_b'] else ""),
                    'Method': s['method'],
                    'Predicted Δ': f"{s['predicted_delta']:+.3f}",
                    'Description': s['description'],
                }
                display_data.append(row)

            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            # Selection
            n_available = min(len(suggestions), 50)
            default_k = min(top_k, n_available)

            st.write(f"**Select suggestions to apply** (top {default_k} pre-selected):")
            selected = st.multiselect(
                "Choose transforms",
                options=list(range(n_available)),
                default=list(range(default_k)),
                format_func=lambda i: f"#{i+1}: {suggestions[i]['column']}"
                                       + (f" × {suggestions[i]['column_b']}" if suggestions[i]['column_b'] else "")
                                       + f" → {suggestions[i]['method']} (Δ={suggestions[i]['predicted_delta']:+.3f})",
            )
            st.session_state.selected_indices = selected

    # =========================================================================
    # STEP 3: TRAIN MODELS
    # =========================================================================
    if st.session_state.selected_indices is not None and st.session_state.suggestions:
        st.divider()
        st.header("③ Train Baseline & Enhanced Models")

        if st.button("🚀 Train Both Models", type="primary"):
            X = st.session_state.X_train
            y = st.session_state.y_train
            n_classes = st.session_state.n_classes
            selected_suggestions = [st.session_state.suggestions[i]
                                     for i in st.session_state.selected_indices]

            # Split for validation
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # --- Apply transforms ---
            with st.spinner(f"Applying {len(selected_suggestions)} transforms..."):
                X_tr_enh, fitted_params = fit_and_apply_suggestions(X_tr, y_tr, selected_suggestions)
                X_vl_enh = apply_fitted_to_test(X_vl, fitted_params)

                # Also fit on full training set for later test evaluation
                X_full_enh, fitted_params_full = fit_and_apply_suggestions(X, y, selected_suggestions)
                st.session_state.fitted_params = fitted_params_full

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Baseline Model")
                with st.spinner("Training baseline..."):
                    baseline_model, baseline_cols = train_lgbm_model(X_tr, y_tr, X_vl, y_vl, n_classes)
                    baseline_metrics = evaluate_on_set(baseline_model, X_vl, y_vl, baseline_cols, n_classes)
                st.session_state.baseline_model = baseline_model
                st.session_state.baseline_train_cols = baseline_cols
                st.session_state.baseline_val_metrics = baseline_metrics

                for metric, val in baseline_metrics.items():
                    if val is not None:
                        st.metric(metric.replace('_', ' ').upper(), f"{val:.4f}")

            with col2:
                st.subheader(f"Enhanced Model (+{len(selected_suggestions)} transforms)")
                with st.spinner("Training enhanced..."):
                    enhanced_model, enhanced_cols = train_lgbm_model(
                        X_tr_enh, y_tr, X_vl_enh, y_vl, n_classes
                    )
                    enhanced_metrics = evaluate_on_set(
                        enhanced_model, X_vl_enh, y_vl, enhanced_cols, n_classes
                    )
                st.session_state.enhanced_model = enhanced_model
                st.session_state.enhanced_train_cols = enhanced_cols
                st.session_state.enhanced_val_metrics = enhanced_metrics

                for metric, val in enhanced_metrics.items():
                    if val is not None:
                        base_val = baseline_metrics.get(metric)
                        delta = val - base_val if base_val is not None else None
                        delta_str = f"{delta:+.4f}" if delta is not None else None
                        st.metric(metric.replace('_', ' ').upper(), f"{val:.4f}", delta=delta_str)

            # Train final models on full training data
            with st.spinner("Training final models on full training set..."):
                X_tr_full, X_vl_dummy = prepare_data_for_model(X, X.iloc[:1])
                final_baseline = lgb.LGBMClassifier(**BASE_PARAMS)
                final_baseline.fit(X_tr_full, y)
                st.session_state.baseline_model = final_baseline
                st.session_state.baseline_train_cols = X_tr_full.columns.tolist()

                X_enh_enc, X_vl_dum_enh = prepare_data_for_model(X_full_enh, X_full_enh.iloc[:1])
                final_enhanced = lgb.LGBMClassifier(**BASE_PARAMS)
                final_enhanced.fit(X_enh_enc, y)
                st.session_state.enhanced_model = final_enhanced
                st.session_state.enhanced_train_cols = X_enh_enc.columns.tolist()

            st.success("Both models trained on full training data and ready for test evaluation.")

    # =========================================================================
    # STEP 4: UPLOAD TEST DATA & COMPARE
    # =========================================================================
    if st.session_state.baseline_model is not None and st.session_state.enhanced_model is not None:
        st.divider()
        st.header("④ Upload Test Data & Compare")

        uploaded_test = st.file_uploader("Upload test CSV (with labels)", type=['csv'], key='test_upload')

        if uploaded_test is not None:
            test_df = pd.read_csv(uploaded_test)
            test_df = sanitize_feature_names(test_df)
            target_col = st.session_state.target_col
            le = st.session_state.label_encoder
            n_classes = st.session_state.n_classes

            if target_col not in test_df.columns:
                st.error(f"Target column '{target_col}' not found in test data. "
                         f"Available: {test_df.columns.tolist()}")
            else:
                y_test = pd.Series(le.transform(test_df[target_col].astype(str)), name=target_col)
                X_test = test_df.drop(columns=[target_col])

                st.write(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} columns")

                if st.button("📊 Evaluate on Test Set", type="primary"):
                    # Enhanced test data
                    X_test_enh = apply_fitted_to_test(X_test, st.session_state.fitted_params)

                    with st.spinner("Evaluating..."):
                        baseline_test = evaluate_on_set(
                            st.session_state.baseline_model, X_test, y_test,
                            st.session_state.baseline_train_cols, n_classes
                        )
                        enhanced_test = evaluate_on_set(
                            st.session_state.enhanced_model, X_test_enh, y_test,
                            st.session_state.enhanced_train_cols, n_classes
                        )

                    # --- Results comparison ---
                    st.subheader("Test Set Results")

                    metrics_order = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss']
                    comp_data = []
                    for m in metrics_order:
                        bv = baseline_test.get(m)
                        ev = enhanced_test.get(m)
                        if bv is not None and ev is not None:
                            diff = ev - bv
                            # For log_loss, lower is better
                            better = diff < 0 if m == 'log_loss' else diff > 0
                            comp_data.append({
                                'Metric': m.replace('_', ' ').upper(),
                                'Baseline': f"{bv:.4f}",
                                'Enhanced': f"{ev:.4f}",
                                'Δ': f"{diff:+.4f}",
                                'Winner': '✅ Enhanced' if better else ('⚖️ Tie' if diff == 0 else '⬅️ Baseline'),
                            })

                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

                    # Highlight headline metric
                    b_auc = baseline_test.get('roc_auc')
                    e_auc = enhanced_test.get('roc_auc')
                    if b_auc is not None and e_auc is not None:
                        delta_auc = e_auc - b_auc
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Baseline ROC-AUC", f"{b_auc:.4f}")
                        col2.metric("Enhanced ROC-AUC", f"{e_auc:.4f}", delta=f"{delta_auc:+.4f}")
                        pct = delta_auc / max(1.0 - b_auc, 0.001) * 100
                        col3.metric("Headroom Captured", f"{pct:+.1f}%")

                    # Applied transforms summary
                    with st.expander("Applied Transforms"):
                        for p in st.session_state.fitted_params:
                            col_str = p['column']
                            if p.get('column_b'):
                                col_str += f" × {p['column_b']}"
                            st.write(f"- **{p['method']}** on `{col_str}`")


if __name__ == '__main__':
    main()
