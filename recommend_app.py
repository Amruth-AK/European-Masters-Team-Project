"""
recommend_app.py — Feature Engineering Recommendation Tool
============================================================

Streamlit application that:
1. Accepts a training CSV + target column
2. Analyzes the dataset via trained meta-models
3. Suggests preprocessing transforms ranked by predicted impact
4. Trains two LightGBM models (baseline vs. enhanced)
5. Accepts a test CSV and compares both models

Self-contained — meta-feature computation and transform logic is inline;
constants, transforms, and UI components are in separate modules.
Usage:
    streamlit run recommend_app.py -- --model_dir ./meta_models
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import argparse
import re
import hashlib
from io import StringIO
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
from chat_component import render_chat_sidebar
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, log_loss, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew, kurtosis, shapiro, spearmanr, f_oneway, chi2_contingency
from scipy.stats import entropy as sp_entropy

from app_constants import (
    SENTINEL_NC, SENTINEL_CC,
    DATASET_FEATURES, NUMERICAL_COLUMN_FEATURES, CATEGORICAL_COLUMN_FEATURES,
    INTERACTION_PAIR_FEATURES, ROW_DATASET_FEATURES, ROW_FAMILIES,
    NUMERICAL_METHODS, CATEGORICAL_METHODS,
    INTERACTION_METHODS_NUM_NUM, INTERACTION_METHODS_CAT_NUM, INTERACTION_METHODS_CAT_CAT,
    BASE_PARAMS, METHOD_DESCRIPTIONS,
    _SUGGESTION_GROUPS, _GROUPS_BY_ID, _METHOD_TO_GROUP, _CUSTOM_METHODS,
    _IMBALANCE_MODERATE, _IMBALANCE_SEVERE,
    _IMBALANCE_MULTICLASS_RATIO_CAP, _IMBALANCE_MULTICLASS_DOMINANT,
)
from transforms import (
    ensure_numeric_target,
    detect_dow_columns, detect_text_columns,
    _DOW_TO_INT, _DOW_ALL,
    _apply_date_features, _apply_date_cyclical, _apply_text_stats,
    fit_and_apply_suggestions, apply_fitted_to_test,
    _to_datetime_safe,
)
from ui_components import (
    _suggestion_label, _delta_color, _on_suggest_change,
    _render_group_card, _render_custom_step_adder,
    _sidebar_progress, _render_locked_step,
)




# =============================================================================
# META-FEATURE COMPUTATION (self-contained from collectors)
# =============================================================================



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
    """Encode categoricals for LightGBM.

    Returns:
        X_tr       – encoded training frame (copy)
        X_vl       – encoded validation frame (copy)
        col_encoders – dict mapping col name → {'encoder': LabelEncoder | None,
                                                  'median': float}
                       Pass this to evaluate_on_set so it can replicate the
                       exact same encoding without refitting on evaluation data.

    Safety: if a transform added columns to X_train that are absent from X_val
    (e.g. TF-IDF after a prior text_stats drop), those columns are zero-filled in
    X_vl so the model always sees a consistent feature set.
    """
    X_tr, X_vl = X_train.copy(), X_val.copy()
    # Align val to train: add any missing columns as zeros
    for _c in X_tr.columns:
        if _c not in X_vl.columns:
            X_vl[_c] = 0.0
    # Drop any val-only columns that the model won't expect
    extra_val = [c for c in X_vl.columns if c not in X_tr.columns]
    if extra_val:
        X_vl = X_vl.drop(columns=extra_val)
    X_vl = X_vl[X_tr.columns]  # guarantee same column order
    col_encoders = {}
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]):
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_vl[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_vl[col] = le.transform(X_vl[col].astype(str))
            col_encoders[col] = {'encoder': le, 'median': None}
        else:
            col_encoders[col] = {'encoder': None, 'median': None}
        med = X_tr[col].median() if pd.api.types.is_numeric_dtype(X_tr[col]) else -999
        col_encoders[col]['median'] = float(med)
        X_tr[col] = X_tr[col].fillna(med)
        X_vl[col] = X_vl[col].fillna(med)
    return X_tr, X_vl, col_encoders


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
        from sklearn.model_selection import cross_val_score
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        scores = cross_val_score(dt, X_enc, y_numeric, cv=3, scoring='accuracy')
        meta['landmarking_score'] = float(scores.mean())
    except Exception:
        meta['landmarking_score'] = 0.5

    return meta


def get_row_dataset_meta(X):
    """
    Compute the row-level dataset meta-features that match SCHEMA_ROW in
    collect_row_features.py.  These are dataset-level aggregates (computed once
    per dataset, not per column) and are used as the feature vector for the row
    meta-model.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    n_numeric = len(numeric_cols)
    meta = {'n_numeric_cols_used': n_numeric}

    if n_numeric == 0:
        return {k: 0.0 for k in ROW_DATASET_FEATURES}

    X_num = X[numeric_cols]

    # Column-wise means / stds
    col_means = X_num.mean()
    col_stds = X_num.std().fillna(0)
    meta['avg_numeric_mean'] = float(col_means.mean())
    meta['avg_numeric_std'] = float(col_stds.mean())

    # Missing
    col_miss = X_num.isnull().mean()
    meta['avg_missing_pct'] = float(col_miss.mean())
    meta['max_missing_pct'] = float(col_miss.max())

    # Row-wise variance (numeric cols, fill NaN with column median first)
    X_num_filled = X_num.apply(lambda s: s.fillna(s.median()))
    row_vars = X_num_filled.var(axis=1)
    meta['avg_row_variance'] = float(row_vars.mean())

    # Missing per row (over all cols, not just numeric)
    meta['pct_rows_with_any_missing'] = float((X.isnull().any(axis=1)).mean())

    # Zero statistics (numeric cols)
    zero_mask = (X_num_filled == 0)
    total_cells = X_num_filled.size
    meta['pct_cells_zero'] = float(zero_mask.values.sum() / max(total_cells, 1))
    meta['pct_rows_with_any_zero'] = float((zero_mask.any(axis=1)).mean())

    # Pairwise correlation among numeric cols
    if n_numeric >= 2:
        try:
            corr_mat = X_num_filled.corr().abs().values.copy()
            np.fill_diagonal(corr_mat, 0)
            meta['numeric_col_corr_mean'] = float(corr_mat.mean())
            meta['numeric_col_corr_max'] = float(corr_mat.max())
        except Exception:
            meta['numeric_col_corr_mean'] = 0.0
            meta['numeric_col_corr_max'] = 0.0
    else:
        meta['numeric_col_corr_mean'] = 0.0
        meta['numeric_col_corr_max'] = 0.0

    # Row-wise Shannon entropy (discretise each row into bins, then compute entropy)
    try:
        row_entropies = []
        arr = X_num_filled.values
        for row in arr[:min(len(arr), 2000)]:  # cap at 2000 rows for speed
            counts, _ = np.histogram(row, bins=min(10, n_numeric))
            total = counts.sum()
            probs = counts / total if total > 0 else counts
            probs = probs[probs > 0]
            row_entropies.append(float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0)
        meta['avg_row_entropy'] = float(np.mean(row_entropies)) if row_entropies else 0.0
    except Exception:
        meta['avg_row_entropy'] = 0.0

    # Numeric range ratio: mean(col_max - col_min) / (global_std + 1e-8)
    try:
        col_ranges = X_num_filled.max() - X_num_filled.min()
        global_std = float(X_num_filled.values.std())
        meta['numeric_range_ratio'] = float(col_ranges.mean() / (global_std + 1e-8))
    except Exception:
        meta['numeric_range_ratio'] = 0.0

    return meta


def get_numeric_column_meta(series, y, importance, importance_rank_pct):
    clean = series.dropna().astype(float)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(enc.reshape(-1, 1), y_numeric, random_state=42)[0]
        meta['mutual_info_score'] = float(mi)
    except Exception:
        meta['mutual_info_score'] = 0.0
    try:
        le = LabelEncoder()
        enc = le.fit_transform(series.astype(str).fillna('NaN'))
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        from sklearn.model_selection import cross_val_score as _cvs
        pps = float(_cvs(dt, enc.reshape(-1, 1), y_numeric, cv=3, scoring='accuracy').mean())
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
    """
    Compute order-invariant pair-level meta-features.
    Must stay in sync with collect_interaction_features.py::get_pair_meta_features.

    Split-sentinel encoding:
      SENTINEL_NC = -10  →  num+cat  (n_numerical == 1)
      SENTINEL_CC = -20  →  cat+cat  (n_numerical == 0)
    """
    y_numeric = ensure_numeric_target(y)
    a_num = pd.api.types.is_numeric_dtype(col_a)
    b_num = pd.api.types.is_numeric_dtype(col_b)
    n_numerical = int(a_num) + int(b_num)
    meta = {'n_numerical_cols': n_numerical}

    # Sentinel for features requiring num+num: -10 if one step away, -20 if two steps
    sent_nn = SENTINEL_NC if n_numerical == 1 else SENTINEL_CC

    # ---- Shared pairwise features ----

    if n_numerical == 2:
        try:
            cl = col_a.notna() & col_b.notna()
            meta['pearson_corr'] = float(abs(col_a[cl].corr(col_b[cl]))) if cl.sum() > 10 else 0.0
        except Exception:
            meta['pearson_corr'] = 0.0
        try:
            cl = col_a.notna() & col_b.notna()
            sp_val, _ = spearmanr(col_a[cl], col_b[cl]) if cl.sum() > 10 else (0, 0)
            meta['spearman_corr'] = float(abs(sp_val)) if not np.isnan(sp_val) else 0.0
        except Exception:
            meta['spearman_corr'] = 0.0
    else:
        meta['pearson_corr'] = sent_nn
        meta['spearman_corr'] = sent_nn

    try:
        ea = _encode_for_mi(col_a)
        eb = _encode_for_mi(col_b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(ea.reshape(-1, 1),
                 (eb * 100).astype(int) if b_num else eb, random_state=42)[0]
        meta['mutual_info_pair'] = float(mi)
    except Exception:
        meta['mutual_info_pair'] = 0.0

    try:
        n = len(col_a)
        nb = min(20, max(5, int(n ** 0.4)))
        if a_num:
            ab = pd.qcut(col_a.fillna(col_a.median()), q=nb, labels=False, duplicates='drop').values
        else:
            ab = LabelEncoder().fit_transform(col_a.astype(str).fillna('NaN'))
        if b_num:
            bb = pd.qcut(col_b.fillna(col_b.median()), q=nb, labels=False, duplicates='drop').values
        else:
            bb = LabelEncoder().fit_transform(col_b.astype(str).fillna('NaN'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_b = mutual_info_classif(ab.reshape(-1, 1), bb, discrete_features=True, random_state=42)[0]
        _, cb = np.unique(bb, return_counts=True)
        hb = sp_entropy(cb / cb.sum(), base=2)
        meta['mic_score'] = min(float(mi_b / max(hb, 1e-10)), 1.0) if hb > 0 else 0.0
    except Exception:
        meta['mic_score'] = 0.0

    if n_numerical == 2:
        try:
            sa, sb = col_a.std(), col_b.std()
            meta['scale_ratio'] = float(max(sa, sb) / min(sa, sb)) if min(sa, sb) > 1e-10 else 0.0
        except Exception:
            meta['scale_ratio'] = 0.0
    else:
        meta['scale_ratio'] = sent_nn

    # ---- Combined order-invariant individual stats (all types) ----

    meta['sum_importance'] = float(imp_a + imp_b)
    meta['max_importance'] = float(max(imp_a, imp_b))
    meta['min_importance'] = float(min(imp_a, imp_b))
    na_, nb_ = float(col_a.isnull().mean()), float(col_b.isnull().mean())
    meta['sum_null_pct'] = na_ + nb_
    meta['max_null_pct'] = max(na_, nb_)
    ua = float(col_a.nunique() / max(len(col_a), 1))
    ub = float(col_b.nunique() / max(len(col_b), 1))
    meta['sum_unique_ratio'] = ua + ub
    meta['abs_diff_unique_ratio'] = abs(ua - ub)
    ea_, eb_ = _column_entropy(col_a), _column_entropy(col_b)
    meta['sum_entropy'] = ea_ + eb_
    meta['abs_diff_entropy'] = abs(ea_ - eb_)

    if n_numerical == 2:
        def _tc(s):
            try:
                cl = s.notna() & y_numeric.notna()
                if cl.sum() > 10:
                    sp_val, _ = spearmanr(s[cl], y_numeric[cl])
                    return float(abs(sp_val)) if not np.isnan(sp_val) else 0.0
                return 0.0
            except Exception:
                return 0.0
        ta, tb = _tc(col_a), _tc(col_b)
        meta['sum_target_corr'] = ta + tb
        meta['abs_diff_target_corr'] = abs(ta - tb)
    else:
        meta['sum_target_corr'] = sent_nn
        meta['abs_diff_target_corr'] = sent_nn

    def _mi_t(s):
        try:
            e = _encode_for_mi(s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(mutual_info_classif(e.reshape(-1, 1), y_numeric, random_state=42)[0])
        except Exception:
            return 0.0
    ma, mb = _mi_t(col_a), _mi_t(col_b)
    meta['sum_mi_target'] = ma + mb
    meta['abs_diff_mi_target'] = abs(ma - mb)
    meta['both_binary'] = int(col_a.nunique() <= 2 and col_b.nunique() <= 2)

    # =========================================================================
    # NEW: num+num specific features
    # =========================================================================
    if n_numerical == 2:
        mean_a = float(col_a.mean())
        mean_b = float(col_b.mean())
        std_a  = float(col_a.std())
        std_b  = float(col_b.std())

        meta['product_of_means'] = mean_a * mean_b

        abs_m_a, abs_m_b = abs(mean_a), abs(mean_b)
        meta['abs_mean_ratio'] = max(abs_m_a, abs_m_b) / (min(abs_m_a, abs_m_b) + 1e-8)

        cv_a = std_a / (abs(mean_a) + 1e-8)
        cv_b = std_b / (abs(mean_b) + 1e-8)
        meta['sum_cv']      = cv_a + cv_b
        meta['abs_diff_cv'] = abs(cv_a - cv_b)

        try:
            sk_a = float(skew(col_a.dropna()))
            sk_b = float(skew(col_b.dropna()))
        except Exception:
            sk_a, sk_b = 0.0, 0.0
        meta['sum_skewness']      = abs(sk_a) + abs(sk_b)
        meta['abs_diff_skewness'] = abs(abs(sk_a) - abs(sk_b))

        try:
            cl = col_a.notna() & col_b.notna()
            meta['sign_concordance'] = float(((col_a[cl] >= 0) == (col_b[cl] >= 0)).mean())
        except Exception:
            meta['sign_concordance'] = 0.0

        meta['n_positive_means'] = int(mean_a > 0) + int(mean_b > 0)
    else:
        for feat in ['product_of_means', 'abs_mean_ratio', 'sum_cv', 'abs_diff_cv',
                     'sum_skewness', 'abs_diff_skewness', 'sign_concordance', 'n_positive_means']:
            meta[feat] = sent_nn

    # =========================================================================
    # NEW: num+cat specific features
    # =========================================================================
    if n_numerical == 1:
        num_s = col_a if a_num else col_b
        cat_s = col_b if a_num else col_a

        meta['n_groups'] = int(cat_s.nunique())

        try:
            grand_mean = float(num_s.mean())
            groups = [num_s[cat_s == g].dropna() for g in cat_s.unique()]
            groups = [g for g in groups if len(g) > 0]
            ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups)
            ss_total   = float(((num_s - grand_mean) ** 2).sum())
            meta['eta_squared'] = float(ss_between / (ss_total + 1e-10))
        except Exception:
            meta['eta_squared'] = 0.0

        try:
            groups_data = [num_s[cat_s == g].dropna().values for g in cat_s.unique()]
            groups_data = [g for g in groups_data if len(g) > 1]
            if len(groups_data) >= 2:
                f_stat, _ = f_oneway(*groups_data)
                meta['anova_f_stat'] = float(f_stat) if not np.isnan(f_stat) else 0.0
            else:
                meta['anova_f_stat'] = 0.0
        except Exception:
            meta['anova_f_stat'] = 0.0
    else:
        sent_nc = -10.0 if n_numerical == 2 else -20.0
        for feat in ['eta_squared', 'anova_f_stat', 'n_groups']:
            meta[feat] = sent_nc

    # =========================================================================
    # NEW: cat+cat specific features
    # =========================================================================
    if n_numerical == 0:
        nu_a = int(col_a.nunique())
        nu_b = int(col_b.nunique())

        try:
            ct = pd.crosstab(col_a.astype(str), col_b.astype(str))
            chi2_val, _, _, _ = chi2_contingency(ct)
            n = len(col_a)
            k = min(ct.shape) - 1
            meta['cramers_v'] = min(float(np.sqrt(chi2_val / (n * max(k, 1)))), 1.0)
        except Exception:
            meta['cramers_v'] = 0.0

        meta['joint_cardinality'] = float(nu_a * nu_b)
        meta['cardinality_ratio'] = float(min(nu_a, nu_b) / (max(nu_a, nu_b) + 1e-10))

        try:
            ct_vals = pd.crosstab(col_a.astype(str), col_b.astype(str)).values
            actual_combos = int((ct_vals > 0).sum())
            theoretical   = nu_a * nu_b
            meta['joint_sparsity'] = float(1.0 - actual_combos / max(theoretical, 1))
        except Exception:
            meta['joint_sparsity'] = 0.0
    else:
        sent_cc = -10.0 if n_numerical == 1 else -20.0
        for feat in ['cramers_v', 'joint_cardinality', 'cardinality_ratio', 'joint_sparsity']:
            meta[feat] = sent_cc

    return meta


# =============================================================================
# METHOD APPLICABILITY GATES
# =============================================================================

def should_test_numerical(method, col_meta, series):
    if method == 'impute_median' and col_meta['null_pct'] == 0: return False
    if method == 'missing_indicator' and col_meta['null_pct'] == 0: return False
    # High-missing columns: only impute/missing-indicator make sense
    if col_meta['null_pct'] > 0.50 and method not in ('impute_median', 'missing_indicator'):
        return False
    if method == 'sqrt_transform':
        clean = series.dropna()
        if len(clean) > 0 and clean.min() < 0: return False
        if col_meta['is_binary']: return False
    if method == 'log_transform' and col_meta['is_binary']: return False
    if method == 'quantile_binning':
        if col_meta['unique_ratio'] < 0.01: return False
        if col_meta['is_binary']: return False
    if method in ('polynomial_square', 'polynomial_cube', 'reciprocal_transform'):
        if col_meta['is_binary']: return False
    return True


def should_test_categorical(method, col_meta):
    nu = col_meta['n_unique']
    if method == 'missing_indicator' and col_meta['null_pct'] == 0: return False
    if method == 'onehot_encoding' and (nu < 2 or nu > 10): return False
    if method == 'hashing_encoding' and nu <= 10: return False
    return True


def detect_problematic_columns(X, known_date_cols=None, known_text_cols=None):
    """
    Detect columns that should be excluded from (or handled carefully in) suggestions.

    Parameters
    ----------
    known_date_cols : set of column names already identified as date/datetime/time columns.
    known_text_cols : set of column names already identified as free-form text columns.

    Returns a dict:
        'id_columns':          {col: reason}  — likely identifiers; skip entirely
        'constant_columns':    {col: reason}  — zero-variance; skip entirely
        'binary_num_columns':  {col: reason}  — binary numerics; already gated in
                                                should_test_numerical, surfaced for UI info
        'high_missing_columns':{col: reason}  — >50% null; only impute/missing-indicator allowed

    Detection logic
    ---------------
    Constant: nunique <= 1 (after dropping NaN).

    ID-like  (numeric columns only, n_unique > 10):
      • unique_ratio >= 0.95 AND near-constant step size (step CV < 5 %) → sequential ID
      • unique_ratio >= 0.80 AND column name matches an ID-name pattern
    String columns: only flagged as ID if the name looks like an identifier.
      Date columns and free-text columns are never flagged as IDs.

    Binary numeric: nunique == 2 (not yet a constant / ID column).

    High-missing: null_pct > 0.50.
    """
    known_date_cols = set(known_date_cols or {})
    known_text_cols = set(known_text_cols or {})
    id_cols = {}
    constant_cols = {}
    binary_num_cols = {}
    high_missing_cols = {}

    n = len(X)

    # Name patterns that suggest an identifier column.
    ID_EXACT = {'id', 'idx', 'index', 'key', 'no', 'num', 'number', 'row', '#'}
    ID_SUBSTRINGS = ['_id', 'id_', '_idx', 'idx_', '_index', 'index_',
                     '_key', 'key_', '_no', '_num', '_number', 'rownum', 'row_num',
                     'record_id', 'sample_id', 'obs_id', 'entry_id']

    def _name_looks_like_id(col):
        c = col.lower().strip()
        if c in ID_EXACT:
            return True
        return any(sub in c for sub in ID_SUBSTRINGS)

    for col in X.columns:
        s = X[col]
        n_unique = s.nunique(dropna=True)
        non_null = s.dropna()
        null_pct = s.isnull().mean()

        # ── Constant ────────────────────────────────────────────────────────
        if n_unique <= 1:
            reason = "all values are null" if n_unique == 0 else "all values are identical"
            constant_cols[col] = reason
            continue

        # ── ID-like ─────────────────────────────────────────────────────────
        unique_ratio = n_unique / max(n, 1)

        if pd.api.types.is_numeric_dtype(s) and n_unique > 10:
            is_id = False
            id_reason = None

            # Check 1: strictly monotonic with all-unique values (row-order index).
            # This catches a 1,2,3,4,… column even when unique_ratio is slightly
            # below 0.95 (e.g. a few gaps) or the step size is irregular.
            try:
                float_vals = non_null.astype(float)
                if (n_unique == len(non_null) and
                        (float_vals.is_monotonic_increasing or
                         float_vals.is_monotonic_decreasing)):
                    is_id = True
                    id_reason = (
                        f"monotonic all-unique sequence (index-like), "
                        f"unique_ratio={unique_ratio:.2f}"
                    )
            except Exception:
                pass

            if not is_id and unique_ratio >= 0.95:
                # Check 2: near-sequential step pattern (uniform steps, any ordering)
                try:
                    sorted_vals = np.sort(non_null.values.astype(float))
                    diffs = np.diff(sorted_vals)
                    if len(diffs) > 0 and diffs.mean() != 0:
                        step_cv = diffs.std() / abs(diffs.mean())
                        if step_cv < 0.05:           # <5 % coefficient of variation
                            is_id = True
                            id_reason = (
                                f"sequential values — unique_ratio={unique_ratio:.2f}, "
                                f"step≈{diffs.mean():.2g} (step CV={step_cv:.3f})"
                            )
                except Exception:
                    pass

                if not is_id and _name_looks_like_id(col):
                    is_id = True
                    id_reason = (
                        f"ID-like column name with near-unique values "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )

            elif unique_ratio >= 0.80 and _name_looks_like_id(col):
                is_id = True
                id_reason = (
                    f"ID-like column name with high uniqueness "
                    f"(unique_ratio={unique_ratio:.2f})"
                )

            if is_id:
                id_cols[col] = id_reason
                continue

        elif not pd.api.types.is_numeric_dtype(s):
            # Skip columns we've already classified as dates or free-text —
            # those have high unique_ratio by nature but are not identifiers.
            if col in known_date_cols or col in known_text_cols:
                pass  # fall through to high_missing / binary checks
            else:
                # For string columns, only flag as ID when BOTH the name looks like
                # an identifier AND the column has near-unique values.
                # A product description or review with unique_ratio=1.0 should NOT be
                # flagged here — it should be handled by the text-features pipeline.
                avg_len = float(s.dropna().astype(str).str.len().mean()) if len(s.dropna()) > 0 else 0
                looks_like_text = avg_len > 25  # long values → likely text, not an ID
                if unique_ratio >= 0.95 and _name_looks_like_id(col) and not looks_like_text:
                    id_cols[col] = (
                        f"string column with near-unique values and ID-like name "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )
                    continue
                elif unique_ratio >= 0.80 and _name_looks_like_id(col) and not looks_like_text:
                    id_cols[col] = (
                        f"ID-like column name with high uniqueness "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )
                    continue

        # ── High-missing ─────────────────────────────────────────────────────
        if null_pct > 0.50:
            high_missing_cols[col] = (
                f"{null_pct*100:.0f}% missing — only impute/missing-indicator transforms considered"
            )

        # ── Binary numeric ───────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(s) and n_unique == 2:
            binary_num_cols[col] = "binary numeric — non-trivial transforms skipped"

    return {
        'id_columns': id_cols,
        'constant_columns': constant_cols,
        'binary_num_columns': binary_num_cols,
        'high_missing_columns': high_missing_cols,
    }


# =============================================================================
# COLUMN TYPE DETECTION — UNIFIED + OVERRIDE SUPPORT
# =============================================================================

_COL_TYPE_ICONS = {
    'Numerical': '🔢', 'Categorical': '🏷️', 'Binary': '⚡',
    'Date': '📅', 'DateTime': '🕐', 'Time only': '⏱️',
    'Day-of-Week': '📆', 'Free Text': '📝', 'ID': '🆔', 'Constant': '⚫',
}


def _override_options_for(info):
    """Return the valid override options for a column based on its dtype.
    Numeric columns cannot be temporal or free text — those options are hidden.
    """
    base = ['Auto', 'Numerical', 'Categorical', 'Binary']
    if not info.get('is_numeric', False):
        base += ['Date', 'DateTime', 'Time only', 'Day-of-Week', 'Free Text']
    return base


def get_column_type_info(X):
    """Run all detectors; return per-column summary dict."""
    date_col_map = detect_date_columns(X)
    dow_cols     = detect_dow_columns(X, already_date_cols=set(date_col_map))
    text_col_map = detect_text_columns(X, date_cols=date_col_map, dow_cols=dow_cols)
    skipped      = detect_problematic_columns(
        X,
        known_date_cols=set(date_col_map) | dow_cols,
        known_text_cols=set(text_col_map),
    )
    result = {}
    for col in X.columns:
        s        = X[col]
        nu       = int(s.nunique(dropna=True))
        mp       = float(s.isnull().mean())
        non_null = s.dropna()
        sample   = str(non_null.iloc[0])[:50] if len(non_null) > 0 else ''
        drop_suggested, drop_reason = False, ''

        if col in skipped['constant_columns']:
            detected = 'Constant'; drop_suggested = True
            drop_reason = skipped['constant_columns'][col]
        elif col in skipped['id_columns']:
            detected = 'ID'; drop_suggested = True
            drop_reason = skipped['id_columns'][col]
        elif col in date_col_map:
            ct = date_col_map[col]['col_type']
            detected = {'datetime': 'DateTime', 'date': 'Date', 'time': 'Time only'}.get(ct, 'Date')
        elif col in dow_cols:
            detected = 'Day-of-Week'
        elif col in text_col_map:
            detected = 'Free Text'
        elif pd.api.types.is_numeric_dtype(s):
            detected = 'Binary' if nu <= 2 else 'Numerical'
        else:
            detected = 'Binary' if nu <= 2 else 'Categorical'

        if not drop_suggested and mp > 0.80:
            drop_suggested = True; drop_reason = f"{mp*100:.0f}% values missing"

        result[col] = {
            'detected': detected, 'icon': _COL_TYPE_ICONS.get(detected, ''),
            'n_unique': nu, 'missing_pct': mp, 'sample': sample,
            'drop_suggested': drop_suggested, 'drop_reason': drop_reason,
            'is_numeric': bool(pd.api.types.is_numeric_dtype(s)),
        }
    return result


def _validate_col_override(col, override_type, info, X):
    """Return (severity, message) if override looks wrong, else None."""
    if override_type in ('Auto',):
        return None
    is_num = info.get('is_numeric', False)
    nu     = info.get('n_unique', 0)
    s      = X[col].dropna().astype(str)

    if override_type in ('Date', 'DateTime', 'Time only') and len(s):
        try:
            pr = float(pd.to_datetime(s, errors='coerce', format='mixed').notna().mean())
            if pr < 0.50:
                return ('warning', f"Only {pr*100:.0f}% of values parse as dates — are you sure?")
        except Exception:
            pass
    if override_type == 'Numerical' and not is_num:
        try:
            pr = float(pd.to_numeric(X[col], errors='coerce').notna().mean())
            if pr < 0.70:
                return ('error', f"Only {pr*100:.0f}% of values are numeric — use Categorical instead.")
            if pr < 0.95:
                return ('warning', f"{(1-pr)*100:.0f}% of values will become NaN.")
        except Exception:
            return ('error', "Values don't appear to be numeric.")
    if override_type == 'Free Text' and not is_num and len(s):
        avg_len = float(s.str.len().mean())
        if avg_len < 10:
            return ('warning', f"Average value length is {avg_len:.1f} chars — looks more like Categorical.")
    if override_type == 'Day-of-Week' and not is_num and len(s):
        recognised = {v.strip().lower() for v in s.unique()} & _DOW_ALL
        pct = len(recognised) / max(s.nunique(), 1)
        if pct < 0.50:
            return ('warning', f"Only {pct*100:.0f}% of unique values match weekday names.")
    if override_type == 'Binary' and nu > 2:
        return ('error', f"Column has {nu} unique values — Binary requires ≤ 2.")
    if override_type == 'Categorical' and nu > 500:
        return ('warning', f"{nu} unique values is very high — consider Free Text or Drop.")
    return None


def _apply_type_reassignments(X, type_reassignments, date_col_map, text_col_map,
                               dow_cols, skipped_info):
    """Apply non-Drop type overrides to detection dicts."""
    _DTYPE_MAP = {'Date': 'date', 'DateTime': 'datetime', 'Time only': 'time'}
    date_col_map = dict(date_col_map); text_col_map = dict(text_col_map)
    dow_cols = set(dow_cols)
    id_cols  = dict(skipped_info.get('id_columns', {}))
    const_cols = dict(skipped_info.get('constant_columns', {}))
    for col, t in type_reassignments.items():
        if col not in X.columns:
            continue
        date_col_map.pop(col, None); text_col_map.pop(col, None)
        dow_cols.discard(col); id_cols.pop(col, None); const_cols.pop(col, None)
        if t in _DTYPE_MAP:
            date_col_map[col] = {'parse_rate': 1.0, 'col_type': _DTYPE_MAP[t]}
        elif t == 'Free Text':
            text_col_map[col] = 'user-specified'
        elif t == 'Day-of-Week':
            dow_cols.add(col)
    skipped_info = dict(skipped_info)
    skipped_info['id_columns'] = id_cols; skipped_info['constant_columns'] = const_cols
    return X, date_col_map, text_col_map, dow_cols, skipped_info




# =============================================================================
# DATE & TEXT COLUMN DETECTION
# =============================================================================

def detect_date_columns(X):
    """
    Return a dict of col -> {'parse_rate', 'col_type'} for temporal columns.

    col_type is one of:
      'datetime'  -- carries both a meaningful date and a time component
      'date'      -- date only (year/month/day varies, hour is always midnight)
      'time'      -- time only (values like "08:47"; the date part is a constant
                    default injected by pd.to_datetime, not real date info)

    Skips numeric columns. Requires >= 70% parse rate.
    """
    date_cols = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        s = X[col].dropna().astype(str)
        if len(s) < 5:
            continue
        try:
            parsed = _to_datetime_safe(s)
            parse_rate = float(parsed.notna().mean())
            if parse_rate < 0.70:
                continue

            valid = parsed.dropna()

            # Has time component?
            has_hour = float((valid.dt.hour != 0).mean()) >= 0.20

            # Has real date component?
            # When the input is time-only (e.g. "08:47"), pd.to_datetime fills in
            # today's date for every row, so the date part has <= 1 unique value.
            # A genuine date column has many distinct dates.
            n_unique_dates = valid.dt.normalize().nunique()
            has_date = n_unique_dates >= max(3, int(len(valid) * 0.02))

            if has_hour and not has_date:
                col_type = 'time'
            elif has_date and has_hour:
                col_type = 'datetime'
            else:
                col_type = 'date'

            date_cols[col] = {'parse_rate': parse_rate, 'col_type': col_type}
        except Exception:
            pass
    return date_cols


def detect_date_has_hour(X, col):
    """
    Return True if a detected date column appears to carry a time component
    (i.e. at least 20 % of parsed values have a non-zero hour).
    Kept for backwards-compat; new code uses detect_date_columns col_type.
    """
    try:
        parsed = pd.to_datetime(X[col].dropna().astype(str), errors='coerce', format='mixed').dropna()
        if len(parsed) == 0:
            return False
        return float((parsed.dt.hour != 0).mean()) >= 0.20
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Day-of-week categorical column detection
# ---------------------------------------------------------------------------



def generate_dataset_advisories(X, y):
    """
    Inspect dataset-level properties and return a list of advisory dicts.
    Each advisory has: {'category', 'severity', 'title', 'detail', 'suggested_params'}

    Current checks
    --------------
    1. Very small dataset (<500 rows) → suggest min_child_samples / num_leaves reduction
    2. Very wide dataset (more features than rows) → suggest colsample_bytree / L1 reg
    3. High global missing rate (>20 %) → suggest imputation transforms first

    Note: class imbalance is handled by the dedicated suggestion group (⚖️ Class Imbalance),
    not as an advisory here.
    """
    advisories = []
    n_rows, n_cols = X.shape
    missing_rate = float(X.isnull().mean().mean())

    # ── 1. Very small dataset ────────────────────────────────────────────────
    if n_rows < 500:
        advisories.append({
            'category': 'Small Dataset',
            'severity': 'medium',
            'title': f"Small dataset ({n_rows} rows) — consider reducing model complexity",
            'detail': (
                f"With only {n_rows} rows, default LightGBM settings risk overfitting. "
                f"The suggested parameters below constrain tree growth and add regularisation."
            ),
            'suggested_params': {
                'num_leaves': 15,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
            },
        })

    # ── 2. Very wide dataset (n_features >> n_rows) ──────────────────────────
    feature_cols = X.shape[1]
    if feature_cols > n_rows * 0.5 and feature_cols > 50:
        advisories.append({
            'category': 'High Dimensionality',
            'severity': 'medium',
            'title': f"Wide dataset ({feature_cols} features, {n_rows} rows) — risk of overfitting",
            'detail': (
                f"The dataset has {feature_cols} features vs {n_rows} rows. "
                f"Reduce `colsample_bytree` (0.5–0.7) and add L1 regularisation "
                f"(`reg_alpha`) to encourage sparsity. Consider feature selection "
                f"before adding interaction or polynomial features."
            ),
            'suggested_params': {
                'colsample_bytree': 0.5,
                'reg_alpha': 0.5,
                'min_child_samples': 20,
            },
        })

    # ── 3. High global missing rate ──────────────────────────────────────────
    if missing_rate > 0.20:
        advisories.append({
            'category': 'High Missingness',
            'severity': 'low',
            'title': f"High global missing rate ({missing_rate*100:.0f}%) — prioritise imputation transforms",
            'detail': (
                f"{missing_rate*100:.0f}% of all cells are missing. "
                f"Prioritise `impute_median` and `missing_indicator` suggestions. "
                f"LightGBM handles NaN natively, but explicit imputation can still help "
                f"downstream feature engineering (e.g. interactions involving NaN columns)."
            ),
            'suggested_params': None,
        })

    return advisories


# =============================================================================
# META-MODEL LOADING & PREDICTION
# =============================================================================

@st.cache_resource
def load_meta_models(model_dir):
    """Load all available meta-models from disk."""
    models = {}
    for ctype in ['numerical', 'categorical', 'interaction', 'row']:
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


def _is_near_bijection(col_a, col_b, threshold=0.95):
    """
    Return True if col_a and col_b encode the same entity under different schemes
    (i.e. they have a near-1:1 mapping in both directions).

    Examples that should be blocked:
      - Recipe Code  ↔  Recipe Number  (different scales, same recipe)
      - Product ID   ↔  Product Name   (numeric vs. string, same product)

    These pairs pass Pearson/Spearman filters because the ordering is unrelated,
    but interactions between them carry zero new information.

    Algorithm
    ---------
    1. Require comparable cardinalities (within 20 % of each other).
    2. Check that ≥ threshold fraction of values in A map to exactly one value of B.
    3. Check the same in reverse (B → A).
    Both directions must hold to conclude a bijection.
    """
    try:
        n_unique_a = col_a.nunique(dropna=True)
        n_unique_b = col_b.nunique(dropna=True)

        # Skip trivially-low-cardinality columns (binary, near-constant)
        if n_unique_a < 3 or n_unique_b < 3:
            return False

        # Cardinalities must be within 20 % of each other
        ratio = min(n_unique_a, n_unique_b) / max(n_unique_a, n_unique_b)
        if ratio < 0.80:
            return False

        tmp = pd.DataFrame({'a': col_a.values, 'b': col_b.values}).dropna()
        if len(tmp) < 10:
            return False

        # A → B: how many distinct B values does each A value map to?
        b_per_a = tmp.groupby('a')['b'].nunique()
        pct_a_to_one_b = float((b_per_a == 1).mean())

        # B → A
        a_per_b = tmp.groupby('b')['a'].nunique()
        pct_b_to_one_a = float((a_per_b == 1).mean())

        return pct_a_to_one_b >= threshold and pct_b_to_one_a >= threshold
    except Exception:
        return False


def generate_suggestions(X, y, meta_models, baseline_score, baseline_std,
                         progress_cb=None, type_reassignments=None,
                         real_n_rows=None):
    """
    Run meta-models on all applicable (column, method) combinations.
    Drops are applied to X before calling; type_reassignments are applied here.

    real_n_rows: when X is a subsample (quick mode), pass the full dataset's
        row count so meta-models receive accurate n_rows / row_col_ratio and
        are not misled into thinking the dataset is small.
    """
    y_numeric = ensure_numeric_target(y)
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = baseline_score
    ds_meta['baseline_std'] = baseline_std
    ds_meta['relative_headroom'] = max(1.0 - baseline_score, 0.001)

    # ── Patch shape-based fields when working on a subsample ────────────────
    # get_dataset_meta computed n_rows / row_col_ratio from the (small) X it
    # received.  The meta-models were trained on full datasets, so feeding them
    # sample-derived shape features would make every quick-mode run look like a
    # small-dataset problem, skewing all predictions.  We overwrite just these
    # two fields with the real numbers while keeping every other stat
    # (correlations, MI scores, etc.) from the sample — those are intentionally
    # approximate and that's fine.
    if real_n_rows is not None and real_n_rows > len(X):
        ds_meta['n_rows']        = real_n_rows
        ds_meta['row_col_ratio'] = real_n_rows / max(X.shape[1], 1)
    # ─────────────────────────────────────────────────────────────────────────

    importances = get_baseline_importances(X, y)
    imp_ranks = importances.rank(ascending=False, pct=True)

    # ── Pre-detect date and text columns ────────────────────────────────────
    date_col_map     = detect_date_columns(X)
    dow_cols_pre     = detect_dow_columns(X, already_date_cols=set(date_col_map))
    text_col_map_pre = detect_text_columns(X, date_cols=date_col_map, dow_cols=dow_cols_pre)

    # ── Column-level safeguards ──────────────────────────────────────────────
    skipped_info = detect_problematic_columns(
        X,
        known_date_cols=set(date_col_map) | dow_cols_pre,
        known_text_cols=set(text_col_map_pre),
    )

    # ── Apply user type reassignments ────────────────────────────────────────
    if type_reassignments:
        X, date_col_map, text_col_map_pre, dow_cols_pre, skipped_info = (
            _apply_type_reassignments(
                X, type_reassignments,
                date_col_map, text_col_map_pre, dow_cols_pre, skipped_info,
            )
        )

    fully_skip = set(skipped_info['id_columns']) | set(skipped_info['constant_columns'])

    suggestions = []
    total_steps = 0
    done_steps = 0
    _leakage_col_metas = {}  # col -> col_meta dict, for leakage detection

    # Count work — only over columns we will actually process
    numeric_cols = [c for c in X.columns
                    if pd.api.types.is_numeric_dtype(X[c]) and c not in fully_skip]
    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c]) and c not in fully_skip]
    if 'numerical' in meta_models: total_steps += len(numeric_cols) * len(NUMERICAL_METHODS)
    if 'categorical' in meta_models: total_steps += len(cat_cols) * len(CATEGORICAL_METHODS)
    if 'interaction' in meta_models: total_steps += 50  # rough estimate
    if 'row' in meta_models: total_steps += len(ROW_FAMILIES)
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
            _leakage_col_metas[col] = col_meta
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
                    'meta': {k: col_meta.get(k) for k in [
                        'null_pct', 'skewness', 'pct_negative', 'outlier_ratio',
                        'zeros_ratio', 'coeff_variation', 'unique_ratio', 'is_binary',
                    ]},
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
            _leakage_col_metas[col] = col_meta
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
                    'meta': {k: col_meta.get(k) for k in [
                        'null_pct', 'n_unique', 'unique_ratio', 'is_high_cardinality',
                        'is_low_cardinality', 'rare_category_pct', 'top_category_dominance',
                    ]},
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
        _num_num_done = False
        for i, a in enumerate(sorted_num):
            if _num_num_done:
                break
            for b in sorted_num[i+1:]:
                if len(pairs) >= 30:
                    _num_num_done = True
                    break
                if _is_near_bijection(X[a], X[b]): continue
                pairs.append((a, b, INTERACTION_METHODS_NUM_NUM))
        # cat+num
        cnt = 0
        for cat_c in sorted_cat:
            for num_c in sorted_num:
                if cnt >= 20: break
                if _is_near_bijection(X[cat_c], X[num_c]): continue
                pairs.append((cat_c, num_c, INTERACTION_METHODS_CAT_NUM))
                cnt += 1
        # cat+cat
        cnt = 0
        for i, a in enumerate(sorted_cat):
            for b in sorted_cat[i+1:]:
                if cnt >= 10: break
                if X[a].nunique() > len(X)*0.5 or X[b].nunique() > len(X)*0.5:
                    continue
                if _is_near_bijection(X[a], X[b]): continue
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
                    # col_b is always the denominator for num+num pairs (the only
                    # type that reaches this method); be explicit to avoid confusion.
                    denom = X[col_b]
                    if (denom.fillna(0) == 0).any(): continue

                if method in ('group_mean', 'group_std'):
                    if a_num == b_num: continue   # need one cat, one num
                    # Skip if the categorical column is nearly unique (groups would be singletons)
                    cat_col = col_a if not a_num else col_b
                    cat_unique_ratio = X[cat_col].nunique() / max(len(X), 1)
                    if cat_unique_ratio > 0.30: continue

                if method == 'cat_concat':
                    # Skip if joint cardinality would be huge (> 50% of rows or > 500 combos)
                    joint_card = X[col_a].nunique() * X[col_b].nunique()
                    if joint_card > min(len(X) * 0.5, 500): continue

                # Skip product/addition/abs_diff when both numerics are binary
                # (result has at most 3–4 distinct values, not meaningful for these ops)
                if method in ('product_interaction', 'addition_interaction', 'abs_diff_interaction'):
                    if a_num and b_num:
                        if X[col_a].nunique() <= 2 and X[col_b].nunique() <= 2: continue

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
                    'meta': {k: pair_meta.get(k) for k in [
                        'pearson_corr', 'spearman_corr', 'mutual_info_pair',
                        'imp_a', 'imp_b',
                    ]},
                })

    # --- Row suggestions ---
    # One prediction per applicable family (method IS the family name).
    if 'row' in meta_models:
        mm = meta_models['row']
        row_meta = get_row_dataset_meta(X)
        combined_meta = {**ds_meta, **row_meta}
        numeric_cols_for_row = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        has_zeros = any(
            (X[c].fillna(0) == 0).any() for c in numeric_cols_for_row
        ) if numeric_cols_for_row else False
        has_missing = X.isnull().any().any()

        for family in ROW_FAMILIES:
            done_steps += 1
            if progress_cb: progress_cb(min(done_steps / total_steps, 1.0))
            if family not in mm['method_vocab']:
                continue

            # Applicability gates (mirror collect_row_features.py requirements)
            if family == 'row_numeric_stats' and len(numeric_cols_for_row) < 2:
                continue
            if family == 'row_zero_stats' and (len(numeric_cols_for_row) < 2 or not has_zeros):
                continue
            if family == 'row_missing_stats' and not has_missing:
                continue

            fv = build_feature_vector(combined_meta, family, mm['config'])
            pred = mm['booster'].predict(fv)[0]
            suggestions.append({
                'type': 'row',
                'column': '(all numeric cols)',
                'column_b': None,
                'method': family,
                'predicted_delta': float(pred),
                'description': METHOD_DESCRIPTIONS.get(family, family),
                'meta': {k: row_meta.get(k) for k in [
                    'avg_missing_pct', 'max_missing_pct', 'pct_cells_zero',
                    'pct_rows_with_any_missing', 'pct_rows_with_any_zero',
                ]},
            })

    # ── Pin all applicable row families so they always survive the threshold ──
    # z-score normalisation can push row family deltas below 0 even when they
    # genuinely help.  Any family that passes the applicability gate is shown
    # regardless of predicted delta — the user can always uncheck them.
    if 'row' in meta_models:
        _row_applicable = {
            'row_numeric_stats': len(numeric_cols_for_row) >= 2,
            'row_zero_stats':    len(numeric_cols_for_row) >= 2 and has_zeros,
            'row_missing_stats': has_missing,
        }
        for _fam, _applicable in _row_applicable.items():
            if not _applicable:
                continue
            already_in = any(s['method'] == _fam for s in suggestions)
            if not already_in:
                try:
                    _row_meta_pin = get_row_dataset_meta(X)
                    _combined_pin = {**ds_meta, **_row_meta_pin}
                    _mm_pin       = meta_models['row']
                    _fv_pin       = build_feature_vector(_combined_pin, _fam, _mm_pin['config'])
                    _raw_pin      = float(_mm_pin['booster'].predict(_fv_pin)[0])
                except Exception:
                    _raw_pin = 0.0
                suggestions.append({
                    'type': 'row', 'column': '(all numeric cols)', 'column_b': None,
                    'method': _fam,
                    'predicted_delta':     0.5,   # above any default threshold
                    'predicted_delta_raw': _raw_pin,
                    'description': METHOD_DESCRIPTIONS.get(_fam, _fam),
                    'pinned': True,
                })
            else:
                for s in suggestions:
                    if s['method'] == _fam:
                        s['pinned'] = True
                        if s['predicted_delta'] < 0:
                            s['predicted_delta'] = 0.0
    # ─────────────────────────────────────────────────────────────────────────

    # ── Inject date-column suggestions ───────────────────────────────────────
    # (date_col_map and dow_cols already computed above before safeguard detection)

    # Per-component cyclical encoding descriptions + guidance
    _CYCLICAL_COMPONENT_META = {
        'date_cyclical_month': {
            'label':  '📅 Cyclic month',
            'icon':   '🔵',
            'hint':   (
                "**When to enable:** your target follows a seasonal calendar pattern "
                "(retail sales, energy demand, weather-driven behaviour, tax deadlines). "
                "Wrapping month 1–12 into sin/cos means the model sees Dec and Jan as "
                "neighbours rather than opposites. "
                "**Skip if** you only need to identify *which* month something happened "
                "and there is no wrap-around seasonality."
            ),
        },
        'date_cyclical_dow': {
            'label':  '📅 Cyclic day-of-week',
            'icon':   '🟢',
            'hint':   (
                "**When to enable:** weekly rhythms matter — footfall, support-ticket "
                "volume, fraud rates, or any pattern where Sunday and Monday are "
                "behaviourally adjacent. "
                "**Skip if** the ordinal weekday identity is more important than "
                "its position in the weekly cycle."
            ),
        },
        'date_cyclical_dom': {
            'label':  '📅 Cyclic day-of-month',
            'icon':   '🟡',
            'hint':   (
                "**When to enable:** billing cycles, salary payment dates, or any "
                "month-end / month-start effect where the 31st and the 1st should be "
                "treated as adjacent. "
                "**Skip if** you only need the raw day number and there is no "
                "meaningful wrap-around between month boundaries."
            ),
        },
        'date_cyclical_hour': {
            'label':  '📅 Cyclic hour',
            'icon':   '🟠',
            'hint':   (
                "**When to enable:** intra-day patterns where 23:00 and 00:00 are "
                "adjacent — rush-hour traffic, overnight fraud, shift-based workloads, "
                "or any circadian rhythm. "
                "**Skip if** the absolute hour is sufficient or if the column contains "
                "only date (no time) information."
            ),
        },
    }

    for col, col_info in date_col_map.items():
        parse_rate = col_info['parse_rate']
        col_type   = col_info['col_type']   # 'date' | 'time' | 'datetime'

        # Base date extraction — always recommended, checked by default
        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'date_features',
            'col_type':            col_type,        # 'date' | 'time' | 'datetime'
            'predicted_delta':     0.50,           # sentinel — always shown
            'predicted_delta_raw': 0.0,
            'description':         (
                (
                    "Time extraction — hour"
                    if col_type == 'time'
                    else METHOD_DESCRIPTIONS['date_features']
                ) + f" (parse rate {parse_rate:.0%})"
            ),
            'pinned':       True,
            'auto_checked': True,
        })

        # Gate cyclical components on what the column actually contains:
        #   'date'     → month, dow, dom  (no hour — it's midnight everywhere)
        #   'time'     → hour only        (no date info; month/dow/dom are meaningless)
        #   'datetime' → month, dow, dom, hour
        if col_type == 'time':
            components_to_offer = ['date_cyclical_hour']
        elif col_type == 'date':
            components_to_offer = ['date_cyclical_month', 'date_cyclical_dow', 'date_cyclical_dom']
        else:  # datetime
            components_to_offer = [
                'date_cyclical_month', 'date_cyclical_dow',
                'date_cyclical_dom',   'date_cyclical_hour',
            ]

        # delta sentinels: slightly below base so they sort after date_features
        for rank, comp_method in enumerate(components_to_offer, start=1):
            cmeta = _CYCLICAL_COMPONENT_META[comp_method]
            suggestions.append({
                'type':                'date',
                'column':              col,
                'column_b':            None,
                'method':              comp_method,
                'predicted_delta':     0.49 - rank * 0.001,
                'predicted_delta_raw': 0.0,
                'description':         (
                    f"{METHOD_DESCRIPTIONS[comp_method]}  ·  {cmeta['hint']}"
                ),
                'pinned':              False,
                'auto_checked':        False,
            })

    # ── Inject day-of-week categorical column suggestions ─────────────────────
    dow_cols = dow_cols_pre
    # Mark them so standard categorical suggestions are suppressed below
    fully_skip |= dow_cols
    for col in dow_cols:
        # Ordinal (0=Mon … 6=Sun) — always recommended
        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'dow_ordinal',
            'predicted_delta':     0.50,
            'predicted_delta_raw': 0.0,
            'description':         (
                "Day-of-week ordinal encoding — maps Mon→0, Tue→1, … Sun→6. "
                "Gives the model a numeric representation that preserves weekday order."
            ),
            'pinned':       True,
            'auto_checked': True,
        })
        # Cyclical encoding — optional
        cmeta = _CYCLICAL_COMPONENT_META['date_cyclical_dow']
        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'dow_cyclical',
            'predicted_delta':     0.49,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"Day-of-week cyclical encoding — sin/cos of weekday (0–6); "
                f"keeps Sunday and Monday adjacent.  ·  {cmeta['hint']}"
            ),
            'pinned':              False,
            'auto_checked':        False,
        })
    # ─────────────────────────────────────────────────────────────────────────

    # ── Inject text-column suggestions ────────────────────────────────────────
    text_col_map = text_col_map_pre
    _TEXT_STAT_FIELDS = [
        ('word_count',      'Word count'),
        ('char_count',      'Char count'),
        ('avg_word_len',    'Avg word length'),
        ('uppercase_ratio', 'Uppercase %'),
        ('digit_ratio',     'Digit %'),
        ('punct_ratio',     'Punctuation %'),
    ]
    for col, reason in text_col_map.items():
        # text_stats: inject a parent suggestion that carries sub-field info
        suggestions.append({
            'type':                'text',
            'column':              col,
            'column_b':            None,
            'method':              'text_stats',
            'predicted_delta':     0.5,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"{METHOD_DESCRIPTIONS['text_stats']} — {reason}"
            ),
            'pinned': True,
            'text_stat_fields': [f for f, _ in _TEXT_STAT_FIELDS],  # all selected by default
        })
        suggestions.append({
            'type':                'text',
            'column':              col,
            'column_b':            None,
            'method':              'text_tfidf',
            'predicted_delta':     0.5,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"{METHOD_DESCRIPTIONS['text_tfidf']} — {reason}"
            ),
            'pinned': True,
        })
    # ─────────────────────────────────────────────────────────────────────────

    # ── Inject imbalance suggestion when class ratio is high ──────────────────
    y_counts_adv       = pd.Series(y_numeric).value_counts()
    imbalance_ratio_adv = float(y_counts_adv.max() / max(y_counts_adv.min(), 1))
    dominant_frac_adv   = float(y_counts_adv.max() / max(len(y_numeric), 1))
    is_binary_adv       = (y_counts_adv.shape[0] == 2)
    if imbalance_ratio_adv >= _IMBALANCE_MODERATE:
        if is_binary_adv:
            _adv_strategy   = 'binary'
            _adv_auto_check = False   # Off by default — baseline first, enable only if metrics show weakness
        else:
            _severe = (
                imbalance_ratio_adv > _IMBALANCE_MULTICLASS_RATIO_CAP
                or dominant_frac_adv > _IMBALANCE_MULTICLASS_DOMINANT
            )
            _adv_strategy   = 'none' if _severe else 'multiclass_moderate'
            _adv_auto_check = not _severe   # uncheck for severe multiclass
        method_hint = {'binary': 'is_unbalance=True',
                       'multiclass_moderate': 'class_weight=balanced',
                       'none': 'no reweighting (severe imbalance)'}[_adv_strategy]
        suggestions.append({
            'type':                'imbalance',
            'column':              '(model parameter)',
            'column_b':            None,
            'method':              'class_weight_balance',
            'predicted_delta':     1.0,           # always shown at top of group
            'predicted_delta_raw': 0.0,
            'description':         (
                f"Class reweighting — ratio {imbalance_ratio_adv:.1f}:1 "
                f"(dominant class {dominant_frac_adv:.0%}) — {method_hint}"
            ),
            'pinned':              True,
            # Extra metadata used by the imbalance card UI and training step
            'imbalance_ratio':     imbalance_ratio_adv,
            'dominant_frac':       dominant_frac_adv,
            'n_classes_imb':       int(y_counts_adv.shape[0]),
            'imbalance_strategy':  _adv_strategy,
            'auto_checked':        _adv_auto_check,
        })
    # ─────────────────────────────────────────────────────────────────────────

    # -------------------------------------------------------------------------
    # Cross-type normalization
    # -------------------------------------------------------------------------
    # Different collector types may be trained on formulas with very different
    # output scales (e.g. weighted_full_indiv produces AUC deltas ~0.001-0.01,
    # while cohens_d_weighted produces Cohen's d values ~0.1-0.8).
    # Sorting the merged list by raw predicted_delta would always push interaction
    # and row suggestions to the top simply because their numbers are larger.
    #
    # Fix: z-score normalize within each type so every type contributes on equal
    # footing to the global ranking. The raw value is preserved as
    # predicted_delta_raw for display (so the user still sees meaningful units).
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in suggestions:
        # Skip sentinels (imbalance, date, text) AND pinned items — they have
        # manually assigned deltas that must survive the threshold filter.
        if s['type'] not in ('imbalance', 'date', 'text') and not s.get('pinned'):
            by_type[s['type']].append(s)

    for type_suggestions in by_type.values():
        vals = np.array([s['predicted_delta'] for s in type_suggestions], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        for s in type_suggestions:
            s['predicted_delta_raw'] = s['predicted_delta']
            s['predicted_delta'] = float((s['predicted_delta'] - mu) / sigma) if sigma > 1e-9 else 0.0

    # Ensure pinned non-sentinel suggestions also get a predicted_delta_raw
    for s in suggestions:
        if s.get('pinned') and 'predicted_delta_raw' not in s:
            s['predicted_delta_raw'] = s['predicted_delta']

    suggestions.sort(key=lambda s: s['predicted_delta'], reverse=True)

    advisories = generate_dataset_advisories(X, y)

    # ── Leakage detection ─────────────────────────────────────────────────────
    # Flag columns whose correlation with the target is suspiciously perfect —
    # this often signals a data-leakage issue (e.g. a derived or post-event
    # column that would not be available at inference time).
    n_classes_lk = int(y_numeric.nunique())
    # Maximum possible mutual information for a perfectly predictive feature is
    # the entropy of the target.  We flag if MI exceeds 95% of that ceiling.
    target_probs = y_numeric.value_counts(normalize=True).values
    target_entropy = float(-np.sum(target_probs * np.log2(np.clip(target_probs, 1e-12, None))))
    mi_leakage_thresh = max(0.95 * target_entropy, 0.95)

    leaky_cols = {}
    for col, cm in _leakage_col_metas.items():
        spear = cm.get('spearman_corr_target', 0.0)
        mi    = cm.get('mutual_info_score', 0.0)
        reasons = []
        if spear >= 0.95:
            reasons.append(f"Spearman |ρ| = {spear:.4f} ≥ 0.95")
        if mi >= mi_leakage_thresh:
            reasons.append(f"Mutual info = {mi:.4f} ≥ {mi_leakage_thresh:.3f} (≥ 95% of target entropy)")
        if reasons:
            leaky_cols[col] = " | ".join(reasons)

    if leaky_cols:
        col_list = "\n".join(f"- `{c}`: {r}" for c, r in leaky_cols.items())
        advisories.insert(0, {
            'category': 'leakage',
            'severity': 'high',
            'title': f'⚠️ Potential Data Leakage — {len(leaky_cols)} column(s) suspiciously correlated with target',
            'detail': (
                "The following features have near-perfect statistical association with the "
                "target. This is a strong signal of **data leakage** — the column may be "
                "derived from the target, recorded after the event, or otherwise unavailable "
                "at real inference time. Consider removing or carefully auditing these columns "
                "before drawing any conclusions from model performance.\n\n"
                + col_list
            ),
            'code_hint': "# Drop suspect columns before training\nX = X.drop(columns=" + str(list(leaky_cols.keys())) + ")",
        })
    # ─────────────────────────────────────────────────────────────────────────

    return suggestions, skipped_info, advisories, ds_meta




# =============================================================================
# POST-TRAINING DIAGNOSIS
# =============================================================================

def _compute_suggestion_verdicts(
    fitted_params,
    suggestions,
    selected_indices,
    enhanced_model,
    enhanced_train_cols,
    baseline_model,
    baseline_train_cols,
    baseline_val_metrics,
    enhanced_val_metrics,
    apply_imbalance=False,
):
    """
    Attribute each applied suggestion's impact using feature importances.

    For transforms that create new columns (new_cols in params): sum the
    importance % of those new columns in the enhanced model.
    For in-place transforms (log, freq-encoding, etc.): compare the column's
    importance before (baseline) and after (enhanced).

    Returns
    -------
    verdicts : list[dict]
        One entry per applied transform + one for imbalance if used.
        Keys: sug_idx, method, column, column_b, type, new_cols, verdict, reason
        verdict ∈ {'good', 'marginal', 'bad'}
    low_imp_orig : dict[str, float]
        Original columns (in baseline) with < 0.5 % importance.
    """
    # Normalise importances to % of total
    enh_imps  = enhanced_model.feature_importances_
    enh_total = max(float(enh_imps.sum()), 1.0)
    enh_pct   = {c: float(v / enh_total * 100)
                 for c, v in zip(enhanced_train_cols, enh_imps)}

    base_imps  = baseline_model.feature_importances_
    base_total = max(float(base_imps.sum()), 1.0)
    base_pct   = {c: float(v / base_total * 100)
                  for c, v in zip(baseline_train_cols, base_imps)}

    original_cols = set(baseline_train_cols)

    verdicts = []

    # ── Dynamic importance thresholds ─────────────────────────────────────────
    # With many features, importance is diluted — fixed thresholds like 1.0%
    # would wrongly flag useful transforms as bad.  Scale with feature count:
    #   threshold_good     = max(0.5%,  50 / n_cols)   → e.g. 0.5% @ 100 cols, 1.7% @ 30 cols
    #   threshold_marginal = threshold_good / 4
    n_enh_cols = max(len(enhanced_train_cols), 1)
    _thr_good     = max(0.5, 50.0 / n_enh_cols)
    _thr_marginal = _thr_good / 4.0

    # ── Per-transform verdicts ────────────────────────────────────────────────
    for p in fitted_params:
        method   = p.get('method', '')
        col      = p.get('column', '')
        col_b    = p.get('column_b')
        sug_type = p.get('type', '')
        new_cols = p.get('new_cols') or []

        # Find the original suggestion index.
        # Search selected_indices first (preferred), then fall back to all
        # suggestions so that stale-index or session-state timing issues
        # don't silently leave sug_idx=None and hide the auto-deselect button.
        sug_idx = None
        search_order = list(selected_indices or []) + [
            j for j in range(len(suggestions)) if j not in set(selected_indices or [])
        ]
        for i in search_order:
            if i < len(suggestions):
                s = suggestions[i]
                if (s['method'] == method
                        and s['column'] == col
                        and s.get('column_b') == col_b):
                    sug_idx = i
                    break

        if new_cols:
            # New columns added by the transform — evaluate by their importance in enhanced model
            total_imp = sum(enh_pct.get(c, 0.0) for c in new_cols)
            col_names = ", ".join(f"`{c}`" for c in new_cols[:2]) + ("…" if len(new_cols) > 2 else "")
            if total_imp >= _thr_good:
                verdict = 'good'
                reason  = f"New column(s) {col_names} hold {total_imp:.2f}% of enhanced model importance"
            elif total_imp >= _thr_marginal:
                verdict = 'marginal'
                reason  = (f"New column(s) {col_names} hold only {total_imp:.2f}% importance "
                           f"— minimal contribution (threshold: {_thr_good:.2f}%)")
            else:
                verdict = 'bad'
                reason  = (f"New column(s) {col_names} hold {total_imp:.2f}% importance "
                           f"— effectively no contribution (threshold: {_thr_good:.2f}%)")
        else:
            # In-place transform (log, freq-enc, etc.): column keeps its name, compare importance
            base_i = base_pct.get(col, 0.0)
            enh_i  = enh_pct.get(col, 0.0)
            delta_i = enh_i - base_i
            delta_str = f"{delta_i:+.2f}%" if abs(delta_i) >= 0.01 else "unchanged"
            if enh_i >= _thr_good:
                verdict = 'good'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline)")
            elif enh_i >= _thr_marginal:
                verdict = 'marginal'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline) — limited gain (threshold: {_thr_good:.2f}%)")
            else:
                verdict = 'bad'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline) — transform had no measurable effect (threshold: {_thr_good:.2f}%)")

        # ── Date sub-feature pruning ──────────────────────────────────────────
        # For date_features transforms that are overall 'good', individually
        # check each extracted sub-feature.  Sub-features with near-zero
        # importance (e.g. is_weekend) are recorded so the auto-deselect
        # handler can remove them from selected_date_features without
        # discarding the whole transform.
        bad_date_subfeatures = []
        col_prefix_for_date = ''
        if method == 'date_features' and new_cols and verdict == 'good':
            col_prefix_for_date = p.get('col_prefix', f'{col}_')
            for _nc in new_cols:
                if enh_pct.get(_nc, 0.0) < _thr_good:
                    _sub_name = (_nc[len(col_prefix_for_date):]
                                 if _nc.startswith(col_prefix_for_date) else _nc)
                    bad_date_subfeatures.append(_sub_name)

        # ── Row-stat sub-feature pruning ──────────────────────────────────────
        # Each stat is held to the same _thr_good bar as standalone interaction
        # columns — if a single interaction column at 2% gets dropped, a row stat
        # at 1.7% should too. Using _thr_marginal was too lenient.
        bad_row_stats = []
        if method == 'row_numeric_stats' and new_cols and verdict == 'good':
            for _nc in new_cols:
                if enh_pct.get(_nc, 0.0) < _thr_good:
                    bad_row_stats.append(_nc)   # e.g. 'row_range', 'row_sum'

        verdicts.append({
            'sug_idx':             sug_idx,
            'method':              method,
            'column':              col,
            'column_b':            col_b,
            'type':                sug_type,
            'new_cols':            new_cols,
            'verdict':             verdict,
            'reason':              reason,
            'bad_date_subfeatures': bad_date_subfeatures,
            'col_prefix':          col_prefix_for_date,
            'bad_row_stats':       bad_row_stats,
        })

    # ── Class-imbalance verdict ───────────────────────────────────────────────
    if apply_imbalance:
        imb_idx = None
        for i in (selected_indices or []):
            if i < len(suggestions) and suggestions[i].get('type') == 'imbalance':
                imb_idx = i
                break

        b_f1   = float(baseline_val_metrics.get('f1')   or 0)
        e_f1   = float(enhanced_val_metrics.get('f1')    or 0)
        b_prec = float(baseline_val_metrics.get('precision') or 0)
        e_prec = float(enhanced_val_metrics.get('precision') or 0)
        b_rec  = float(baseline_val_metrics.get('recall') or 0)
        e_rec  = float(enhanced_val_metrics.get('recall') or 0)

        # Thresholds: if baseline is already this strong, imbalance handling is not needed
        _STRONG_BASELINE_THRESHOLD = 0.90

        b_already_strong = (
            b_f1   >= _STRONG_BASELINE_THRESHOLD and
            b_prec >= _STRONG_BASELINE_THRESHOLD and
            b_rec  >= _STRONG_BASELINE_THRESHOLD
        )

        if b_already_strong and e_f1 <= b_f1 + 0.005:
            imb_verdict = 'marginal'
            imb_reason  = (
                f"Baseline was already strong (F1={b_f1:.3f}, P={b_prec:.3f}, R={b_rec:.3f}) — "
                f"class reweighting had no meaningful benefit (F1: {b_f1:.3f} → {e_f1:.3f}). "
                f"Consider disabling it."
            )
        elif b_already_strong and e_f1 < b_f1 - 0.005:
            imb_verdict = 'bad'
            imb_reason  = (
                f"Baseline was already strong (F1={b_f1:.3f}, P={b_prec:.3f}, R={b_rec:.3f}) — "
                f"class reweighting hurt performance (F1: {b_f1:.3f} → {e_f1:.3f}). "
                f"Disable it; the model handles the imbalance on its own."
            )
        elif e_f1 < b_f1 - 0.02:
            imb_verdict = 'bad'
            imb_reason  = (f"F1 degraded: {b_f1:.3f} → {e_f1:.3f}. "
                           f"Class reweighting may be hurting overall performance.")
        elif e_rec > b_rec + 0.15 and e_prec < b_prec - 0.15:
            imb_verdict = 'marginal'
            imb_reason  = (f"Recall improved ({b_rec:.3f} → {e_rec:.3f}) but precision "
                           f"dropped ({b_prec:.3f} → {e_prec:.3f}). "
                           f"Minority class is being over-predicted.")
        elif e_f1 >= b_f1 - 0.005:
            imb_verdict = 'good'
            imb_reason  = f"F1 maintained or improved: {b_f1:.3f} → {e_f1:.3f}"
        else:
            imb_verdict = 'marginal'
            imb_reason  = (f"Mixed results — F1: {b_f1:.3f} → {e_f1:.3f}, "
                           f"Recall: {b_rec:.3f} → {e_rec:.3f}")

        verdicts.append({
            'sug_idx':  imb_idx,
            'method':   'class_weight_balance',
            'column':   '(model param)',
            'column_b': None,
            'type':     'imbalance',
            'new_cols': [],
            'verdict':  imb_verdict,
            'reason':   imb_reason,
        })

    # ── Low-importance original columns ───────────────────────────────────────
    low_imp_orig = {
        c: pct
        for c, pct in base_pct.items()
        if pct < 0.5 and c in original_cols
    }

    return verdicts, low_imp_orig


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_lgbm_model(X_train, y_train, X_val, y_val, n_classes,
                     apply_imbalance: bool = False,
                     imbalance_strategy: str = 'none',
                     base_params: dict = None):
    """Train a LightGBM classifier with early stopping on validation set.

    apply_imbalance: if True, apply imbalance handling based on imbalance_strategy.
    imbalance_strategy:
        'binary'               → is_unbalance=True  (safe for binary at any ratio)
        'multiclass_moderate'  → class_weight='balanced'  (moderate multiclass only)
        'none'                 → no reweighting (baseline, or severe multiclass)
    base_params: override the default BASE_PARAMS (e.g. for small-dataset HP tuning)
    """
    X_tr, X_vl, col_encoders = prepare_data_for_model(X_train, X_val)

    params = (base_params or BASE_PARAMS).copy()
    if apply_imbalance:
        if imbalance_strategy == 'binary':
            params['is_unbalance'] = True
        elif imbalance_strategy == 'multiclass_moderate':
            params['class_weight'] = 'balanced'
        # 'none' → severe multiclass or baseline: no reweighting added

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model, X_tr.columns.tolist(), col_encoders


# =============================================================================
# THRESHOLD OPTIMISATION HELPERS
# =============================================================================

def _metrics_at_threshold(y_true, y_proba, threshold):
    """Compute classification metrics using a custom probability threshold.

    Returns a dict with accuracy, f1, precision, recall, and the threshold used.
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        'threshold': float(threshold),
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _find_optimal_thresholds(y_true, y_proba, n_steps=200):
    """Sweep thresholds and return the optimal one for each metric.

    Returns a dict mapping metric name → {'threshold': float, 'value': float}.
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.01, 0.99, n_steps)

    best = {
        'f1':        {'threshold': 0.5, 'value': 0.0},
        'precision': {'threshold': 0.5, 'value': 0.0},
        'recall':    {'threshold': 0.5, 'value': 0.0},
        'accuracy':  {'threshold': 0.5, 'value': 0.0},
    }

    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, t)
        for key in best:
            if m[key] > best[key]['value']:
                best[key] = {'threshold': float(t), 'value': float(m[key])}

    return best


def evaluate_on_set(model, X, y, train_columns, n_classes, col_encoders=None):
    """Evaluate model on a dataset. Returns metrics dict.

    col_encoders: dict returned by prepare_data_for_model / train_lgbm_model.
    When provided, categorical columns are encoded with the *training* encoder
    so integer codes are consistent with what the model saw during training.
    When None (legacy path), a fresh LabelEncoder is fit on the evaluation
    data — this is incorrect if eval categories differ from training, but is
    kept as a fallback for callers that don't yet pass encoders.
    """
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            if col_encoders and col in col_encoders and col_encoders[col]['encoder'] is not None:
                le = col_encoders[col]['encoder']
                # Map unseen categories to the first known class to avoid
                # transform errors, then let the model treat them as that class.
                known = set(le.classes_)
                fallback = le.classes_[0]
                X_enc[col] = le.transform(
                    X_enc[col].astype(str).map(lambda v: v if v in known else fallback)
                )
            else:
                # Fallback: refit on eval data (less correct but won't crash)
                le = LabelEncoder()
                le.fit(X_enc[col].astype(str))
                X_enc[col] = le.transform(X_enc[col].astype(str))
        # Fill NaN with the training median when available, else eval median
        if col_encoders and col in col_encoders and col_encoders[col]['median'] is not None:
            fill_val = col_encoders[col]['median']
        elif pd.api.types.is_numeric_dtype(X_enc[col]):
            fill_val = X_enc[col].median()
        else:
            fill_val = -999
        X_enc[col] = X_enc[col].fillna(fill_val)

    # Align columns
    for c in train_columns:
        if c not in X_enc.columns:
            X_enc[c] = 0
    X_enc = X_enc[train_columns]

    y_pred = model.predict(X_enc)
    y_pred_proba = model.predict_proba(X_enc)

    metrics = {}
    # Store raw predictions for the download-predictions section
    metrics['_y_pred'] = y_pred
    metrics['_y_pred_proba'] = y_pred_proba
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

    # ── Confusion matrix ─────────────────────────────────────────────────────
    try:
        from sklearn.metrics import confusion_matrix as _cm
        metrics['confusion_matrix'] = _cm(y, y_pred).tolist()
        metrics['y_classes'] = [str(c) for c in sorted(set(y))]
    except Exception:
        metrics['confusion_matrix'] = None
        metrics['y_classes'] = None

    # ── ROC curve data (binary only) ─────────────────────────────────────────
    try:
        if n_classes == 2:
            from sklearn.metrics import roc_curve as _roc_curve
            fpr, tpr, _ = _roc_curve(y, y_pred_proba[:, 1])
            metrics['roc_data'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': metrics.get('roc_auc') or 0.0,
            }
        else:
            metrics['roc_data'] = None
    except Exception:
        metrics['roc_data'] = None

    # ── Precision-Recall curve data (binary only) ─────────────────────────────
    try:
        if n_classes == 2:
            from sklearn.metrics import precision_recall_curve as _pr_curve
            from sklearn.metrics import average_precision_score as _ap_score
            _pr_prec, _pr_rec, _pr_thresh = _pr_curve(y, y_pred_proba[:, 1])
            metrics['pr_data'] = {
                'precision': _pr_prec.tolist(),
                'recall':    _pr_rec.tolist(),
                'thresholds': _pr_thresh.tolist(),
                'avg_precision': float(_ap_score(y, y_pred_proba[:, 1])),
            }
            # Store raw arrays for threshold optimisation
            metrics['_y_true']  = np.array(y).tolist()
            metrics['_y_proba'] = y_pred_proba[:, 1].tolist()
        else:
            metrics['pr_data'] = None
    except Exception:
        metrics['pr_data'] = None

    return metrics


def predict_on_set(model, X, train_columns, n_classes, col_encoders=None):
    """Generate predictions without ground-truth labels (predict-only mode).

    Returns a dict with '_y_pred' and '_y_pred_proba' keys only — no
    evaluation metrics since there is no y to compare against.

    col_encoders: same format as evaluate_on_set.
    """
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            if col_encoders and col in col_encoders and col_encoders[col]['encoder'] is not None:
                le = col_encoders[col]['encoder']
                known = set(le.classes_)
                fallback = le.classes_[0]
                X_enc[col] = le.transform(
                    X_enc[col].astype(str).map(lambda v: v if v in known else fallback)
                )
            else:
                le_local = LabelEncoder()
                le_local.fit(X_enc[col].astype(str))
                X_enc[col] = le_local.transform(X_enc[col].astype(str))
        if col_encoders and col in col_encoders and col_encoders[col]['median'] is not None:
            fill_val = col_encoders[col]['median']
        elif pd.api.types.is_numeric_dtype(X_enc[col]):
            fill_val = X_enc[col].median()
        else:
            fill_val = -999
        X_enc[col] = X_enc[col].fillna(fill_val)

    # Align columns
    for c in train_columns:
        if c not in X_enc.columns:
            X_enc[c] = 0
    X_enc = X_enc[train_columns]

    y_pred = model.predict(X_enc)
    y_pred_proba = model.predict_proba(X_enc)

    return {
        '_y_pred': y_pred,
        '_y_pred_proba': y_pred_proba,
    }


# =============================================================================
# DEDUPLICATE SUGGESTIONS
# =============================================================================

def deduplicate_suggestions(suggestions):
    """
    For single-column transforms, keep only the best method per column among
    in-place transforms (those that overwrite the original column).  Additive
    transforms — those that always create a *new* column and therefore do not
    conflict with each other or with any in-place transform — are kept
    unconditionally so that, for example, missing_indicator and impute_median
    are both surfaced for a column with missing values.

    For interactions, keep only the best method per pair.
    For row/date/text families, keep as-is (one entry per family by design).
    """
    # Methods that add a brand-new column and never overwrite the original.
    # These are genuinely complementary with any in-place transform on the
    # same column, so they must NOT compete in the per-column best-of bucket.
    ADDITIVE_METHODS = {
        'missing_indicator',      # adds <col>_is_na
        'polynomial_square',      # adds <col>_sq
        'polynomial_cube',        # adds <col>_cube
        'reciprocal_transform',   # adds <col>_recip
    }

    best_single = {}       # (type, col) -> best in-place suggestion
    additive_suggestions = []
    best_interaction = {}  # (pair, method) -> best suggestion
    row_suggestions = []   # row families are already deduplicated

    for s in suggestions:
        if s['type'] in ('numerical', 'categorical'):
            if s['method'] in ADDITIVE_METHODS:
                # Always keep — it adds a new column and cannot conflict.
                additive_suggestions.append(s)
            else:
                key = (s['type'], s['column'])
                if key not in best_single or s['predicted_delta'] > best_single[key]['predicted_delta']:
                    best_single[key] = s
        elif s['type'] in ('row', 'date', 'text', 'imbalance'):
            # row/date/text/imbalance are already unique by design — keep as-is.
            row_suggestions.append(s)
        else:
            col_a = s['column'] or ''
            col_b = s.get('column_b') or ''
            pair = tuple(sorted([col_a, col_b]))
            key = (pair, s['method'])  # keep best per pair+method
            if key not in best_interaction or s['predicted_delta'] > best_interaction[key]['predicted_delta']:
                best_interaction[key] = s

    deduped = (list(best_single.values()) + additive_suggestions
               + list(best_interaction.values()) + row_suggestions)
    deduped.sort(key=lambda s: s['predicted_delta'], reverse=True)
    return deduped


# =============================================================================
# STREAMLIT UI
# =============================================================================



def main():
    st.set_page_config(
        page_title="ML Compass",
        page_icon="🧭",
        layout="wide",
    )

    # ── Inject subtle style improvements + scroll-position preserver ────────
    st.markdown("""
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        div[data-testid="stMetricDelta"] svg { display: none; }
        /* Prevent Streamlit from greying out the UI during script reruns */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main .block-container,
        .stApp { opacity: 1 !important; transition: none !important; }
        /* Hide the "running" top-bar decoration flash */
        [data-testid="stStatusWidget"] { visibility: hidden; }
        </style>
        <script>
        (function () {
          /* Preserve the main-panel scroll position across Streamlit reruns.
             Streamlit's own JS resets scroll on every rerun; we save it to
             sessionStorage before the reset and restore it 120 ms later. */
          const KEY = '__st_scroll_y';
          function getMain() {
            return (
              document.querySelector('[data-testid="stMain"]') ||
              document.querySelector('section.main')
            );
          }
          function save() {
            const el = getMain();
            if (el) sessionStorage.setItem(KEY, el.scrollTop);
          }
          function restore() {
            const el = getMain();
            const saved = parseInt(sessionStorage.getItem(KEY) || '0', 10);
            if (el && saved > 0) el.scrollTop = saved;
          }
          /* Attach scroll listener once main element is in DOM. */
          function attachListener() {
            const el = getMain();
            if (!el) { setTimeout(attachListener, 80); return; }
            el.addEventListener('scroll', save, { passive: true });
          }
          /* On every script execution (= every Streamlit rerun) restore scroll
             after a short delay so Streamlit's own reset fires first. */
          restore();
          setTimeout(restore, 120);
          attachListener();

          /* ── Sidebar scroll preservation ──────────────────────────────
             Approach: use a single, stable scrollable element and restore
             its scroll position after every Streamlit rerun.
             Key insight: on resize, Streamlit replaces sidebar DOM but the
             scrollable container selector stays predictable. We use a 
             MutationObserver only to detect when the container is replaced,
             then re-attach and immediately restore. */
          const SB_KEY = '__st_sidebar_scroll_y';
          let _sbLastSaved = 0;
          let _sbRestoreTimer = null;
          let _sbListener = null;

          function getSidebarScroller() {
            // Walk up from stSidebarContent to find the actual scrolling container
            const candidates = [
              document.querySelector('section[data-testid="stSidebar"] > div:first-child'),
              document.querySelector('[data-testid="stSidebarContent"]'),
              document.querySelector('[data-testid="stSidebar"] > div'),
            ];
            for (const el of candidates) {
              if (el && el.scrollHeight > el.clientHeight) return el;
            }
            return candidates.find(Boolean) || null;
          }

          function saveSidebarScroll() {
            const sb = getSidebarScroller();
            if (sb && sb.scrollTop > 0) {
              _sbLastSaved = sb.scrollTop;
              try { sessionStorage.setItem(SB_KEY, _sbLastSaved); } catch(e) {}
            }
          }

          function restoreSidebarScroll() {
            const saved = parseInt(sessionStorage.getItem(SB_KEY) || '0', 10);
            if (saved <= 0) return;
            const sb = getSidebarScroller();
            if (sb) {
              sb.scrollTop = saved;
              // Double-tap after layout settles
              setTimeout(() => { const s2 = getSidebarScroller(); if (s2) s2.scrollTop = saved; }, 80);
            }
          }

          function attachSidebarListener() {
            const sb = getSidebarScroller();
            if (!sb) return;
            if (_sbListener) sb.removeEventListener('scroll', _sbListener);
            _sbListener = () => saveSidebarScroll();
            sb.addEventListener('scroll', _sbListener, { passive: true });
            restoreSidebarScroll();
          }

          // Watch the sidebar root for DOM changes (resize / rerun replaces content)
          const _sidebarRoot = document.querySelector('[data-testid="stSidebar"]');
          if (_sidebarRoot) {
            new MutationObserver(() => {
              // Debounce: wait until DOM settles before re-attaching
              clearTimeout(_sbRestoreTimer);
              _sbRestoreTimer = setTimeout(() => {
                attachSidebarListener();
              }, 50);
            }).observe(_sidebarRoot, { childList: true, subtree: false });
          }

          restoreSidebarScroll();
          setTimeout(attachSidebarListener, 100);
          setTimeout(restoreSidebarScroll, 300);
        })();
        </script>
    """, unsafe_allow_html=True)

    st.title("🧭 ML Compass")
    st.caption("Upload a classification dataset → get data-driven transform suggestions → compare baseline vs. enhanced LightGBM model")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown("""
**What it does:** This tool uses trained *meta-models* — models that have learned which feature engineering transforms tend to help on datasets with similar characteristics — to recommend preprocessing steps tailored to your specific dataset.

**The workflow:**
1. **Upload** your training CSV and pick the target column
2. **Analyze** — meta-models score each potential transform (log-transform, frequency encoding, interaction features, etc.) against your dataset's properties
3. **Review** the ranked suggestions, check/uncheck what to apply
4. **Train** a baseline LightGBM and an enhanced one (with your selected transforms), then compare validation metrics
5. **Test** — upload a held-out test CSV to get a clean head-to-head comparison

**Meta-models directory:** Set this in the sidebar to point at your trained models (run `train_meta_models.py` to generate them).
        """)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        model_dir = st.text_input("Meta-models directory", value="./meta_models")

        # Dynamic default: scale with dataset rows AND columns
        _ds = st.session_state.get('X_train')
        if _ds is not None:
            _nr, _nc = _ds.shape
            if   _nr < 500:   _auto_k = 3
            elif _nr < 2_000: _auto_k = 5
            elif _nr < 10_000: _auto_k = 8
            elif _nr < 50_000: _auto_k = 12
            else:             _auto_k = 15
            # Wider datasets → more FE opportunities → bump up
            if   _nc > 30: _auto_k = min(_auto_k + 3, 20)
            elif _nc > 15: _auto_k = min(_auto_k + 1, 20)
        else:
            _auto_k = 10

        # Reset slider to new auto value when a fresh dataset is loaded
        _top_k_sig = f"{getattr(_ds, 'shape', None)}"
        if st.session_state.get('_top_k_shape_sig') != _top_k_sig:
            st.session_state['_top_k_shape_sig'] = _top_k_sig
            st.session_state['_top_k_slider']    = _auto_k

        top_k = st.slider(
            "Pre-select top N suggestions", 1, 30,
            key='_top_k_slider',
            help=(
                f"Auto-set to **{_auto_k}** based on dataset size "
                + "("
                + (f"{_ds.shape[0]:,} rows × {_ds.shape[1]} cols" if hasattr(_ds, 'shape') else "? rows × ? cols")
                + "). Adjust as needed."
            ),
        )
        delta_threshold = st.number_input(
            "Confidence filter",
            value=0.0, step=0.05, format="%.2f",
            help="Hides suggestions below this z-score rank within their type. "
                 "0 = show all above-average, 1 = top ~16% only. "
                 "Raw Δ AUC values are shown in the results table.",
        )
        st.divider()
        st.markdown("**⚡ Quick Mode**")
        _quick_mode = st.toggle(
            "Sample rows for faster analysis",
            value=st.session_state.get('_quick_mode', False),
            key='_quick_mode',
            help=(
                "Sub-samples the dataset before running meta-feature analysis "
                "(correlations, MI scores, landmarking). Useful for wide/large "
                "datasets where the Analyze step is slow. The sample is used only "
                "for *analysis*; training always uses the full data unless you "
                "explicitly opt in below."
            ),
        )
        if _quick_mode:
            _xt = st.session_state.get('X_train')
            _n_full = _xt.shape[0] if _xt is not None else 0
            # Dynamic default: 10% of the dataset, clamped between 2 000 and 20 000.
            # This keeps the sample large enough for stable meta-feature estimates
            # while scaling sensibly with dataset size.
            #   20 k rows  → 2 000 (lower clamp)
            #   50 k rows  → 5 000
            #   100 k rows → 10 000
            #   300 k rows → 20 000 (upper clamp)
            _default_n = min(20_000, max(2_000, _n_full // 10))
            st.number_input(
                "Sample size (rows)",
                min_value=500, max_value=max(50_000, _n_full),
                value=st.session_state.get('_quick_n', _default_n), step=500,
                key='_quick_n',
                help=(
                    f"Default is 10% of your dataset ({_default_n:,} rows), "
                    f"clamped between 2 000 and 20 000. "
                    "Increase for more accurate meta-feature estimates; "
                    "decrease for faster analysis on very wide datasets."
                ),
            )
            st.toggle(
                "Also train on sample (not full data)",
                value=st.session_state.get('_quick_train_sample', False),
                key='_quick_train_sample',
                help="If on, both analysis AND model training use the sample. Faster but less representative.",
            )

        st.divider()
        _sidebar_progress(st.session_state)
        st.divider()
        if st.button("🔄 Reset session", help="Clear all state and start over", use_container_width=True):
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()
        render_chat_sidebar(st.session_state)

    # Load meta-models
    if os.path.isdir(model_dir):
        meta_models = load_meta_models(model_dir)
        if meta_models:
            st.sidebar.success(f"✅ Models loaded: {', '.join(meta_models.keys())}")
        else:
            st.sidebar.warning("No model files found in that directory.")
            meta_models = {}
    else:
        with st.sidebar.expander("⚠️ Setup required", expanded=True):
            st.warning(
                f"Directory **`{model_dir}`** not found.  \n"
                "Run `train_meta_models.py` to generate models, "
                "then point the path above to that directory."
            )
        meta_models = {}

    # --- Session state init ---
    for key in ['train_df', 'target_col', 'X_train', 'y_train', 'suggestions',
                'selected_indices', 'baseline_model', 'enhanced_model',
                'baseline_train_cols', 'enhanced_train_cols', 'fitted_params',
                'n_classes', 'label_encoder', 'baseline_val_metrics', 'enhanced_val_metrics',
                'baseline_col_encoders', 'enhanced_col_encoders',
                'skipped_info', 'advisories', 'X_train_enhanced',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_baseline_metrics', '_test_enhanced_metrics',
                '_test_file_name']:
        if key not in st.session_state:
            st.session_state[key] = None

    # =========================================================================
    # STEP 1: UPLOAD TRAINING DATA
    # =========================================================================
    st.header("① Upload Training Data")

    uploaded_train = st.file_uploader("Upload training CSV", type=['csv'], key='train_upload')

    if uploaded_train is not None:
        try:
            try:
                df = pd.read_csv(uploaded_train, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_train.seek(0)
                df = pd.read_csv(uploaded_train, encoding='latin-1')
        except Exception as _parse_err:
            st.error(
                f"**Could not read your file:** {_parse_err}  \n"
                "Make sure it's a valid CSV with a header row and comma-separated values."
            )
            st.stop()
        df = sanitize_feature_names(df)
        st.session_state.train_df = df

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            st.dataframe(df.head(5), use_container_width=True, height=200)

        # Target selection — auto-detect by common name, fallback to last column
        _col_list = df.columns.tolist()
        _TARGET_NAMES = {
            'target', 'label', 'labels', 'class', 'classes',
            'y', 'outcome', 'response', 'output', 'result',
            'dependent', 'dependent_variable', 'churn', 'fraud',
            'survived', 'survival', 'defaulted', 'default',
        }
        _auto_target_idx = len(_col_list) - 1  # fallback: last column
        for _ci, _cn in enumerate(_col_list):
            if _cn.lower().strip().replace('-', '_').replace(' ', '_') in _TARGET_NAMES:
                _auto_target_idx = _ci
                break
        _auto_detected = (_auto_target_idx < len(_col_list) - 1)
        target_col = st.selectbox(
            "Select target column",
            options=_col_list,
            index=_auto_target_idx,
            help=(
                f"🎯 Auto-detected from column name: **`{_col_list[_auto_target_idx]}`**"
                if _auto_detected
                else "Defaulted to the last column — adjust if needed."
            ),
        )
        if _auto_detected and target_col == _col_list[_auto_target_idx]:
            st.caption(f"🎯 Target auto-detected: `{target_col}`")
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
        # Colour-coded bar chart: largest class grey, minority classes amber/red
        try:
            import plotly.graph_objects as go
            _max_count = class_dist.max()
            _bar_colors = [
                "#3fb950" if v == _max_count else ("#f0883e" if v >= _max_count * 0.2 else "#f85149")
                for v in class_dist.values
            ]
            _fig_dist = go.Figure(go.Bar(
                x=[str(c) for c in class_dist.index],
                y=class_dist.values,
                marker_color=_bar_colors,
                text=[f"{v:,}" for v in class_dist.values],
                textposition="outside",
            ))
            _fig_dist.update_layout(
                margin=dict(t=10, b=10, l=0, r=0),
                height=180,
                xaxis_title="Class",
                yaxis_title="Count",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=11),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#30363d"),
            )
            st.plotly_chart(_fig_dist, use_container_width=True)
        except Exception:
            st.bar_chart(class_dist, height=150)

        # Only (re-)set X_train from the raw file when no column changes have been
        # applied yet.  Once Apply Changes has run, X_train holds the modified
        # dataset and must not be overwritten by the raw file on subsequent reruns.
        if not st.session_state.get('_col_type_applied', False):
            st.session_state.X_train = X
        st.session_state.y_train = y

        # Snapshot raw X for reset capability (once per unique dataset)
        _raw_sig = f"{X.shape}__{list(X.columns)}"
        if st.session_state.get('_raw_sig') != _raw_sig:
            st.session_state['X_train_raw']     = X
            st.session_state['_raw_sig']         = _raw_sig
            st.session_state['_applied_drops']   = []
            st.session_state['_applied_types']   = {}
            st.session_state['_col_type_applied'] = False
            # Clear all downstream state so the new dataset starts fresh
            for _stale_key in [
                'suggestions', 'skipped_info', 'advisories', 'selected_indices',
                'baseline_model', 'enhanced_model',
                'baseline_train_cols', 'enhanced_train_cols',
                'baseline_val_metrics', 'enhanced_val_metrics',
                'baseline_col_encoders', 'enhanced_col_encoders',
                'fitted_params', 'X_train_enhanced',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_baseline_metrics', '_test_enhanced_metrics',
                '_test_file_sig', '_test_file_name',
                # Presentation / report keys that would show stale data
                '_val_metrics_rows', '_report_class_dist',
                '_fi_b_pct', '_fi_e_original', '_fi_e_new',
                '_fi_orig_pct', '_fi_new_pct',
                '_suggestion_verdicts', '_low_imp_cols', '_verdicts_stale',
                '_analyze_baseline_score', '_analyze_baseline_std',
                'apply_imbalance',
                '_optuna_best_params', '_optuna_best_score',
            ]:
                st.session_state[_stale_key] = None

    # =========================================================================
    # COLUMN TYPE REVIEW — detect → review → Apply Changes → then Analyze
    # =========================================================================
    if st.session_state.X_train is not None:
        X_raw = st.session_state.get('X_train_raw', st.session_state.X_train)

        # Detect types on raw X, cache by signature
        _xt_sig_ct = f"{X_raw.shape}__{list(X_raw.columns[:5])}"
        if st.session_state.get('_col_type_sig') != _xt_sig_ct:
            with st.spinner("Detecting column types…"):
                st.session_state['_col_type_info'] = get_column_type_info(X_raw)
                st.session_state['_col_type_sig']  = _xt_sig_ct
                # Bump version so per-row widget keys reset on new dataset
                st.session_state['_col_type_ver'] = (
                    st.session_state.get('_col_type_ver', 0) + 1
                )

        col_type_info    = st.session_state.get('_col_type_info', {})
        _ct_ver          = st.session_state.get('_col_type_ver', 0)
        _already_applied = st.session_state.get('_col_type_applied', False)
        _applied_drops   = st.session_state.get('_applied_drops', [])
        _applied_types   = st.session_state.get('_applied_types', {})

        # ── Expander title ────────────────────────────────────────────────────
        _n_drop_sug  = sum(1 for v in col_type_info.values() if v['drop_suggested'])
        _type_counts = {}
        for v in col_type_info.values():
            _type_counts[v['detected']] = _type_counts.get(v['detected'], 0) + 1
        _summary_parts = []
        if _already_applied and (_applied_drops or _applied_types):
            _summary_parts.append("✅ Changes applied")
        elif _n_drop_sug:
            _summary_parts.append(f"⚠️ {_n_drop_sug} drop suggestion{'s' if _n_drop_sug!=1 else ''}")
        for _t in ('Free Text', 'Date', 'DateTime', 'Time only', 'Day-of-Week'):
            if _type_counts.get(_t):
                _summary_parts.append(f"{_COL_TYPE_ICONS.get(_t,'')} {_type_counts[_t]} {_t}")
        _ct_label = "🔬 Column Types & Drop Suggestions"
        if _summary_parts:
            _ct_label += "  —  " + "  ·  ".join(_summary_parts)
        _ct_label += "  *(optional)*"

        with st.expander(_ct_label, expanded=bool(_n_drop_sug)):
            # ── Applied state banner + preview ────────────────────────────────
            if _already_applied and (_applied_drops or _applied_types):
                _banner_parts = []
                if _applied_drops:
                    _banner_parts.append(
                        f"**{len(_applied_drops)}** column{'s' if len(_applied_drops)!=1 else ''} dropped"
                    )
                if _applied_types:
                    _banner_parts.append(
                        f"**{len(_applied_types)}** type override{'s' if len(_applied_types)!=1 else ''}"
                    )
                st.success("Changes applied: " + "  ·  ".join(_banner_parts) +
                           ". Click **Analyze Dataset** below to regenerate suggestions.")

                # ── Preview of the working dataset ───────────────────────────
                X_working_preview = st.session_state['X_train']
                _prev_cols = st.columns([1, 3])
                _prev_cols[0].metric("Remaining columns",
                                     f"{X_working_preview.shape[1]} / {X_raw.shape[1]}")
                _prev_cols[0].metric("Rows", X_working_preview.shape[0])
                with _prev_cols[1]:
                    st.caption("Working dataset preview (first 5 rows):")
                    st.dataframe(X_working_preview.head(5), use_container_width=True,
                                 height=195)

                if st.button("↩️ Reset to original columns", key="_ct_reset_btn"):
                    st.session_state['X_train']           = st.session_state['X_train_raw']
                    st.session_state['_applied_drops']    = []
                    st.session_state['_applied_types']    = {}
                    st.session_state['_col_type_applied'] = False
                    st.session_state['_col_type_ver']    += 1   # reset row widget keys
                    st.rerun()
                st.divider()

            st.caption(
                "Auto-detected column types below. Use **Override** to correct a misdetection, "
                "or tick **Drop** to exclude a column. Override options are filtered to "
                "what's valid for each column's data type. "
                "Click **Apply Changes** to update the working dataset before analyzing."
            )

            # ── Pre-initialise ALL widget keys so hidden columns keep their state ──
            for _col_init, _info_init in col_type_info.items():
                _k_ovr  = f"_ct_ovr_{_col_init}_v{_ct_ver}"
                _k_drop = f"_ct_drop_{_col_init}_v{_ct_ver}"
                if _k_ovr  not in st.session_state:
                    st.session_state[_k_ovr]  = 'Auto'
                if _k_drop not in st.session_state:
                    st.session_state[_k_drop] = _info_init['drop_suggested']

            # ── Compute pending state from ALL columns (reads previous-run values) ──
            _pending_drops = []
            _pending_types = {}
            _val_issues    = []
            for _col_p, _info_p in col_type_info.items():
                _dv = st.session_state.get(f"_ct_drop_{_col_p}_v{_ct_ver}",
                                           _info_p['drop_suggested'])
                _ov = st.session_state.get(f"_ct_ovr_{_col_p}_v{_ct_ver}", 'Auto')
                if _dv:
                    _pending_drops.append(_col_p)
                elif _ov != 'Auto':
                    _pending_types[_col_p] = _ov
                    _res = _validate_col_override(_col_p, _ov, _info_p, X_raw)
                    if _res:
                        _val_issues.append((_col_p, _ov, _res[0], _res[1]))

            _has_pending = bool(_pending_drops or _pending_types)
            _has_errors  = any(s == 'error' for *_, s, _ in _val_issues)
            _parts: list = []
            if _pending_drops:
                _parts.append(f"drop **{len(_pending_drops)}** column{'s' if len(_pending_drops)!=1 else ''}")
            if _pending_types:
                _parts.append(f"**{len(_pending_types)}** type override{'s' if len(_pending_types)!=1 else ''}")

            def _do_apply_changes():
                _X_work = st.session_state['X_train_raw'].copy()
                if _pending_drops:
                    _X_work = _X_work.drop(
                        columns=[c for c in _pending_drops if c in _X_work.columns]
                    )
                st.session_state['X_train']           = _X_work
                st.session_state['_applied_drops']    = _pending_drops
                st.session_state['_applied_types']    = _pending_types
                st.session_state['_col_type_applied'] = True
                st.session_state.suggestions      = None
                st.session_state.skipped_info     = None
                st.session_state.selected_indices = None
                st.rerun()

            # ── Apply Changes button — TOP ────────────────────────────────────
            _ap_top = st.columns([3, 1])
            _ap_top[0].caption(
                ("Pending: " + ", ".join(_parts) + ".") if _has_pending
                else "No changes pending — all columns use auto-detected types."
            )
            if _ap_top[1].button("✅ Apply Changes", type="primary",
                                  disabled=(not _has_pending or _has_errors),
                                  key="_ct_apply_btn_top"):
                _do_apply_changes()

            st.markdown("<hr style='margin:6px 0 10px 0;border-color:#30363d'>",
                        unsafe_allow_html=True)

            # ── Search + Tag filter ───────────────────────────────────────────
            _tag_type_counts: dict = {}
            for _ti in col_type_info.values():
                _td = _ti['detected']
                _tag_type_counts[_td] = _tag_type_counts.get(_td, 0) + 1
            _n_drop_total = sum(1 for _v in col_type_info.values() if _v['drop_suggested'])

            _TAG_DEFS = [
                ('⚠️ Drop suggested', '__drop__'),
                ('📝 Free Text',      'Free Text'),
                ('📅 Date',           'Date'),
                ('🕐 DateTime',       'DateTime'),
                ('⏱️ Time only',      'Time only'),
                ('📆 Day-of-Week',    'Day-of-Week'),
                ('🔢 Numerical',      'Numerical'),
                ('🏷️ Categorical',    'Categorical'),
                ('⚡ Binary',         'Binary'),
                ('🆔 ID',             'ID'),
                ('⚫ Constant',       'Constant'),
            ]
            _available_tags = []
            for _tlabel, _tkey in _TAG_DEFS:
                _cnt = _n_drop_total if _tkey == '__drop__' else _tag_type_counts.get(_tkey, 0)
                if _cnt > 0:
                    _available_tags.append(f"{_tlabel} ({_cnt})")

            _filt_row = st.columns([2, 4])
            with _filt_row[0]:
                _search_val = st.text_input(
                    "_ct_search_lbl", key="_ct_search",
                    placeholder="🔍  Search column name…",
                    label_visibility="collapsed",
                )
            with _filt_row[1]:
                _active_tags = st.multiselect(
                    "_ct_tags_lbl", options=_available_tags,
                    key="_ct_tags",
                    placeholder="🏷️  Filter by type tag — none = show all…",
                    label_visibility="collapsed",
                )

            # Map active tag labels back to type keys
            _active_tag_keys: set = set()
            _filter_drop_flag = False
            for _atag in _active_tags:
                for _tlabel, _tkey in _TAG_DEFS:
                    if _atag.startswith(_tlabel):
                        if _tkey == '__drop__':
                            _filter_drop_flag = True
                        else:
                            _active_tag_keys.add(_tkey)

            def _col_visible(col, info):
                if _search_val and _search_val.lower() not in col.lower():
                    return False
                if not _active_tags:
                    return True  # no filter → show all
                if _filter_drop_flag and info['drop_suggested']:
                    return True
                return info['detected'] in _active_tag_keys

            # ── Sort: problematic / special columns first ─────────────────────
            def _col_sort_key(item):
                _c, _i = item
                _d = _i['detected']
                if   _d == 'Constant':                                    _r = 0
                elif _d == 'ID':                                          _r = 1
                elif _i['drop_suggested']:                                _r = 2
                elif _d == 'Free Text':                                   _r = 3
                elif _d in ('Date', 'DateTime', 'Time only', 'Day-of-Week'): _r = 4
                elif _d == 'Binary':                                      _r = 5
                elif _d == 'Categorical':                                 _r = 6
                else:                                                     _r = 7
                return (_r, _c)

            _sorted_items = sorted(col_type_info.items(), key=_col_sort_key)
            _n_visible    = sum(1 for _c, _i in _sorted_items if _col_visible(_c, _i))
            _n_total      = len(_sorted_items)

            if _n_visible < _n_total:
                st.caption(
                    f"Showing **{_n_visible}** of {_n_total} columns "
                    f"— clear filters above to see all."
                )

            # ── Per-row table (manual layout so each row gets filtered options) ──
            _hc = st.columns([2.0, 1.8, 2.2, 0.7, 0.8, 0.9, 2.0, 2.5])
            for _hl, _hcol in zip(
                ["Column", "Detected", "Override", "Drop", "Unique", "Missing", "Sample", "Drop Reason"],
                _hc
            ):
                _hcol.markdown(
                    f"<span style='font-size:0.72rem;color:#8b949e;"
                    f"text-transform:uppercase;letter-spacing:0.06em'>{_hl}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<hr style='margin:4px 0 6px 0;border-color:#30363d'>",
                        unsafe_allow_html=True)

            for col, info in _sorted_items:
                if not _col_visible(col, info):
                    continue

                _row = st.columns([2.0, 1.8, 2.2, 0.7, 0.8, 0.9, 2.0, 2.5])
                _name_color = "color:#f0883e;" if info['drop_suggested'] else ""
                _row[0].markdown(
                    f"<span style='font-family:monospace;font-size:0.82rem;{_name_color}'>`{col}`</span>",
                    unsafe_allow_html=True,
                )
                _row[1].markdown(
                    f"<span style='font-size:0.82rem'>{info['icon']} {info['detected']}</span>",
                    unsafe_allow_html=True,
                )

                # Filtered override options — no temporal/text for numeric columns
                _opts    = _override_options_for(info)
                _ovr_key  = f"_ct_ovr_{col}_v{_ct_ver}"
                _drop_key = f"_ct_drop_{col}_v{_ct_ver}"

                _ovr_val  = _row[2].selectbox(
                    "##", options=_opts, key=_ovr_key,
                    label_visibility="collapsed",
                )
                _drop_val = _row[3].checkbox(
                    "##", key=_drop_key,
                    label_visibility="collapsed",
                )

                _row[4].markdown(
                    f"<span style='font-size:0.82rem'>{info['n_unique']}</span>",
                    unsafe_allow_html=True,
                )
                _row[5].markdown(
                    f"<span style='font-size:0.82rem'>{info['missing_pct']*100:.0f}%</span>",
                    unsafe_allow_html=True,
                )
                _row[6].markdown(
                    f"<span style='font-size:0.78rem;color:#8b949e'>{info['sample'][:30]}</span>",
                    unsafe_allow_html=True,
                )
                if info['drop_reason']:
                    _row[7].markdown(
                        f"<span style='font-size:0.75rem;color:#f0883e'>{info['drop_reason']}</span>",
                        unsafe_allow_html=True,
                    )

            # ── Validation messages ───────────────────────────────────────────
            if _val_issues:
                st.markdown("")
                for _vc, _vt, _vs, _vm in _val_issues:
                    if _vs == 'error':
                        st.error(f"**`{_vc}` → {_vt}:** {_vm}", icon="🚫")
                    else:
                        st.warning(f"**`{_vc}` → {_vt}:** {_vm}", icon="⚠️")

            # ── Apply Changes button — BOTTOM ─────────────────────────────────
            st.markdown("")
            _ap_cols = st.columns([3, 1])
            _ap_cols[0].caption(
                ("Pending: " + ", ".join(_parts) + ".") if _has_pending
                else "No changes pending — all columns use auto-detected types."
            )
            if _ap_cols[1].button("✅ Apply Changes", type="primary",
                                   disabled=(not _has_pending or _has_errors),
                                   key="_ct_apply_btn"):
                _do_apply_changes()

    # =========================================================================
    # STEP 2: ANALYZE & SUGGEST
    # =========================================================================
    st.divider()
    st.header("② Analyze & Get Suggestions")
    if st.session_state.X_train is not None:

        if not meta_models:
            st.warning("No meta-models loaded. Please check the model directory.")
        else:
            _analyze_cols = st.columns([2, 1])
            if _analyze_cols[0].button("🔍 Analyze Dataset", type="primary"):
                # Clear stale MODEL results only — deliberately keep suggestions/
                # selected_indices alive so Step ③ stays visible while the new
                # analysis runs (prevents the "Step 3 disappears" flash).
                for _stale_k in [
                    'baseline_model', 'enhanced_model',
                    'baseline_train_cols', 'enhanced_train_cols',
                    'baseline_val_metrics', 'enhanced_val_metrics',
                    'baseline_col_encoders', 'enhanced_col_encoders',
                    'fitted_params', 'X_train_enhanced',
                    'X_test_raw', 'X_test_enhanced', '_test_df_original',
                    '_test_baseline_metrics', '_test_enhanced_metrics',
                    '_test_file_sig', '_test_file_name',
                    '_suggestion_verdicts', '_low_imp_cols',
                    '_val_metrics_rows', '_fi_b_pct', '_fi_e_original',
                    '_fi_e_new', '_fi_orig_pct', '_fi_new_pct',
                    'apply_imbalance',
                ]:
                    st.session_state[_stale_k] = None
                # Note: suggestions / selected_indices are NOT cleared here.
                # They will be overwritten below once the new analysis finishes.

                X = st.session_state.X_train
                y = st.session_state.y_train
                n_classes = st.session_state.n_classes

                # ── Quick-mode sub-sampling ──────────────────────────────────
                _qm_on      = st.session_state.get('_quick_mode', False)
                _qm_n       = int(st.session_state.get('_quick_n', max(2_000, len(X) // 10)))
                _qm_sampled = False          # did we actually subsample?
                X_for_analysis, y_for_analysis = X, y
                if _qm_on and len(X) > _qm_n:
                    # Use stratified sampling to preserve class balance
                    try:
                        from sklearn.model_selection import train_test_split as _tts
                        _keep_frac = _qm_n / len(X)
                        _, _qm_X_s, _, _qm_y_s = _tts(
                            X, y, test_size=_keep_frac, random_state=42,
                            stratify=y if y.value_counts().min() >= 2 else None,
                        )
                        X_for_analysis = _qm_X_s.reset_index(drop=True)
                        y_for_analysis = _qm_y_s.reset_index(drop=True)
                    except Exception:
                        # Fallback to random sample if stratification fails
                        _qm_idx        = X.sample(n=_qm_n, random_state=42).index
                        X_for_analysis = X.loc[_qm_idx].reset_index(drop=True)
                        y_for_analysis = y.loc[_qm_idx].reset_index(drop=True)
                    _qm_sampled    = True
                    st.info(
                        f"⚡ Quick mode — analyzing **{_qm_n:,}** of **{len(X):,}** rows "
                        f"({_qm_n/len(X)*100:.0f}%) with stratified sampling (class balance preserved). "
                        "Dataset-level shape (n_rows, row_col_ratio) uses the real count "
                        "so meta-model predictions are not skewed."
                    )
                # ─────────────────────────────────────────────────────────────
                # Compute quick baseline on the analysis subset so meta-models
                # receive the real baseline_score / headroom.
                with st.spinner("Computing quick baseline..."):
                    try:
                        _stratify = y_for_analysis if y_for_analysis.value_counts().min() >= 2 else None
                        if _stratify is None:
                            st.warning("Some classes have only 1 sample — stratified split disabled for quick baseline.")
                        _X_tr, _X_vl, _y_tr, _y_vl = train_test_split(
                            X_for_analysis, y_for_analysis, test_size=0.2, random_state=42, stratify=_stratify
                        )
                        _X_tr_enc, _X_vl_enc, _ = prepare_data_for_model(_X_tr, _X_vl)
                        _quick = lgb.LGBMClassifier(**{**BASE_PARAMS, 'n_estimators': 100})
                        _quick.fit(_X_tr_enc, _y_tr,
                                   eval_set=[(_X_vl_enc, _y_vl)],
                                   callbacks=[lgb.early_stopping(10, verbose=False)])
                        if n_classes == 2:
                            _bs = roc_auc_score(_y_vl, _quick.predict_proba(_X_vl_enc)[:, 1])
                        else:
                            _bs = roc_auc_score(_y_vl, _quick.predict_proba(_X_vl_enc),
                                                multi_class='ovr', average='weighted')
                        _bs_std = float(abs(min(_bs - 0.5, 1.0 - _bs)) * 0.1 + 0.01)
                    except Exception as _e:
                        st.warning(f"Quick baseline failed ({_e}), defaulting to 0.5")
                        _bs, _bs_std = 0.5, 0.05
                st.session_state['_analyze_baseline_score'] = _bs
                st.session_state['_analyze_baseline_std']   = _bs_std
                st.info(f"Quick baseline ROC-AUC: **{_bs:.4f}**")

                progress_bar = st.progress(0, text="Generating suggestions...")

                def update_progress(pct):
                    progress_bar.progress(min(pct, 1.0), text=f"Analyzing... {pct*100:.0f}%")

                suggestions, skipped_info, advisories, ds_meta = generate_suggestions(
                    X_for_analysis, y_for_analysis, meta_models,
                    baseline_score=_bs,
                    baseline_std=_bs_std,
                    progress_cb=update_progress,
                    type_reassignments=st.session_state.get('_applied_types', {}),
                    real_n_rows=len(X) if _qm_sampled else None,
                )
                progress_bar.empty()

                # Filter and deduplicate
                suggestions = [s for s in suggestions if s['predicted_delta'] >= delta_threshold]
                suggestions = deduplicate_suggestions(suggestions)

                st.session_state.suggestions = suggestions
                st.session_state.skipped_info = skipped_info
                st.session_state.advisories = advisories
                st.session_state.ds_meta = ds_meta
                st.success(f"Generated {len(suggestions)} suggestions")
                # Inform the user that interaction search is capped so they
                # know the list may not be exhaustive on wide datasets.
                _n_interact = sum(1 for s in suggestions if s['type'] == 'interaction')
                if _n_interact > 0:
                    st.caption(
                        f"ℹ️ Interaction search is capped at the top-30 num×num, "
                        f"20 cat×num, and 10 cat×cat pairs (ranked by baseline importance). "
                        f"On wide datasets some column pairs may not have been evaluated."
                    )

        # Display suggestions
        if st.session_state.suggestions:
            suggestions = st.session_state.suggestions

            # ── Reset analysis button ─────────────────────────────────────────
            if _analyze_cols[1].button("🔄 Reset Analysis", help="Clear suggestions and start fresh"):
                for _k in ['suggestions', 'skipped_info', 'advisories', 'selected_indices',
                            'baseline_model', 'enhanced_model',
                            'baseline_train_cols', 'enhanced_train_cols',
                            'baseline_val_metrics', 'enhanced_val_metrics',
                            'baseline_col_encoders', 'enhanced_col_encoders',
                            'fitted_params', 'X_train_enhanced',
                            'X_test_raw', 'X_test_enhanced', '_test_df_original',
                            '_test_baseline_metrics', '_test_enhanced_metrics',
                            '_test_file_sig', '_test_file_name']:
                    st.session_state[_k] = None
                for _ek in list(st.session_state.keys()):
                    if _ek.startswith("_expander_open_"):
                        del st.session_state[_ek]
                st.rerun()
            # ─────────────────────────────────────────────────────────────────

            # ── Initialise checkbox states on new analysis ────────────────────
            _sig = (
                f"{len(suggestions)}"
                f"__{suggestions[0]['method'] if suggestions else ''}"
                f"__{st.session_state.X_train.shape if st.session_state.X_train is not None else ''}"
                f"__{st.session_state.target_col or ''}"
            )
            if st.session_state.get("_suggestions_sig") != _sig:
                st.session_state._suggestions_sig = _sig
                default_k = min(top_k, len(suggestions))
                for i, s in enumerate(suggestions):
                    _raw = s.get("predicted_delta_raw", s.get("predicted_delta", 0))
                    val = s['auto_checked'] if 'auto_checked' in s else (i < default_k and _raw > 0)
                    st.session_state[f"suggest_check_{i}"]         = val
                    st.session_state[f"_ck_persist_{i}"]           = val   # seed persistent key
                    st.session_state[f"_initial_auto_checked_{i}"] = val   # immutable: was this system-ticked?
                # Reset group expansion so groups start collapsed for the new analysis
                for _ek in list(st.session_state.keys()):
                    if _ek.startswith("_expander_open_"):
                        del st.session_state[_ek]
            # ─────────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────

            # ── Dataset advisories ────────────────────────────────────────────
            advisories = st.session_state.get('advisories') or []
            if advisories:
                _SEVERITY_ICON  = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}
                _SEVERITY_COLOR = {'high': '#4a1010', 'medium': '#3d3010', 'low': '#0e2a3d'}
                for adv in advisories:
                    icon = _SEVERITY_ICON.get(adv['severity'], '💡')
                    _adv_cat = adv['category'].replace(' ', '_')
                    _adv_open_key = f"_adv_open_{_adv_cat}"
                    _adv_expanded = st.session_state.get(_adv_open_key, adv['severity'] == 'high')
                    with st.expander(f"{icon} **{adv['title']}**", expanded=_adv_expanded):
                        st.markdown(adv['detail'])
                        suggested = adv.get('suggested_params')
                        if suggested:
                            _hp_key = f"_custom_hp_{_adv_cat}"
                            if _hp_key not in st.session_state:
                                st.session_state[_hp_key] = BASE_PARAMS.copy()
                            st.markdown("**Model hyperparameters** *(used for the enhanced model)*")
                            _hp_btn_key = f"_hp_btn_{_adv_cat}"
                            if st.button(f"⚡ Apply suggested hyperparameters", key=_hp_btn_key):
                                merged = BASE_PARAMS.copy()
                                merged.update(suggested)
                                st.session_state[_hp_key] = merged
                                for _pk in ['num_leaves', 'min_child_samples', 'reg_alpha',
                                            'reg_lambda', 'subsample', 'colsample_bytree']:
                                    if _pk in merged:
                                        st.session_state[f"_hp_{_adv_cat}_{_pk}"] = merged[_pk]
                                st.session_state[_adv_open_key] = True
                                st.rerun()

                            _cur_hp = st.session_state[_hp_key]
                            _hp_cols = st.columns(3)
                            _editable_params = [
                                ('num_leaves',        'Num leaves',        int,   4,   256),
                                ('min_child_samples', 'Min child samples',  int,   1,   500),
                                ('reg_alpha',         'reg_alpha (L1)',     float, 0.0, 10.0),
                                ('reg_lambda',        'reg_lambda (L2)',    float, 0.0, 10.0),
                                ('subsample',         'Subsample',          float, 0.1, 1.0),
                                ('colsample_bytree',  'Colsample bytree',   float, 0.1, 1.0),
                            ]
                            for _pi, (_pk, _pl, _ptype, _pmin, _pmax) in enumerate(_editable_params):
                                _col = _hp_cols[_pi % 3]
                                _wkey = f"_hp_{_adv_cat}_{_pk}"
                                # Use session_state widget value if present (persists across reruns),
                                # otherwise fall back to the stored HP dict value
                                _default_val = st.session_state.get(_wkey, _cur_hp.get(_pk, BASE_PARAMS.get(_pk, _pmin)))
                                if _ptype == int:
                                    _new_val = _col.number_input(
                                        _pl, min_value=_pmin, max_value=_pmax,
                                        value=int(_default_val), step=1, key=_wkey,
                                    )
                                else:
                                    _new_val = _col.number_input(
                                        _pl, min_value=float(_pmin), max_value=float(_pmax),
                                        value=float(_default_val), step=0.05,
                                        format="%.2f", key=_wkey,
                                    )
                                st.session_state[_hp_key][_pk] = _new_val
            # ─────────────────────────────────────────────────────────────────

            # ── Quick stats row ───────────────────────────────────────────────
            n_pos = sum(
                1 for i, s in enumerate(suggestions)
                if s.get("predicted_delta_raw", s.get("predicted_delta", 0)) > 0
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total suggestions", len(suggestions))
            mc2.metric("Positive Δ AUC", n_pos)
            mc3.metric("Pre-selected", sum(
                st.session_state.get(f"_ck_persist_{i}", False)
                for i in range(len(suggestions))
            ))
            # ─────────────────────────────────────────────────────────────────

            # ── Bulk-action bar ───────────────────────────────────────────────
            _n_currently_selected = sum(
                st.session_state.get(f"_ck_persist_{i}", False)
                for i in range(len(suggestions))
            )
            _bulk_label_col, _bulk_btn_col = st.columns([5, 3])
            _bulk_label_col.markdown(
                f"<div style='padding:8px 0 2px 0'>"
                f"<span style='font-size:0.92rem;color:#e6edf3'>"
                f"<b>{_n_currently_selected}</b>"
                f"<span style='color:#8b949e'> of {len(suggestions)} transforms selected</span>"
                f"</span></div>",
                unsafe_allow_html=True,
            )

            # Buttons in their own styled row
            st.markdown(
                "<p style='font-size:0.75rem;color:#8b949e;"
                "text-transform:uppercase;letter-spacing:0.07em;"
                "margin:0 0 4px 0'>Quick selection</p>",
                unsafe_allow_html=True,
            )
            _bc1, _bc2, _bc3 = st.columns(3)
            if _bc1.button(
                "✅ Select All",
                help="Check every suggestion in the list",
                use_container_width=True,
            ):
                for i in range(len(suggestions)):
                    st.session_state[f"suggest_check_{i}"] = True
                    st.session_state[f"_ck_persist_{i}"]   = True
                st.rerun()
            if _bc2.button(
                "☐ Deselect All",
                help="Uncheck everything — start from a clean slate",
                use_container_width=True,
            ):
                for i in range(len(suggestions)):
                    st.session_state[f"suggest_check_{i}"] = False
                    st.session_state[f"_ck_persist_{i}"]   = False
                st.rerun()
            if _bc3.button(
                f"⭐ Top {top_k} Only",
                help=f"Select only the top {top_k} suggestions ranked by predicted Δ AUC (auto-sized from your dataset)",
                use_container_width=True,
                type="primary",
            ):
                _sorted_idxs = sorted(
                    range(len(suggestions)),
                    key=lambda i: suggestions[i].get("predicted_delta_raw", suggestions[i].get("predicted_delta", 0)),
                    reverse=True,
                )
                for i in range(len(suggestions)):
                    _val = i in _sorted_idxs[:top_k]
                    st.session_state[f"suggest_check_{i}"] = _val
                    st.session_state[f"_ck_persist_{i}"]   = _val
                st.rerun()
            # ─────────────────────────────────────────────────────────────────

            # ── Problem-grouped suggestion cards ─────────────────────────────
            st.markdown("### Suggestions by Problem Type")
            st.caption("Check or uncheck individual transforms. Custom steps can be added below.")

            # Bucket suggestions into groups (ungrouped methods fall into "other")
            groups_seen = {}
            ungrouped   = []
            for s in suggestions:
                gid = _METHOD_TO_GROUP.get(s["method"])
                if gid:
                    groups_seen.setdefault(gid, []).append(s)
                else:
                    ungrouped.append(s)

            # Render in defined order
            for g in _SUGGESTION_GROUPS:
                gid = g["id"]
                if gid not in groups_seen:
                    continue
                _render_group_card(gid, groups_seen[gid], suggestions)

            # Ungrouped fallback
            if ungrouped:
                st.markdown("**Other transforms**")
                for s in ungrouped:
                    idx = suggestions.index(s)
                    col_disp, method, delta_str, desc, delta_val = _suggestion_label(s)
                    ck          = f"suggest_check_{idx}"
                    _persist_ck = f"_ck_persist_{idx}"
                    st.session_state[ck] = st.session_state.get(
                        _persist_ck, s.get('auto_checked', True)
                    )
                    cc, nc, mc, dc, dsc = st.columns([0.4, 2.2, 1.8, 1.1, 3.5])
                    cc.checkbox(
                        " ", key=ck, label_visibility="collapsed",
                        on_change=_on_suggest_change, args=(ck, _persist_ck),
                    )
                    nc.markdown(f"`{col_disp}`")
                    mc.markdown(f"<span style='color:#79c0ff;font-size:0.8rem'>{method}</span>",
                                unsafe_allow_html=True)
                    dc.markdown(
                        f"<span style='color:{_delta_color(delta_val)};font-weight:700;"
                        f"font-family:monospace'>{delta_str}</span>",
                        unsafe_allow_html=True,
                    )
                    dsc.markdown(f"<span style='color:#8b949e;font-size:0.78rem'>{desc}</span>",
                                 unsafe_allow_html=True)
                st.markdown("---")
            # ─────────────────────────────────────────────────────────────────

            # ── Custom step adder ─────────────────────────────────────────────
            _render_custom_step_adder(st.session_state.X_train, suggestions)
            # ─────────────────────────────────────────────────────────────────

            # ── Resolve selected_indices — read from persistent keys ──────────
            st.session_state.selected_indices = [
                i for i in range(len(suggestions))
                if st.session_state.get(f"_ck_persist_{i}", False)
            ]
            n_sel = len(st.session_state.selected_indices)
            if n_sel:
                st.info(f"**{n_sel} transform(s) selected** — scroll down to train models.")
            else:
                st.warning("No transforms selected. Check at least one suggestion above.")
            # ─────────────────────────────────────────────────────────────────
    else:
        _render_locked_step("Upload a training CSV in **①** to unlock this step.")

    # =========================================================================
    # STEP 3: TRAIN MODELS
    # =========================================================================
    st.divider()
    st.header("③ Train Baseline & Enhanced Models")
    if st.session_state.selected_indices is not None and st.session_state.suggestions:

        # ── Hyperparameters + Optuna (unified expander) ───────────────────────
        # Key design decision: number_inputs use NO `key=` argument.
        # Their values are read from / written to _user_hp_overrides (a plain
        # session_state dict).  This avoids Streamlit's "cannot modify widget
        # state after instantiation" error entirely — there are no widget-bound
        # keys to conflict with when Apply / Reset updates the dict and reruns.
        _HP_EDIT_KEY = "_user_hp_overrides"
        _ALL_EDITABLE_PARAMS = [
            # (param_key,          label,               type,  min,   max,   step,  help)
            ('n_estimators',      'N Estimators',       int,   50,    2000,  1,     'Number of boosting rounds'),
            ('learning_rate',     'Learning Rate',      float, 0.001, 0.5,   0.005, 'Step size shrinkage'),
            ('num_leaves',        'Num Leaves',         int,   4,     512,   1,     'Max leaves per tree'),
            ('max_depth',         'Max Depth',          int,   -1,    30,    1,     '-1 = unlimited'),
            ('subsample',         'Subsample',          float, 0.1,   1.0,   0.05,  'Row subsampling ratio'),
            ('colsample_bytree',  'Colsample Bytree',   float, 0.1,   1.0,   0.05,  'Feature subsampling ratio'),
            ('min_child_samples', 'Min Child Samples',  int,   1,     500,   1,     'Min samples per leaf'),
            ('reg_alpha',         'reg_alpha (L1)',      float, 0.0,   20.0,  0.1,   'L1 regularisation'),
            ('reg_lambda',        'reg_lambda (L2)',     float, 0.0,   20.0,  0.1,   'L2 regularisation'),
        ]
        if _HP_EDIT_KEY not in st.session_state:
            st.session_state[_HP_EDIT_KEY] = BASE_PARAMS.copy()

        with st.expander("⚙️ Hyperparameters & Optimization (Enhanced Model)", expanded=False):

            # ── Section 1: HP editor ─────────────────────────────────────────
            st.caption(
                "💡 **Recommended: keep the defaults** — they are well-tuned for most "
                "tabular classification tasks. The **baseline model always uses fixed "
                "defaults** so comparisons stay fair. Edit only if you have a specific "
                "reason, or use Optuna below to search automatically."
            )

            _hpc1, _hpc2 = st.columns([1, 5])
            _do_reset = _hpc1.button("↺ Reset to defaults", key="_hp_reset_btn")
            _hpc2.markdown(
                "<span style='color:#8b949e;font-size:0.82rem'>"
                "Defaults: n_estimators=300 · lr=0.05 · num_leaves=31 · "
                "max_depth=6 · subsample=0.8 · colsample_bytree=0.8"
                "</span>",
                unsafe_allow_html=True,
            )
            if _do_reset:
                st.session_state[_HP_EDIT_KEY] = BASE_PARAMS.copy()
                st.rerun()

            # Render number_inputs WITHOUT key= to avoid widget-state conflicts.
            # value= is always driven from our dict; return values are written back.
            _hp_grid = st.columns(3)
            for _pi, (_pk, _pl, _ptype, _pmin, _pmax, _pstep, _phelp) in enumerate(_ALL_EDITABLE_PARAMS):
                _cur_val = st.session_state[_HP_EDIT_KEY].get(_pk, BASE_PARAMS.get(_pk, _pmin))
                with _hp_grid[_pi % 3]:
                    if _ptype == int:
                        _ret = st.number_input(
                            _pl, min_value=_pmin, max_value=_pmax,
                            value=int(_cur_val), step=_pstep, help=_phelp,
                        )
                    else:
                        _ret = st.number_input(
                            _pl, min_value=float(_pmin), max_value=float(_pmax),
                            value=float(_cur_val), step=float(_pstep),
                            format="%.3f", help=_phelp,
                        )
                    st.session_state[_HP_EDIT_KEY][_pk] = _ret

            # Live diff banner
            _changed = {
                k: (BASE_PARAMS.get(k), st.session_state[_HP_EDIT_KEY].get(k))
                for k, *_ in _ALL_EDITABLE_PARAMS
                if st.session_state[_HP_EDIT_KEY].get(k) != BASE_PARAMS.get(k)
            }
            if _changed:
                _diff_parts = [f"`{k}`: {v[0]} → **{v[1]}**" for k, v in _changed.items()]
                st.info("📝 Modified from defaults: " + " · ".join(_diff_parts))
            else:
                st.success("✅ Using default parameters")

            # ── Section 2: Optuna ────────────────────────────────────────────
            st.divider()
            st.markdown("**🔍 Optuna Hyperparameter Search** *(optional)*")

            if not _OPTUNA_AVAILABLE:
                st.warning("Optuna is not installed. Run `pip install optuna` to enable.")
            else:
                st.caption(
                    "Runs a Bayesian (TPE) search and finds the best parameter combination. "
                    "Hit **Apply** afterwards to load the result into the editor above."
                )

                _oc1, _oc2 = st.columns([1, 3])
                _n_optuna_trials = _oc1.number_input(
                    "Trials", min_value=5, max_value=500,
                    value=st.session_state.get("_optuna_n_trials_val", 30),
                    step=5,
                    help="More trials → better params, but slower. 20–50 is a good start.",
                )
                st.session_state["_optuna_n_trials_val"] = int(_n_optuna_trials)

                # Show previous result + Apply button (always visible once a run is done)
                if st.session_state.get('_optuna_best_params'):
                    _ob   = st.session_state['_optuna_best_params']
                    _oscr = st.session_state.get('_optuna_best_score', 0)
                    _oc2.success(
                        f"✅ Best ROC-AUC: **{_oscr:.4f}** · "
                        + ", ".join(f"`{k}={round(v, 4) if isinstance(v, float) else v}`"
                                    for k, v in _ob.items())
                    )
                    if st.button("📥 Apply Optuna params", key="_optuna_apply_btn", type="secondary"):
                        merged = BASE_PARAMS.copy()
                        merged.update(_ob)
                        # Simply overwrite our dict and rerun — no widget-key manipulation needed
                        st.session_state[_HP_EDIT_KEY] = merged
                        st.rerun()

                if st.button("⚡ Run Optuna Search", key="_optuna_run_btn", type="primary"):
                    _X_opt = st.session_state.get('X_train')
                    _y_opt = st.session_state.get('y_train')
                    if _X_opt is None or _y_opt is None:
                        st.warning("Upload a training CSV first.")
                    else:
                        _n_trials_run = int(st.session_state["_optuna_n_trials_val"])
                        _opt_prog  = st.progress(0.0, text="Starting Optuna search…")
                        _opt_log   = st.empty()
                        _trial_log = []

                        def _optuna_objective(trial):
                            _p = {
                                'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
                                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                                'num_leaves':        trial.suggest_int('num_leaves', 10, 200),
                                'max_depth':         trial.suggest_int('max_depth', 3, 12),
                                'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
                                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                                'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),
                                'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 5.0),
                                'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
                            }
                            try:
                                _yo = ensure_numeric_target(_y_opt.copy())
                                _strat = _yo if _yo.value_counts().min() >= 2 else None
                                _Xtr_o, _Xvl_o, _ytr_o, _yvl_o = train_test_split(
                                    _X_opt.copy(), _yo, test_size=0.2,
                                    random_state=42, stratify=_strat,
                                )
                                _Xtr_enc, _Xvl_enc, _ = prepare_data_for_model(_Xtr_o, _Xvl_o)
                                _m = lgb.LGBMClassifier(**_p)
                                _m.fit(_Xtr_enc, _ytr_o,
                                       eval_set=[(_Xvl_enc, _yvl_o)],
                                       callbacks=[lgb.early_stopping(20, verbose=False)])
                                if int(_yo.nunique()) == 2:
                                    return roc_auc_score(_yvl_o, _m.predict_proba(_Xvl_enc)[:, 1])
                                return roc_auc_score(_yvl_o, _m.predict_proba(_Xvl_enc),
                                                     multi_class='ovr', average='weighted')
                            except Exception:
                                return 0.0

                        def _optuna_cb(study, trial):
                            _frac = (trial.number + 1) / _n_trials_run
                            _opt_prog.progress(
                                min(_frac, 1.0),
                                text=f"Trial {trial.number + 1}/{_n_trials_run} "
                                     f"— best so far: {study.best_value:.4f}",
                            )
                            _trial_log.append(
                                f"Trial {trial.number + 1:03d}: {trial.value:.4f}"
                                + (" ★" if trial.value == study.best_value else "")
                            )
                            _opt_log.markdown(
                                "<details><summary style='font-size:0.8rem;color:#8b949e'>"
                                "Trial log</summary>"
                                "<pre style='font-size:0.75rem;max-height:160px;overflow:auto'>"
                                + "\n".join(_trial_log[-20:])
                                + "</pre></details>",
                                unsafe_allow_html=True,
                            )

                        _study = optuna.create_study(
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42),
                        )
                        _study.optimize(_optuna_objective, n_trials=_n_trials_run,
                                        callbacks=[_optuna_cb])

                        st.session_state['_optuna_best_params'] = _study.best_params
                        st.session_state['_optuna_best_score']  = _study.best_value
                        # Rerun immediately so the Apply button appears without extra clicks
                        st.rerun()
        # ─────────────────────────────────────────────────────────────────────

        if st.button("🚀 Train Both Models", type="primary"):
            # Clear any stale test results from a previous training run so that
            # step ④ starts fresh and doesn't show outdated comparisons.
            for _stale_k in [
                '_test_baseline_metrics', '_test_enhanced_metrics',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_file_sig', '_test_file_name',
                '_suggestion_verdicts',
            ]:
                st.session_state[_stale_k] = None

            X = st.session_state.X_train
            y = st.session_state.y_train
            n_classes = st.session_state.n_classes

            # ── Quick-mode: optionally train on sample too ────────────────────
            if (st.session_state.get('_quick_mode', False)
                    and st.session_state.get('_quick_train_sample', False)):
                _qm_n_tr = int(st.session_state.get('_quick_n', 5_000))
                if len(X) > _qm_n_tr:
                    _qm_tr_idx = X.sample(n=_qm_n_tr, random_state=42).index
                    X = X.loc[_qm_tr_idx].reset_index(drop=True)
                    y = y.loc[_qm_tr_idx].reset_index(drop=True)
                    st.info(f"⚡ Training on quick-mode sample ({_qm_n_tr:,} rows)")
            # ─────────────────────────────────────────────────────────────────

            selected_suggestions = [st.session_state.suggestions[i]
                                     for i in st.session_state.selected_indices]

            # Feature engineering suggestions only (imbalance is a model param, not a transform)
            fe_suggestions = [s for s in selected_suggestions if s.get('type') != 'imbalance']

            # Resolve imbalance strategy: start from the suggestion's auto-detection
            # (which is already encoded in the suggestion metadata), then respect
            # the user's checkbox override.
            _imb_suggestion = next(
                (s for s in st.session_state.suggestions if s.get('type') == 'imbalance'),
                None,
            )
            _imb_idx = (
                st.session_state.suggestions.index(_imb_suggestion)
                if _imb_suggestion else None
            )
            # User checked/unchecked the box?  Absent → fall back to auto_checked default.
            _user_wants_imbalance = (
                st.session_state.get(f"_ck_persist_{_imb_idx}",
                                     _imb_suggestion.get('auto_checked', False))
                if _imb_suggestion else False
            )
            _imbalance_strategy = (
                _imb_suggestion.get('imbalance_strategy', 'none')
                if _imb_suggestion else 'none'
            )
            apply_imbalance     = _user_wants_imbalance and _imbalance_strategy != 'none'
            st.session_state['apply_imbalance'] = apply_imbalance  # persisted for post-training analysis

            # Compute ratio/dominant_frac for the caption (may differ from suggestion
            # if the dataset was changed without re-analyzing).
            _y_counts_enh        = pd.Series(y).value_counts()
            _imbalance_ratio_enh = float(_y_counts_enh.max() / max(_y_counts_enh.min(), 1))
            _dominant_class_frac = float(_y_counts_enh.max() / len(y))

            # Resolve effective model params for the enhanced model.
            # Priority (lowest → highest):
            #   BASE_PARAMS → advisory panel overrides → always-visible HP editor
            _effective_params = BASE_PARAMS.copy()
            # Advisory panels (small dataset, high dimensionality)
            for _adv_key in ['_custom_hp_Small_Dataset', '_custom_hp_High_Dimensionality']:
                if _adv_key in st.session_state and st.session_state[_adv_key]:
                    _effective_params.update(st.session_state[_adv_key])
            # Always-visible HP editor is the final word
            if st.session_state.get('_user_hp_overrides'):
                _effective_params.update(st.session_state['_user_hp_overrides'])

            # Split for validation
            # Store class distribution snapshot for the report
            try:
                st.session_state['_report_class_dist'] = {
                    str(k): int(v)
                    for k, v in pd.Series(y).value_counts().sort_index().items()
                }
            except Exception:
                pass

            _stratify = y if y.value_counts().min() >= 2 else None
            if _stratify is None:
                st.warning("Some classes have only 1 sample — stratified split disabled.")
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=_stratify
            )

            # --- Compute baseline score ---
            with st.spinner("Computing baseline score...", show_time=True):
                try:
                    X_tr_enc, X_vl_enc, _ = prepare_data_for_model(X_tr, X_vl)
                    quick_model = lgb.LGBMClassifier(**{**BASE_PARAMS, 'n_estimators': 100})
                    quick_model.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)],
                                    callbacks=[lgb.early_stopping(10, verbose=False)])
                    if n_classes == 2:
                        baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc)[:, 1])
                    else:
                        baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc),
                                                        multi_class='ovr', average='weighted')
                except Exception as e:
                    st.error(f"Baseline failed: {e}")
                    baseline_score = 0.5
            st.info(f"Quick baseline ROC-AUC: **{baseline_score:.4f}**")

            # --- Apply transforms ---
            with st.spinner(f"Applying {len(fe_suggestions)} transforms...", show_time=True):
                X_tr_enh, fitted_params = fit_and_apply_suggestions(X_tr, y_tr, fe_suggestions)
                X_vl_enh = apply_fitted_to_test(X_vl, fitted_params)

                # The enhanced model is trained with transforms fitted on X_tr.
                # We use those same fitted_params at test time so the feature
                # statistics (medians, encoding maps, …) always match what the
                # model saw during training.
                st.session_state.fitted_params = fitted_params

                # Fit on the full dataset solely to produce the download artefact
                # (a fully-transformed version of the training data the user can
                # export).  This is NOT used for test evaluation.
                X_full_enh, _ = fit_and_apply_suggestions(X, y, fe_suggestions)
                st.session_state.X_train_enhanced = X_full_enh  # saved for download

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Baseline Model")
                st.caption("No class reweighting, no feature engineering")
                with st.spinner("Training baseline...", show_time=True):
                    baseline_model, baseline_cols, baseline_col_enc = train_lgbm_model(
                        X_tr, y_tr, X_vl, y_vl, n_classes, apply_imbalance=False,
                        base_params=BASE_PARAMS.copy(),
                    )
                    baseline_metrics = evaluate_on_set(baseline_model, X_vl, y_vl, baseline_cols, n_classes, baseline_col_enc)
                st.session_state.baseline_model = baseline_model
                st.session_state.baseline_train_cols = baseline_cols
                st.session_state.baseline_val_metrics = baseline_metrics

            with col2:
                enh_label_parts = [f"+{len(fe_suggestions)} transforms"]
                if apply_imbalance:
                    enh_label_parts.append("+ class reweighting (auto)")
                if _effective_params != BASE_PARAMS:
                    enh_label_parts.append("+ custom HP")
                st.subheader(f"Enhanced Model ({', '.join(enh_label_parts)})")
                if apply_imbalance:
                    _strat_label = '`is_unbalance=True`' if _imbalance_strategy == 'binary' else '`class_weight=balanced`'
                    st.caption(
                        f"Class imbalance **auto-detected** ({_imbalance_ratio_enh:.1f}:1, "
                        f"dominant class {_dominant_class_frac:.0%}) — "
                        f"{_strat_label} applied automatically"
                    )
                elif n_classes > 2 and _imbalance_ratio_enh >= _IMBALANCE_MODERATE:
                    st.caption(
                        f"⚠️ Severe multiclass imbalance detected ({_imbalance_ratio_enh:.1f}:1, "
                        f"dominant class {_dominant_class_frac:.0%}) — "
                        f"class reweighting **skipped** (would over-penalise minority classes "
                        f"~{_imbalance_ratio_enh:.0f}× and collapse model on dominant class)"
                    )
                with st.spinner("Training enhanced...", show_time=True):
                    enhanced_model, enhanced_cols, enhanced_col_enc = train_lgbm_model(
                        X_tr_enh, y_tr, X_vl_enh, y_vl, n_classes,
                        apply_imbalance=apply_imbalance,
                        imbalance_strategy=_imbalance_strategy,
                        base_params=_effective_params,
                    )
                    enhanced_metrics = evaluate_on_set(
                        enhanced_model, X_vl_enh, y_vl, enhanced_cols, n_classes, enhanced_col_enc
                    )
                st.session_state.enhanced_model = enhanced_model
                st.session_state.enhanced_train_cols = enhanced_cols
                st.session_state.enhanced_val_metrics = enhanced_metrics

            # ── Compact validation metrics table ──────────────────────────────
            _metrics_order_val = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss']
            _val_rows = []
            for _m in _metrics_order_val:
                _bv = baseline_metrics.get(_m)
                _ev = enhanced_metrics.get(_m)
                if _bv is not None and _ev is not None:
                    _diff = _ev - _bv
                    _better = _diff < 0 if _m == 'log_loss' else _diff > 0
                    _val_rows.append({
                        'Metric': _m.replace('_', ' ').upper(),
                        'Baseline': f"{_bv:.4f}",
                        'Enhanced': f"{_ev:.4f}",
                        'Δ': f"{_diff:+.4f}",
                        '': '✅' if _better else ('⚖️' if _diff == 0 else '↘️'),
                    })
            # Save to session state so the table persists across reruns (e.g. test upload)
            st.session_state['_val_metrics_rows'] = _val_rows

            # ── Feature Importance — stored for persistent display ─────────────
            try:
                # Normalise to % for both models
                _b_imps = baseline_model.feature_importances_
                _e_imps = enhanced_model.feature_importances_
                _b_pct = pd.Series(
                    (_b_imps / max(_b_imps.sum(), 1)) * 100, index=baseline_cols
                )
                _e_pct = pd.Series(
                    (_e_imps / max(_e_imps.sum(), 1)) * 100, index=enhanced_cols
                )

                # Split enhanced features into original vs new
                _orig_col_set = set(baseline_cols)
                _e_original = _e_pct[[c for c in enhanced_cols if c in _orig_col_set]]
                _e_new      = _e_pct[[c for c in enhanced_cols if c not in _orig_col_set]]

                _orig_pct_total = float(_e_original.sum())
                _new_pct_total  = float(_e_new.sum())

                # Save FI data to session state so it stays visible after test evaluation
                st.session_state['_fi_b_pct']       = _b_pct
                st.session_state['_fi_e_original']   = _e_original
                st.session_state['_fi_e_new']        = _e_new
                st.session_state['_fi_orig_pct']     = _orig_pct_total
                st.session_state['_fi_new_pct']      = _new_pct_total

            except Exception as _fi_err:
                st.info(f"Could not build importance display: {_fi_err}")
            # ─────────────────────────────────────────────────────────────────
            # Train final models on full training data
            with st.spinner("Training final models on full training set...", show_time=True):
                X_tr_full, _, baseline_full_enc = prepare_data_for_model(X, X.iloc[:1])
                # Baseline intentionally uses BASE_PARAMS (no user overrides) so it
                # measures the raw starting point unaffected by any tuning decisions.
                final_baseline = lgb.LGBMClassifier(**BASE_PARAMS)
                final_baseline.fit(X_tr_full, y)
                st.session_state.baseline_model = final_baseline
                st.session_state.baseline_train_cols = X_tr_full.columns.tolist()
                st.session_state.baseline_col_encoders = baseline_full_enc

                # Enhanced model uses _effective_params (which may include advisory HP
                # overrides chosen by the user) so the test comparison reflects the full
                # effect of every suggestion — transforms + hyperparameter tuning.
                enh_params = _effective_params.copy()
                if apply_imbalance:
                    if _imbalance_strategy == 'binary':
                        enh_params['is_unbalance'] = True
                    elif _imbalance_strategy == 'multiclass_moderate':
                        enh_params['class_weight'] = 'balanced'
                    # 'none' → severe multiclass: no reweighting
                X_enh_enc, _, enhanced_full_enc = prepare_data_for_model(X_full_enh, X_full_enh.iloc[:1])
                final_enhanced = lgb.LGBMClassifier(**enh_params)
                final_enhanced.fit(X_enh_enc, y)
                st.session_state.enhanced_model = final_enhanced
                st.session_state.enhanced_train_cols = X_enh_enc.columns.tolist()
                st.session_state.enhanced_col_encoders = enhanced_full_enc

            st.success("Both models trained on full training data and ready for test evaluation.")
            # (download buttons are rendered below, outside this block, so they persist)

            # ── Compute post-training diagnosis (store for persistent display) ──
            try:
                _verdicts, _low_imp = _compute_suggestion_verdicts(
                    fitted_params        = st.session_state.fitted_params or [],
                    suggestions          = st.session_state.suggestions or [],
                    selected_indices     = st.session_state.selected_indices or [],
                    enhanced_model       = enhanced_model,
                    enhanced_train_cols  = enhanced_cols,
                    baseline_model       = baseline_model,
                    baseline_train_cols  = baseline_cols,
                    baseline_val_metrics = baseline_metrics,
                    enhanced_val_metrics = enhanced_metrics,
                    apply_imbalance      = apply_imbalance,
                )
                st.session_state['_suggestion_verdicts'] = _verdicts
                st.session_state['_low_imp_cols']        = _low_imp
                st.session_state['_verdicts_stale']      = False  # fresh results
            except Exception as _diag_err:
                st.session_state['_suggestion_verdicts'] = []
                st.session_state['_low_imp_cols']        = {}
                st.warning(f"Post-training diagnosis could not be computed: {_diag_err}")
            # ─────────────────────────────────────────────────────────────────

        # ── Persistent Validation Metrics (shown whenever models are trained) ──
        if st.session_state.baseline_model is not None and st.session_state.get('_val_metrics_rows'):
            st.divider()
            st.markdown("**📋 Validation Metrics**")
            st.dataframe(
                pd.DataFrame(st.session_state['_val_metrics_rows']),
                hide_index=True, use_container_width=True,
            )

        # ── Persistent Feature Importance (shown whenever models are trained) ──
        if st.session_state.baseline_model is not None and st.session_state.get('_fi_b_pct') is not None:
            st.divider()
            with st.expander("📊 Feature Importance", expanded=True):
                _fi_b_pct     = st.session_state['_fi_b_pct']
                _fi_e_orig    = st.session_state['_fi_e_original']
                _fi_e_new     = st.session_state['_fi_e_new']
                _fi_orig_pct  = st.session_state['_fi_orig_pct']
                _fi_new_pct   = st.session_state['_fi_new_pct']

                st.caption(
                    "Sorted by importance. Click any column header to re-sort.  "
                    f"Original features hold **{_fi_orig_pct:.1f}%** | "
                    f"Engineered features hold **{_fi_new_pct:.1f}%** "
                    f"({len(_fi_e_new)} new columns)"
                )

                def _make_fi_table(series, top_n=25):
                    s = series.sort_values(ascending=False).head(top_n).round(3)
                    df_fi = s.reset_index()
                    df_fi.columns = ['Feature', 'Importance %']
                    return df_fi

                def _make_fi_table_styled(series, top_n=25, flagged_cols=None):
                    """Like _make_fi_table but returns a Styler with flagged rows greyed out."""
                    df_fi = _make_fi_table(series, top_n=top_n)
                    if not flagged_cols:
                        return df_fi
                    flagged = set(flagged_cols)

                    def _style_row(row):
                        if row['Feature'] in flagged:
                            return [
                                'background-color:#1e1e1e;color:#ffffff',
                                'background-color:#1e1e1e;color:#ffffff',
                            ]
                        return ['', '']

                    return df_fi.style.apply(_style_row, axis=1)

                # Build the set of flagged engineered column names from verdicts.
                # Includes: (a) all new_cols from bad/marginal transforms,
                #           (b) individual bad_row_stats,
                #           (c) individual bad_date_subfeatures.
                _verdicts_for_fi = st.session_state.get('_suggestion_verdicts') or []
                _flagged_eng_cols = set()
                for _vv in _verdicts_for_fi:
                    if _vv.get('verdict') in ('bad', 'marginal'):
                        _flagged_eng_cols.update(_vv.get('new_cols') or [])
                    # Partial bad row stats (parent is good, individual stats aren't)
                    _flagged_eng_cols.update(_vv.get('bad_row_stats') or [])
                    # Partial bad date sub-features
                    _prefix = _vv.get('col_prefix', '')
                    for _sf in (_vv.get('bad_date_subfeatures') or []):
                        _flagged_eng_cols.add(f"{_prefix}{_sf}")

                _max_imp = max(
                    float(_fi_b_pct.max()) if not _fi_b_pct.empty else 0,
                    float(_fi_e_orig.max()) if not _fi_e_orig.empty else 0,
                    float(_fi_e_new.max())  if not _fi_e_new.empty  else 0,
                    1.0,
                )

                _tab_b, _tab_orig, _tab_eng = st.tabs([
                    "Baseline model",
                    f"Enhanced — original features ({_fi_orig_pct:.1f}%)",
                    f"Enhanced — engineered features ({_fi_new_pct:.1f}%, {len(_fi_e_new)} cols)",
                ])

                _fi_col_cfg = {
                    "Importance %": st.column_config.ProgressColumn(
                        "Importance %",
                        format="%.3f%%",
                        min_value=0.0,
                        max_value=_max_imp,
                    )
                }

                with _tab_b:
                    st.dataframe(
                        _make_fi_table(_fi_b_pct),
                        hide_index=True,
                        use_container_width=True,
                        column_config=_fi_col_cfg,
                    )
                with _tab_orig:
                    if not _fi_e_orig.empty:
                        st.dataframe(
                            _make_fi_table(_fi_e_orig),
                            hide_index=True,
                            use_container_width=True,
                            column_config=_fi_col_cfg,
                        )
                    else:
                        st.info("No original features retained in the enhanced model.")
                with _tab_eng:
                    if not _fi_e_new.empty:
                        _top_new = _fi_e_new.sort_values(ascending=False)
                        _n_flagged_eng = sum(1 for c in _top_new.index if c in _flagged_eng_cols)
                        _flagged_caption = (
                            f"  ·  **{_n_flagged_eng} greyed out** — from underperforming transforms "
                            f"(grey = transform failed, not just low rank)"
                            if _n_flagged_eng else ""
                        )
                        st.caption(
                            f"Top engineered feature: **{_top_new.index[0]}** "
                            f"({_top_new.iloc[0]:.2f}%). "
                            "Drop in original-feature importance is expected — signal is shared across more columns."
                            + _flagged_caption
                        )
                        st.dataframe(
                            _make_fi_table_styled(_fi_e_new, flagged_cols=_flagged_eng_cols),
                            hide_index=True,
                            use_container_width=True,
                            column_config=_fi_col_cfg,
                        )
                    else:
                        st.info("No engineered features were added.")

            # ── Stale results banner ───────────────────────────────────────────
            if st.session_state.get('_verdicts_stale'):
                st.warning(
                    "⚠️ These results are from the **previous** training run. "
                    "Scroll up to **③ Train Models** and re-train to refresh them.",
                    icon=None,
                )

            # ── Auto-deselect shortcut below Feature Importance ────────────────
            _verdicts_fi = st.session_state.get('_suggestion_verdicts') or []
            _bad_marginal_fi = [
                v for v in _verdicts_fi
                if v.get('verdict') in ('bad', 'marginal')
                and v.get('type') != 'imbalance'   # imbalance is never auto-deselected
            ]
            _deselectable_fi = [
                v for v in _bad_marginal_fi
                if v.get('sug_idx') is not None and v.get('type') != 'imbalance'
            ]
            # Also collect date_features verdicts that are overall 'good' but have
            # individual zero-importance sub-features (e.g. is_weekend) to prune.
            _date_subfeature_fi = [
                v for v in _verdicts_fi
                if v.get('bad_date_subfeatures') and v.get('sug_idx') is not None
                and v.get('verdict') not in ('bad', 'marginal')
            ]
            # Also collect row_numeric_stats verdicts that are overall 'good' but have
            # individual low-importance stats (e.g. row_range) to prune.
            _row_stat_fi = [
                v for v in _verdicts_fi
                if v.get('bad_row_stats') and v.get('sug_idx') is not None
                and v.get('verdict') not in ('bad', 'marginal')
            ]
            _total_fi_actions = (
                len(_deselectable_fi)
                + len(_date_subfeature_fi)
                + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_fi)
            )
            # Show button whenever there are bad/marginal transforms or bad date/row sub-features
            if _bad_marginal_fi or _date_subfeature_fi or _row_stat_fi:
                st.markdown("")
                if st.button(
                    f"⚡ Auto-deselect {_total_fi_actions} bad/marginal transform(s) & re-train",
                    help="Unticks underperforming transforms in Step ② and resets Step ③ — scroll up to re-train.",
                    type="primary",
                    key="_auto_deselect_fi",
                ):
                    for _v in _deselectable_fi:
                        st.session_state[f"_ck_persist_{_v['sug_idx']}"] = False
                    # Stage date sub-feature deselections in a non-widget key.
                    # ui_components applies them before st.checkbox is called on the next run.
                    _pending = st.session_state.get('_pending_date_deselect', {})
                    for _v in _date_subfeature_fi:
                        _si = _v['sug_idx']
                        _pending[_si] = list(_v['bad_date_subfeatures'])
                    st.session_state['_pending_date_deselect'] = _pending
                    # Stage row-stat sub-feature deselections in a non-widget key.
                    _pending_rs = st.session_state.get('_pending_row_stat_deselect', {})
                    for _v in _row_stat_fi:
                        _si = _v['sug_idx']
                        _pending_rs[_si] = list(_v['bad_row_stats'])
                    st.session_state['_pending_row_stat_deselect'] = _pending_rs
                    for _rk in [
                        'baseline_model', 'enhanced_model',
                        'baseline_train_cols', 'enhanced_train_cols',
                        'baseline_val_metrics', 'enhanced_val_metrics',
                        'baseline_col_encoders', 'enhanced_col_encoders',
                        'fitted_params', 'X_train_enhanced',
                        'X_test_raw', 'X_test_enhanced', '_test_df_original',
                        '_test_baseline_metrics', '_test_enhanced_metrics',
                    ]:
                        st.session_state[_rk] = None
                    st.session_state['_verdicts_stale'] = True
                    st.rerun()

            # ── Bottom-2% feature importance warning ──────────────────────────
            _fi_b_pct_warn = st.session_state.get('_fi_b_pct')
            if _fi_b_pct_warn is not None and not st.session_state.get('_verdicts_stale'):
                try:
                    _n_feats = len(_fi_b_pct_warn)
                    # Bottom 2% means columns below the 2nd percentile of importance
                    _p2_thresh = float(np.percentile(_fi_b_pct_warn.values, 2))
                    _useless = _fi_b_pct_warn[
                        (_fi_b_pct_warn <= _p2_thresh) & (_fi_b_pct_warn < 0.05)
                    ]
                    if len(_useless) >= 2 and len(_useless) < _n_feats:
                        _useless_names = _useless.sort_values().index.tolist()
                        with st.expander(
                            f"🗑️ {len(_useless_names)} potentially useless feature(s) "
                            f"(bottom 2% of baseline importance)",
                            expanded=False,
                        ):
                            st.markdown(
                                "These columns contribute less than **0.05%** importance "
                                "in the baseline model and fall in the bottom 2% of all "
                                "features. They are unlikely to help and may introduce noise. "
                                "Consider dropping them in the **Column Types** panel above."
                            )
                            _useless_df = _useless.sort_values().reset_index()
                            _useless_df.columns = ['Column', 'Importance %']
                            _useless_df['Importance %'] = _useless_df['Importance %'].round(4)
                            st.dataframe(_useless_df, hide_index=True, use_container_width=True)
                            st.caption(
                                "To drop these columns: open **🔬 Column Types & Drop Suggestions** "
                                "above → check the drop box for each → Apply Changes → Re-analyze."
                            )
                except Exception:
                    pass
            # ─────────────────────────────────────────────────────────────────
        if st.session_state.baseline_model is not None:
            st.divider()
            st.subheader("📥 Download Training Data")

            # ── Preview: raw vs enhanced ──────────────────────────────────────
            _X_raw_prev = st.session_state.get('X_train')
            _X_enh_prev = st.session_state.get('X_train_enhanced')
            if _X_raw_prev is not None and _X_enh_prev is not None:
                _n_new_cols = _X_enh_prev.shape[1] - _X_raw_prev.shape[1]
                with st.expander(
                    f"🔍 Dataset preview after all transforms "
                    f"({_X_raw_prev.shape[1]} → {_X_enh_prev.shape[1]} columns, "
                    f"+{_n_new_cols} engineered)",
                    expanded=False,
                ):
                    _pv1, _pv2 = st.columns(2)
                    with _pv1:
                        st.caption(
                            f"**Baseline (raw)** — {_X_raw_prev.shape[0]:,} rows "
                            f"× {_X_raw_prev.shape[1]} cols"
                        )
                        st.dataframe(_X_raw_prev.head(8), use_container_width=True, height=260)
                    with _pv2:
                        st.caption(
                            f"**Enhanced (post-transform)** — {_X_enh_prev.shape[0]:,} rows "
                            f"× {_X_enh_prev.shape[1]} cols"
                        )
                        st.dataframe(_X_enh_prev.head(8), use_container_width=True, height=260)

                    # Highlight the newly added columns
                    _orig_cols = set(_X_raw_prev.columns)
                    _new_col_names = [c for c in _X_enh_prev.columns if c not in _orig_cols]
                    if _new_col_names:
                        st.caption(
                            "**Engineered columns added:** "
                            + ", ".join(f"`{c}`" for c in _new_col_names[:20])
                            + (f" (+{len(_new_col_names)-20} more)" if len(_new_col_names) > 20 else "")
                        )
            # ─────────────────────────────────────────────────────────────────

            # ── Download transformed datasets ──────────────────────────────────
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if st.session_state.get('X_train') is not None:
                    raw_csv = st.session_state.X_train.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download baseline X_train (raw)",
                        data=raw_csv,
                        file_name="X_train_baseline.csv",
                        mime="text/csv",
                        help="Original training features before any transforms",
                        key="dl_train_raw",
                    )
            with dl_col2:
                if st.session_state.get('X_train_enhanced') is not None:
                    enh_csv = st.session_state.X_train_enhanced.to_csv(index=False).encode('utf-8')
                    n_sel = len(st.session_state.get('selected_indices') or [])
                    st.download_button(
                        label="⬇️ Download enhanced X_train (post-transform)",
                        data=enh_csv,
                        file_name="X_train_enhanced.csv",
                        mime="text/csv",
                        help=f"Training features after applying {n_sel} selected transforms",
                        key="dl_train_enh",
                    )
            # ──────────────────────────────────────────────────────────────────

            # ── Export validation report (available immediately after training) ──
            st.divider()
            st.subheader("📥 Export Validation Report")
            st.caption("Report uses validation-set metrics. Upload a test CSV below for final held-out results.")
            try:
                from report_generator import add_report_download_buttons
                _lbl_train = getattr(uploaded_train, "name", "dataset").replace(".csv", "")
                add_report_download_buttons(
                    st.session_state,
                    dataset_name=_lbl_train,
                    key_suffix="val",
                    report_stage="validation",
                )
            except ImportError:
                st.info("Place `report_generator.py` in the same directory to enable report export.")
            # ──────────────────────────────────────────────────────────────────
    else:
        # Show context-aware locked message
        if st.session_state.X_train is None:
            _render_locked_step("Upload training data in **①** first.")
        elif not st.session_state.suggestions:
            _render_locked_step("Click **Analyze Dataset** in **②** to generate transform suggestions.")
        else:
            _render_locked_step("Select at least one transform in **②** and click **Analyze Dataset**.")

    # =========================================================================
    # STEP 4: UPLOAD TEST DATA & COMPARE
    # =========================================================================
    st.divider()
    _step4_ready = (
        st.session_state.baseline_model is not None
        and st.session_state.enhanced_model is not None
    )
    _step4_has_results = bool(
        st.session_state.get('_test_baseline_metrics')
        or st.session_state.get('_test_enhanced_metrics')
    )
    if _step4_ready and _step4_has_results:
        _test_hdr_cols = st.columns([3, 1])
        _po_header = "④ Upload Test Data & Predict" if st.session_state.get('_predict_only') else "④ Upload Test Data & Compare"
        _test_hdr_cols[0].header(_po_header)
        _reset_label = "🔄 Reset Predictions" if st.session_state.get('_predict_only') else "🔄 Reset Test Results"
        if _test_hdr_cols[1].button(_reset_label,
                                     help="Clear test evaluation results"):
            st.session_state['_test_baseline_metrics'] = None
            st.session_state['_test_enhanced_metrics'] = None
            st.session_state['X_test_raw']             = None
            st.session_state['X_test_enhanced']        = None
            st.session_state['_test_file_sig']         = None
            st.session_state['_test_df_original']      = None
            st.session_state['_predict_only']          = False
            st.session_state.pop('_pred_dl_cols', None)
            st.rerun()

    else:
        st.header("④ Upload Test Data & Compare / Predict")

    if _step4_ready:
        uploaded_test = st.file_uploader(
            "Upload test CSV (with or without labels)",
            type=['csv'], key='test_upload',
            help="If the target column is absent, the app enters **predict-only** mode — "
                 "you can download predictions but evaluation metrics won't be computed.",
        )

        if uploaded_test is not None:
            # Detect when the user swaps to a different test file and clear stale results
            _new_test_sig = f"{uploaded_test.name}__{uploaded_test.size}"
            if st.session_state.get('_test_file_sig') != _new_test_sig:
                st.session_state['_test_file_sig']         = _new_test_sig
                st.session_state['_test_file_name']        = uploaded_test.name
                st.session_state['_test_baseline_metrics'] = None
                st.session_state['_test_enhanced_metrics'] = None
                st.session_state['X_test_raw']             = None
                st.session_state['X_test_enhanced']        = None
                st.session_state['_test_df_original']      = None
                st.session_state['_predict_only']          = False
                st.session_state.pop('_pred_dl_cols', None)

            try:
                test_df = pd.read_csv(uploaded_test, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_test.seek(0)
                test_df = pd.read_csv(uploaded_test, encoding='latin-1')
            test_df = sanitize_feature_names(test_df)
            target_col = st.session_state.target_col
            le = st.session_state.label_encoder
            n_classes = st.session_state.n_classes

            # ── Detect predict-only mode ──────────────────────────────────
            _predict_only = target_col not in test_df.columns

            if _predict_only:
                st.info(
                    f"🔮 **Predict-only mode** — target column `{target_col}` not found in test data. "
                    f"Predictions will be generated using the enhanced model, but evaluation "
                    f"metrics cannot be computed without ground-truth labels.",
                    icon=None,
                )
                y_test = None
                X_test = test_df.copy()
            else:
                # Check for labels in the test target that were never seen during training.
                # LabelEncoder.transform crashes hard on unseen values — surface a clear
                # error so the user knows which classes to remove or remap.
                _test_labels = set(test_df[target_col].astype(str).unique())
                _train_labels = set(le.classes_)
                _unseen = _test_labels - _train_labels
                if _unseen:
                    st.error(
                        f"**Test target contains class labels not seen during training: "
                        f"`{sorted(_unseen)}`**\n\n"
                        f"Training classes: `{sorted(_train_labels)}`\n\n"
                        f"Please either:\n"
                        f"- Remove rows with these labels from the test CSV, or\n"
                        f"- Map them to the closest training class before uploading."
                    )
                    st.stop()

                y_test = pd.Series(le.transform(test_df[target_col].astype(str)), name=target_col)
                X_test = test_df.drop(columns=[target_col])

            # Save the full test features (before any drops) so that ID
            # columns can be re-attached in the prediction download section.
            st.session_state['_test_df_original'] = X_test.copy()

            # Apply the same column drops the user made on the training set so
            # the test features are consistent with what the model was trained on.
            _applied_drops = st.session_state.get('_applied_drops') or []
            _test_drops = [c for c in _applied_drops if c in X_test.columns]
            if _test_drops:
                X_test = X_test.drop(columns=_test_drops)

            # ── Column alignment check ───────────────────────────────────
            _train_cols = set(st.session_state.get('baseline_train_cols') or [])
            if _train_cols:
                _test_feature_cols = set(X_test.columns)
                _missing_from_test = _train_cols - _test_feature_cols
                _extra_in_test     = _test_feature_cols - _train_cols
                if _missing_from_test:
                    st.warning(
                        f"⚠️ **{len(_missing_from_test)} training column(s) missing from test set** "
                        f"(will be zero-filled): "
                        + ", ".join(f"`{c}`" for c in sorted(_missing_from_test)[:10])
                        + (f" +{len(_missing_from_test)-10} more" if len(_missing_from_test) > 10 else "")
                    )
                elif _extra_in_test:
                    st.info(
                        f"ℹ️ {len(_extra_in_test)} extra column(s) in test set will be ignored."
                    )
                else:
                    st.success(f"✅ All {len(_train_cols)} training columns found in test set.")
            # ─────────────────────────────────────────────────────────────

            st.write(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} columns")

            _btn_label = "🔮 Generate Predictions" if _predict_only else "📊 Evaluate on Test Set"
            if st.button(_btn_label, type="primary"):
                # Enhanced test data
                X_test_enh = apply_fitted_to_test(X_test, st.session_state.fitted_params)
                st.session_state['X_test_enhanced'] = X_test_enh
                st.session_state['X_test_raw']      = X_test
                st.session_state['_predict_only']   = _predict_only

                if _predict_only:
                    with st.spinner("Generating predictions..."):
                        baseline_test = predict_on_set(
                            st.session_state.baseline_model, X_test,
                            st.session_state.baseline_train_cols, n_classes,
                            st.session_state.get('baseline_col_encoders'),
                        )
                        enhanced_test = predict_on_set(
                            st.session_state.enhanced_model, X_test_enh,
                            st.session_state.enhanced_train_cols, n_classes,
                            st.session_state.get('enhanced_col_encoders'),
                        )
                    st.session_state['_test_baseline_metrics'] = baseline_test
                    st.session_state['_test_enhanced_metrics'] = enhanced_test
                else:
                    with st.spinner("Evaluating..."):
                        baseline_test = evaluate_on_set(
                            st.session_state.baseline_model, X_test, y_test,
                            st.session_state.baseline_train_cols, n_classes,
                            st.session_state.get('baseline_col_encoders'),
                        )
                        enhanced_test = evaluate_on_set(
                            st.session_state.enhanced_model, X_test_enh, y_test,
                            st.session_state.enhanced_train_cols, n_classes,
                            st.session_state.get('enhanced_col_encoders'),
                        )
                    st.session_state['_test_baseline_metrics'] = baseline_test
                    st.session_state['_test_enhanced_metrics'] = enhanced_test
                # Reset threshold slider so it reinitialises from new test data
                st.session_state.pop('_pr_threshold_slider', None)

        # Show results if evaluation has been run
        if st.session_state.get('_test_baseline_metrics') and st.session_state.get('_test_enhanced_metrics'):
            baseline_test = st.session_state['_test_baseline_metrics']
            enhanced_test = st.session_state['_test_enhanced_metrics']

            # --- Results ---
            _is_predict_only = st.session_state.get('_predict_only', False)

            if _is_predict_only:
                st.subheader("🔮 Predict-Only Results")
                st.success(
                    f"Predictions generated for **{len(enhanced_test.get('_y_pred', []))}** rows. "
                    f"Download them below."
                )
                st.caption(
                    "Evaluation metrics (ROC-AUC, F1, confusion matrix, etc.) are not available "
                    "because the test set does not contain the target column. Upload a test set "
                    "with the target column to compare baseline vs. enhanced models."
                )

            if not _is_predict_only:
                # --- Full evaluation results comparison ---
                st.subheader("Test Set Results")

                # Highlight headline metrics FIRST
                b_auc = baseline_test.get('roc_auc')
                e_auc = enhanced_test.get('roc_auc')
                if b_auc is not None and e_auc is not None:
                    delta_auc = e_auc - b_auc
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Baseline ROC-AUC", f"{b_auc:.4f}")
                    col2.metric("Enhanced ROC-AUC", f"{e_auc:.4f}", delta=f"{delta_auc:+.4f}")
                    pct = delta_auc / max(1.0 - b_auc, 0.001) * 100
                    _hdr_label = "Headroom Captured" if pct >= 0 else "Headroom Lost"
                    col3.metric(_hdr_label, f"{abs(pct):.1f}%", delta=f"{pct:+.1f}%")

                st.markdown("---")

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

            # ── Precision-Recall Curve & Threshold Optimisation (binary, test set) ──
            _pr_n_classes = st.session_state.get('n_classes', 0)
            _pr_test_enh  = st.session_state.get('_test_enhanced_metrics') or {}
            _pr_test_base = st.session_state.get('_test_baseline_metrics') or {}
            _pr_data_enh  = _pr_test_enh.get('pr_data')
            _pr_data_base = _pr_test_base.get('pr_data')

            if not _is_predict_only and _pr_n_classes == 2 and _pr_data_enh is not None:
                st.divider()

                with st.expander(
                    "🎯 Precision-Recall Curve & Threshold Tuning",
                    expanded=False,
                ):
                    st.info(
                        "**How does this work?**  \n"
                        "By default, models predict class 1 when the probability is ≥ 0.5 — "
                        "but 0.5 isn't always the best cutoff.  \n\n"
                        "• **Raising** the threshold → model is pickier: fewer positives, but more of them correct (higher precision).  \n"
                        "• **Lowering** it → model is more inclusive: catches more true positives, but also more false alarms (higher recall).  \n\n"
                        "The PR curve shows this tradeoff at every threshold. Use the **optimal threshold finder** to pick "
                        "the cutoff that maximises the metric you care about most.  \n\n"
                        "💡 *Useful when the cost of a false positive is very different from the cost of a missed detection.  \n"
                        "Note: ROC-AUC is not listed in the threshold table because it measures ranking quality across **all** "
                        "thresholds at once — it stays the same no matter where you set the cutoff.*",
                        icon="ℹ️",
                    )
                    # ── PR Curve plot ─────────────────────────────────────────
                    try:
                        import plotly.graph_objects as go

                        _fig_pr = go.Figure()

                        # Enhanced model curve
                        _fig_pr.add_trace(go.Scatter(
                            x=_pr_data_enh['recall'],
                            y=_pr_data_enh['precision'],
                            mode='lines',
                            name=f"Enhanced (AP={_pr_data_enh['avg_precision']:.3f})",
                            line=dict(color='#58a6ff', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(88,166,255,0.08)',
                        ))

                        # Baseline model curve (if available)
                        if _pr_data_base is not None:
                            _fig_pr.add_trace(go.Scatter(
                                x=_pr_data_base['recall'],
                                y=_pr_data_base['precision'],
                                mode='lines',
                                name=f"Baseline (AP={_pr_data_base['avg_precision']:.3f})",
                                line=dict(color='#8b949e', width=1.5, dash='dot'),
                            ))

                        # Default threshold marker (0.5)
                        _pr_enh_y_true  = _pr_test_enh.get('_y_true')
                        _pr_enh_y_proba = _pr_test_enh.get('_y_proba')
                        if _pr_enh_y_true is not None:
                            _m05 = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, 0.5)
                            _fig_pr.add_trace(go.Scatter(
                                x=[_m05['recall']],
                                y=[_m05['precision']],
                                mode='markers',
                                name='Default (t=0.5)',
                                marker=dict(color='#f0883e', size=10, symbol='diamond'),
                                showlegend=True,
                            ))

                        _fig_pr.update_layout(
                            xaxis_title='Recall',
                            yaxis_title='Precision',
                            xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#30363d'),
                            yaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#30363d'),
                            margin=dict(t=30, b=40, l=50, r=20),
                            height=360,
                            legend=dict(
                                yanchor='bottom', y=0.02,
                                xanchor='right', x=0.98,
                                bgcolor='rgba(0,0,0,0.3)',
                                font=dict(size=11),
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#8b949e', size=11),
                        )

                        st.plotly_chart(_fig_pr, use_container_width=True)
                    except Exception as _pr_plot_err:
                        st.warning(f"Could not render PR curve: {_pr_plot_err}")

                    # ── Optimal threshold finder ──────────────────────────────
                    if _pr_enh_y_true is not None and _pr_enh_y_proba is not None:
                        _optimal = _find_optimal_thresholds(_pr_enh_y_true, _pr_enh_y_proba)

                        st.markdown("**Optimal thresholds** *(enhanced model, test set)*")
                        _opt_rows = []
                        for _opt_m in ['f1', 'precision', 'recall', 'accuracy']:
                            _opt_rows.append({
                                'Optimise for': _opt_m.upper(),
                                'Best threshold': f"{_optimal[_opt_m]['threshold']:.3f}",
                                'Best value': f"{_optimal[_opt_m]['value']:.4f}",
                            })
                        st.dataframe(
                            pd.DataFrame(_opt_rows),
                            hide_index=True, use_container_width=True,
                        )

                        # ── Interactive threshold slider ──────────────────────
                        st.markdown("---")
                        st.markdown("**Explore a custom threshold**")

                        # Initialise slider value once; after that the widget
                        # manages its own state via the key.
                        if '_pr_threshold_slider' not in st.session_state:
                            st.session_state['_pr_threshold_slider'] = round(
                                _optimal['f1']['threshold'], 2,
                            )

                        # Button callbacks — on_click fires BEFORE widgets render
                        # on the next rerun, so setting the widget key here is safe.
                        def _pr_set_threshold(val):
                            st.session_state['_pr_threshold_slider'] = val

                        _pr_sl_col1, _pr_sl_col2 = st.columns([0.6, 0.4])
                        with _pr_sl_col1:
                            _custom_t = st.slider(
                                "Decision threshold",
                                min_value=0.01, max_value=0.99,
                                step=0.01,
                                key="_pr_threshold_slider",
                                help="Drag to see how metrics change at different thresholds.",
                            )
                        with _pr_sl_col2:
                            _quick_btns = st.columns(3)
                            with _quick_btns[0]:
                                st.button(
                                    "Max F1", key="_pr_btn_f1",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['f1']['threshold'], 2),),
                                )
                            with _quick_btns[1]:
                                st.button(
                                    "Max Prec", key="_pr_btn_prec",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['precision']['threshold'], 2),),
                                )
                            with _quick_btns[2]:
                                st.button(
                                    "Max Recall", key="_pr_btn_rec",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['recall']['threshold'], 2),),
                                )

                        _m_custom = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, _custom_t)
                        _m_default = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, 0.5)

                        _pr_mc1, _pr_mc2, _pr_mc3, _pr_mc4 = st.columns(4)
                        _pr_mc1.metric(
                            "F1", f"{_m_custom['f1']:.4f}",
                            delta=f"{_m_custom['f1'] - _m_default['f1']:+.4f} vs 0.5",
                        )
                        _pr_mc2.metric(
                            "Precision", f"{_m_custom['precision']:.4f}",
                            delta=f"{_m_custom['precision'] - _m_default['precision']:+.4f} vs 0.5",
                        )
                        _pr_mc3.metric(
                            "Recall", f"{_m_custom['recall']:.4f}",
                            delta=f"{_m_custom['recall'] - _m_default['recall']:+.4f} vs 0.5",
                        )
                        _pr_mc4.metric(
                            "Accuracy", f"{_m_custom['accuracy']:.4f}",
                            delta=f"{_m_custom['accuracy'] - _m_default['accuracy']:+.4f} vs 0.5",
                        )

                        # ── Mini confusion matrix at chosen threshold ─────────
                        try:
                            _y_t = np.asarray(_pr_enh_y_true)
                            _y_p = (np.asarray(_pr_enh_y_proba) >= _custom_t).astype(int)
                            _cm = confusion_matrix(_y_t, _y_p)
                            _tn, _fp, _fn, _tp = _cm.ravel()
                            _cm_col1, _cm_col2 = st.columns(2)
                            with _cm_col1:
                                st.caption(f"Confusion matrix at threshold **{_custom_t:.2f}**")
                                _cm_df = pd.DataFrame(
                                    _cm,
                                    index=[f"Actual 0", f"Actual 1"],
                                    columns=[f"Pred 0", f"Pred 1"],
                                )
                                st.dataframe(_cm_df, use_container_width=True)
                            with _cm_col2:
                                _total = len(_y_t)
                                _pos_rate = float(_y_p.sum()) / max(_total, 1) * 100
                                st.caption("Prediction summary")
                                st.markdown(
                                    f"- **Predicted positive rate**: {_pos_rate:.1f}%  \n"
                                    f"- **True positives**: {_tp:,} &nbsp;|&nbsp; **False positives**: {_fp:,}  \n"
                                    f"- **True negatives**: {_tn:,} &nbsp;|&nbsp; **False negatives**: {_fn:,}"
                                )
                        except Exception:
                            pass

                        st.caption(
                            "💡 *All metrics above are computed on the **held-out test set** — "
                            "these are unbiased estimates of real-world performance at your chosen threshold.*"
                        )

            # ── ⑤ Post-Training Analysis (shown after test evaluation) ──
            if not _is_predict_only and st.session_state.get('_suggestion_verdicts') is not None:
                st.divider()
                st.header("⑤ Analysis & Recommendations")

                _verdicts  = st.session_state['_suggestion_verdicts']
                _low_imp   = st.session_state.get('_low_imp_cols') or {}

                _VCOLOR = {'good': '#3fb950', 'marginal': '#f0883e', 'bad': '#f85149'}
                _VLABEL = {'good': 'Contributed', 'marginal': 'Marginal', 'bad': 'No contribution'}
                _TICON  = {
                    'numerical': '🔢', 'categorical': '🏷️', 'interaction': '🔗',
                    'row': '📊', 'date': '📅', 'text': '📝', 'imbalance': '⚖️',
                }

                # Exclude imbalance entries from counts — the imbalance verdict here is
                # based on *validation* metrics.  Once test results are available (this
                # block only renders post-test), Section ⑥ supersedes it with test F1.
                _n_good     = sum(1 for v in _verdicts if v['verdict'] == 'good'     and v.get('type') != 'imbalance')
                _n_marginal = sum(1 for v in _verdicts if v['verdict'] == 'marginal' and v.get('type') != 'imbalance')
                _n_bad      = sum(1 for v in _verdicts if v['verdict'] == 'bad'      and v.get('type') != 'imbalance')
                _n_remove   = _n_marginal + _n_bad

                # ── Compact summary banner ─────────────────────────────
                if _n_remove == 0 and not _low_imp:
                    st.success(
                        f"✅ All {len(_verdicts)} transforms contributed — "
                        f"the pipeline is well-optimized."
                    )
                else:
                    _parts = []
                    if _n_good:     _parts.append(f"<span style='color:#3fb950'>✅ {_n_good} contributed</span>")
                    if _n_marginal: _parts.append(f"<span style='color:#f0883e'>⚠️ {_n_marginal} marginal</span>")
                    if _n_bad:      _parts.append(f"<span style='color:#f85149'>❌ {_n_bad} not contributing</span>")
                    if _low_imp:    _parts.append(f"<span style='color:#8b949e'>🗑️ {len(_low_imp)} low-importance cols</span>")
                    st.markdown(
                        f"<div style='padding:10px 16px;background:#161b22;border:1px solid #30363d;"
                        f"border-radius:6px;font-size:0.88rem;margin-bottom:12px'>"
                        + " &nbsp;&nbsp;·&nbsp;&nbsp; ".join(_parts)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Transforms that need attention (only shown if any) ─
                # Imbalance is excluded here — its validation-based verdict is
                # superseded by the test-metrics assessment in Section ⑥ below.
                _problem_verdicts = [
                    v for v in _verdicts
                    if v['verdict'] in ('bad', 'marginal') and v.get('type') != 'imbalance'
                ]
                # Surface a brief note if the imbalance verdict would have appeared
                _imb_val_verdict = next(
                    (v for v in _verdicts if v.get('type') == 'imbalance'), None
                )
                if _imb_val_verdict and _imb_val_verdict['verdict'] in ('bad', 'marginal'):
                    st.info(
                        "⚖️ **Class reweighting** had a mixed/negative verdict on the "
                        "**validation** set — see the **Imbalance handling** row in "
                        "Section ⑥ below for the definitive test-set assessment.",
                        icon=None,
                    )
                if _problem_verdicts:
                    st.markdown(
                        "<p style='font-size:0.82rem;color:#8b949e;margin:0 0 6px 0'>"
                        "The following transforms did not pay off on the held-out test set. "
                        "Consider deselecting them and re-training.</p>",
                        unsafe_allow_html=True,
                    )
                    for _v in _problem_verdicts:
                        _color = _VCOLOR[_v['verdict']]
                        _label = _VLABEL[_v['verdict']]
                        _col_str = _v['column']
                        if _v.get('column_b'):
                            _col_str += f" × {_v['column_b']}"
                        _v_icon = '⚠️' if _v['verdict'] == 'marginal' else '❌'
                        st.markdown(
                            f"<div style='padding:7px 12px;margin:4px 0;background:#0d1117;"
                            f"border-left:3px solid {_color};border-radius:4px;font-size:0.82rem'>"
                            f"{_v_icon} <code style='font-size:0.82rem'>{_col_str}</code>"
                            f"&ensp;<span style='color:#79c0ff'>{_v['method']}</span>"
                            f"&ensp;—&ensp;<span style='color:{_color}'>{_label}</span><br>"
                            f"<span style='color:#6e7681;font-size:0.76rem'>{_v['reason']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Row-stat sub-feature notices ──────────────────
                    # Surface which individual stats have near-zero importance
                    # even though the parent transform is overall 'good'.
                    for _v in _verdicts:
                        _brs = _v.get('bad_row_stats')
                        if not _brs or _v.get('verdict') in ('bad', 'marginal'):
                            continue
                        _brs_names = ", ".join(f"`{s}`" for s in _brs)
                        st.markdown(
                            f"<div style='padding:7px 12px;margin:4px 0;background:#0d1117;"
                            f"border-left:3px solid #d29922;border-radius:4px;font-size:0.82rem'>"
                            f"⚠️ <span style='color:#79c0ff'>row_numeric_stats</span>"
                            f"&ensp;—&ensp;<span style='color:#d29922'>partial: "
                            f"{len(_brs)} of {len(_v['new_cols'])} stat(s) underperforming</span><br>"
                            f"<span style='color:#6e7681;font-size:0.76rem'>"
                            f"{_brs_names} has near-zero importance — auto-deselect will untick "
                            f"{'it' if len(_brs) == 1 else 'them'} while keeping the useful stats.</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Auto-deselect button ───────────────────────────
                    # Show whenever any bad/marginal transforms exist.
                    # Items without a resolved sug_idx are listed but can't be
                    # unticked automatically — show count in label so user knows.
                    # Auto-deselect button: imbalance is explicitly excluded — it is a
                    # model-level parameter, not a transform checkbox, and its effectiveness
                    # is assessed separately in Section ⑥ using test metrics.
                    _deselectable = [
                        v for v in _problem_verdicts
                        if v.get('sug_idx') is not None and v.get('type') != 'imbalance'
                    ]
                    # Also collect good date_features verdicts with bad sub-features
                    _date_subfeature_s5 = [
                        v for v in _verdicts
                        if v.get('bad_date_subfeatures') and v.get('sug_idx') is not None
                        and v.get('verdict') not in ('bad', 'marginal')
                    ]
                    # Also collect good row_numeric_stats verdicts with bad individual stats
                    _row_stat_s5 = [
                        v for v in _verdicts
                        if v.get('bad_row_stats') and v.get('sug_idx') is not None
                        and v.get('verdict') not in ('bad', 'marginal')
                    ]
                    _n_act = (
                        len(_deselectable)
                        + len(_date_subfeature_s5)
                        + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_s5)
                    )
                    _n_tot = (
                        len(_problem_verdicts)
                        + len(_date_subfeature_s5)
                        + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_s5)
                    )
                    if _problem_verdicts or _date_subfeature_s5 or _row_stat_s5:
                        st.markdown("")
                        _btn_label = (
                            f"⚡ Auto-deselect {_n_act} bad/marginal transform(s) & re-train"
                            if _n_act == _n_tot
                            else f"⚡ Auto-deselect {_n_act} of {_n_tot} bad/marginal transform(s) & re-train"
                        )
                        if st.button(
                            _btn_label,
                            help="Unticks these transforms in Step ② and resets Step ③ — scroll up to re-train.",
                            type="primary",
                            key="_auto_deselect_step5",
                        ):
                            for _v in _deselectable:
                                st.session_state[f"_ck_persist_{_v['sug_idx']}"] = False
                            # Stage date sub-feature deselections in a non-widget key.
                            # ui_components applies them before st.checkbox is called on the next run.
                            _pending_s5 = st.session_state.get('_pending_date_deselect', {})
                            for _v in _date_subfeature_s5:
                                _si = _v['sug_idx']
                                _pending_s5[_si] = list(_v['bad_date_subfeatures'])
                            st.session_state['_pending_date_deselect'] = _pending_s5
                            # Stage row-stat sub-feature deselections
                            _pending_rs_s5 = st.session_state.get('_pending_row_stat_deselect', {})
                            for _v in _row_stat_s5:
                                _si = _v['sug_idx']
                                _pending_rs_s5[_si] = list(_v['bad_row_stats'])
                            st.session_state['_pending_row_stat_deselect'] = _pending_rs_s5
                            # Only clear model/training state — keep verdicts & FI visible
                            for _rk in [
                                'baseline_model', 'enhanced_model',
                                'baseline_train_cols', 'enhanced_train_cols',
                                'baseline_val_metrics', 'enhanced_val_metrics',
                                'baseline_col_encoders', 'enhanced_col_encoders',
                                'fitted_params', 'X_train_enhanced',
                                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                                '_test_baseline_metrics', '_test_enhanced_metrics',
                            ]:
                                st.session_state[_rk] = None
                            st.session_state['_verdicts_stale'] = True
                            st.rerun()
                    st.markdown("")

                # ── Low-importance original columns (compact) ──────────
                if _low_imp:
                    _sorted_low = sorted(_low_imp.items(), key=lambda x: x[1])
                    _pills = " ".join(
                        f"<code style='font-size:0.78rem;background:#1c2128;padding:2px 7px;"
                        f"border-radius:3px;border:1px solid #30363d'>{c}&nbsp;{p:.2f}%</code>"
                        for c, p in _sorted_low
                    )
                    st.markdown(
                        f"<details style='margin-bottom:12px'>"
                        f"<summary style='cursor:pointer;color:#8b949e;font-size:0.83rem;"
                        f"user-select:none'>🗑️ {len(_low_imp)} low-importance original "
                        f"column(s) — near-zero baseline importance</summary>"
                        f"<div style='margin-top:8px;color:#6e7681;font-size:0.80rem'>"
                        f"Removing these may reduce noise and speed up training.<br><br>"
                        f"{_pills}</div></details>",
                        unsafe_allow_html=True,
                    )

                # ── Full verdict table (collapsed) ─────────────────────
                with st.expander(f"📋 All {len(_verdicts)} transform verdicts", expanded=False):
                    for _vk, _vl, _vi in [
                        ('good',     'Contributed',      '✅'),
                        ('marginal', 'Marginal',         '⚠️'),
                        ('bad',      'No contribution',  '❌'),
                    ]:
                        _grp = [v for v in _verdicts if v['verdict'] == _vk]
                        if not _grp:
                            continue
                        st.markdown(
                            f"<p style='font-size:0.8rem;font-weight:600;color:#8b949e;"
                            f"margin:10px 0 4px 0'>{_vi} {_vl} ({len(_grp)})</p>",
                            unsafe_allow_html=True,
                        )
                        for _v in _grp:
                            _col_str = _v['column']
                            if _v.get('column_b'):
                                _col_str += f" × {_v['column_b']}"
                            _ticon = _TICON.get(_v.get('type', ''), '•')
                            st.markdown(
                                f"<div style='padding:3px 10px;margin:2px 0;background:#0d1117;"
                                f"border-radius:4px;font-size:0.79rem;color:#c9d1d9'>"
                                f"{_ticon}&ensp;<code style='font-size:0.79rem'>{_col_str}</code>"
                                f"&ensp;<span style='color:#79c0ff'>{_v['method']}</span>"
                                f"&ensp;<span style='color:#6e7681'>— {_v['reason']}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # ── Recommendations ────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 💡 Recommendations")
                _rec_items = []

                # Use test metrics for recommendations (more reliable than val)
                _t_base_metrics = st.session_state.get('_test_baseline_metrics') or {}
                _t_enh_metrics  = st.session_state.get('_test_enhanced_metrics') or {}
                _tb_f1   = _t_base_metrics.get('f1')
                _te_f1   = _t_enh_metrics.get('f1')
                _tb_auc  = _t_base_metrics.get('roc_auc')
                _te_auc  = _t_enh_metrics.get('roc_auc')

                # 1. Underperforming transforms
                if _n_remove > 0:
                    _bad_names = [
                        (f"`{v['column']}`" + (f" × `{v['column_b']}`" if v.get('column_b') else "")
                         + f" — *{v['method']}*")
                        for v in _verdicts
                        if v['verdict'] in ('bad', 'marginal') and v.get('type') != 'imbalance'
                    ]
                    _rec_items.append((
                        "🔁",
                        f"**Re-run without {_n_remove} underperforming transform(s)**  \n"
                        f"Use the **⚡ Auto-deselect** button above, or untick {'it' if _n_remove == 1 else 'them'} "
                        f"manually in Step ②, then re-train for a leaner pipeline:  \n"
                        + "  \n".join(f"  - {n}" for n in _bad_names),
                        "normal",
                    ))
                else:
                    _rec_items.append((
                        "✅",
                        "**All transforms contributed** — the pipeline is already well-pruned.",
                        "good",
                    ))

                # 2. Low-importance columns
                if _low_imp:
                    _sorted_low_names = sorted(_low_imp.items(), key=lambda x: x[1])
                    _col_list_str = ", ".join(f"`{c}` ({p:.2f}%)" for c, p in _sorted_low_names)
                    _rec_items.append((
                        "🗑️",
                        f"**Consider dropping {len(_low_imp)} low-importance original column(s)**  \n"
                        f"{_col_list_str}",
                        "normal",
                    ))

                # 3. Imbalance effectiveness (using test F1)
                _imb_suggestion_exists = any(
                    s.get('type') == 'imbalance'
                    for s in st.session_state.get('suggestions', [])
                )
                if st.session_state.get('apply_imbalance') and _tb_f1 is not None and _te_f1 is not None:
                    _f1_delta  = _te_f1 - _tb_f1
                    _auc_delta = (_te_auc - _tb_auc) if (_tb_auc is not None and _te_auc is not None) else None
                    if _f1_delta > 0.02:
                        _imb_rating = "🟢 **Effective**"
                        if _auc_delta is None:
                            _auc_clause = "."
                        elif _auc_delta >= 0:
                            _auc_clause = f", and ROC-AUC by **+{_auc_delta:.4f}**."
                        else:
                            _auc_clause = f", though ROC-AUC decreased by **{abs(_auc_delta):.4f}**."
                        _imb_detail = (
                            f"Class reweighting improved test F1 by **+{_f1_delta:.4f}**"
                            + _auc_clause
                            + " Keep it enabled."
                        )
                    elif _f1_delta > 0:
                        _imb_rating = "🟡 **Marginal**"
                        _imb_detail = (
                            f"Class reweighting gave a small test F1 gain of **+{_f1_delta:.4f}**. "
                            "Consider a sampling strategy (e.g. SMOTE) for more impact."
                        )
                    else:
                        _imb_rating = "🔴 **Ineffective on test set**"
                        _imb_detail = (
                            f"Class reweighting did not improve test F1 (Δ = **{_f1_delta:+.4f}**). "
                            "Consider disabling it or using an oversampling approach instead."
                        )
                    _rec_items.append((
                        "⚖️",
                        f"**Imbalance handling** — {_imb_rating}  \n{_imb_detail}",
                        "normal",
                    ))
                elif _imb_suggestion_exists and not st.session_state.get('apply_imbalance') and _tb_f1 is not None:
                    _tb_prec = _t_base_metrics.get('precision')
                    _tb_rec  = _t_base_metrics.get('recall')
                    _WEAK_THRESHOLD = 0.80
                    _weak_metrics = {
                        k: v for k, v in [('F1', _tb_f1), ('Precision', _tb_prec), ('Recall', _tb_rec)]
                        if v is not None and v < _WEAK_THRESHOLD
                    }
                    if _weak_metrics:
                        _weak_str = ", ".join(f"**{k}={v:.3f}**" for k, v in _weak_metrics.items())
                        _rec_items.append((
                            "⚖️",
                            f"**Consider enabling imbalance handling**  \n"
                            f"Baseline test metrics are low: {_weak_str}. "
                            f"Go back to Step ② and tick the imbalance checkbox, then re-train.",
                            "normal",
                        ))
                    else:
                        _rec_items.append((
                            "⚖️",
                            f"**Imbalance handling not needed** — baseline metrics are strong "
                            f"(F1={_tb_f1:.3f}"
                            + (f", Precision={_tb_prec:.3f}" if _tb_prec is not None else "")
                            + (f", Recall={_tb_rec:.3f}" if _tb_rec is not None else "")
                            + f"). The model handles the class imbalance on its own.",
                            "good",
                        ))

                # 4. HP tuning headroom (using test AUC)
                if _tb_auc is not None and _te_auc is not None:
                    _headroom_pct = (_te_auc - _tb_auc) / max(1.0 - _tb_auc, 0.001) * 100
                    if _headroom_pct >= 20:
                        _rec_items.append((
                            "⚙️",
                            f"**HP tuning is likely worthwhile** ({_headroom_pct:.1f}% of gap captured)  \n"
                            f"Significant headroom remains. Consider a systematic HP search or "
                            f"increasing `n_estimators` / reducing `learning_rate`.",
                            "normal",
                        ))
                    elif _headroom_pct >= 5:
                        _rec_items.append((
                            "⚙️",
                            f"**Moderate HP tuning headroom** ({_headroom_pct:.1f}% of gap captured)  \n"
                            f"A light search on `num_leaves`, `learning_rate`, `min_child_samples` "
                            f"could further close the gap.",
                            "normal",
                        ))

                for _r_icon, _r_text, _r_type in _rec_items:
                    if _r_type == "good":
                        st.success(f"{_r_icon} {_r_text}")
                    else:
                        st.info(f"{_r_icon} {_r_text}")
            # ── End post-training analysis ────────────────────────────

            # ── Download Predictions ───────────────────────────────
            st.divider()
            st.subheader("📥 Download Predictions")

            # We need predictions from the enhanced model (primary) and
            # access to the original test columns (including dropped ID cols).
            _enh_metrics = st.session_state.get('_test_enhanced_metrics') or {}
            _y_pred      = _enh_metrics.get('_y_pred')
            _y_proba     = _enh_metrics.get('_y_pred_proba')
            _le          = st.session_state.get('label_encoder')
            _test_orig   = st.session_state.get('_test_df_original')  # before column drops

            if _y_pred is not None and _le is not None and _test_orig is not None:
                # ── Build available columns ──────────────────────────────
                # Decode predictions back to original class labels
                _pred_labels = _le.inverse_transform(_y_pred)
                _n_classes   = st.session_state.get('n_classes', 2)
                _class_names = list(_le.classes_)

                # Prediction columns that will be appended
                _pred_col_name = 'predicted_class'
                _prob_col_names = [f'prob_{c}' for c in _class_names]

                # Actual class column (available when test set had a target)
                _actual_col_name = 'actual_class'
                _y_true_raw = _enh_metrics.get('_y_true')
                _has_actual = _y_true_raw is not None
                if _has_actual:
                    _actual_labels = _le.inverse_transform(
                        np.array(_y_true_raw).astype(int)
                    )

                # Identify which original columns are ID columns (were
                # auto-dropped) vs regular feature columns.
                _applied_drops = st.session_state.get('_applied_drops') or []
                _skipped_info  = st.session_state.get('skipped_info') or {}
                _id_col_names  = set(_skipped_info.get('id_columns', {}).keys())

                # Columns from the original test set that were dropped
                # (these include IDs, constants, user-dropped cols).
                _dropped_cols = [c for c in _test_orig.columns
                                 if c in _applied_drops]
                # Columns that survived into the model
                _feature_cols = [c for c in _test_orig.columns
                                 if c not in _applied_drops]

                # ── Default selection logic ──────────────────────────────
                # Pre-select: ID columns + prediction + probabilities
                # Deselect: all feature columns (user already has those
                # in the raw/enhanced downloads and typically only wants
                # an ID + prediction mapping).
                _all_original_cols = list(_test_orig.columns)
                _all_pred_cols     = [_pred_col_name] + ([_actual_col_name] if _has_actual else []) + _prob_col_names

                # Group options for clarity
                _id_options = [c for c in _all_original_cols if c in _id_col_names]
                _other_dropped = [c for c in _dropped_cols if c not in _id_col_names]
                _non_dropped   = [c for c in _feature_cols]

                # Build the full option list with logical grouping
                _all_options = []
                _default_selection = []

                # ID columns first (pre-selected)
                for c in _id_options:
                    _all_options.append(f"🆔 {c}")
                    _default_selection.append(f"🆔 {c}")

                # Other dropped columns (not pre-selected)
                for c in _other_dropped:
                    _all_options.append(f"🗑️ {c}")

                # Feature columns (not pre-selected)
                for c in _non_dropped:
                    _all_options.append(c)

                # Prediction columns (pre-selected)
                for c in _all_pred_cols:
                    _lbl = f"🎯 {c}"
                    _all_options.append(_lbl)
                    _default_selection.append(_lbl)

                st.caption(
                    "Select which columns to include in the prediction export. "
                    "🆔 = ID column (auto-detected, dropped during training), "
                    "🗑️ = other dropped column, "
                    "🎯 = model prediction.  \n"
                    "By default only ID and prediction columns are selected."
                )

                _selected = st.multiselect(
                    "Columns to include",
                    options=_all_options,
                    default=_default_selection,
                    key="_pred_dl_cols",
                )

                if _selected:
                    # ── Assemble the export DataFrame ────────────────────
                    _export_df = pd.DataFrame(index=range(len(_y_pred)))

                    for _opt in _selected:
                        # Strip prefix emoji labels to get the real column name
                        if _opt.startswith("🆔 "):
                            _real = _opt[2:].strip()
                            _export_df[_real] = _test_orig[_real].values
                        elif _opt.startswith("🗑️ "):
                            _real = _opt[2:].strip()
                            _export_df[_real] = _test_orig[_real].values
                        elif _opt.startswith("🎯 "):
                            _real = _opt[2:].strip()
                            if _real == _pred_col_name:
                                _export_df[_pred_col_name] = _pred_labels
                            elif _real == _actual_col_name and _has_actual:
                                _export_df[_actual_col_name] = _actual_labels
                            elif _real in _prob_col_names:
                                _ci = _prob_col_names.index(_real)
                                _export_df[_real] = _y_proba[:, _ci]
                        else:
                            # Regular feature column
                            if _opt in _test_orig.columns:
                                _export_df[_opt] = _test_orig[_opt].values

                    # Preview
                    st.dataframe(_export_df.head(10), use_container_width=True, height=250)
                    st.caption(f"Showing first 10 of {len(_export_df):,} rows  ·  {len(_export_df.columns)} columns selected")

                    _pred_csv = _export_df.to_csv(index=False).encode('utf-8')
                    _fname = (st.session_state.get('_test_file_name') or 'test').replace('.csv', '')
                    st.download_button(
                        label=f"⬇️ Download predictions ({len(_export_df.columns)} columns)",
                        data=_pred_csv,
                        file_name=f"{_fname}_predictions.csv",
                        mime="text/csv",
                        key="dl_pred_csv",
                        type="primary",
                    )
                else:
                    st.warning("Select at least one column to download.")

                # ── Collapsible: raw / enhanced test data ────────────────
                with st.expander("💾 Download raw / transformed test features", expanded=False):
                    _tdl1, _tdl2 = st.columns(2)
                    with _tdl1:
                        if st.session_state.get('X_test_raw') is not None:
                            _traw_csv = st.session_state.X_test_raw.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Test set (raw features)",
                                data=_traw_csv,
                                file_name="X_test_raw.csv",
                                mime="text/csv",
                                key="dl_test_raw",
                            )
                    with _tdl2:
                        if st.session_state.get('X_test_enhanced') is not None:
                            _tenh_csv = st.session_state.X_test_enhanced.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Test set (post-transform)",
                                data=_tenh_csv,
                                file_name="X_test_enhanced.csv",
                                mime="text/csv",
                                key="dl_test_enh",
                            )

                # ── Collapsible: baseline model predictions (predict-only) ───
                if _is_predict_only:
                    _base_metrics = st.session_state.get('_test_baseline_metrics') or {}
                    _base_y_pred  = _base_metrics.get('_y_pred')
                    _base_y_proba = _base_metrics.get('_y_pred_proba')
                    if _base_y_pred is not None and _le is not None:
                        with st.expander("📊 Compare: Baseline model predictions", expanded=False):
                            st.caption(
                                "These are predictions from the **baseline** model (no feature engineering). "
                                "Compare with the enhanced predictions above to see the effect of your transforms."
                            )
                            _base_pred_labels = _le.inverse_transform(_base_y_pred)
                            _base_export = pd.DataFrame()
                            # Include ID columns
                            for _idc in _id_options:
                                _real = _idc[2:].strip() if _idc.startswith("🆔 ") else _idc
                                if _real in _test_orig.columns:
                                    _base_export[_real] = _test_orig[_real].values
                            _base_export['predicted_class_baseline'] = _base_pred_labels
                            _base_export['predicted_class_enhanced'] = _pred_labels
                            if _base_y_proba is not None and _y_proba is not None:
                                for _ci, _cn in enumerate(_class_names):
                                    _base_export[f'prob_{_cn}_baseline'] = _base_y_proba[:, _ci]
                                    _base_export[f'prob_{_cn}_enhanced'] = _y_proba[:, _ci]
                            # Flag rows where baseline and enhanced disagree
                            _base_export['models_agree'] = (
                                _base_export['predicted_class_baseline'] == _base_export['predicted_class_enhanced']
                            )
                            _n_disagree = int((~_base_export['models_agree']).sum())
                            st.write(
                                f"Models **agree** on {len(_base_export) - _n_disagree:,} / "
                                f"{len(_base_export):,} rows "
                                f"(**{_n_disagree:,}** disagreements, "
                                f"{_n_disagree / max(len(_base_export), 1) * 100:.1f}%)."
                            )
                            st.dataframe(_base_export.head(10), use_container_width=True, height=250)
                            _base_csv = _base_export.to_csv(index=False).encode('utf-8')
                            _fname_b = (st.session_state.get('_test_file_name') or 'test').replace('.csv', '')
                            st.download_button(
                                label=f"⬇️ Download baseline vs enhanced comparison",
                                data=_base_csv,
                                file_name=f"{_fname_b}_baseline_vs_enhanced.csv",
                                mime="text/csv",
                                key="dl_pred_compare",
                            )
            else:
                _po_msg = "generating predictions" if st.session_state.get('_predict_only') else "evaluating on the test set"
                st.info(f"Predictions will be available after {_po_msg}.")

            # ── Export full report with test metrics ──────────────────
            if not _is_predict_only:
                st.divider()
                st.subheader("📥 Export Full Report")
                try:
                    from report_generator import add_report_download_buttons
                    _lbl = (st.session_state.get('_test_file_name') or 'dataset').replace('.csv', '')
                    add_report_download_buttons(
                        st.session_state,
                        dataset_name=_lbl,
                        key_suffix="test",
                        report_stage="test",
                        test_baseline_metrics=baseline_test,
                        test_enhanced_metrics=enhanced_test,
                    )
                except ImportError:
                    st.info("Place `report_generator.py` in the same directory to enable report export.")
            # ─────────────────────────────────────────────────────────
    else:
        _render_locked_step("Train both models in **③** first.")
                    


if __name__ == '__main__':
    main()