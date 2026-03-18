"""
mlcompass.analysis.meta_features — Dataset / column / pair meta-feature extraction
=================================================================================
Pure functions that compute the meta-feature vectors consumed by the meta-models.
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew, kurtosis, shapiro, spearmanr, f_oneway, chi2_contingency
from scipy.stats import entropy as sp_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from mlcompass.constants import (
    SENTINEL_NC, SENTINEL_CC, BASE_PARAMS, ROW_DATASET_FEATURES,
)
from mlcompass.transforms.helpers import ensure_numeric_target


# ---------------------------------------------------------------------------
# Baseline importances
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset-level meta-features
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Column-level meta-features
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pair-level meta-features (interactions)
# ---------------------------------------------------------------------------

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

    Split-sentinel encoding:
      SENTINEL_NC = -10  ->  num+cat  (n_numerical == 1)
      SENTINEL_CC = -20  ->  cat+cat  (n_numerical == 0)
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

    # num+num specific features
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

    # num+cat specific features
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

    # cat+cat specific features
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


# ---------------------------------------------------------------------------
# Method applicability gates
# ---------------------------------------------------------------------------

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
