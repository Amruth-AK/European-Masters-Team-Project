"""
collect_interaction_features.py — Two-Column Interaction Feature Collector
==========================================================================

Collects meta-learning data for two-column interaction features.
Tests interaction transforms (product, division, addition, group_mean, 
cat_concat) and records:
  - Dataset-level meta-features
  - Pair-level meta-features (ORDER-INVARIANT, computed from both columns)
  - Intervention outcome (delta, p-value, effect size)

KEY DESIGN DECISIONS:
  1. Only pair-level meta-features that REQUIRE both columns are stored.
     No individual column stats (e.g., mean_a, mean_b) — these are 
     interchangeable and would leak column ordering into the model.
  
  2. All pair features are SYMMETRIC / order-invariant:
     - Correlations: corr(A,B) == corr(B,A)
     - MI: MI(A,B) == MI(B,A) 
     - Combined stats: sum, max, min, abs_diff of individual properties
  
  3. Column type indicator: n_numerical_cols ∈ {0, 1, 2}
     - 2 = num + num (arithmetic interactions)
     - 1 = num + cat (group-by interactions)
     - 0 = cat + cat (concatenation interactions)
  
  4. Features that only apply to certain type combinations use SENTINEL 
     value -10 when not applicable (e.g., Pearson correlation for cat+cat).
  
  5. No 3-way interactions — cut for simplicity.

Classification tasks only. Evaluation via ROC-AUC with repeated k-fold CV.

Usage:
    python collect_interaction_features.py --task_list ./task_list.json --output_dir ./output_interactions
"""

import pandas as pd
import numpy as np
import os
import json
import time
import gc
import traceback
import argparse

from scipy.stats import spearmanr, skew, f_oneway, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

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

# Sentinel values for features not applicable to a given column-type combination.
# Different values encode "how far" the current pair is from the applicable regime,
# giving tree-based meta-models an extra signal beyond n_numerical_cols.
#   SENTINEL_NC = -10 : num+cat pair  (one numeric, one categorical)
#   SENTINEL_CC = -20 : cat+cat pair  (both categorical)
# Rule: for features requiring n_numerical==k, use -10 * (distance + 1) where
#       distance = |n_numerical - k|, capped so NC=-10, CC=-20.
SENTINEL_NC = -10.0   # num+cat  (n_numerical == 1)
SENTINEL_CC = -20.0   # cat+cat  (n_numerical == 0)


# =============================================================================
# CSV SCHEMA — Interaction Features
# =============================================================================

SCHEMA_INTERACTION = [
    # --- Dataset-level (13 fields) ---
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',
    
    # --- Pair-level meta-features: ORDER-INVARIANT (20 fields) ---
    'n_numerical_cols',               # 2=num+num, 1=num+cat, 0=cat+cat
    
    # Pure pairwise features (require both columns to compute)
    'pearson_corr',                   # num+num only, else SENTINEL_NC/CC
    'spearman_corr',                  # num+num only, else SENTINEL_NC/CC
    'mutual_info_pair',               # MI between the two columns (all types)
    'mic_score',                      # Maximal Information Coefficient (all types)
    'scale_ratio',                    # max(std)/min(std), num+num only, else SENTINEL_NC/CC

    # Combined individual stats (order-invariant: sum, max, min, abs_diff)
    'sum_importance',                 # importance_a + importance_b
    'max_importance',                 # max(importance_a, importance_b)
    'min_importance',                 # min(importance_a, importance_b)
    'sum_null_pct',                   # null_pct_a + null_pct_b
    'max_null_pct',                   # max(null_pct_a, null_pct_b)
    'sum_unique_ratio',               # unique_ratio_a + unique_ratio_b
    'abs_diff_unique_ratio',          # |unique_ratio_a - unique_ratio_b|
    'sum_entropy',                    # entropy_a + entropy_b
    'abs_diff_entropy',               # |entropy_a - entropy_b|
    'sum_target_corr',                # num+num: sum of |spearman with target|, else SENTINEL_NC/CC
    'abs_diff_target_corr',           # num+num: abs diff of target correlations, else SENTINEL_NC/CC
    'sum_mi_target',                  # MI(a, target) + MI(b, target)
    'abs_diff_mi_target',             # |MI(a, target) - MI(b, target)|
    'both_binary',                    # 1 if both columns have ≤2 unique values

    # --- num+num specific features (else SENTINEL_NC/CC) ---
    'product_of_means',               # mean_a * mean_b; sign encodes joint direction
    'abs_mean_ratio',                 # max(|mean|) / (min(|mean|) + ε); scale asymmetry
    'sum_cv',                         # cv_a + cv_b  (cv = std / |mean + ε|)
    'abs_diff_cv',                    # |cv_a - cv_b|
    'sum_skewness',                   # |skew_a| + |skew_b|
    'abs_diff_skewness',              # ||skew_a| - |skew_b||
    'sign_concordance',               # fraction of rows where sign(a) == sign(b)
    'n_positive_means',               # 0/1/2: how many column means are positive

    # --- num+cat specific features (else -10 if num+num, -20 if cat+cat) ---
    'eta_squared',                    # between-group variance / total variance
    'anova_f_stat',                   # ANOVA F-statistic (numeric by categorical groups)
    'n_groups',                       # cardinality of the categorical column

    # --- cat+cat specific features (else -10 if num+cat, -20 if num+num) ---
    'cramers_v',                      # Cramér's V association statistic [0, 1]
    'joint_cardinality',              # n_unique_a * n_unique_b
    'cardinality_ratio',              # min(n_unique) / max(n_unique); similarity of granularity
    'joint_sparsity',                 # 1 - (observed combos / theoretical combos)
    
    # --- Intervention (3 fields) ---
    'method',
    'interaction_col_a', 'interaction_col_b',
    
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
# INTERACTION TRANSFORMS
# =============================================================================

def apply_interaction_transform(X_train, X_val, y_train, col_a, col_b, method):
    """
    Apply a two-column interaction transform inside CV folds.
    Returns: (success, X_train, X_val)
    """
    y_train_numeric = ensure_numeric_target(y_train)
    
    try:
        if method == 'product_interaction':
            med_a = X_train[col_a].median()
            med_b = X_train[col_b].median()
            new_col = f"{col_a}_x_{col_b}"
            X_train[new_col] = X_train[col_a].fillna(med_a) * X_train[col_b].fillna(med_b)
            X_val[new_col] = X_val[col_a].fillna(med_a) * X_val[col_b].fillna(med_b)
        
        elif method == 'division_interaction':
            eps = 1e-5
            med_a = X_train[col_a].median()
            med_b = X_train[col_b].median()
            new_col = f"{col_a}_div_{col_b}"
            X_train[new_col] = X_train[col_a].fillna(med_a) / (X_train[col_b].fillna(med_b).abs() + eps)
            X_val[new_col] = X_val[col_a].fillna(med_a) / (X_val[col_b].fillna(med_b).abs() + eps)
        
        elif method == 'addition_interaction':
            med_a = X_train[col_a].median()
            med_b = X_train[col_b].median()
            new_col = f"{col_a}_plus_{col_b}"
            X_train[new_col] = X_train[col_a].fillna(med_a) + X_train[col_b].fillna(med_b)
            X_val[new_col] = X_val[col_a].fillna(med_a) + X_val[col_b].fillna(med_b)
        
        elif method == 'abs_diff_interaction':
            med_a = X_train[col_a].median()
            med_b = X_train[col_b].median()
            new_col = f"{col_a}_absdiff_{col_b}"
            X_train[new_col] = (X_train[col_a].fillna(med_a) - X_train[col_b].fillna(med_b)).abs()
            X_val[new_col] = (X_val[col_a].fillna(med_a) - X_val[col_b].fillna(med_b)).abs()
        
        elif method == 'group_mean':
            # col_a = categorical (grouper), col_b = numeric (value)
            cat_str_train = X_train[col_a].astype(str)
            cat_str_val = X_val[col_a].astype(str)
            num_series = X_train[col_b]
            grp_map = num_series.groupby(cat_str_train).mean()
            fill_val = float(num_series.mean())
            new_col = f"group_mean_{col_b}_by_{col_a}"
            X_train[new_col] = cat_str_train.map(grp_map).fillna(fill_val).astype(float)
            X_val[new_col] = cat_str_val.map(grp_map).fillna(fill_val).astype(float)
        
        elif method == 'group_std':
            cat_str_train = X_train[col_a].astype(str)
            cat_str_val = X_val[col_a].astype(str)
            num_series = X_train[col_b]
            grp_map = num_series.groupby(cat_str_train).std()
            fill_val = float(num_series.std())
            new_col = f"group_std_{col_b}_by_{col_a}"
            X_train[new_col] = cat_str_train.map(grp_map).fillna(fill_val).astype(float)
            X_val[new_col] = cat_str_val.map(grp_map).fillna(fill_val).astype(float)
        
        elif method == 'cat_concat':
            new_col = f"{col_a}_concat_{col_b}"
            combined_train = X_train[col_a].astype(str) + "_" + X_train[col_b].astype(str)
            combined_val = X_val[col_a].astype(str) + "_" + X_val[col_b].astype(str)
            le = LabelEncoder()
            le.fit(combined_train)
            classes_set = set(le.classes_)
            combined_val_safe = combined_val.apply(
                lambda x: x if x in classes_set else '__unseen__'
            )
            if '__unseen__' in combined_val_safe.values and '__unseen__' not in classes_set:
                le.classes_ = np.append(le.classes_, '__unseen__')
            X_train[new_col] = le.transform(combined_train)
            X_val[new_col] = le.transform(combined_val_safe)
        
        return True, X_train, X_val
    
    except Exception as e:
        return False, X_train, X_val


# =============================================================================
# PAIR-LEVEL META-FEATURES (ORDER-INVARIANT)
# =============================================================================

def _encode_for_mi(series):
    """Label-encode a series for MI computation."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.median()).values
    else:
        le = LabelEncoder()
        return le.fit_transform(series.astype(str).fillna('NaN'))


def _column_entropy(series):
    """Compute entropy of a column."""
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) < 5:
            return 0.0
        try:
            counts, _ = np.histogram(clean, bins=min(50, max(int(len(clean)**0.5), 5)))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))
        except Exception:
            return 0.0
    else:
        vc = series.value_counts(normalize=True, dropna=True)
        if len(vc) == 0:
            return 0.0
        probs = vc.values
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


def _column_target_corr(series, y_numeric):
    """Spearman correlation with target (numeric only). Returns 0.0 for non-numeric
    (callers apply the appropriate SENTINEL_NC / SENTINEL_CC at the pair level)."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0.0
    try:
        clean = series.notna() & y_numeric.notna()
        if clean.sum() > 10:
            sp, _ = spearmanr(series[clean], y_numeric[clean])
            return float(abs(sp)) if not np.isnan(sp) else 0.0
        return 0.0
    except Exception:
        return 0.0


def _column_mi_target(series, y_numeric):
    """Mutual information with target."""
    try:
        encoded = _encode_for_mi(series)
        mi = mutual_info_classif(encoded.reshape(-1, 1), y_numeric, random_state=42)[0]
        return float(mi)
    except Exception:
        return 0.0


def get_pair_meta_features(col_a_series, col_b_series, y, imp_a, imp_b):
    """
    Compute order-invariant pair-level meta-features.

    All features are symmetric: swap(col_a, col_b) produces the same values.
    Features inapplicable to the current column-type combination use split sentinels:
      SENTINEL_NC = -10  →  num+cat  (n_numerical == 1)
      SENTINEL_CC = -20  →  cat+cat  (n_numerical == 0)
    For type-specific NEW features, the same distance-encoding logic applies:
      a feature requiring n_num==k uses -10 if |current - k| == 1, else -20.
    """
    y_numeric = ensure_numeric_target(y)
    a_is_num = pd.api.types.is_numeric_dtype(col_a_series)
    b_is_num = pd.api.types.is_numeric_dtype(col_b_series)

    n_numerical = int(a_is_num) + int(b_is_num)

    # Sentinel for features that require num+num (n_numerical == 2)
    sent_nn = SENTINEL_NC if n_numerical == 1 else SENTINEL_CC   # only used when n_numerical < 2

    meta = {'n_numerical_cols': n_numerical}

    # =========================================================================
    # EXISTING SHARED PAIRWISE FEATURES
    # =========================================================================

    # Pearson correlation (num+num only)
    if n_numerical == 2:
        try:
            clean = col_a_series.notna() & col_b_series.notna()
            if clean.sum() > 10:
                meta['pearson_corr'] = float(abs(col_a_series[clean].corr(col_b_series[clean])))
            else:
                meta['pearson_corr'] = 0.0
        except Exception:
            meta['pearson_corr'] = 0.0
    else:
        meta['pearson_corr'] = sent_nn

    # Spearman correlation (num+num only)
    if n_numerical == 2:
        try:
            clean = col_a_series.notna() & col_b_series.notna()
            if clean.sum() > 10:
                sp, _ = spearmanr(col_a_series[clean], col_b_series[clean])
                meta['spearman_corr'] = float(abs(sp)) if not np.isnan(sp) else 0.0
            else:
                meta['spearman_corr'] = 0.0
        except Exception:
            meta['spearman_corr'] = 0.0
    else:
        meta['spearman_corr'] = sent_nn

    # Mutual information between the two columns (all types)
    try:
        enc_a = _encode_for_mi(col_a_series)
        enc_b = _encode_for_mi(col_b_series)
        mi = mutual_info_classif(
            enc_a.reshape(-1, 1),
            (enc_b * 100).astype(int) if b_is_num else enc_b,
            random_state=42
        )[0]
        meta['mutual_info_pair'] = float(mi)
    except Exception:
        meta['mutual_info_pair'] = 0.0

    # MIC approximation via binned MI (all types)
    try:
        n = len(col_a_series)
        n_bins = min(20, max(5, int(n ** 0.4)))
        if a_is_num:
            a_clean = col_a_series.fillna(col_a_series.median())
            a_binned = pd.qcut(a_clean, q=n_bins, labels=False, duplicates='drop').values
        else:
            le_a = LabelEncoder()
            a_binned = le_a.fit_transform(col_a_series.astype(str).fillna('NaN'))
        if b_is_num:
            b_clean = col_b_series.fillna(col_b_series.median())
            b_binned = pd.qcut(b_clean, q=n_bins, labels=False, duplicates='drop').values
        else:
            le_b = LabelEncoder()
            b_binned = le_b.fit_transform(col_b_series.astype(str).fillna('NaN'))
        mi_binned = mutual_info_classif(
            a_binned.reshape(-1, 1), b_binned,
            discrete_features=True, random_state=42
        )[0]
        from scipy.stats import entropy as sp_entropy
        _, counts_b = np.unique(b_binned, return_counts=True)
        h_b = sp_entropy(counts_b / counts_b.sum(), base=2)
        meta['mic_score'] = min(float(mi_binned / max(h_b, 1e-10)) if h_b > 0 else 0.0, 1.0)
    except Exception:
        meta['mic_score'] = 0.0

    # Scale ratio (num+num only)
    if n_numerical == 2:
        try:
            std_a = col_a_series.std()
            std_b = col_b_series.std()
            min_std = min(std_a, std_b)
            max_std = max(std_a, std_b)
            meta['scale_ratio'] = float(max_std / min_std) if min_std > 1e-10 else 0.0
        except Exception:
            meta['scale_ratio'] = 0.0
    else:
        meta['scale_ratio'] = sent_nn

    # ---- Combined individual stats (order-invariant, all types) ----
    meta['sum_importance'] = float(imp_a + imp_b)
    meta['max_importance'] = float(max(imp_a, imp_b))
    meta['min_importance'] = float(min(imp_a, imp_b))

    null_a = float(col_a_series.isnull().mean())
    null_b = float(col_b_series.isnull().mean())
    meta['sum_null_pct'] = null_a + null_b
    meta['max_null_pct'] = max(null_a, null_b)

    ur_a = float(col_a_series.nunique() / max(len(col_a_series), 1))
    ur_b = float(col_b_series.nunique() / max(len(col_b_series), 1))
    meta['sum_unique_ratio'] = ur_a + ur_b
    meta['abs_diff_unique_ratio'] = abs(ur_a - ur_b)

    ent_a = _column_entropy(col_a_series)
    ent_b = _column_entropy(col_b_series)
    meta['sum_entropy'] = ent_a + ent_b
    meta['abs_diff_entropy'] = abs(ent_a - ent_b)

    # Target correlations (num+num only)
    tc_a = _column_target_corr(col_a_series, y_numeric)
    tc_b = _column_target_corr(col_b_series, y_numeric)
    if n_numerical == 2:
        meta['sum_target_corr'] = tc_a + tc_b
        meta['abs_diff_target_corr'] = abs(tc_a - tc_b)
    else:
        meta['sum_target_corr'] = sent_nn
        meta['abs_diff_target_corr'] = sent_nn

    # MI with target (all types)
    mi_a = _column_mi_target(col_a_series, y_numeric)
    mi_b = _column_mi_target(col_b_series, y_numeric)
    meta['sum_mi_target'] = mi_a + mi_b
    meta['abs_diff_mi_target'] = abs(mi_a - mi_b)

    # Both binary indicator (all types)
    meta['both_binary'] = int(col_a_series.nunique() <= 2 and col_b_series.nunique() <= 2)

    # =========================================================================
    # NEW: num+num SPECIFIC FEATURES
    # Sentinel rule: -10 if n_numerical==1 (num+cat), -20 if n_numerical==0 (cat+cat)
    # =========================================================================
    if n_numerical == 2:
        mean_a = float(col_a_series.mean())
        mean_b = float(col_b_series.mean())
        std_a  = float(col_a_series.std())
        std_b  = float(col_b_series.std())

        # product_of_means: sign tells us joint direction, magnitude encodes scale
        meta['product_of_means'] = mean_a * mean_b

        # abs_mean_ratio: order-invariant scale asymmetry between the two means
        # Always >= 1.  Use absolute values so sign doesn't affect ordering.
        abs_m_a = abs(mean_a)
        abs_m_b = abs(mean_b)
        meta['abs_mean_ratio'] = (
            max(abs_m_a, abs_m_b) / (min(abs_m_a, abs_m_b) + 1e-8)
        )

        # Coefficient of variation (std / |mean|)
        cv_a = std_a / (abs(mean_a) + 1e-8)
        cv_b = std_b / (abs(mean_b) + 1e-8)
        meta['sum_cv']       = cv_a + cv_b
        meta['abs_diff_cv']  = abs(cv_a - cv_b)

        # Skewness (use absolute value so order-invariant aggregations make sense)
        try:
            sk_a = float(skew(col_a_series.dropna()))
            sk_b = float(skew(col_b_series.dropna()))
        except Exception:
            sk_a, sk_b = 0.0, 0.0
        meta['sum_skewness']      = abs(sk_a) + abs(sk_b)
        meta['abs_diff_skewness'] = abs(abs(sk_a) - abs(sk_b))

        # Sign concordance: fraction of rows where sign(a) == sign(b)
        try:
            clean = col_a_series.notna() & col_b_series.notna()
            a_c = col_a_series[clean]
            b_c = col_b_series[clean]
            meta['sign_concordance'] = float(((a_c >= 0) == (b_c >= 0)).mean())
        except Exception:
            meta['sign_concordance'] = 0.0

        # n_positive_means: 2 = both positive, 1 = one positive, 0 = neither
        meta['n_positive_means'] = int(mean_a > 0) + int(mean_b > 0)

    else:
        for feat in ['product_of_means', 'abs_mean_ratio', 'sum_cv', 'abs_diff_cv',
                     'sum_skewness', 'abs_diff_skewness', 'sign_concordance', 'n_positive_means']:
            meta[feat] = sent_nn   # -10 for num+cat, -20 for cat+cat

    # =========================================================================
    # NEW: num+cat SPECIFIC FEATURES
    # Sentinel rule: -10 if n_numerical==2 (num+num), -20 if n_numerical==0 (cat+cat)
    # =========================================================================
    if n_numerical == 1:
        num_s = col_a_series if a_is_num else col_b_series
        cat_s = col_b_series if a_is_num else col_a_series

        n_groups = int(cat_s.nunique())
        meta['n_groups'] = n_groups

        # Eta-squared: between-group variance / total variance ∈ [0, 1]
        try:
            grand_mean = float(num_s.mean())
            groups = [num_s[cat_s == g].dropna() for g in cat_s.unique()]
            groups = [g for g in groups if len(g) > 0]
            ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups)
            ss_total   = float(((num_s - grand_mean) ** 2).sum())
            meta['eta_squared'] = float(ss_between / (ss_total + 1e-10))
        except Exception:
            meta['eta_squared'] = 0.0

        # ANOVA F-statistic
        try:
            groups_data = [
                num_s[cat_s == g].dropna().values
                for g in cat_s.unique()
            ]
            groups_data = [g for g in groups_data if len(g) > 1]
            if len(groups_data) >= 2:
                f_stat, _ = f_oneway(*groups_data)
                meta['anova_f_stat'] = float(f_stat) if not np.isnan(f_stat) else 0.0
            else:
                meta['anova_f_stat'] = 0.0
        except Exception:
            meta['anova_f_stat'] = 0.0

    else:
        # -10 if one step away (num+num), -20 if two steps away (cat+cat)
        sent_nc = -10.0 if n_numerical == 2 else -20.0
        for feat in ['eta_squared', 'anova_f_stat', 'n_groups']:
            meta[feat] = sent_nc

    # =========================================================================
    # NEW: cat+cat SPECIFIC FEATURES
    # Sentinel rule: -10 if n_numerical==1 (num+cat), -20 if n_numerical==2 (num+num)
    # =========================================================================
    if n_numerical == 0:
        nu_a = int(col_a_series.nunique())
        nu_b = int(col_b_series.nunique())

        # Cramér's V: symmetric association measure for two categoricals ∈ [0, 1]
        try:
            ct = pd.crosstab(col_a_series.astype(str), col_b_series.astype(str))
            chi2, _, _, _ = chi2_contingency(ct)
            n = len(col_a_series)
            k = min(ct.shape) - 1
            meta['cramers_v'] = min(float(np.sqrt(chi2 / (n * max(k, 1)))), 1.0)
        except Exception:
            meta['cramers_v'] = 0.0

        # joint_cardinality: total possible combinations
        meta['joint_cardinality'] = float(nu_a * nu_b)

        # cardinality_ratio: how similar the two columns are in granularity ∈ (0, 1]
        meta['cardinality_ratio'] = float(min(nu_a, nu_b) / (max(nu_a, nu_b) + 1e-10))

        # joint_sparsity: fraction of theoretical combos NOT observed ∈ [0, 1]
        try:
            ct_vals = pd.crosstab(col_a_series.astype(str), col_b_series.astype(str)).values
            actual_combos = int((ct_vals > 0).sum())
            theoretical   = nu_a * nu_b
            meta['joint_sparsity'] = float(1.0 - actual_combos / max(theoretical, 1))
        except Exception:
            meta['joint_sparsity'] = 0.0

    else:
        # -10 if one step away (num+cat), -20 if two steps away (num+num)
        sent_cc = -10.0 if n_numerical == 1 else -20.0
        for feat in ['cramers_v', 'joint_cardinality', 'cardinality_ratio', 'joint_sparsity']:
            meta[feat] = sent_cc

    return meta


# =============================================================================
# PAIR SELECTION: WHICH COLUMNS TO INTERACT
# =============================================================================

def get_interaction_pairs(X, importances, max_pairs_per_type=30):
    """
    Select pairs of columns to test interactions on.
    
    Strategy: Top columns by importance, tested pairwise.
    Returns: list of (col_a, col_b, methods_to_test)
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    
    # Sort by importance
    sorted_num = sorted(numeric_cols, key=lambda c: importances.get(c, 0), reverse=True)
    sorted_cat = sorted(cat_cols, key=lambda c: importances.get(c, 0), reverse=True)
    
    # Cap column pools
    top_num = sorted_num[:8]
    top_cat = sorted_cat[:6]
    
    pairs = []
    tested = set()
    
    # Num + Num pairs
    count = 0
    for i, a in enumerate(top_num):
        for b in top_num[i+1:]:
            if count >= max_pairs_per_type:
                break
            pair_key = tuple(sorted([a, b]))
            if pair_key in tested:
                continue
            tested.add(pair_key)
            
            methods = ['product_interaction', 'division_interaction',
                       'addition_interaction', 'abs_diff_interaction']
            pairs.append((a, b, methods))
            count += 1
    
    # Cat + Num pairs (group_mean, group_std)
    count = 0
    for cat_col in top_cat:
        for num_col in top_num:
            if count >= max_pairs_per_type:
                break
            pair_key = tuple(sorted([cat_col, num_col]))
            if pair_key in tested:
                continue
            tested.add(pair_key)
            
            pairs.append((cat_col, num_col, ['group_mean', 'group_std']))
            count += 1
    
    # Cat + Cat pairs (cat_concat)
    count = 0
    for i, a in enumerate(top_cat):
        for b in top_cat[i+1:]:
            if count >= max_pairs_per_type:
                break
            pair_key = tuple(sorted([a, b]))
            if pair_key in tested:
                continue
            tested.add(pair_key)
            
            # Skip if either is very high cardinality
            if X[a].nunique() > len(X) * 0.5 or X[b].nunique() > len(X) * 0.5:
                continue
            
            pairs.append((a, b, ['cat_concat']))
            count += 1
    
    return pairs


# =============================================================================
# TREE-GUIDED PAIR SELECTION
# =============================================================================

def get_tree_guided_pairs(X, y, numeric_cols, max_depth=4, n_trees=5):
    """
    Fit shallow decision trees and extract parent→child split pairs.
    These represent conditional dependencies = ideal interaction targets.
    
    Returns: list of (col_a, col_b, tree_score) sorted by score desc.
    """
    y_numeric = ensure_numeric_target(y)
    
    if len(numeric_cols) < 2:
        return []
    
    X_num = X[numeric_cols].copy()
    for c in X_num.columns:
        X_num[c] = X_num[c].fillna(X_num[c].median())
    
    pair_scores = {}
    
    for seed in range(n_trees):
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42 + seed,
            max_features='sqrt' if seed > 0 else None
        )
        dt.fit(X_num, y_numeric)
        
        tree = dt.tree_
        features = tree.feature
        
        for node_id in range(tree.node_count):
            if features[node_id] < 0:  # leaf
                continue
            parent_feat = numeric_cols[features[node_id]]
            
            # Check children
            for child_id in [tree.children_left[node_id], tree.children_right[node_id]]:
                if child_id >= 0 and features[child_id] >= 0:
                    child_feat = numeric_cols[features[child_id]]
                    if parent_feat != child_feat:
                        pair_key = tuple(sorted([parent_feat, child_feat]))
                        depth = 1  # simplified
                        score = 1.0 / depth
                        pair_scores[pair_key] = pair_scores.get(pair_key, 0) + score
    
    # Sort by score
    ranked = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    return [(a, b, s) for (a, b), s in ranked]


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

def collect_interaction_features(X, y, dataset_name="unknown",
                                  n_folds=5, n_repeats=3,
                                  time_budget=7200):
    """
    Collect meta-learning data for two-column interaction features.
    """
    start_time = time.time()
    y = ensure_numeric_target(y)
    
    print(f"[INT] Starting: {dataset_name} | Shape: {X.shape}")
    
    # --- Column pruning ---
    X, dropped = prune_columns(X, y)
    if dropped:
        print(f"[INT]   Dropped {len(dropped)} columns")
    
    if X.shape[1] < 2:
        print(f"[INT]   Need at least 2 columns for interactions")
        return pd.DataFrame()
    
    # --- Baseline evaluation ---
    print(f"[INT]   Computing baseline...")
    base_score, base_std, baseline_folds, importances = evaluate_model(
        X, y, n_folds=n_folds, n_repeats=n_repeats, return_importance=True
    )
    
    if base_score is None:
        print(f"[INT]   Baseline failed")
        return pd.DataFrame()
    
    print(f"[INT]   Baseline AUC: {base_score:.5f}")
    
    should_skip, _, reason = check_ceiling(base_score)
    if should_skip:
        print(f"[INT]   Skipping: {reason}")
        return pd.DataFrame()
    
    # --- Dataset meta-features ---
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = base_score
    ds_meta['baseline_std'] = base_std
    ds_meta['relative_headroom'] = max(1.0 - base_score, 0.001)
    
    null_std, _ = compute_noise_floor(baseline_folds)
    
    imp = pd.Series(importances, index=X.columns)
    imp_dict = imp.to_dict()
    
    # --- Select interaction pairs ---
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    
    # Get tree-guided pairs for numeric columns
    tree_pairs = get_tree_guided_pairs(X, y, numeric_cols)
    
    # Get importance-based pairs (all types)
    importance_pairs = get_interaction_pairs(X, imp_dict)
    
    # Merge: tree-guided pairs get priority
    tested_pair_keys = set()
    all_pairs = []
    
    # Add tree-guided pairs first (num+num only)
    for col_a, col_b, t_score in tree_pairs[:30]:
        pair_key = tuple(sorted([col_a, col_b]))
        if pair_key not in tested_pair_keys:
            tested_pair_keys.add(pair_key)
            methods = ['product_interaction', 'division_interaction',
                       'addition_interaction', 'abs_diff_interaction']
            all_pairs.append((col_a, col_b, methods))
    
    # Add importance-based pairs
    for col_a, col_b, methods in importance_pairs:
        pair_key = tuple(sorted([col_a, col_b]))
        if pair_key not in tested_pair_keys:
            tested_pair_keys.add(pair_key)
            all_pairs.append((col_a, col_b, methods))
    
    print(f"[INT]   Pairs to test: {len(all_pairs)} "
          f"(tree-guided: {len(tree_pairs[:30])}, total unique: {len(tested_pair_keys)})")
    
    # --- Test interactions ---
    results = []
    n_tests = 0
    
    for pair_idx, (col_a, col_b, methods) in enumerate(all_pairs):
        if time.time() - start_time > time_budget:
            print(f"[INT]   Time budget exceeded at pair {pair_idx}/{len(all_pairs)}")
            break
        
        # Compute pair meta-features (order-invariant)
        pair_meta = get_pair_meta_features(
            X[col_a], X[col_b], y,
            imp_a=float(imp_dict.get(col_a, 0)),
            imp_b=float(imp_dict.get(col_b, 0))
        )
        
        # Individual baseline (pair of columns)
        indiv_base_score, indiv_base_std, indiv_base_folds = evaluate_model(
            X[[col_a, col_b]], y, params=INDIVIDUAL_PARAMS,
            n_folds=n_folds, n_repeats=n_repeats
        )
        
        for method in methods:
            if time.time() - start_time > time_budget:
                break
            
            # Skip degenerate cases
            a_is_num = pd.api.types.is_numeric_dtype(X[col_a])
            b_is_num = pd.api.types.is_numeric_dtype(X[col_b])
            
            if method == 'division_interaction':
                # Skip if either is binary (0/x or 1/x is trivial)
                if X[col_a].nunique() <= 2 or X[col_b].nunique() <= 2:
                    continue
            
            if method in ('group_mean', 'group_std'):
                # col_a should be categorical, col_b should be numeric
                if a_is_num and not b_is_num:
                    col_a, col_b = col_b, col_a  # Swap so cat is first
                elif a_is_num and b_is_num:
                    continue  # Both numeric, skip group operations
                elif not a_is_num and not b_is_num:
                    continue  # Both categorical, skip group operations
                
                # group_std needs enough observations per group
                if method == 'group_std':
                    n_groups = X[col_a].nunique()
                    if len(X) / max(n_groups, 1) < 10:
                        continue
            
            # Full-model evaluation
            full_score, full_std, full_folds = evaluate_with_intervention(
                X, y, col_to_transform=col_a, col_second=col_b,
                method=method, apply_fn=apply_interaction_transform,
                n_folds=n_folds, n_repeats=n_repeats
            )
            
            if full_score is None:
                continue
            
            # Individual evaluation (pair + interaction vs pair alone)
            indiv_score, _, indiv_folds = evaluate_with_intervention(
                X[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b,
                method=method, apply_fn=apply_interaction_transform,
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
            
            # Store column names in sorted order (matches order-invariant features)
            sorted_cols = sorted([col_a, col_b])
            
            row = {**ds_meta, **pair_meta}
            row.update({
                'method': method,
                'interaction_col_a': sanitize_string(sorted_cols[0]),
                'interaction_col_b': sanitize_string(sorted_cols[1]),
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
            print(f"[INT]     {col_a}×{col_b}.{method}: full={delta_norm:+.2f}% p={p_val:.4f} {sig_str} | {indiv_str}")
    
    # Post-hoc Bonferroni correction
    df = pd.DataFrame(results)
    if not df.empty and n_tests > 0:
        df['p_value_bonferroni'] = (df['p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['is_significant_bonferroni'] = df['p_value_bonferroni'] < 0.05
        df['individual_p_value_bonferroni'] = (df['individual_p_value'].astype(float) * n_tests).clip(upper=1.0)
        df['individual_is_significant_bonferroni'] = df['individual_p_value_bonferroni'] < 0.05
    
    elapsed = time.time() - start_time
    print(f"[INT]   Done: {len(df)} rows, {n_tests} tests in {elapsed:.1f}s")
    
    return df


# =============================================================================
# OPENML PIPELINE RUNNER
# =============================================================================

def run_openml_collection(output_dir='./output_interactions',
                           task_list_file=None,
                           n_folds=5, n_repeats=3,
                           time_budget=7200):
    """Run interaction feature collection across OpenML datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    with open(task_list_file, 'r') as f:
        task_list = json.load(f)
    
    task_list = [t for t in task_list if t.get('task_type') == 'classification']
    print(f"[INT] Tasks: {len(task_list)} classification tasks")
    
    csv_file = os.path.join(output_dir, 'interaction_features.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint_interactions.json')
    
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
            
            df_result = collect_interaction_features(
                X, y, dataset_name=dataset.name,
                n_folds=n_folds, n_repeats=n_repeats,
                time_budget=time_budget
            )
            
            if not df_result.empty:
                df_result['openml_task_id'] = task_id
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                write_csv(df_result, csv_file, SCHEMA_INTERACTION)
            
            processed.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed': list(processed)}, f)
            
            gc.collect()
        
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            processed.add(ckpt_key)
            continue
    
    print(f"\n[INT] Collection complete. Results: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interaction Feature Data Collection")
    parser.add_argument('--output_dir', default='./output_interactions')
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