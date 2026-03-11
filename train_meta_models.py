"""
train_meta_models.py — Train Meta-Models for Feature Engineering Recommendation
=================================================================================

Trains four independent LightGBM models (numerical, categorical, interaction, row)
on the collected meta-learning data. Each model learns:

    (dataset meta-features + column/pair/row meta-features + method) → calibrated_delta

Additionally trains a lightweight binary classifier per type for:
    → is_significant_bonferroni  (should we bother?)

Evaluation uses GroupKFold by dataset_name so no data from the same dataset
leaks between train and validation — this measures true generalization.

Usage:
    python train_meta_models.py \
        --numerical_csv   ./output_numerical/numerical_transforms_merged.csv \
        --categorical_csv ./output_categorical/categorical_transforms_merged.csv \
        --interaction_csv ./output_interactions/interaction_features_merged.csv \
        --row_csv         ./output_row/row_features_merged.csv \
        --output_dir      ./meta_models

Outputs per collector type:
    - {type}_regressor.txt          LightGBM regression model (calibrated_delta)
    - {type}_classifier.txt         LightGBM binary classifier (is_significant)
    - {type}_config.json            Feature lists, method vocab, training stats
    - training_report.json          Combined evaluation summary
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import argparse
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# FEATURE DEFINITIONS — exactly matching collector schemas
# =============================================================================

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
    # Type indicator
    'n_numerical_cols',
    # Shared pairwise features (all types)
    'pearson_corr', 'spearman_corr',
    'mutual_info_pair', 'mic_score', 'scale_ratio',
    # Combined order-invariant individual stats (all types)
    'sum_importance', 'max_importance', 'min_importance',
    'sum_null_pct', 'max_null_pct',
    'sum_unique_ratio', 'abs_diff_unique_ratio',
    'sum_entropy', 'abs_diff_entropy',
    'sum_target_corr', 'abs_diff_target_corr',
    'sum_mi_target', 'abs_diff_mi_target',
    'both_binary',
    # num+num specific (SENTINEL_NC=-10 for num+cat, SENTINEL_CC=-20 for cat+cat)
    'product_of_means',
    'abs_mean_ratio',
    'sum_cv', 'abs_diff_cv',
    'sum_skewness', 'abs_diff_skewness',
    'sign_concordance',
    'n_positive_means',
    # num+cat specific (-10 for num+num, -20 for cat+cat)
    'eta_squared',
    'anova_f_stat',
    'n_groups',
    # cat+cat specific (-10 for num+cat, -20 for num+num)
    'cramers_v',
    'joint_cardinality',
    'cardinality_ratio',
    'joint_sparsity',
]

# Row collector: dataset-level features computed once per dataset (no column-level features).
# One CSV row per applicable family (not per individual method).
ROW_DATASET_FEATURES = [
    'n_numeric_cols_used',
    'avg_numeric_mean',
    'avg_numeric_std',
    'avg_missing_pct',
    'max_missing_pct',
    'avg_row_variance',
    'pct_rows_with_any_missing',
    'pct_cells_zero',
    'pct_rows_with_any_zero',
    'numeric_col_corr_mean',
    'numeric_col_corr_max',
    'avg_row_entropy',
    'numeric_range_ratio',
]

# Regression target
TARGET_REGRESSION = 'calibrated_delta'

# Classification target
TARGET_CLASSIFICATION = 'is_significant_bonferroni'


# =============================================================================
# COMPOSITE TARGET FORMULAS
# =============================================================================
#
# Each formula takes a DataFrame (with all raw outcome columns available) and
# returns a pd.Series of float values to use as the regression target.
#
# Design principles:
#   - full-model delta (delta) > individual delta: the full-model signal captures
#     real-world utility — does this transform help in context with other features?
#   - p-value weighting (1 - p) smoothly penalises statistically unreliable deltas
#     instead of the hard Bonferroni binary, giving a continuous signal
#   - calibrated_delta = delta / null_std is good SNR but null_std clamping at 0.001
#     can inflate low-noise datasets; combining with p-value damps this
#   - individual_delta is useful as a prior / sanity check, but gets lower weight
#
# Available raw columns (from the collector CSVs):
#   delta, delta_normalized          — full-model AUC change + headroom-normalised
#   p_value, p_value_bonferroni      — paired t-test significance (full model)
#   individual_delta                 — single-column-only AUC change
#   individual_delta_normalized      — headroom-normalised version
#   individual_p_value               — paired t-test (individual eval)
#   individual_p_value_bonferroni    — Bonferroni-corrected (individual)
#   cohens_d                         — effect size (full model)
#   individual_cohens_d              — effect size (individual)
#   calibrated_delta                 — delta / null_std (current default)
#   individual_calibrated_delta      — individual_delta / null_std
#   null_std                         — noise floor estimate
# =============================================================================

def _safe(df, col, fill=0.0):
    """Return column as float Series, filling missing/NaN with fill."""
    if col not in df.columns:
        return pd.Series(fill, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(fill)


def _clip_p(p: pd.Series) -> pd.Series:
    """Clip p-values to [0, 1] and fill NaN with 0.5 (neutral weight)."""
    return p.clip(0.0, 1.0).fillna(0.5)


COMPOSITE_FORMULAS = {

    # ==========================================================================
    # DELTA FAMILY  (raw AUC delta, p-value confidence weighting)
    # ==========================================================================

    # ---- Pure deltas, no statistical tests ----
    # Cleanest possible signal: trust the measured deltas directly.
    # No p-value shrinkage — removes the risk of over-penalising noisy-but-real gains.
    # w_indiv=0.0: baseline with no individual signal at all.
    'delta_only': lambda df: _safe(df, 'delta'),

    # w_indiv=0.2: individual delta as a weak sanity-check prior.
    'delta_w02': lambda df: (
        _safe(df, 'delta')
        + 0.2 * _safe(df, 'individual_delta')
    ),

    # w_indiv=0.4: original weight, no p-value damping.
    'delta_w04': lambda df: (
        _safe(df, 'delta')
        + 0.4 * _safe(df, 'individual_delta')
    ),

    # w_indiv=0.6: individual delta gets more say.
    'delta_w06': lambda df: (
        _safe(df, 'delta')
        + 0.6 * _safe(df, 'individual_delta')
    ),

    # w_indiv=0.8: heavy individual delta — useful if full-model signal is noisier.
    'delta_w08': lambda df: (
        _safe(df, 'delta')
        + 0.8 * _safe(df, 'individual_delta')
    ),

    # ---- Confidence-weighted full delta only ----
    # "How much did the full model improve, scaled by how sure we are?"
    # High delta + low p → high value. Noisy result → shrinks toward 0.
    'delta_x_conf': lambda df: (
        _safe(df, 'delta') * (1.0 - _clip_p(_safe(df, 'p_value', 0.5)))
    ),

    # ---- Weighted full + individual, p-value damped (current best for num/cat) ----
    # w_indiv=0.4 — the ablation winner. Kept as the reference point.
    'weighted_full_indiv': lambda df: (
        1.0 * _safe(df, 'delta') * (1.0 - _clip_p(_safe(df, 'p_value', 0.5)))
        + 0.4 * _safe(df, 'individual_delta') * (1.0 - _clip_p(_safe(df, 'individual_p_value', 0.5)))
    ),

    # ---- Soft confidence: p-value floors at 0.5 instead of going to 0 ----
    # (1 - p) → (0.5 + 0.5*(1-p)):  uncertain results are discounted but not zeroed.
    # Less aggressive than weighted_full_indiv when p-values are unreliable.
    'delta_soft_conf': lambda df: (
        _safe(df, 'delta') * (0.5 + 0.5 * (1.0 - _clip_p(_safe(df, 'p_value', 0.5))))
        + 0.4 * _safe(df, 'individual_delta') * (0.5 + 0.5 * (1.0 - _clip_p(_safe(df, 'individual_p_value', 0.5))))
    ),

    # ==========================================================================
    # COHEN'S D FAMILY  (effect size — avoids null_std clamping inflation)
    # ==========================================================================

    # ---- Pure Cohen's d (full model only) ----
    'cohens_d': lambda df: _safe(df, 'cohens_d'),

    # w_indiv=0.2
    'cohens_d_w02': lambda df: (
        _safe(df, 'cohens_d')
        + 0.2 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=0.4 — current ablation winner for interaction
    'cohens_d_weighted': lambda df: (
        1.0 * _safe(df, 'cohens_d')
        + 0.4 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=0.6
    'cohens_d_w06': lambda df: (
        _safe(df, 'cohens_d')
        + 0.6 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=0.8
    'cohens_d_w08': lambda df: (
        _safe(df, 'cohens_d')
        + 0.8 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=1.0: individual gets equal weight to full
    'cohens_d_w10': lambda df: (
        _safe(df, 'cohens_d')
        + 1.0 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=1.2
    'cohens_d_w12': lambda df: (
        _safe(df, 'cohens_d')
        + 1.2 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=1.5: individual dominates — tests whether full-model cohens_d adds much
    'cohens_d_w15': lambda df: (
        _safe(df, 'cohens_d')
        + 1.5 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=2.0
    'cohens_d_w20': lambda df: (
        _safe(df, 'cohens_d')
        + 2.0 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=2.5
    'cohens_d_w25': lambda df: (
        _safe(df, 'cohens_d')
        + 2.5 * _safe(df, 'individual_cohens_d')
    ),

    # w_indiv=3.0
    'cohens_d_w30': lambda df: (
        _safe(df, 'cohens_d')
        + 3.0 * _safe(df, 'individual_cohens_d')
    ),

    # Pure individual Cohen's d — no full-model term at all.
    # If this beats cohens_d_w*, full-model cohens_d is just noise for that collector.
    'individual_cohens_d_only': lambda df: _safe(df, 'individual_cohens_d'),
}

# Default formula to use for training (can be overridden via CLI)
DEFAULT_FORMULA = 'weighted_full_indiv'

# Classification target: instead of hard Bonferroni binary, offer a softer option
# that uses the composite target thresholded at a meaningful level.
CLASSIFICATION_STRATEGIES = {
    'bonferroni':     lambda df, _: _safe(df, 'is_significant_bonferroni', 0).astype(int),
    'p_value_05':     lambda df, _: (_safe(df, 'p_value', 1.0) < 0.05).astype(int),
    'composite_pos':  lambda df, y_reg: (y_reg > 0).astype(int),   # composite target > 0
    'composite_q75':  lambda df, y_reg: (y_reg > y_reg.quantile(0.75)).astype(int),  # top quartile
}

DEFAULT_CLS_STRATEGY = 'composite_pos'


def compute_composite_target(df: pd.DataFrame, formula_name: str) -> pd.Series:
    """
    Compute a composite regression target from raw collector outcome columns.

    Args:
        df: DataFrame with raw outcome columns present
        formula_name: Key in COMPOSITE_FORMULAS

    Returns:
        pd.Series of float values (the new regression target)
    """
    if formula_name not in COMPOSITE_FORMULAS:
        raise ValueError(
            f"Unknown formula '{formula_name}'. "
            f"Available: {list(COMPOSITE_FORMULAS.keys())}"
        )
    result = COMPOSITE_FORMULAS[formula_name](df).reset_index(drop=True)
    return result.astype(float)


def compute_classification_target(df: pd.DataFrame, y_reg: pd.Series,
                                   strategy: str) -> pd.Series:
    """
    Compute classification target using the given strategy.

    Args:
        df: Raw collector DataFrame
        y_reg: Already-computed composite regression target (for threshold strategies)
        strategy: Key in CLASSIFICATION_STRATEGIES

    Returns:
        pd.Series of int (0/1)
    """
    if strategy not in CLASSIFICATION_STRATEGIES:
        raise ValueError(
            f"Unknown cls strategy '{strategy}'. "
            f"Available: {list(CLASSIFICATION_STRATEGIES.keys())}"
        )
    return CLASSIFICATION_STRATEGIES[strategy](df, y_reg).reset_index(drop=True)

# Group column for CV splits
GROUP_COL = 'dataset_name'

# Identifier columns to always exclude from features
ID_COLS = [
    'openml_task_id', 'dataset_name', 'dataset_id',
    'column_name', 'interaction_col_a', 'interaction_col_b',
]

# Outcome columns to exclude from features
OUTCOME_COLS = [
    'delta', 'delta_normalized', 'absolute_score',
    't_statistic', 'p_value', 'p_value_bonferroni',
    'is_significant', 'is_significant_bonferroni',
    'individual_baseline_score', 'individual_intervention_score',
    'individual_delta', 'individual_delta_normalized',
    'individual_p_value', 'individual_p_value_bonferroni',
    'individual_is_significant', 'individual_is_significant_bonferroni',
    'cohens_d', 'individual_cohens_d',
    'calibrated_delta', 'individual_calibrated_delta',
    'null_std',
]


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

REGRESSOR_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
}

CLASSIFIER_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
    'is_unbalance': True,
}


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_prepare(csv_path, collector_type,
                     formula=DEFAULT_FORMULA,
                     cls_strategy=DEFAULT_CLS_STRATEGY):
    """
    Load a merged CSV and prepare features + targets.

    Args:
        csv_path: Path to merged collector CSV
        collector_type: One of 'numerical', 'categorical', 'interaction', 'row'
        formula: Key in COMPOSITE_FORMULAS — defines the regression target
        cls_strategy: Key in CLASSIFICATION_STRATEGIES — defines the binary target

    Returns:
        X (pd.DataFrame): Feature matrix (dataset meta + column/pair meta + method one-hot)
        y_reg (pd.Series): Regression target (composite formula)
        y_cls (pd.Series): Classification target
        groups (pd.Series): Dataset name for grouped CV
        method_vocab (list): Ordered method names used for one-hot encoding
        feature_names (list): Final feature column names
    """
    print(f"\n{'='*60}")
    print(f"Loading {collector_type}: {csv_path}")
    print(f"  Target formula:    {formula}")
    print(f"  Cls strategy:      {cls_strategy}")

    df = pd.read_csv(csv_path)
    print(f"  Raw rows: {len(df)}, columns: {len(df.columns)}")
    print(f"  Unique datasets: {df[GROUP_COL].nunique()}")
    print(f"  Methods: {df['method'].value_counts().to_dict()}")

    # --- Select meta-feature columns by type ---
    if collector_type == 'numerical':
        meta_features = DATASET_FEATURES + NUMERICAL_COLUMN_FEATURES
    elif collector_type == 'categorical':
        meta_features = DATASET_FEATURES + CATEGORICAL_COLUMN_FEATURES
    elif collector_type == 'interaction':
        meta_features = DATASET_FEATURES + INTERACTION_PAIR_FEATURES
    elif collector_type == 'row':
        # Row collector is dataset-level: no per-column features, only dataset-level
        # stats computed once per dataset + the row-specific aggregate stats.
        meta_features = DATASET_FEATURES + ROW_DATASET_FEATURES
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")
    
    # Verify all expected columns exist
    missing = [c for c in meta_features if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns: {missing}")
        for c in missing:
            df[c] = 0.0
    
    # --- Drop rows where calibrated_delta is missing (data integrity check) ---
    # We still need calibrated_delta present as a fallback sanity check column.
    # For composite formulas, we only strictly require 'delta' to exist.
    n_before = len(df)
    required_for_formula = 'delta'  # minimum requirement
    df = df.dropna(subset=[required_for_formula])
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with missing '{required_for_formula}'")

    # --- Compute composite regression target ---
    df = df.reset_index(drop=True)
    y_reg = compute_composite_target(df, formula)
    print(f"  Composite target '{formula}': "
          f"mean={y_reg.mean():.4f}, std={y_reg.std():.4f}, "
          f"median={y_reg.median():.4f}, "
          f"q25={y_reg.quantile(0.25):.4f}, q75={y_reg.quantile(0.75):.4f}")

    # --- Compute classification target ---
    y_cls = compute_classification_target(df, y_reg, cls_strategy)
    print(f"  Cls target '{cls_strategy}': "
          f"{y_cls.sum()} positive / {len(y_cls)} total ({y_cls.mean()*100:.1f}%)")

    # --- One-hot encode method ---
    method_vocab = sorted(df['method'].unique().tolist())
    method_dummies = pd.get_dummies(df['method'], prefix='method')
    # Ensure consistent column order
    expected_method_cols = [f'method_{m}' for m in method_vocab]
    for c in expected_method_cols:
        if c not in method_dummies.columns:
            method_dummies[c] = 0
    method_dummies = method_dummies[expected_method_cols]
    
    # --- Build feature matrix ---
    X = df[meta_features].copy()
    
    # Convert any remaining non-numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    
    # Concatenate method dummies
    X = pd.concat([X.reset_index(drop=True), method_dummies.reset_index(drop=True)], axis=1)
    
    feature_names = X.columns.tolist()
    
    # Fill NaN features with -999 (LightGBM handles this well as a sentinel)
    X = X.fillna(-999)
    
    y_reg = y_reg.reset_index(drop=True)
    y_cls = y_cls.reset_index(drop=True)
    groups = df[GROUP_COL].reset_index(drop=True)
    
    print(f"  Final: {len(X)} rows, {len(feature_names)} features")
    print(f"  Target (regression): mean={y_reg.mean():.4f}, std={y_reg.std():.4f}, "
          f"median={y_reg.median():.4f}")
    print(f"  Target (classification): {y_cls.sum()} significant / {len(y_cls)} total "
          f"({y_cls.mean()*100:.1f}%)")
    
    return X, y_reg, y_cls, groups, method_vocab, feature_names


# =============================================================================
# GROUPED CROSS-VALIDATION EVALUATION
# =============================================================================

def evaluate_grouped_cv(X, y_reg, y_cls, groups, n_splits=5,
                        reg_params=None, cls_params=None):
    """
    Evaluate models with GroupKFold — no dataset leaks between folds.
    
    Returns dict with regression and classification metrics per fold.
    """
    reg_params = reg_params or REGRESSOR_PARAMS
    cls_params = cls_params or CLASSIFIER_PARAMS
    gkf = GroupKFold(n_splits=n_splits)
    
    reg_metrics = {'mae': [], 'rmse': [], 'r2': [], 'median_ae': []}
    cls_metrics = {'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
    
    feature_importances = np.zeros(X.shape[1])
    
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y_reg, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        # --- Regression ---
        y_train_reg, y_val_reg = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
        
        reg_model = lgb.LGBMRegressor(**reg_params)
        reg_model.fit(
            X_train, y_train_reg,
            eval_set=[(X_val, y_val_reg)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        y_pred_reg = reg_model.predict(X_val)
        
        reg_metrics['mae'].append(mean_absolute_error(y_val_reg, y_pred_reg))
        reg_metrics['rmse'].append(np.sqrt(mean_squared_error(y_val_reg, y_pred_reg)))
        reg_metrics['r2'].append(r2_score(y_val_reg, y_pred_reg))
        reg_metrics['median_ae'].append(float(np.median(np.abs(y_val_reg - y_pred_reg))))
        
        feature_importances += reg_model.feature_importances_
        
        # --- Classification ---
        y_train_cls, y_val_cls = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
        
        # Skip if validation has only one class
        if y_val_cls.nunique() < 2:
            continue
        
        cls_model = lgb.LGBMClassifier(**cls_params)
        cls_model.fit(
            X_train, y_train_cls,
            eval_set=[(X_val, y_val_cls)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        y_pred_cls_proba = cls_model.predict_proba(X_val)[:, 1]
        y_pred_cls = (y_pred_cls_proba >= 0.5).astype(int)
        
        cls_metrics['auc'].append(roc_auc_score(y_val_cls, y_pred_cls_proba))
        cls_metrics['f1'].append(f1_score(y_val_cls, y_pred_cls))
        cls_metrics['precision'].append(precision_score(y_val_cls, y_pred_cls, zero_division=0))
        cls_metrics['recall'].append(recall_score(y_val_cls, y_pred_cls, zero_division=0))
        cls_metrics['accuracy'].append(accuracy_score(y_val_cls, y_pred_cls))
    
    feature_importances /= n_splits
    
    # Aggregate
    results = {'regression': {}, 'classification': {}, 'n_splits': n_splits}
    
    for metric, values in reg_metrics.items():
        results['regression'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'per_fold': [float(v) for v in values],
        }
    
    for metric, values in cls_metrics.items():
        if values:
            results['classification'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'per_fold': [float(v) for v in values],
            }
    
    return results, feature_importances


# =============================================================================
# OPTUNA HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(X, y_reg, y_cls, groups, n_trials=50, n_splits=5):
    """
    Use Optuna to find optimal LightGBM hyperparameters for both the regressor
    and classifier. Optimises over GroupKFold CV (same split strategy as
    evaluate_grouped_cv) so there is no dataset leakage during search.

    Returns:
        best_reg_params (dict): Tuned regressor params (merged with fixed params)
        best_cls_params (dict): Tuned classifier params (merged with fixed params)
        tuning_report (dict): Best values + trial counts for the config
    """
    if not OPTUNA_AVAILABLE:
        print("  WARNING: optuna not installed — skipping tuning, using defaults.")
        return REGRESSOR_PARAMS.copy(), CLASSIFIER_PARAMS.copy(), {}

    n_datasets = groups.nunique()
    actual_splits = min(n_splits, n_datasets)
    if actual_splits < 2:
        print("  WARNING: too few datasets for tuning CV — skipping.")
        return REGRESSOR_PARAMS.copy(), CLASSIFIER_PARAMS.copy(), {}

    gkf = GroupKFold(n_splits=actual_splits)

    # ---- Regressor objective (minimise mean MAE across folds) ----
    def regressor_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
        }
        fold_maes = []
        for train_idx, val_idx in gkf.split(X, y_reg, groups):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X.iloc[train_idx], y_reg.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y_reg.iloc[val_idx])],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            preds = model.predict(X.iloc[val_idx])
            fold_maes.append(mean_absolute_error(y_reg.iloc[val_idx], preds))
        return float(np.mean(fold_maes))

    # ---- Classifier objective (maximise mean ROC-AUC across folds) ----
    def classifier_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'is_unbalance': True,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
        }
        fold_aucs = []
        for train_idx, val_idx in gkf.split(X, y_cls, groups):
            y_val_cls = y_cls.iloc[val_idx]
            if y_val_cls.nunique() < 2:
                continue
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[train_idx], y_cls.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y_val_cls)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_val_cls, proba))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.5

    print(f"\n  Optuna tuning: {n_trials} trials each for regressor and classifier...")

    reg_study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    reg_study.optimize(regressor_objective, n_trials=n_trials, show_progress_bar=False)

    cls_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    cls_study.optimize(classifier_objective, n_trials=n_trials, show_progress_bar=False)

    # Merge tuned params with fixed keys (random_state, verbosity, n_jobs)
    fixed_keys = {'random_state': 42, 'verbosity': -1, 'n_jobs': -1}
    best_reg_params = {**reg_study.best_params, **fixed_keys}
    best_cls_params = {**cls_study.best_params, 'is_unbalance': True, **fixed_keys}

    print(f"  Best regressor MAE:  {reg_study.best_value:.4f}")
    print(f"  Best classifier AUC: {cls_study.best_value:.4f}")
    print(f"  Best regressor params: {reg_study.best_params}")
    print(f"  Best classifier params: {cls_study.best_params}")

    tuning_report = {
        'n_trials': n_trials,
        'regressor': {
            'best_mae': reg_study.best_value,
            'best_params': reg_study.best_params,
            'n_trials_completed': len(reg_study.trials),
        },
        'classifier': {
            'best_auc': cls_study.best_value,
            'best_params': cls_study.best_params,
            'n_trials_completed': len(cls_study.trials),
        },
    }

    return best_reg_params, best_cls_params, tuning_report




def train_final_models(X, y_reg, y_cls, reg_params=None, cls_params=None):
    """Train final regressor and classifier on all available data."""
    reg_params = reg_params or REGRESSOR_PARAMS
    cls_params = cls_params or CLASSIFIER_PARAMS

    reg_model = lgb.LGBMRegressor(**reg_params)
    reg_model.fit(X, y_reg)
    
    cls_model = lgb.LGBMClassifier(**cls_params)
    cls_model.fit(X, y_cls)
    
    return reg_model, cls_model


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_collector_type(csv_path, collector_type, output_dir,
                         n_cv_splits=5, tune=False, n_trials=50,
                         formula=DEFAULT_FORMULA,
                         cls_strategy=DEFAULT_CLS_STRATEGY):
    """
    Full pipeline for one collector type:
    1. Load data (with composite target)
    2. (Optional) Optuna HP tuning on regressor + classifier
    3. Grouped CV evaluation with final params
    4. Train final models on all data
    5. Save models + config
    """
    if csv_path is None or not os.path.exists(csv_path):
        print(f"\n  Skipping {collector_type}: CSV not found at {csv_path}")
        return None

    # --- Load ---
    X, y_reg, y_cls, groups, method_vocab, feature_names = load_and_prepare(
        csv_path, collector_type,
        formula=formula,
        cls_strategy=cls_strategy,
    )
    
    if len(X) < 50:
        print(f"  Skipping {collector_type}: only {len(X)} rows (need ≥50)")
        return None
    
    n_datasets = groups.nunique()
    actual_splits = min(n_cv_splits, n_datasets)

    # --- HP Tuning (optional) ---
    tuning_report = {}
    if tune:
        reg_params, cls_params, tuning_report = tune_hyperparameters(
            X, y_reg, y_cls, groups,
            n_trials=n_trials,
            n_splits=actual_splits,
        )
    else:
        reg_params, cls_params = REGRESSOR_PARAMS.copy(), CLASSIFIER_PARAMS.copy()

    # --- CV Evaluation ---
    if actual_splits < 2:
        print(f"  Skipping CV: only {n_datasets} unique dataset(s)")
        cv_results = None
        feat_importances = np.zeros(X.shape[1])
    else:
        print(f"\n  Evaluating with {actual_splits}-fold GroupKFold ({n_datasets} datasets)...")
        cv_results, feat_importances = evaluate_grouped_cv(
            X, y_reg, y_cls, groups,
            n_splits=actual_splits,
            reg_params=reg_params,
            cls_params=cls_params,
        )
        
        print(f"\n  REGRESSION target='{formula}':")
        for metric in ['mae', 'rmse', 'r2', 'median_ae']:
            vals = cv_results['regression'][metric]
            print(f"    {metric:>10}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        print(f"\n  CLASSIFICATION strategy='{cls_strategy}':")
        for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
            if metric in cv_results['classification']:
                vals = cv_results['classification'][metric]
                print(f"    {metric:>10}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    
    # --- Top features ---
    top_k = min(15, len(feature_names))
    imp_series = pd.Series(feat_importances, index=feature_names).sort_values(ascending=False)
    print(f"\n  Top {top_k} features (regression):")
    for fname, fval in imp_series.head(top_k).items():
        print(f"    {fname:>35}: {fval:.1f}")
    
    # --- Train final models ---
    print(f"\n  Training final models on all {len(X)} rows...")
    reg_model, cls_model = train_final_models(X, y_reg, y_cls,
                                              reg_params=reg_params,
                                              cls_params=cls_params)
    
    # --- Save ---
    type_dir = os.path.join(output_dir, collector_type)
    os.makedirs(type_dir, exist_ok=True)
    
    reg_path = os.path.join(type_dir, f'{collector_type}_regressor.txt')
    cls_path = os.path.join(type_dir, f'{collector_type}_classifier.txt')
    config_path = os.path.join(type_dir, f'{collector_type}_config.json')
    
    reg_model.booster_.save_model(reg_path)
    cls_model.booster_.save_model(cls_path)
    
    # Determine which meta features belong to this type
    if collector_type == 'numerical':
        column_features = NUMERICAL_COLUMN_FEATURES
    elif collector_type == 'categorical':
        column_features = CATEGORICAL_COLUMN_FEATURES
    elif collector_type == 'interaction':
        column_features = INTERACTION_PAIR_FEATURES
    elif collector_type == 'row':
        column_features = ROW_DATASET_FEATURES
    else:
        raise ValueError(f"Unknown collector_type: {collector_type!r}")
    
    config = {
        'collector_type': collector_type,
        'dataset_features': DATASET_FEATURES,
        'column_features': column_features,
        'method_vocab': method_vocab,
        'feature_names': feature_names,
        'n_training_rows': len(X),
        'n_training_datasets': int(groups.nunique()),
        'target_formula': formula,
        'cls_strategy': cls_strategy,
        'target_regression': f'composite:{formula}',
        'target_classification': f'composite:{cls_strategy}',
        'regressor_params': reg_params,
        'classifier_params': cls_params,
        'tuning': tuning_report,
        'cv_results': cv_results,
        'feature_importances': {
            fname: float(fval) 
            for fname, fval in imp_series.head(30).items()
        },
        'target_stats': {
            'target_mean': float(y_reg.mean()),
            'target_std': float(y_reg.std()),
            'target_median': float(y_reg.median()),
            'target_q25': float(y_reg.quantile(0.25)),
            'target_q75': float(y_reg.quantile(0.75)),
            'pct_positive_cls': float(y_cls.mean()),
        },
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Saved:")
    print(f"    Regressor:  {reg_path}")
    print(f"    Classifier: {cls_path}")
    print(f"    Config:     {config_path}")
    
    return {
        'collector_type': collector_type,
        'n_rows': len(X),
        'n_datasets': int(groups.nunique()),
        'n_features': len(feature_names),
        'method_vocab': method_vocab,
        'cv_results': cv_results,
        'tuning': tuning_report,
        'formula': formula,
        'cls_strategy': cls_strategy,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def ablate_formulas(collector_csvs, types, n_cv_splits=5):
    """
    Run GroupKFold CV across all COMPOSITE_FORMULAS x CLASSIFICATION_STRATEGIES
    for the given collector types, and print a comparison table.
    No models are saved — this is purely for target selection.
    """
    print("\n" + "=" * 70)
    print("ABLATION: Composite Target Formula Comparison")
    print("=" * 70)

    all_results = {}  # {ctype: {formula: {cls_strategy: cv_metrics}}}

    for ctype in types:
        csv_path = collector_csvs.get(ctype)
        if csv_path is None or not os.path.exists(csv_path):
            print(f"\nSkipping {ctype}: CSV not found")
            continue

        all_results[ctype] = {}
        print(f"\n{'─'*60}")
        print(f"  Collector: {ctype}")

        for formula_name in COMPOSITE_FORMULAS:
            all_results[ctype][formula_name] = {}

            # Load data once per formula (targets differ)
            try:
                # Use composite_pos as cls strategy for all reg ablations
                X, y_reg, y_cls, groups, _, _ = load_and_prepare(
                    csv_path, ctype,
                    formula=formula_name,
                    cls_strategy='composite_pos',
                )
            except Exception as e:
                print(f"    ERROR loading formula={formula_name}: {e}")
                continue

            # Guard: constant target (std ≈ 0) means the formula references a column
            # absent from this collector's CSV (e.g. individual_delta in row collector).
            # sklearn r2_score silently returns 1.0 for constant targets — a false positive.
            if y_reg.std() < 1e-6:
                print(f"    SKIP {formula_name}: target is constant (std={y_reg.std():.2e}) "
                      f"— formula likely uses a column missing from the '{ctype}' CSV")
                continue

            n_datasets = groups.nunique()
            actual_splits = min(n_cv_splits, n_datasets)
            if actual_splits < 2:
                print(f"    Skipping CV: only {n_datasets} datasets")
                continue

            cv_results, _ = evaluate_grouped_cv(
                X, y_reg, y_cls, groups,
                n_splits=actual_splits,
            )

            r2_mean = cv_results['regression']['r2']['mean']
            r2_std  = cv_results['regression']['r2']['std']
            mae_mean = cv_results['regression']['mae']['mean']
            auc_mean = cv_results['classification']['auc']['mean']
            f1_mean  = cv_results['classification']['f1']['mean']
            cls_pct  = float(y_cls.mean()) * 100

            all_results[ctype][formula_name] = {
                'r2': r2_mean, 'r2_std': r2_std,
                'mae': mae_mean,
                'cls_auc': auc_mean,
                'cls_f1': f1_mean,
                'cls_pct_pos': cls_pct,
            }
            print(f"    {formula_name:>30}  R²={r2_mean:+.3f}±{r2_std:.3f}  "
                  f"MAE={mae_mean:.4f}  CLS-AUC={auc_mean:.3f}  F1={f1_mean:.3f}  "
                  f"pos={cls_pct:.1f}%")

    # Summary table
    print("\n\n" + "=" * 70)
    print("ABLATION SUMMARY (sorted by R²)")
    print("=" * 70)
    rows = []
    for ctype, formulas in all_results.items():
        for fname, m in formulas.items():
            if m:
                rows.append((ctype, fname, m.get('r2', -999), m))
    rows.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'Type':>15}  {'Formula':>30}  {'R²':>7}  {'MAE':>7}  {'AUC':>6}  {'F1':>6}")
    print(f"  {'-'*15}  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
    for ctype, fname, r2, m in rows:
        print(f"  {ctype:>15}  {fname:>30}  {r2:+.3f}  {m['mae']:.4f}  "
              f"{m['cls_auc']:.3f}  {m['cls_f1']:.3f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Train meta-models for feature engineering recommendation'
    )
    parser.add_argument('--numerical_csv',   type=str, default=None)
    parser.add_argument('--categorical_csv', type=str, default=None)
    parser.add_argument('--interaction_csv', type=str, default=None)
    parser.add_argument('--row_csv',         type=str, default=None)
    parser.add_argument(
        '--output_dir', type=str, default='./meta_models',
        help='Directory to save trained models and configs'
    )
    parser.add_argument('--n_cv_splits', type=int, default=5)
    parser.add_argument(
        '--types', type=str, nargs='+',
        choices=['numerical', 'categorical', 'interaction', 'row'],
        default=['numerical', 'categorical', 'interaction', 'row'],
    )
    parser.add_argument('--tune', action='store_true',
                        help='Run Optuna hyperparameter tuning before final training')
    parser.add_argument('--n_trials', type=int, default=50)

    # --- Composite target options ---
    parser.add_argument(
        '--formula', type=str, default=DEFAULT_FORMULA,
        choices=list(COMPOSITE_FORMULAS.keys()),
        help=(
            f'Composite regression target formula (default: {DEFAULT_FORMULA}). '
            f'Options: {list(COMPOSITE_FORMULAS.keys())}'
        )
    )
    parser.add_argument(
        '--cls_strategy', type=str, default=DEFAULT_CLS_STRATEGY,
        choices=list(CLASSIFICATION_STRATEGIES.keys()),
        help=(
            f'Classification target strategy (default: {DEFAULT_CLS_STRATEGY}). '
            f'Options: {list(CLASSIFICATION_STRATEGIES.keys())}'
        )
    )

    # --- Ablation mode ---
    parser.add_argument(
        '--ablate', action='store_true',
        help=(
            'Run CV ablation across ALL formulas and print comparison table. '
            'Does NOT save any models. Use this first to pick the best formula.'
        )
    )

    args = parser.parse_args()

    if args.tune and not OPTUNA_AVAILABLE:
        print("WARNING: --tune requested but optuna is not installed.")

    collector_csvs = {
        'numerical':   args.numerical_csv,
        'categorical': args.categorical_csv,
        'interaction': args.interaction_csv,
        'row':         args.row_csv,
    }

    # --- Ablation mode: compare all formulas, exit ---
    if args.ablate:
        ablate_formulas(collector_csvs, args.types, n_cv_splits=args.n_cv_splits)
        return

    # --- Normal training mode ---
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("META-MODEL TRAINING")
    print(f"Types:        {args.types}")
    print(f"Formula:      {args.formula}")
    print(f"Cls strategy: {args.cls_strategy}")
    if args.tune:
        print(f"Optuna tuning: {args.n_trials} trials per model")
    print("=" * 70)

    all_reports = {}
    for ctype in args.types:
        report = train_collector_type(
            csv_path=collector_csvs[ctype],
            collector_type=ctype,
            output_dir=args.output_dir,
            n_cv_splits=args.n_cv_splits,
            tune=args.tune,
            n_trials=args.n_trials,
            formula=args.formula,
            cls_strategy=args.cls_strategy,
        )
        if report:
            all_reports[ctype] = report

    # --- Save combined report ---
    report_path = os.path.join(args.output_dir, 'training_report.json')
    existing = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(all_reports)
    with open(report_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Models saved to: {args.output_dir}")
    print(f"Training report: {report_path}")

    for ctype, report in all_reports.items():
        r2_str = "N/A"
        auc_str = "N/A"
        if report.get('cv_results'):
            r2_vals = report['cv_results']['regression'].get('r2', {})
            r2_str = f"{r2_vals.get('mean', 0):.3f}" if r2_vals else "N/A"
            auc_vals = report['cv_results']['classification'].get('auc', {})
            auc_str = f"{auc_vals.get('mean', 0):.3f}" if auc_vals else "N/A"
        tuned = " [tuned]" if report.get('tuning') else ""
        print(f"  {ctype:>15}: {report['n_rows']:>6} rows, "
              f"{report['n_datasets']:>4} datasets, "
              f"R²={r2_str}, AUC={auc_str}{tuned}")


if __name__ == '__main__':
    main()