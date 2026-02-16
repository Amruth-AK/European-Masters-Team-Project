"""
Meta-Model Trainer v5 (train_meta_model_3.py)
==============================================

v5 improvements over v4:
1. METHOD FAMILY FEATURE: method_family_encoded now actually included as a
   model feature (was computed but unused in v4).

2. MISSINGNESS INDICATORS: Binary _was_missing flags for features with >5%
   missing values, so the model can learn imputation impact instead of just
   seeing median-filled values.

3. FEATURE INTERACTIONS: Pairwise interactions between key dataset-level and
   column-level features (e.g., n_rows*null_pct, baseline_score*skewness).

4. UPSIDE MODEL (Quantile Regression): New 4th model predicts the 90th
   percentile of effect size. Captures "high risk/high reward" methods that
   the mean regressor washes out (e.g., Arithmetic Interactions).

5. CONTINUOUS RANKER RELEVANCE: Uses scaled continuous effect signals instead
   of 5-level discretization for finer-grained lambdarank training.

6. LEARNED ENSEMBLE WEIGHTS: Optimizes blending weights for regressor, gate,
   and upside models via OOF predictions + Nelder-Mead minimization.

7. RICHER METHOD PRIORS: Now include effect_p90 (upside potential), effect_p10
   (downside risk), and effect_range (variance indicator) per method.

Backward compatible: outputs same .pkl file format. New models (upside_model.pkl,
ensemble_weights.json) are optional — the app gracefully falls back if absent.

Usage:
    python train_meta_model_3.py --meta-db ./meta_learning_output/meta_learning_db.csv --output-dir ./meta_model
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
import argparse
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             f1_score, precision_recall_curve,
                             balanced_accuracy_score, roc_auc_score,
                             ndcg_score)
from scipy.stats import spearmanr
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Using hand-tuned hyperparameters.")

warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

# Methods where individual evaluation is structurally uninformative
INDIVIDUAL_UNINFORMATIVE_METHODS = {
    'target_encoding', 'frequency_encoding', 'group_mean', 'group_std',
    'cat_concat', 'hashing_encoding', 'row_stats',
}

# Method family groupings â€” gives the model a coarser signal than 31 methods
METHOD_FAMILIES = {
    'arithmetic_interaction': ['addition_interaction', 'subtraction_interaction',
                               'product_interaction', 'division_interaction', 'abs_diff_interaction'],
    'three_way': ['three_way_addition', 'three_way_ratio', 'three_way_interaction'],
    'group_agg': ['group_mean', 'group_std'],
    'encoding': ['frequency_encoding', 'target_encoding', 'onehot_encoding',
                 'hashing_encoding', 'cat_concat'],
    'transform': ['log_transform', 'sqrt_transform', 'polynomial_square'],
    'temporal': ['cyclical_encode', 'date_extract_basic', 'date_cyclical_hour',
                 'date_cyclical_day', 'date_cyclical_dow', 'date_cyclical_month',
                 'date_elapsed_days'],
    'cleaning': ['impute_median', 'missing_indicator', 'quantile_binning'],
    'row_level': ['row_stats', 'text_stats'],
}
METHOD_TO_FAMILY = {}
for family, methods in METHOD_FAMILIES.items():
    for m in methods:
        METHOD_TO_FAMILY[m] = family


# =============================================================================
# SCORE COMPUTATION â€” v4: Effect-size based
# =============================================================================

def compute_effect_signal(row):
    """
    Compute the primary signal for how helpful a transform is.
    
    Priority order:
    1. Cohen's d (paired effect size) â€” best measure of practical significance
    2. Calibrated delta (z-score vs noise floor) â€” good alternative
    3. Raw delta with p-value moderation â€” fallback for old data
    
    Returns a single float: positive = helpful, negative = harmful, 0 = neutral.
    """
    def safe_float(val, default=np.nan):
        if val is None:
            return default
        try:
            f = float(val)
            return default if np.isnan(f) else f
        except (TypeError, ValueError):
            return default

    method = str(row.get('method', ''))
    
    # --- Try Cohen's d first (best signal) ---
    cohens_d_full = safe_float(row.get('cohens_d'))
    cohens_d_indiv = safe_float(row.get('individual_cohens_d'))
    
    if not np.isnan(cohens_d_full):
        if method in INDIVIDUAL_UNINFORMATIVE_METHODS or np.isnan(cohens_d_indiv):
            effect = cohens_d_full
        else:
            # Blend: individual is more targeted, full-model captures interaction effects
            effect = 0.4 * cohens_d_full + 0.6 * cohens_d_indiv
        return float(np.clip(effect, -5, 5))
    
    # --- Fallback: calibrated delta ---
    cal_full = safe_float(row.get('calibrated_delta'))
    cal_indiv = safe_float(row.get('individual_calibrated_delta'))
    
    if not np.isnan(cal_full):
        if method in INDIVIDUAL_UNINFORMATIVE_METHODS or np.isnan(cal_indiv):
            # Calibrated delta is in z-score units; divide by ~2 to approximate Cohen's d scale
            effect = cal_full / 2.0
        else:
            effect = (0.4 * cal_full + 0.6 * cal_indiv) / 2.0
        return float(np.clip(effect, -5, 5))
    
    # --- Fallback: raw delta (v3 style) ---
    delta_full = safe_float(row.get('delta_normalized'), 0.0)
    delta_indiv = safe_float(row.get('individual_delta_normalized'), 0.0)
    p_full = safe_float(row.get('p_value'), 1.0)
    p_indiv = safe_float(row.get('individual_p_value'), 1.0)
    
    if method in INDIVIDUAL_UNINFORMATIVE_METHODS:
        delta = delta_full
        p_best = p_full
    else:
        delta = 0.4 * delta_full + 0.6 * delta_indiv
        p_best = min(p_full, p_indiv)
    
    # Soft p-value moderation
    if p_best < 0.05:
        delta *= 1.3
    elif p_best < 0.10:
        delta *= 1.1
    elif p_best >= 0.50:
        delta *= 0.7
    
    # Map raw delta to approximate Cohen's d scale
    # Raw delta is in "percentage point" units; 1% AUC change â‰ˆ d=0.3 typically
    return float(np.clip(delta * 0.3, -5, 5))


def effect_to_score(effect_values, center=50, scale=10):
    """
    Map effect sizes (centered at 0) to 0-100 scores (centered at 50).
    
    Uses LINEAR mapping instead of tanh:
      score = 50 + clip(effect, -5, 5) * 10
    
    This gives meaningful spread:
      effect = 0.0 â†’ score = 50 (neutral)
      effect = 0.5 â†’ score = 55 (small positive)
      effect = 1.0 â†’ score = 60 (medium positive)
      effect = 2.0 â†’ score = 70 (large positive)
      effect = -1.0 â†’ score = 40 (harmful)
    """
    clipped = np.clip(effect_values, -5, 5)
    return np.clip(center + clipped * scale, 0, 100)


def effect_to_score_percentile(effect_values):
    """
    Map effect sizes to 0-100 scores using rank-based percentile mapping.
    
    This spreads scores uniformly across the full 0-100 range, giving the
    regressor a much richer target to learn from (std ~29 instead of ~3.7).
    
    At inference time, the original linear mapping is still used because
    we don't have access to the training distribution. But the model learns
    to rank correctly because the percentile mapping preserves order.
    """
    from scipy.stats import rankdata
    ranks = rankdata(effect_values, method='average')
    # Scale ranks to 0-100 (min rank -> ~0, max rank -> ~100)
    n = len(effect_values)
    if n <= 1:
        return np.full(n, 50.0)
    scores = (ranks - 1) / (n - 1) * 100.0
    return scores


def compute_is_helpful(row):
    """
    Determine if a transform is genuinely helpful using effect size + significance.
    
    A transform is "helpful" if:
    - Effect size (Cohen's d) > 0.2 (small practical effect), OR
    - Calibrated delta > 2.0 (2 std above noise floor)
    AND:
    - At least marginal statistical significance (p < 0.20)
    
    This is MUCH stricter than "top 20% of compressed scores" â€” it requires
    both practical significance AND statistical evidence.
    """
    def safe_float(val, default=np.nan):
        if val is None:
            return default
        try:
            f = float(val)
            return default if np.isnan(f) else f
        except (TypeError, ValueError):
            return default
    
    method = str(row.get('method', ''))
    
    # Get best p-value
    p_full = safe_float(row.get('p_value'), 1.0)
    p_indiv = safe_float(row.get('individual_p_value'), 1.0)
    if method in INDIVIDUAL_UNINFORMATIVE_METHODS:
        p_best = p_full
    else:
        p_best = min(p_full, p_indiv)
    
    # Statistical significance gate (liberal: p < 0.20)
    is_significant = p_best < 0.20
    
    # Effect size gate
    cohens_d_full = safe_float(row.get('cohens_d'), 0.0)
    cohens_d_indiv = safe_float(row.get('individual_cohens_d'), 0.0)
    cal_full = safe_float(row.get('calibrated_delta'), 0.0)
    
    if method in INDIVIDUAL_UNINFORMATIVE_METHODS:
        best_d = cohens_d_full
    else:
        best_d = max(cohens_d_full, cohens_d_indiv)
    
    has_effect = (best_d > 0.2) or (cal_full > 2.0)
    
    # Must have BOTH significance AND effect size
    if has_effect and is_significant:
        return 1
    
    # Fallback for data without Cohen's d: use effect signal
    if np.isnan(safe_float(row.get('cohens_d'))):
        effect = compute_effect_signal(row)
        return 1 if effect > 0.3 else 0
    
    return 0


# =============================================================================
# META-DB PREPARATION â€” v4
# =============================================================================

class MetaDBPreparator:
    """
    Prepares the raw meta_learning_db.csv for meta-model training.
    
    v4 changes:
    - Uses Cohen's d / calibrated_delta as the primary signal
    - Linear score mapping instead of tanh compression
    - Effect-size + significance based gate target
    - Method family as additional feature
    """
    
    DATASET_FEATURES = [
        'n_rows', 'n_cols', 'cat_ratio', 'missing_ratio', 'row_col_ratio',
        'class_imbalance_ratio', 'n_classes', 'target_std', 'target_skew',
        'landmarking_score', 'landmarking_score_norm',
        'avg_feature_corr', 'max_feature_corr', 'avg_target_corr', 'max_target_corr',
        'avg_numeric_sparsity',
        'linearity_gap', 'corr_graph_components', 'corr_graph_clustering',
        'corr_graph_density', 'matrix_rank_ratio', 'baseline_score', 'baseline_std',
        'n_cols_before_selection', 'n_cols_selected',
        # Aggregate features
        'n_numeric_cols', 'n_cat_cols',
        'std_feature_importance', 'max_minus_min_importance',
        'pct_features_above_median_importance',
        # Headroom
        'relative_headroom',
    ]
    
    COLUMN_FEATURES = [
        'null_pct', 'unique_ratio', 'is_numeric', 'outlier_ratio', 'entropy',
        'baseline_feature_importance', 'skewness', 'kurtosis', 'coeff_variation',
        'zeros_ratio', 'shapiro_p_value', 'has_multiple_modes',
        'bimodality_proxy_heuristic', 'range_iqr_ratio', 'dominant_quartile_pct',
        'pct_in_0_1_range', 'spearman_corr_target', 'hartigan_dip_pval',
        'is_multimodal', 'top_category_dominance', 'normalized_entropy',
        'is_binary', 'is_low_cardinality', 'is_high_cardinality',
        'top3_category_concentration', 'rare_category_pct', 'conditional_entropy',
        'pps_score', 'mutual_information_score',
    ]
    
    def __init__(self):
        self.method_encoder = LabelEncoder()
        self.task_type_encoder = LabelEncoder()
        self.feature_columns = None
        self.fill_values = None
    
    def prepare(self, df: pd.DataFrame):
        """
        Returns: (X, y, is_helpful, dataset_ids, metadata_dict)
        """
        df = df.copy()
        
        # Filter null_intervention rows (these are noise-floor calibrations, not real transforms)
        df = df[~df['method'].isin(['null_intervention'])].reset_index(drop=True)
        
        # â”€â”€ Step 1: Compute effect signal (Cohen's d based) â”€â”€
        has_cohens_d = 'cohens_d' in df.columns and df['cohens_d'].notna().mean() > 0.1
        has_calibrated = 'calibrated_delta' in df.columns and df['calibrated_delta'].notna().mean() > 0.1
        
        signal_source = 'cohens_d' if has_cohens_d else ('calibrated_delta' if has_calibrated else 'raw_delta')
        print(f"\n  Signal source: {signal_source} "
              f"(cohens_d available: {has_cohens_d}, calibrated_delta available: {has_calibrated})")
        
        df['effect_signal'] = df.apply(compute_effect_signal, axis=1)
        
        # â”€â”€ Step 2: Map to 0-100 scores (LINEAR, not tanh) â”€â”€
        df['recommendation_score'] = effect_to_score(df['effect_signal'].values)
        
        self._score_normalization = {
            'method': 'linear_from_effect_size',
            'signal_source': signal_source,
            'effect_mean': float(df['effect_signal'].mean()),
            'effect_std': float(df['effect_signal'].std()),
            'effect_median': float(df['effect_signal'].median()),
        }
        
        print(f"  Effect signal: mean={df['effect_signal'].mean():.4f}, "
              f"std={df['effect_signal'].std():.4f}, "
              f"median={df['effect_signal'].median():.4f}")
        print(f"  Score after mapping: mean={df['recommendation_score'].mean():.1f}, "
              f"std={df['recommendation_score'].std():.1f}, "
              f"range=[{df['recommendation_score'].min():.1f}, {df['recommendation_score'].max():.1f}]")
        
        # â”€â”€ Step 3: Effect-size based gate target â”€â”€
        df['is_helpful'] = df.apply(compute_is_helpful, axis=1)
        n_helpful = df['is_helpful'].sum()
        pct_helpful = df['is_helpful'].mean() * 100
        
        # Sanity check: if too few or too many helpful, fall back to quantile-based
        if pct_helpful < 2.0:
            print(f"  âš  Only {pct_helpful:.1f}% helpful â€” too strict. "
                  f"Relaxing to effect_signal > 0.15")
            df['is_helpful'] = (df['effect_signal'] > 0.15).astype(int)
            n_helpful = df['is_helpful'].sum()
            pct_helpful = df['is_helpful'].mean() * 100
        elif pct_helpful > 30.0:
            print(f"  âš  {pct_helpful:.1f}% helpful â€” too generous. "
                  f"Tightening to effect_signal > 0.5 AND significance")
            # Tighten: require larger effect
            stricter = df['effect_signal'] > 0.5
            if stricter.mean() > 0.05:
                df['is_helpful'] = stricter.astype(int)
                n_helpful = df['is_helpful'].sum()
                pct_helpful = df['is_helpful'].mean() * 100
        
        print(f"  Gate target: helpful={n_helpful} ({pct_helpful:.1f}%)")
        
        # â”€â”€ Step 4: Encode features â”€â”€
        df['method_encoded'] = self.method_encoder.fit_transform(df['method'].astype(str))
        
        if 'task_type' in df.columns:
            df['task_type_encoded'] = self.task_type_encoder.fit_transform(df['task_type'].astype(str))
        else:
            df['task_type_encoded'] = 0
        
        # NEW: Method family encoding
        df['method_family'] = df['method'].map(METHOD_TO_FAMILY).fillna('other')
        family_encoder = LabelEncoder()
        df['method_family_encoded'] = family_encoder.fit_transform(df['method_family'])
        self._family_encoder_classes = family_encoder.classes_.tolist()
        
        # TARGET ENCODING for method (leave-one-out to prevent leakage)
        method_target_means = df.groupby('method')['recommendation_score'].transform('mean')
        # Leave-one-out: subtract current row's contribution
        method_counts = df.groupby('method')['recommendation_score'].transform('count')
        df['method_target_enc'] = (
            (method_target_means * method_counts - df['recommendation_score']) 
            / (method_counts - 1).clip(lower=1)
        )
        # Smooth toward global mean for rare methods (Bayesian shrinkage)
        global_mean = df['recommendation_score'].mean()
        smoothing = 50  # regularization strength
        df['method_target_enc'] = (
            (df['method_target_enc'] * method_counts + global_mean * smoothing) 
            / (method_counts + smoothing)
        )
        self._method_target_enc_map = df.groupby('method')['recommendation_score'].mean().to_dict()
        self._method_target_enc_global = global_mean

        # METHOD × KEY DATASET PROPERTY interactions
        # These let the model learn "method X works when property Y is high"
        key_dataset_props = ['baseline_score', 'n_rows', 'n_cols', 'cat_ratio', 
                             'missing_ratio', 'target_std', 'n_classes']
        for prop in key_dataset_props:
            if prop in df.columns:
                col_name = f'mx_{prop}'
                df[col_name] = df['method_target_enc'] * pd.to_numeric(df[prop], errors='coerce').fillna(0)
        
        # METHOD FAMILY × COLUMN PROPERTY interactions  
        key_col_props = ['null_pct', 'unique_ratio', 'skewness', 'entropy',
                         'baseline_feature_importance', 'spearman_corr_target']
        for prop in key_col_props:
            if prop in df.columns:
                col_name = f'fx_{prop}'
                df[col_name] = df['method_family_encoded'] * pd.to_numeric(df[prop], errors='coerce').fillna(0)


        # â”€â”€ Step 5: Assemble feature columns â”€â”€
        feature_cols = []
        for col in self.DATASET_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        for col in self.COLUMN_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        feature_cols.extend(['method_encoded', 'task_type_encoded',
                             'method_family_encoded', 'method_target_enc'])
        for col in df.columns:
            if col.startswith('mx_') or col.startswith('fx_'):
                feature_cols.append(col)
        
        # Boolean fields
        bool_fields = ['is_interaction', 'near_ceiling_flag']
        for bf in bool_fields:
            if bf in df.columns:
                df[bf] = df[bf].map(
                    lambda v: 1 if str(v).strip().lower() in ('true', '1', '1.0') else 0
                )
                feature_cols.append(bf)
        
        # Extra numeric features
        for extra in ['vif', 'composite_predictive_score',
                     'pairwise_corr_ab', 'pairwise_spearman_ab',
                     'pairwise_mi_ab', 'interaction_scale_ratio',
                     'is_temporal_component', 'temporal_period',
                     # v6: col_b metadata (partner column in 2-way/3-way interactions)
                     'col_b_is_numeric', 'col_b_skewness', 'col_b_unique_ratio',
                     'col_b_null_pct', 'col_b_outlier_ratio', 'col_b_baseline_importance',
                     'col_b_entropy', 'col_b_composite_predictive_score',
                     # v6: col_c metadata (third column in 3-way interactions)
                     'col_c_is_numeric', 'col_c_skewness', 'col_c_unique_ratio',
                     'col_c_null_pct', 'col_c_outlier_ratio', 'col_c_baseline_importance',
                     'col_c_entropy', 'col_c_composite_predictive_score',
                     # v5: tree-guided pair score
                     'tree_pair_score']:
            if extra in df.columns:
                feature_cols.append(extra)
        
        # Sentinel value for non-interaction pair features
        # For single-column transforms, interaction-specific features are meaningless,
        # so we set them to -1.0 as a sentinel the model can learn from.
        INTERACTION_ONLY_FEATURES = [
            'pairwise_corr_ab', 'pairwise_spearman_ab',
            'pairwise_mi_ab', 'interaction_scale_ratio', 'tree_pair_score',
            # col_b metadata
            'col_b_is_numeric', 'col_b_skewness', 'col_b_unique_ratio',
            'col_b_null_pct', 'col_b_outlier_ratio', 'col_b_baseline_importance',
            'col_b_entropy', 'col_b_composite_predictive_score',
            # col_c metadata
            'col_c_is_numeric', 'col_c_skewness', 'col_c_unique_ratio',
            'col_c_null_pct', 'col_c_outlier_ratio', 'col_c_baseline_importance',
            'col_c_entropy', 'col_c_composite_predictive_score',
        ]
        is_interaction_col = df.get('is_interaction')
        if is_interaction_col is not None:
            non_interaction_mask = is_interaction_col.astype(str).str.lower().isin(
                ['false', '0', '0.0', 'nan', ''])
            for ipf in INTERACTION_ONLY_FEATURES:
                if ipf in df.columns:
                    df.loc[non_interaction_mask, ipf] = -1.0
        
        # NOISE REDUCTION: Drop near-zero-variance features
        # (These add noise without signal, especially harmful for the gate)
        for col in feature_cols[:]:  # iterate a copy
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                if vals.nunique() <= 1:
                    feature_cols.remove(col)


        X = df[feature_cols].copy()
        y = df['recommendation_score']
        is_helpful = df['is_helpful']
        
        # Dataset grouping
        dataset_ids = None
        if 'dataset_name' in df.columns:
            dataset_id_enc = LabelEncoder()
            dataset_ids = pd.Series(dataset_id_enc.fit_transform(df['dataset_name'].astype(str)),
                                     index=df.index)
        
        # Clean numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # --- Missingness indicators for high-missing features ---
        # Instead of silently imputing, let the model learn that a value was missing
        MISSINGNESS_THRESHOLD = 0.05  # Add indicator if > 5% missing
        missingness_cols = []
        for col in X.columns:
            miss_pct = X[col].isna().mean()
            if miss_pct > MISSINGNESS_THRESHOLD and miss_pct < 0.95:
                indicator_col = f'{col}_was_missing'
                X[indicator_col] = X[col].isna().astype(int)
                missingness_cols.append(indicator_col)
        if missingness_cols:
            feature_cols.extend(missingness_cols)
            print(f"  Added {len(missingness_cols)} missingness indicators "
                  f"(threshold={MISSINGNESS_THRESHOLD})")
        self._missingness_cols = missingness_cols
        
        # --- Feature interactions (dataset-level × column-level) ---
        # These capture important conditional relationships the model might miss
        INTERACTION_PAIRS = [
            ('n_rows', 'null_pct'),          # Large datasets tolerate missing vals better
            ('n_rows', 'unique_ratio'),       # Cardinality relative to dataset size
            ('baseline_score', 'skewness'),   # Skew matters more when baseline is good
            ('baseline_score', 'entropy'),    # Entropy impact depends on headroom
            ('n_cols', 'avg_feature_corr'),   # Multi-collinearity in wide datasets
            ('relative_headroom', 'baseline_feature_importance'),  # Important features in easy datasets
            ('cat_ratio', 'unique_ratio'),    # Categorical complexity
            ('missing_ratio', 'null_pct'),    # Per-col vs dataset-wide missingness
        ]
        interaction_cols_added = []
        for col_a, col_b in INTERACTION_PAIRS:
            if col_a in X.columns and col_b in X.columns:
                interaction_name = f'ix_{col_a}_x_{col_b}'
                X[interaction_name] = X[col_a] * X[col_b]
                interaction_cols_added.append(interaction_name)
        if interaction_cols_added:
            feature_cols.extend(interaction_cols_added)
            print(f"  Added {len(interaction_cols_added)} feature interactions")
        self._interaction_pairs = INTERACTION_PAIRS
        
        self.feature_columns = feature_cols
        
        self.fill_values = X.median()
        X = X.fillna(self.fill_values)
        
        # Store gate threshold for backward compat with app (score-based threshold)
        # This is the score above which is_helpful=1. With our linear mapping:
        # effect > 0.2 â†’ score > 52, but we use the actual quantile of helpful scores
        helpful_scores = y[is_helpful == 1]
        gate_score_threshold = float(helpful_scores.min()) if len(helpful_scores) > 0 else 52.0
        self._gate_score_threshold = gate_score_threshold
        
        metadata = {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'score_normalization': self._score_normalization,
            'score_distribution': {
                'mean': float(y.mean()), 'std': float(y.std()),
                'min': float(y.min()), 'max': float(y.max()),
                'pct_above_55': float((y > 55).mean()),
                'pct_above_60': float((y > 60).mean()),
                'pct_below_45': float((y < 45).mean()),
            },
            'gate_distribution': {
                'gate_score_threshold': gate_score_threshold,
                'pct_helpful': float(is_helpful.mean()),
                'n_helpful': int(is_helpful.sum()),
                'n_neutral': int((~is_helpful.astype(bool)).sum()),
                'gate_method': 'effect_size_plus_significance',
            },
            'method_mapping': dict(zip(
                self.method_encoder.classes_.tolist(),
                self.method_encoder.transform(self.method_encoder.classes_).tolist()
            )),
            'family_mapping': dict(zip(
                self._family_encoder_classes,
                range(len(self._family_encoder_classes))
            )),
            'methods_in_db': df['method'].value_counts().to_dict(),
            'datasets_in_db': int(df['dataset_name'].nunique()) if 'dataset_name' in df.columns else -1,
            'version': 'v4',
        }
        
        return X, y, is_helpful, dataset_ids, metadata
    
    def save(self, path: str):
        state = {
            'method_encoder_classes': self.method_encoder.classes_.tolist(),
            'task_type_encoder_classes': self.task_type_encoder.classes_.tolist(),
            'feature_columns': self.feature_columns,
            'fill_values': self.fill_values.to_dict(),
            'family_encoder_classes': getattr(self, '_family_encoder_classes', []),
            'interaction_pairs': getattr(self, '_interaction_pairs', []),
            'missingness_cols': getattr(self, '_missingness_cols', []),
            'method_target_enc_map': getattr(self, '_method_target_enc_map', {}),
            'method_target_enc_global': getattr(self, '_method_target_enc_global', 50.0),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.method_encoder.classes_ = np.array(state['method_encoder_classes'])
        self.task_type_encoder.classes_ = np.array(state['task_type_encoder_classes'])
        self.feature_columns = state['feature_columns']
        self.fill_values = pd.Series(state['fill_values'])
        self._family_encoder_classes = state.get('family_encoder_classes', [])
        self._interaction_pairs = state.get('interaction_pairs', [])
        self._missingness_cols = state.get('missingness_cols', [])
        self._method_target_enc_map = state.get('method_target_enc_map', {})
        self._method_target_enc_global = state.get('method_target_enc_global', 50.0)


# =============================================================================
# OPTUNA UTILITIES
# =============================================================================

def _optuna_lgbm_regressor_objective(trial, X, y, dataset_ids, n_folds=5,
                                      sample_weights=None):
    """Optuna objective for LightGBM regressor."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 80),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
    }
    
    if dataset_ids is not None:
        n_groups = dataset_ids.nunique()
        actual_folds = min(n_folds, n_groups)
        kf = GroupKFold(n_splits=actual_folds)
        split_iter = list(kf.split(X, y, groups=dataset_ids))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = list(kf.split(X))
    
    cv_mae = []
    for fold_i, (train_idx, val_idx) in enumerate(split_iter):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = sample_weights[train_idx] if sample_weights is not None else None
        
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_vl, y_vl)], callbacks=callbacks)
        
        preds = model.predict(X_vl)
        cv_mae.append(mean_absolute_error(y_vl, preds))
        
        # Optuna pruning
        trial.report(np.mean(cv_mae), fold_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return np.mean(cv_mae)


def _optuna_lgbm_classifier_objective(trial, X, y_binary, dataset_ids, n_folds=5):
    """Optuna objective for LightGBM classifier (gate model)."""
    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 80),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'scale_pos_weight': scale_pos,
        'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
    }
    
    if dataset_ids is not None:
        n_groups = dataset_ids.nunique()
        actual_folds = min(n_folds, n_groups)
        kf = GroupKFold(n_splits=actual_folds)
        split_iter = list(kf.split(X, y_binary, groups=dataset_ids))
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = list(kf.split(X, y_binary))
    
    cv_auc = []
    for fold_i, (train_idx, val_idx) in enumerate(split_iter):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  eval_metric='auc', callbacks=callbacks)
        
        probs = model.predict_proba(X_vl)[:, 1]
        try:
            fold_auc = roc_auc_score(y_vl, probs)
        except ValueError:
            fold_auc = 0.5
        cv_auc.append(fold_auc)
        
        trial.report(np.mean(cv_auc), fold_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return np.mean(cv_auc)


# =============================================================================
# META-MODEL TRAINER â€” v4 with Optuna
# =============================================================================

class MetaModelTrainer:
    """
    Score regressor: predicts recommendation score (0-100).
    v4: Optuna-tuned hyperparameters.
    """
    
    def __init__(self, model_type='auto'):
        if model_type == 'auto':
            self.model_type = 'xgboost' if HAS_XGB else 'lightgbm'
        else:
            self.model_type = model_type
        self.model = None
        self.cv_scores = None
        self.best_params = None
    
    def train(self, X, y, n_folds=5, dataset_ids=None, n_optuna_trials=60,
             method_labels=None):
        print(f"\nTraining meta-model ({self.model_type})...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        # — Compute method-stratified sample weights —
        sample_weights = None
        if method_labels is not None:
            method_counts = pd.Series(method_labels).value_counts()
            weight_map = 1.0 / method_counts
            raw_weights = pd.Series(method_labels).map(weight_map).values
            sample_weights = raw_weights / raw_weights.mean()
            sample_weights = np.clip(sample_weights, 0.1, 10.0)
            print(f"  Using method-stratified sample weights "
                  f"(min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, "
                  f"{len(method_counts)} methods)")
        
        # â”€â”€ Optuna HPO â”€â”€
        if HAS_OPTUNA and n_optuna_trials > 0:
            print(f"  Running Optuna ({n_optuna_trials} trials)...")
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial: _optuna_lgbm_regressor_objective(
                    trial, X, y, dataset_ids, n_folds,
                    sample_weights=sample_weights),
                n_trials=n_optuna_trials,
                show_progress_bar=True,
            )
            self.best_params = study.best_params
            print(f"  Best MAE: {study.best_value:.4f}")
            print(f"  Best params: {json.dumps(self.best_params, indent=4)}")
        else:
            # Fallback: hand-tuned params (same as v3)
            self.best_params = {
                'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'reg_lambda': 1.0,
            }
        
        # â”€â”€ Cross-validate with best params â”€â”€
        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = list(kf.split(X, y, groups=dataset_ids))
            print(f"  Using GroupKFold ({actual_folds} folds, {n_groups} datasets)")
        else:
            actual_folds = n_folds
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = kf.split(X)
        
        cv_mae, cv_rmse = [], []
        cv_spearman = []  # NEW: ranking-aware metric
        
        for train_idx, val_idx in split_iter:
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None
            
            model = self._create_model(self.best_params)
            if self.model_type == 'lightgbm':
                callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_vl, y_vl)], callbacks=callbacks)
            else:
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_vl, y_vl)], verbose=False)
            
            preds = model.predict(X_vl)
            cv_mae.append(mean_absolute_error(y_vl, preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y_vl, preds)))
            
            # NEW: Per-dataset Spearman rank correlation
            if dataset_ids is not None:
                val_ds = dataset_ids.iloc[val_idx]
                sp_corrs = []
                for ds_id in val_ds.unique():
                    mask = val_ds == ds_id
                    if mask.sum() >= 3:
                        y_ds = y_vl[mask].values
                        p_ds = preds[mask.values]
                        corr, _ = spearmanr(y_ds, p_ds)
                        if not np.isnan(corr):
                            sp_corrs.append(corr)
                if sp_corrs:
                    cv_spearman.append(float(np.mean(sp_corrs)))
        
        self.cv_scores = {
            'mae_mean': float(np.mean(cv_mae)), 'mae_std': float(np.std(cv_mae)),
            'rmse_mean': float(np.mean(cv_rmse)), 'rmse_std': float(np.std(cv_rmse)),
        }
        if cv_spearman:
            self.cv_scores['spearman_mean'] = float(np.mean(cv_spearman))
            self.cv_scores['spearman_std'] = float(np.std(cv_spearman))
        
        print(f"  CV MAE:  {self.cv_scores['mae_mean']:.3f} ± {self.cv_scores['mae_std']:.3f}")
        print(f"  CV RMSE: {self.cv_scores['rmse_mean']:.3f} ± {self.cv_scores['rmse_std']:.3f}")
        if cv_spearman:
            print(f"  CV Spearman (per-dataset ranking): "
                  f"{self.cv_scores['spearman_mean']:.3f} ± {self.cv_scores['spearman_std']:.3f}")
        
        # Final model on all data
        self.model = self._create_model(self.best_params)
        if self.model_type == 'lightgbm':
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y, sample_weight=sample_weights, verbose=False)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            imp = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
            imp_pct = imp / imp.sum() * 100 if imp.sum() > 0 else imp
            print(f"\n  Top 10 features:")
            for feat, val in imp_pct.head(10).items():
                print(f"    {feat}: {val:.1f}%")
        
        return self.cv_scores
    
    def _create_model(self, params):
        p = {k: v for k, v in params.items()}
        p['random_state'] = 42
        p['verbosity'] = -1 if self.model_type == 'lightgbm' else 0
        p['n_jobs'] = -1
        
        if self.model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBRegressor(**p)
        else:
            return lgb.LGBMRegressor(**p)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def calibrate_per_method(self, X, y, method_labels):
        """Learn per-method residual corrections from OOF predictions."""
        from sklearn.model_selection import KFold
        oof_preds = np.zeros(len(X))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X):
            model = self._create_model(self.best_params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False if self.model_type == 'xgboost' else None)
            oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        
        residuals = y.values - oof_preds
        residual_df = pd.DataFrame({'method': method_labels, 'residual': residuals})
        # Per-method mean residual, shrunk toward 0
        method_counts = residual_df.groupby('method')['residual'].count()
        method_mean_resid = residual_df.groupby('method')['residual'].mean()
        shrink = method_counts / (method_counts + 30)  # regularization
        self._method_calibration = (method_mean_resid * shrink).to_dict()
    
    def predict_calibrated(self, X, method_labels):
        raw = self.predict(X)
        offsets = np.array([self._method_calibration.get(m, 0.0) for m in method_labels])
        return raw + offsets
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model, 
                'model_type': self.model_type,
                'cv_scores': self.cv_scores, 
                'best_params': self.best_params,
                # ADD THIS LINE: Persist the calibration dictionary
                'method_calibration': getattr(self, '_method_calibration', {}) 
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.model_type = state['model_type']
        self.cv_scores = state.get('cv_scores')
        self.best_params = state.get('best_params')
        # ADD THIS LINE: Load the calibration dictionary (default to empty if missing)
        self._method_calibration = state.get('method_calibration', {})


# =============================================================================
# GATE MODEL â€” v4: Optuna-tuned, effect-size target
# =============================================================================

class GateModelTrainer:
    """
    Binary classifier predicting P(helpful) for each candidate transform.
    
    v4 improvements:
    - Optuna-tuned hyperparameters
    - Trained on effect-size + significance target (not quantile-based)
    - Two-stage threshold: first maximize F1, then fine-tune recall â‰¥ 0.70
    """
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.cv_scores = None
        self.best_params = None
    
    def train(self, X, y_binary, n_folds=5, dataset_ids=None, n_optuna_trials=60):
        print(f"\nTraining gate model (binary classifier)...")
        n_pos = int(y_binary.sum())
        n_neg = len(y_binary) - n_pos
        print(f"  Samples: {len(X)}, Positive: {n_pos} ({y_binary.mean()*100:.1f}%)")
        
        # â”€â”€ Optuna HPO â”€â”€
        if HAS_OPTUNA and n_optuna_trials > 0:
            print(f"  Running Optuna ({n_optuna_trials} trials)...")
            study = optuna.create_study(
                direction='maximize',  # maximize AUC
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial: _optuna_lgbm_classifier_objective(trial, X, y_binary, dataset_ids, n_folds),
                n_trials=n_optuna_trials,
                show_progress_bar=True,
            )
            self.best_params = study.best_params
            print(f"  Best AUC: {study.best_value:.4f}")
            print(f"  Best params: {json.dumps(self.best_params, indent=4)}")
        else:
            scale_pos = n_neg / max(n_pos, 1)
            self.best_params = {
                'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
                'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'min_child_samples': max(5, n_pos // 50),
                'reg_alpha': 0.1, 'reg_lambda': 1.0,
                'scale_pos_weight': scale_pos,
            }
        
        # â”€â”€ Cross-validate with best params for final metrics â”€â”€
        scale_pos = n_neg / max(n_pos, 1)
        final_params = {k: v for k, v in self.best_params.items()}
        final_params['scale_pos_weight'] = scale_pos
        final_params['random_state'] = 42
        final_params['verbosity'] = -1
        final_params['n_jobs'] = -1
        
        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = list(kf.split(X, y_binary, groups=dataset_ids))
            print(f"  Using GroupKFold ({actual_folds} folds, {n_groups} datasets)")
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = list(kf.split(X, y_binary))
        
        cv_auc = []
        all_probs, all_true = [], []
        
        for train_idx, val_idx in split_iter:
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**final_params)
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                    eval_metric='auc', callbacks=callbacks)
            
            probs = model.predict_proba(X_vl)[:, 1]
            all_probs.extend(probs.tolist())
            all_true.extend(y_vl.tolist())
            
            try:
                fold_auc = roc_auc_score(y_vl, probs)
            except ValueError:
                fold_auc = 0.5
            cv_auc.append(fold_auc)
        
        # â”€â”€ Calibrate threshold â”€â”€
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        precision, recall, thresholds = precision_recall_curve(all_true, all_probs)
        
        # Strategy: find threshold that maximizes F1 with recall >= 0.70
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        valid_mask = recall[:-1] >= 0.70
        if valid_mask.any():
            # Among those with recall >= 0.70, pick highest F1
            f1_valid = f1_scores[:-1].copy()
            f1_valid[~valid_mask] = 0
            best_idx = np.argmax(f1_valid)
            self.threshold = float(thresholds[best_idx])
        else:
            # Fallback: find closest to recall=0.70
            closest = np.argmin(np.abs(recall[:-1] - 0.70))
            self.threshold = float(thresholds[closest])
        
        # Compute metrics at calibrated threshold
        cal_preds = (all_probs >= self.threshold).astype(int)
        cal_f1 = f1_score(all_true, cal_preds)
        cal_recall = cal_preds[all_true == 1].mean() if all_true.sum() > 0 else 0
        cal_precision = all_true[cal_preds == 1].mean() if cal_preds.sum() > 0 else 0
        
        self.cv_scores = {
            'auc_mean': float(np.mean(cv_auc)), 'auc_std': float(np.std(cv_auc)),
            'f1_at_threshold': float(cal_f1),
            'threshold': float(self.threshold),
        }
        print(f"  CV AUC:  {self.cv_scores['auc_mean']:.3f} Â± {self.cv_scores['auc_std']:.3f}")
        print(f"  Calibrated threshold: {self.threshold:.3f}")
        print(f"  At threshold: F1={cal_f1:.3f}, precision={cal_precision:.3f}, recall={cal_recall:.3f}")
        
        # â”€â”€ Final model on all data â”€â”€
        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X, y_binary)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            imp = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
            imp_pct = imp / imp.sum() * 100 if imp.sum() > 0 else imp
            top5 = imp_pct.head(5)
            if top5.sum() > 0:
                print(f"  Gate top-5 features: {', '.join(f'{k}={v:.1f}%' for k, v in top5.items())}")
        
        return self.cv_scores
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.threshold,
                        'cv_scores': self.cv_scores, 'best_params': self.best_params}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.threshold = state['threshold']
        self.cv_scores = state.get('cv_scores')
        self.best_params = state.get('best_params')


# =============================================================================
# RANKING MODEL â€” v4: Tighter training set, Optuna-tuned
# =============================================================================

class RankingModelTrainer:
    """
    LightGBM ranker with lambdarank objective.
    
    v4 improvements:
    - Only trained on transforms with meaningful positive effect (cohens_d > 0.3)
    - Uses raw effect signal as relevance label (not quantile-binned compressed scores)
    - Optuna-tuned hyperparameters for the ranker
    """
    
    def __init__(self):
        self.model = None
        self.cv_scores = None
        self.best_params = None
    
    def train(self, X, y_scores, dataset_ids, effect_signals=None, n_folds=5, n_optuna_trials=30):
        """
        Train ranking model on helpful rows.
        
        Args:
            X: features (only helpful rows)
            y_scores: recommendation scores (0-100)
            dataset_ids: integer dataset identifiers
            effect_signals: raw effect signals for better relevance labels
        """
        print(f"\nTraining ranking model (lambdarank)...")
        print(f"  Samples: {len(X)}, Datasets: {dataset_ids.nunique()}")
        
        if len(X) < 20:
            print("  âš  Too few helpful samples. Skipping.")
            self.model = None
            return None
        
        # â”€â”€ Relevance labels â”€â”€
        # Use continuous effect signal for finer-grained ranking
        # LightGBM lambdarank supports float relevance labels
        if effect_signals is not None:
            # Clip to non-negative (only rank positive-effect transforms)
            shifted = effect_signals.clip(lower=0)
            # Scale to 0-100 range for better gradient resolution
            max_eff = shifted.max()
            if max_eff > 0:
                # Scale to range 0-30
                relevance = (shifted / max_eff * 30)
            else:
                relevance = shifted
            
        else:
            # Fallback: Scale 0-100 scores to 0-30
            relevance = y_scores / 100.0 * 30
        
        # CRITICAL FIX: Round, CLIP strictly to 0-30, and cast to int
        relevance = relevance.round().clip(0, 30).astype(int)
        
        # â”€â”€ Build groups â”€â”€
        group_df = pd.DataFrame({'dataset_id': dataset_ids.values, 'idx': range(len(X))})
        group_df = group_df.sort_values('dataset_id')
        sorted_idx = group_df['idx'].values
        X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
        relevance_sorted = relevance.iloc[sorted_idx].reset_index(drop=True)
        group_sizes = group_df.groupby('dataset_id').size().values
        
        # Filter single-sample groups
        valid_groups = group_sizes > 1
        if valid_groups.sum() < 3:
            print("  âš  Too few multi-sample groups. Skipping.")
            self.model = None
            return None
        
        cumsum = np.cumsum(np.concatenate([[0], group_sizes]))
        valid_indices = []
        valid_group_sizes = []
        for i, v in enumerate(valid_groups):
            if v:
                valid_indices.extend(range(cumsum[i], cumsum[i+1]))
                valid_group_sizes.append(group_sizes[i])
        
        X_rank = X_sorted.iloc[valid_indices].reset_index(drop=True)
        rel_rank = relevance_sorted.iloc[valid_indices].reset_index(drop=True)
        group_sizes_valid = np.array(valid_group_sizes)
        
        print(f"  After filtering: {len(X_rank)} samples in {len(group_sizes_valid)} groups")
        print(f"  Relevance distribution: {dict(zip(*np.unique(rel_rank, return_counts=True)))}")
        
        # â”€â”€ Optuna HPO for ranker â”€â”€
        # Ranker uses LightGBM native API, so Optuna needs special handling
        # For now, use sensible defaults since ranker Optuna is complex
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5],
            'num_leaves': 31, 'learning_rate': 0.05,
            'max_depth': 6, 'min_child_samples': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
        }
        
        # If Optuna is available, do a simple grid search on key params
        if HAS_OPTUNA and n_optuna_trials > 0:
            n_groups = len(group_sizes_valid)
            n_train_groups = max(int(n_groups * 0.8), 1)
            train_end = sum(group_sizes_valid[:n_train_groups])
            
            def ranker_objective(trial):
                test_params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'ndcg_eval_at': [3, 5],
                    'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
                    'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
                }
                train_set = lgb.Dataset(X_rank.iloc[:train_end], label=rel_rank.iloc[:train_end],
                                        group=group_sizes_valid[:n_train_groups])
                val_set = lgb.Dataset(X_rank.iloc[train_end:], label=rel_rank.iloc[train_end:],
                                      group=group_sizes_valid[n_train_groups:], reference=train_set)
                cbs = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
                try:
                    model = lgb.train(test_params, train_set, num_boost_round=400,
                                     valid_sets=[val_set], callbacks=cbs)
                    return model.best_score['valid_0'].get('ndcg@3', 0)
                except:
                    return 0.0
            
            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(ranker_objective, n_trials=n_optuna_trials, show_progress_bar=True)
            params.update(study.best_params)
            print(f"  Best ranker NDCG@3: {study.best_value:.4f}")
        
        # â”€â”€ Train final ranker â”€â”€
        n_groups = len(group_sizes_valid)
        n_train_groups = max(int(n_groups * 0.8), 1)
        train_end = sum(group_sizes_valid[:n_train_groups])
        
        train_set = lgb.Dataset(X_rank.iloc[:train_end], label=rel_rank.iloc[:train_end],
                                group=group_sizes_valid[:n_train_groups])
        val_set = lgb.Dataset(X_rank.iloc[train_end:], label=rel_rank.iloc[train_end:],
                              group=group_sizes_valid[n_train_groups:], reference=train_set)
        
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        self.model = lgb.train(params, train_set, num_boost_round=300,
                               valid_sets=[val_set], callbacks=callbacks)
        
        if self.model.best_score and 'valid_0' in self.model.best_score:
            ndcg_scores = self.model.best_score['valid_0']
            self.cv_scores = {k: float(v) for k, v in ndcg_scores.items()}
            print(f"  Validation NDCG: {self.cv_scores}")
        else:
            self.cv_scores = {}
        
        # Retrain on all data
        full_set = lgb.Dataset(X_rank, label=rel_rank, group=group_sizes_valid)
        self.model = lgb.train(params, full_set,
                               num_boost_round=self.model.best_iteration or 100)
        
        return self.cv_scores
    
    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'cv_scores': self.cv_scores,
                        'best_params': self.best_params}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')
        self.best_params = state.get('best_params')


# =============================================================================
# UPSIDE MODEL — Quantile Regression for potential upside (90th percentile)
# =============================================================================

class UpsideModelTrainer:
    """
    LightGBM quantile regressor predicting the 90th percentile of effect.
    
    Rationale: Some methods are "high variance" (e.g., Arithmetic Interactions).
    They usually do nothing, but when they work, they work well. If you predict
    the mean, you wash out these "high risk/high reward" methods. Predicting
    the 90th percentile captures the *potential upside* of a method.
    
    This model complements the mean regressor:
    - Mean regressor → expected outcome (safe recommendations)
    - Upside model → best-case outcome (adventurous recommendations)
    """
    
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.model = None
        self.cv_scores = None
        self.best_params = None
    
    def train(self, X, y_effect, n_folds=5, dataset_ids=None, n_optuna_trials=30):
        """
        Train quantile regressor on raw effect signals (NOT mapped scores).
        
        Args:
            X: feature matrix
            y_effect: raw effect signals (centered ~0, positive = helpful)
            dataset_ids: for GroupKFold
        """
        print(f"\nTraining upside model (quantile={self.alpha})...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Effect signal: mean={y_effect.mean():.4f}, "
              f"p90={np.percentile(y_effect, 90):.4f}, "
              f"max={y_effect.max():.4f}")
        
        # Optuna HPO
        if HAS_OPTUNA and n_optuna_trials > 0:
            print(f"  Running Optuna ({n_optuna_trials} trials)...")
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(
                lambda trial: self._optuna_objective(trial, X, y_effect, dataset_ids, n_folds),
                n_trials=n_optuna_trials,
                show_progress_bar=True,
            )
            self.best_params = study.best_params
            print(f"  Best pinball loss: {study.best_value:.4f}")
        else:
            self.best_params = {
                'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
                'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'reg_lambda': 1.0,
            }
        
        # Cross-validate
        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = list(kf.split(X, y_effect, groups=dataset_ids))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = list(kf.split(X))
        
        cv_pinball = []
        cv_coverage = []
        
        for train_idx, val_idx in split_iter:
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y_effect.iloc[train_idx], y_effect.iloc[val_idx]
            
            model = self._create_model(self.best_params)
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)
            
            preds = model.predict(X_vl)
            # Pinball loss (quantile loss)
            errors = y_vl.values - preds
            pinball = np.mean(np.where(errors >= 0, self.alpha * errors,
                                        (1 - self.alpha) * (-errors)))
            cv_pinball.append(pinball)
            # Coverage: what fraction of actual values are below the predicted quantile?
            coverage = (y_vl.values <= preds).mean()
            cv_coverage.append(coverage)
        
        self.cv_scores = {
            'pinball_mean': float(np.mean(cv_pinball)),
            'pinball_std': float(np.std(cv_pinball)),
            'coverage_mean': float(np.mean(cv_coverage)),
            'coverage_std': float(np.std(cv_coverage)),
        }
        print(f"  CV Pinball: {self.cv_scores['pinball_mean']:.4f} "
              f"± {self.cv_scores['pinball_std']:.4f}")
        print(f"  CV Coverage: {self.cv_scores['coverage_mean']:.2%} "
              f"(target: {self.alpha:.0%})")
        
        # Final model on all data
        self.model = self._create_model(self.best_params)
        self.model.fit(X, y_effect)
        
        return self.cv_scores
    
    def _optuna_objective(self, trial, X, y, dataset_ids, n_folds):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 60),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = list(kf.split(X, y, groups=dataset_ids))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = list(kf.split(X))
        
        cv_pinball = []
        for fold_i, (train_idx, val_idx) in enumerate(split_iter):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._create_model(params)
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)
            
            preds = model.predict(X_vl)
            errors = y_vl.values - preds
            pinball = np.mean(np.where(errors >= 0, self.alpha * errors,
                                        (1 - self.alpha) * (-errors)))
            cv_pinball.append(pinball)
            
            trial.report(np.mean(cv_pinball), fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(cv_pinball)
    
    def _create_model(self, params):
        p = {k: v for k, v in params.items()}
        p['objective'] = 'quantile'
        p['alpha'] = self.alpha
        p['random_state'] = 42
        p['verbosity'] = -1
        p['n_jobs'] = -1
        return lgb.LGBMRegressor(**p)
    
    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'alpha': self.alpha,
                        'cv_scores': self.cv_scores, 'best_params': self.best_params}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.alpha = state.get('alpha', 0.9)
        self.cv_scores = state.get('cv_scores')
        self.best_params = state.get('best_params')


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_meta_model(meta_db_path: str, output_dir: str = './meta_model',
                     n_optuna_trials: int = 60):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("META-MODEL TRAINING v5 (Upside + Interactions + Ensemble)")
    print("=" * 70)
    
    # â”€â”€ Load â”€â”€
    print(f"\nLoading: {meta_db_path}")
    df = pd.read_csv(meta_db_path)
    n_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else '?'
    print(f"  {len(df)} rows, {n_datasets} datasets")
    
    # Quick data quality report
    for col in ['cohens_d', 'calibrated_delta', 'individual_cohens_d']:
        if col in df.columns:
            pct_avail = df[col].notna().mean() * 100
            print(f"  {col}: {pct_avail:.0f}% available")
    
    # â”€â”€ Prepare â”€â”€
    preparator = MetaDBPreparator()
    X, y, is_helpful, dataset_ids, metadata = preparator.prepare(df)
    
    print(f"\nScore distribution (after linear mapping):")
    print(f"  Mean: {y.mean():.1f}  Std: {y.std():.1f}  Range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  >55 (helpful): {(y > 55).mean()*100:.1f}% ({(y > 55).sum()} rows)")
    print(f"  >60 (strong):  {(y > 60).mean()*100:.1f}%")
    print(f"  <45 (negative): {(y < 45).mean()*100:.1f}%")
    print(f"  Gate positive class: {is_helpful.sum()} ({is_helpful.mean()*100:.1f}%)")
    
    # Also compute effect signals for ranker later
    df_filtered = df[~df['method'].isin(['null_intervention'])].reset_index(drop=True)
    effect_signals = df_filtered.apply(compute_effect_signal, axis=1)
    effect_signals.index = X.index  # Align indices
    
    # Extract method labels for sample weighting and calibration
    method_labels = df_filtered['method'].values[:len(X)]
    
    # â”€â”€ Model 1: Score regressor â”€â”€
    print("\n" + "â”€" * 50)
    trainer = MetaModelTrainer(model_type='auto')
    cv_scores = trainer.train(X, y, dataset_ids=dataset_ids,
                              n_optuna_trials=n_optuna_trials,
                              method_labels=method_labels)
    
    print("Calibrating residuals per method...")
    trainer.calibrate_per_method(X, y, method_labels)

    # â”€â”€ Model 2: Gate classifier â”€â”€
    print("\n" + "â”€" * 50)
    gate_trainer = GateModelTrainer()
    gate_cv = gate_trainer.train(X, is_helpful, dataset_ids=dataset_ids, n_optuna_trials=n_optuna_trials)
    
    # â”€â”€ Model 3: Ranking model (on helpful-only rows) â”€â”€
    print("\n" + "â”€" * 50)
    ranker_trainer = RankingModelTrainer()
    ranker_cv = None
    if dataset_ids is not None:
        helpful_mask = is_helpful.astype(bool)
        if helpful_mask.sum() >= 20:
            X_helpful = X[helpful_mask].reset_index(drop=True)
            y_helpful = y[helpful_mask].reset_index(drop=True)
            ds_helpful = dataset_ids[helpful_mask].reset_index(drop=True)
            eff_helpful = effect_signals[helpful_mask].reset_index(drop=True)
            ranker_cv = ranker_trainer.train(X_helpful, y_helpful, ds_helpful,
                                             effect_signals=eff_helpful,
                                             n_optuna_trials=n_optuna_trials // 2)
        else:
            print(f"\nâš  Only {helpful_mask.sum()} helpful rows â€” skipping ranker training")
    else:
        print("\nNo dataset_name column — skipping ranker training")
    
    # —— Model 4: Upside model (quantile regression, 90th percentile) ——
    print("\n" + "—" * 50)
    upside_trainer = UpsideModelTrainer(alpha=0.9)
    upside_cv = upside_trainer.train(
        X, effect_signals, dataset_ids=dataset_ids,
        n_optuna_trials=max(n_optuna_trials // 2, 10)
    )
    
    # —— Compute ensemble weights via hold-out correlation ——
    print("\nComputing ensemble weights...")
    ensemble_weights = {'regressor': 0.7, 'gate': 0.0, 'upside': 0.3}  # gate=0 (used as filter only)
    if dataset_ids is not None:
        try:
            n_groups = dataset_ids.nunique()
            if n_groups >= 5:
                kf = GroupKFold(n_splits=min(5, n_groups))
                oof_reg = np.zeros(len(X))
                oof_upside = np.zeros(len(X))
                
                for train_idx, val_idx in kf.split(X, y, groups=dataset_ids):
                    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr = y.iloc[train_idx]
                    eff_tr = effect_signals.iloc[train_idx]
                    
                    # Quick retrain for OOF predictions
                    tmp_reg = MetaModelTrainer(model_type='auto')
                    tmp_reg.best_params = trainer.best_params
                    tmp_model = tmp_reg._create_model(trainer.best_params)
                    tmp_model.fit(X_tr, y_tr)
                    oof_reg[val_idx] = tmp_model.predict(X_vl)
                    
                    tmp_up = UpsideModelTrainer(alpha=0.9)
                    up_params = upside_trainer.best_params or {
                        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05}
                    up_model = tmp_up._create_model(up_params)
                    up_model.fit(X_tr, eff_tr)
                    oof_upside[val_idx] = up_model.predict(X_vl)
                
                # Convert upside effect predictions to score scale
                oof_upside_score = effect_to_score(oof_upside)
                
                # Optimize weights for regressor + upside only (gate used as filter)
                from scipy.optimize import minimize
                
                def weighted_mae(w):
                    w = np.abs(w)
                    w = w / w.sum()
                    blended = w[0] * oof_reg + w[1] * oof_upside_score
                    return mean_absolute_error(y, blended)
                
                result = minimize(weighted_mae, [0.7, 0.3],
                                method='Nelder-Mead', options={'maxiter': 1000})
                w = np.abs(result.x)
                w = w / w.sum()
                ensemble_weights = {
                    'regressor': float(round(w[0], 3)),
                    'gate': 0.0,  # gate used as filter only, not in score blend
                    'upside': float(round(w[1], 3)),
                }
                print(f"  Learned ensemble weights: {ensemble_weights}")
                print(f"  Ensemble MAE: {result.fun:.3f} vs regressor-only: "
                      f"{mean_absolute_error(y, oof_reg):.3f}")
        except Exception as e:
            print(f"  Ensemble weight learning failed ({e}), using defaults")
    
    # —— Save models ——
    preparator.save(os.path.join(output_dir, 'preparator.pkl'))
    trainer.save(os.path.join(output_dir, 'meta_model.pkl'))
    gate_trainer.save(os.path.join(output_dir, 'gate_model.pkl'))
    ranker_trainer.save(os.path.join(output_dir, 'ranker_model.pkl'))
    upside_trainer.save(os.path.join(output_dir, 'upside_model.pkl'))
    
    with open(os.path.join(output_dir, 'ensemble_weights.json'), 'w') as f:
        json.dump(ensemble_weights, f, indent=2)
    
    all_cv = {'regressor': cv_scores, 'gate': gate_cv, 'ranker': ranker_cv,
              'upside': upside_cv}
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'cv_scores.json'), 'w') as f:
        json.dump(all_cv, f, indent=2, default=str)
    
    # â”€â”€ Method priors (v4: effect-size based) â”€â”€
    print("\n" + "â”€" * 50)
    print("Computing method priors (effect-size based)...")
    
    method_scores = pd.DataFrame({
        'method': df_filtered['method'],
        'recommendation_score': y,
        'effect_signal': effect_signals,
        'is_helpful': is_helpful,
    })
    
    score_summary = method_scores.groupby('method').agg(
        score_mean=('recommendation_score', 'mean'),
        score_std=('recommendation_score', 'std'),
        effect_mean=('effect_signal', 'mean'),
        effect_std=('effect_signal', 'std'),
        effect_p90=('effect_signal', lambda x: np.percentile(x, 90)),
        effect_p10=('effect_signal', lambda x: np.percentile(x, 10)),
        pct_helpful=('is_helpful', 'mean'),
        count=('recommendation_score', 'count'),
    ).round(4)
    score_summary = score_summary.sort_values('effect_mean', ascending=False)
    score_summary.to_csv(os.path.join(output_dir, 'method_score_summary.csv'))
    
    print(f"\n  Method score summary (sorted by effect size):")
    print(score_summary.to_string())
    
    # Build priors dict
    gate_threshold_val = preparator._gate_score_threshold
    method_priors = {}
    for method_name, row in score_summary.iterrows():
        mean_score = row['score_mean']
        effect_mean = row['effect_mean']
        pct_helpful = row['pct_helpful']
        count = row['count']
        
        # Prior adjustment based on mean score deviation from 50
        prior_adjustment = mean_score - 50.0
        
        # v4: Also flag methods that are empirically harmful by effect size
        n_harmful = method_scores[
            (method_scores['method'] == method_name) &
            (method_scores['effect_signal'] < -0.2)
        ].shape[0]
        pct_harmful = n_harmful / max(count, 1)
        
        effect_p90 = row['effect_p90']
        effect_p10 = row['effect_p10']
        effect_range = effect_p90 - effect_p10
        
        method_priors[method_name] = {
            'mean_score': float(mean_score),
            'prior_adjustment': float(prior_adjustment),
            'effect_mean': float(effect_mean),
            'effect_p90': float(effect_p90),   # Upside potential
            'effect_p10': float(effect_p10),   # Downside risk
            'effect_range': float(effect_range),  # Variance indicator
            'pct_helpful': float(pct_helpful),
            'pct_harmful': float(pct_harmful),
            'count': int(count),
        }
    
    with open(os.path.join(output_dir, 'method_priors.json'), 'w') as f:
        json.dump(method_priors, f, indent=2)
    print(f"\n  Method priors saved ({len(method_priors)} methods)")
    
    # â”€â”€ Summary â”€â”€
    print(f"\n{'=' * 70}")
    print(f"v5 TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Output: {output_dir}/")
    print(f"  - preparator.pkl")
    print(f"  - meta_model.pkl (Optuna-tuned score regressor)")
    print(f"  - gate_model.pkl (Optuna-tuned gate classifier)")
    print(f"  - ranker_model.pkl (continuous-relevance lambdarank ranker)")
    print(f"  - upside_model.pkl (quantile={upside_trainer.alpha} upside regressor)")
    print(f"  - ensemble_weights.json (learned model blend weights)")
    print(f"  - method_priors.json (effect-size based priors)")
    print(f"  - method_score_summary.csv")
    print(f"  - metadata.json / cv_scores.json")
    
    print(f"\n  Key metrics:")
    print(f"    Regressor MAE:  {cv_scores['mae_mean']:.3f} Â± {cv_scores['mae_std']:.3f}")
    print(f"    Gate AUC:       {gate_cv['auc_mean']:.3f} Â± {gate_cv['auc_std']:.3f}")
    if ranker_cv:
        ndcg3 = ranker_cv.get('ndcg@3', 'N/A')
        print(f"    Ranker NDCG@3:  {ndcg3}")
    if upside_cv:
        print(f"    Upside pinball: {upside_cv['pinball_mean']:.4f} "
              f"(coverage: {upside_cv['coverage_mean']:.2%})")
    print(f"    Ensemble wts:   {ensemble_weights}")
    
    # v5 improvement notes
    print(f"\n  v4â†’v4 improvements:")
    print(f"    Score spread: {y.std():.1f} std (v3 was ~3.0 â€” most rows at 50)")
    print(f"    Gate class: {is_helpful.mean()*100:.1f}% positive "
          f"(using effect size + significance, not quantile)")
    if HAS_OPTUNA:
        print(f"    All models Optuna-tuned ({n_optuna_trials} trials each)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the meta-model for Auto-FE (v4)')
    parser.add_argument('--meta-db', required=True, help='Path to meta_learning_db.csv')
    parser.add_argument('--output-dir', default='./meta_model', help='Where to save the model')
    parser.add_argument('--n-optuna-trials', type=int, default=60,
                        help='Number of Optuna trials per model (0 to disable)')
    args = parser.parse_args()
    
    train_meta_model(args.meta_db, args.output_dir, n_optuna_trials=args.n_optuna_trials)