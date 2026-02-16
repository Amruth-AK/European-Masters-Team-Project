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

from archetype_guidance import load_pipeline_guidance_models, ArchetypeGuidance


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
    'date_cyclical_hour': 'encoding', 'date_cyclical_day': 'encoding',
    'cyclical_encode': 'feature_creation',
}

# Phase 4.3: Impact type classification
# additive = creates genuinely NEW information LightGBM can't derive
# encoding = re-encodes existing info (LightGBM's histogram binning already approximates most)
# cleaning = handles data quality issues
METHOD_IMPACT_TYPE = {
    # Additive Ã¢â‚¬â€ genuinely new signal
    'product_interaction': 'additive', 'division_interaction': 'additive',
    'addition_interaction': 'additive', 'subtraction_interaction': 'additive',
    'abs_diff_interaction': 'additive',
    'three_way_interaction': 'additive', 'three_way_addition': 'additive',
    'three_way_ratio': 'additive',
    'group_mean': 'additive', 'group_std': 'additive',
    'cat_concat': 'additive', 'text_stats': 'additive',
    'row_stats': 'additive',
    'polynomial_square': 'additive', 'cyclical_encode': 'additive',
    'target_encoding': 'additive',
    # Encoding Ã¢â‚¬â€ re-encodes existing information
    'log_transform': 'encoding', 'sqrt_transform': 'encoding',
    'frequency_encoding': 'encoding', 'onehot_encoding': 'encoding',
    'hashing_encoding': 'encoding',
    'date_extract_basic': 'encoding', 'date_cyclical_month': 'encoding',
    'date_cyclical_dow': 'encoding', 'date_elapsed_days': 'encoding',
    'date_cyclical_hour': 'encoding', 'date_cyclical_day': 'encoding',
    'impute_median': 'cleaning', 'missing_indicator': 'cleaning',
    'quantile_binning': 'cleaning',
}

IMPACT_TYPE_EMOJI = {'additive': 'ðŸŸ¢', 'encoding': 'ðŸ”„', 'cleaning': 'ðŸ§¹'}


# Method family groupings â€” matches train_meta_model_3.py
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
for _fam, _methods in METHOD_FAMILIES.items():
    for _m in _methods:
        METHOD_TO_FAMILY[_m] = _fam

# Feature interaction pairs â€” matches train_meta_model_3.py
META_FEATURE_INTERACTION_PAIRS = [
    ('n_rows', 'null_pct'),
    ('n_rows', 'unique_ratio'),
    ('baseline_score', 'skewness'),
    ('baseline_score', 'entropy'),
    ('n_cols', 'avg_feature_corr'),
    ('relative_headroom', 'baseline_feature_importance'),
    ('cat_ratio', 'unique_ratio'),
    ('missing_ratio', 'null_pct'),
]


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


def deduplicate_and_diversify(candidates, diversity_penalty=0.8):
    """
    Phase 0.2: Deduplicate and diversify recommendations.
    
    1. Deduplicate: For interactions using the same set of columns, keep only the
       highest-scored one UNLESS the methods are fundamentally different
       (e.g., product vs. division Ã¢â‚¬â€ keep both; but three_way_interaction vs
       three_way_addition on same 3 cols Ã¢â‚¬â€ keep only best).
    2. Diversify: After selecting the top interaction, downweight remaining 
       interactions that share >=2 columns with already-selected ones.
    """
    # Separate single-column from interactions
    singles = [c for c in candidates if not c.get('is_interaction', False)]
    interactions = [c for c in candidates if c.get('is_interaction', False)]
    
    # --- Step 1: Deduplicate interactions ---
    # Group methods by "functional family" Ã¢â‚¬â€ same family on same columns = duplicate
    METHOD_FAMILY = {
        'product_interaction': 'multiplicative',
        'three_way_interaction': 'multiplicative_3',
        'three_way_ratio': 'ratio_3',
        'division_interaction': 'ratio',
        'addition_interaction': 'additive',
        'three_way_addition': 'additive_3',
        'subtraction_interaction': 'difference',
        'abs_diff_interaction': 'difference',
        'group_mean': 'group_agg',
        'group_std': 'group_agg_spread',
        'cat_concat': 'cat_combine',
    }
    
    seen_keys = {}  # (family, frozenset(cols)) -> best candidate
    deduped_interactions = []
    
    for inter in interactions:
        col_a = inter.get('col_a', inter.get('column', ''))
        col_b = inter.get('col_b', '')
        col_c = inter.get('col_c', '')
        cols = frozenset(filter(None, [col_a, col_b, col_c]))
        family = METHOD_FAMILY.get(inter['method'], inter['method'])
        key = (family, cols)
        
        if key in seen_keys:
            # Keep higher score
            if inter.get('predicted_score', 0) > seen_keys[key].get('predicted_score', 0):
                seen_keys[key] = inter
        else:
            seen_keys[key] = inter
    
    deduped_interactions = list(seen_keys.values())
    deduped_interactions.sort(key=lambda x: x.get('predicted_score', 0), reverse=True)
    
    # --- Step 2: Diversity constraint for interactions ---
    selected_cols_used = []  # list of column sets already selected
    diversified = []
    
    for inter in deduped_interactions:
        col_a = inter.get('col_a', inter.get('column', ''))
        col_b = inter.get('col_b', '')
        col_c = inter.get('col_c', '')
        cols = set(filter(None, [col_a, col_b, col_c]))
        
        # Check overlap with already-selected interactions
        overlap_count = sum(1 for used_cols in selected_cols_used if len(cols & used_cols) >= 2)
        if overlap_count > 0:
            # Apply diversity penalty for each overlapping selection
            inter = dict(inter)  # copy to avoid mutation
            inter['predicted_score'] = inter.get('predicted_score', 0) * (diversity_penalty ** overlap_count)
        
        diversified.append(inter)
        selected_cols_used.append(cols)
    
    # Re-sort after diversity penalties
    diversified.sort(key=lambda x: x.get('predicted_score', 0), reverse=True)
    
    # --- Step 3: Deduplicate single-column transforms ---
    seen_singles = set()
    deduped_singles = []
    for s in singles:
        key = (s.get('method'), s.get('column'))
        if key not in seen_singles:
            seen_singles.add(key)
            deduped_singles.append(s)
    
    return deduped_singles + diversified


# =============================================================================
# HP ARCHETYPE SYSTEM (Phase 3.1 + 3.2)
# =============================================================================

# Each archetype is a battle-tested HP config derived from Kaggle defaults
# and published benchmarks.  All use early stopping with patience=50.
HP_ARCHETYPES = {
    'Tiny': {
        'description': 'Small dataset (< 2k rows) Ã¢â‚¬â€ regularize heavily, prevent overfitting',
        'params': {
            'num_leaves': 20, 'learning_rate': 0.08, 'n_estimators': 500,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.5, 'max_bin': 127, 'max_depth': 5,
        },
    },
    'Small-wide': {
        'description': 'Small dataset with many features (< 5k rows, > 50 cols) Ã¢â‚¬â€ aggressive feature subsampling',
        'params': {
            'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 1000,
            'min_child_samples': 15, 'subsample': 0.8, 'colsample_bytree': 0.5,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'max_bin': 127, 'max_depth': 6,
        },
    },
    'Medium': {
        'description': 'Medium dataset (5k-50k rows, < 100 cols) Ã¢â‚¬â€ balanced defaults',
        'params': {
            'num_leaves': 40, 'learning_rate': 0.05, 'n_estimators': 1500,
            'min_child_samples': 15, 'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.05, 'reg_lambda': 0.5, 'max_bin': 255, 'max_depth': 6,
        },
    },
    'Medium high-cat': {
        'description': 'Medium dataset, heavy categorical (5k-50k rows, cat > 50%) Ã¢â‚¬â€ stronger regularization',
        'params': {
            'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 1500,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'max_bin': 255, 'max_depth': 6,
        },
    },
    'Large': {
        'description': 'Large dataset (50k-500k rows) Ã¢â‚¬â€ more capacity, moderate subsampling',
        'params': {
            'num_leaves': 63, 'learning_rate': 0.05, 'n_estimators': 2000,
            'min_child_samples': 20, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.05, 'reg_lambda': 0.5, 'max_bin': 255, 'max_depth': 7,
        },
    },
    'Huge / high-dim': {
        'description': 'Very large or wide dataset (> 500k rows OR > 500 cols) Ã¢â‚¬â€ aggressive subsampling',
        'params': {
            'num_leaves': 80, 'learning_rate': 0.03, 'n_estimators': 3000,
            'min_child_samples': 30, 'subsample': 0.6, 'colsample_bytree': 0.5,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'max_bin': 255, 'max_depth': 8,
        },
    },
}


def _match_archetype(n_rows, n_cols, cat_ratio=0.0):
    """
    Phase 3.2: Score each archetype by how well it matches the dataset.
    Returns (archetype_name, archetype_dict).
    """
    # Hard-rule matching in priority order (most specific first)
    if n_rows > 500000 or n_cols > 500:
        return 'Huge / high-dim', HP_ARCHETYPES['Huge / high-dim']
    if n_rows > 50000:
        return 'Large', HP_ARCHETYPES['Large']
    if 5000 <= n_rows <= 50000 and cat_ratio > 0.5:
        return 'Medium high-cat', HP_ARCHETYPES['Medium high-cat']
    if 5000 <= n_rows <= 50000:
        return 'Medium', HP_ARCHETYPES['Medium']
    if n_rows < 5000 and n_cols > 50:
        return 'Small-wide', HP_ARCHETYPES['Small-wide']
    if n_rows < 2000:
        return 'Tiny', HP_ARCHETYPES['Tiny']
    # Fallback: small dataset with few columns â†’ Tiny is safe
    return 'Tiny', HP_ARCHETYPES['Tiny']


def compute_lgbm_params(n_rows, n_cols, task_type='classification',
                        n_classes=2, missing_ratio=0.0, cat_ratio=0.0,
                        avg_feature_corr=0.0, class_imbalance_ratio=1.0):
    """
    Phase 3.1+3.2: Select LightGBM hyperparameters via dataset archetype matching.
    
    Instead of formula-based adjustments, matches the dataset to one of 6
    battle-tested archetypes. Each archetype has a fixed HP config derived from
    Kaggle competition defaults and published benchmarks.
    
    All archetypes use early stopping with patience=50 on the n_estimators budget.
    
    Returns: (params_dict, explanation_dict)
    """
    archetype_name, archetype = _match_archetype(n_rows, n_cols, cat_ratio)
    params = archetype['params'].copy()
    params.update({'random_state': 42, 'verbosity': -1, 'n_jobs': -1})
    
    # Build explanation
    explanation = {
        'archetype': f"**{archetype_name}** Ã¢â‚¬â€ {archetype['description']}",
        'match_reason': f"n_rows={n_rows:,}, n_cols={n_cols}, cat_ratio={cat_ratio:.0%}",
    }
    for key in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'max_bin', 'max_depth']:
        explanation[key] = f"{params[key]}"
    explanation['n_estimators'] += " (early stopping at patience=50 picks actual count)"
    
    # Task-specific additions
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
# HP META-MODEL INTEGRATION (Phase 5)
# =============================================================================

# HP parameter bounds (from HPCollector.HP_SPACE, inlined to avoid import dependency)
HP_BOUNDS = {
    'num_leaves':        {'low': 4,     'high': 256,   'log': False, 'type': 'int'},
    'max_depth':         {'low': 3,     'high': 15,    'log': False, 'type': 'int'},
    'learning_rate':     {'low': 0.005, 'high': 0.3,   'log': True,  'type': 'float'},
    'n_estimators':      {'low': 50,    'high': 3000,  'log': True,  'type': 'int'},
    'min_child_samples': {'low': 5,     'high': 100,   'log': False, 'type': 'int'},
    'subsample':         {'low': 0.5,   'high': 1.0,   'log': False, 'type': 'float'},
    'colsample_bytree':  {'low': 0.3,   'high': 1.0,   'log': False, 'type': 'float'},
    'reg_alpha':         {'low': 1e-8,  'high': 10.0,  'log': True,  'type': 'float'},
    'reg_lambda':        {'low': 1e-8,  'high': 10.0,  'log': True,  'type': 'float'},
    'max_bin':           {'low': 63,    'high': 511,   'log': False, 'type': 'int'},
}


def _clip_hp_config(config):
    """Clip HP values to valid bounds and enforce types."""
    clipped = {}
    for param, val in config.items():
        if param in HP_BOUNDS:
            spec = HP_BOUNDS[param]
            val = max(spec['low'], min(spec['high'], val))
            if spec['type'] == 'int':
                val = int(round(val))
            else:
                val = round(val, 6)
        clipped[param] = val
    return clipped


def _compute_hp_derived_features(hp_config):
    """Compute derived HP features for the scorer model (matches HPCollector.py)."""
    num_leaves = hp_config.get('num_leaves', 31)
    max_depth = hp_config.get('max_depth', 6)
    learning_rate = hp_config.get('learning_rate', 0.05)
    n_estimators = hp_config.get('n_estimators', 1000)
    subsample = hp_config.get('subsample', 0.8)
    colsample_bytree = hp_config.get('colsample_bytree', 0.8)
    reg_alpha = hp_config.get('reg_alpha', 0.05)
    reg_lambda = hp_config.get('reg_lambda', 0.5)

    return {
        'hp_leaves_depth_ratio': num_leaves / max(2 ** max_depth, 1),
        'hp_regularization_strength': np.log1p(reg_alpha + reg_lambda),
        'hp_sample_ratio': subsample * colsample_bytree,
        'hp_lr_estimators_product': learning_rate * n_estimators,
    }


def _generate_hp_perturbations(base_config, n_candidates=30, seed=42):
    """
    Generate candidate HP configs by perturbing the base prediction.

    Uses +/-30% perturbations (in log-space for log-scale params),
    sampled randomly for good coverage.
    """
    rng = np.random.RandomState(seed)
    candidates = [base_config.copy()]  # Always include the base prediction

    for i in range(n_candidates - 1):
        config = {}
        for param, base_val in base_config.items():
            if param not in HP_BOUNDS:
                config[param] = base_val
                continue
            spec = HP_BOUNDS[param]
            if spec['log'] and base_val > 0:
                log_val = np.log(max(base_val, 1e-10))
                perturbed = log_val + rng.uniform(-0.35, 0.35)
                config[param] = np.exp(perturbed)
            else:
                delta = base_val * rng.uniform(-0.30, 0.30)
                config[param] = base_val + delta
        config = _clip_hp_config(config)
        candidates.append(config)

    return candidates


def compute_lgbm_params_meta(ds_meta, task_type, n_classes, class_imbalance_ratio,
                              hp_preparator, hp_predictor, hp_scorer=None,
                              hp_ranker=None, n_candidates=30):
    """
    Phase 5: Meta-model-driven HP selection.

    Strategy (combined Path A + B):
    1. Direct Predictor gives a strong starting config
    2. Generate perturbations around it
    3. Scorer ranks all candidates -> pick the best
    4. Falls back to rule-based archetypes if models are unavailable

    Returns: (params_dict, explanation_dict)
    """
    explanation = {}

    # --- Step 1: Direct HP Prediction (Path A) ---
    pred_features = {}
    pred_features.update(ds_meta)
    if task_type in hp_preparator.task_type_encoder.classes_:
        pred_features['task_type_encoded'] = int(
            hp_preparator.task_type_encoder.transform([task_type])[0])
    else:
        pred_features['task_type_encoded'] = 0

    X_pred = pd.DataFrame([pred_features])
    X_pred = hp_preparator.transform_predictor(X_pred)

    # Direct prediction: one forward pass -> complete HP config
    direct_preds = hp_predictor.predict(X_pred)
    direct_config = {}
    for hp_name, values in direct_preds.items():
        param_name = hp_name.replace('hp_', '')
        direct_config[param_name] = float(values[0])
    direct_config = _clip_hp_config(direct_config)

    explanation['strategy'] = 'meta-model'
    explanation['direct_prediction'] = {k: v for k, v in direct_config.items()}

    # --- Step 2: Score-and-Select (Path B) ---
    best_config = direct_config.copy()
    best_score = None

    if hp_scorer is not None and hp_scorer.model is not None:
        candidates = _generate_hp_perturbations(direct_config, n_candidates=n_candidates)

        scorer_rows = []
        for config in candidates:
            row = {}
            row.update(ds_meta)
            row['task_type_encoded'] = pred_features['task_type_encoded']
            for param, val in config.items():
                row[f'hp_{param}'] = val
            row.update(_compute_hp_derived_features(config))
            scorer_rows.append(row)

        X_score = pd.DataFrame(scorer_rows)
        X_score = hp_preparator.transform_scorer(X_score)

        scores = hp_scorer.predict(X_score)

        # Optional: re-rank with the LambdaRank model
        if hp_ranker is not None and hp_ranker.model is not None:
            rank_scores = hp_ranker.predict(X_score)
            combined = list(zip(range(len(candidates)), rank_scores, scores))
            combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best_idx = combined[0][0]
            explanation['refinement'] = 'scorer + ranker'
        else:
            best_idx = int(np.argmax(scores))
            explanation['refinement'] = 'scorer only'

        best_config = candidates[best_idx]
        best_score = float(scores[best_idx])

        direct_score = float(scores[0])
        if best_idx == 0:
            explanation['search_result'] = 'Direct prediction was optimal among candidates'
        else:
            explanation['search_result'] = (
                f'Perturbation #{best_idx} improved predicted score: '
                f'{direct_score:.5f} -> {best_score:.5f} '
                f'(+{best_score - direct_score:.5f})'
            )
        explanation['n_candidates_scored'] = len(candidates)
        explanation['predicted_primary_score'] = best_score
        # Denormalize score for display (scorer predicts z-scores since v2)
        if hasattr(hp_preparator, 'denormalize_score'):
            explanation['predicted_primary_score_raw'] = hp_preparator.denormalize_score(best_score, task_type)
    else:
        explanation['refinement'] = 'direct prediction only (no scorer available)'

    # --- Build final params ---
    params = best_config.copy()
    params.update({'random_state': 42, 'verbosity': -1, 'n_jobs': -1})

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

    # Explanation summary for UI
    explanation['archetype'] = '**Meta-Model** Ã¢â‚¬â€ data-driven HP prediction'
    explanation['match_reason'] = (
        f"Direct predictor + {'scorer refinement' if best_score is not None else 'no refinement'} "
        f"(n_rows={ds_meta.get('n_rows', '?'):,}, n_cols={ds_meta.get('n_cols', '?')})"
    )
    for key in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'max_bin', 'max_depth']:
        if key in params:
            explanation[key] = f"{params[key]}"
    explanation['n_estimators'] = f"{params.get('n_estimators', '')} (early stopping at patience=50 picks actual count)"

    if hp_predictor.cv_scores:
        r2_scores = [v.get('r2_mean', 0) for v in hp_predictor.cv_scores.values()]
        explanation['predictor_avg_r2'] = float(np.mean(r2_scores))
    if hp_scorer is not None and hp_scorer.cv_scores:
        explanation['scorer_r2'] = hp_scorer.cv_scores.get('r2_mean', 0)

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
            meta['target_kurtosis'] = np.nan
            meta['target_nunique_ratio'] = float(len(vc)) / max(X.shape[0], 1)
        else:
            meta['class_imbalance_ratio'] = -1.0
            meta['n_classes'] = -1
            meta['target_std'] = float(y_num.std())
            meta['target_skew'] = float(skew(y_num.dropna()))
            meta['target_kurtosis'] = float(kurtosis(y_num.dropna()))
            meta['target_nunique_ratio'] = float(y_num.nunique()) / max(X.shape[0], 1)
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
        
        # Phase 5: HP meta-model features (avg_skewness, avg_kurtosis over numeric cols)
        if len(numeric_cols) > 0:
            col_skews = []
            col_kurts = []
            for c in numeric_cols:
                clean = X[c].dropna()
                if len(clean) > 2:
                    col_skews.append(float(skew(clean)))
                    col_kurts.append(float(kurtosis(clean)))
            meta['avg_skewness'] = float(np.mean(col_skews)) if col_skews else 0.0
            meta['avg_kurtosis'] = float(np.mean(col_kurts)) if col_kurts else 0.0
        else:
            meta['avg_skewness'] = 0.0
            meta['avg_kurtosis'] = 0.0
        
        # Phase 2.1: Aggregate column features
        meta['n_numeric_cols'] = len(numeric_cols)
        meta['n_cat_cols'] = len(X.select_dtypes(include=['object', 'category', 'bool']).columns)
        # Importance aggregates are filled in after baseline evaluation in recommend()
        meta['std_feature_importance'] = 0.0
        meta['max_minus_min_importance'] = 0.0
        meta['pct_features_above_median_importance'] = 0.5
        # Phase 2.5: Headroom Ã¢â‚¬â€ filled after baseline evaluation
        meta['relative_headroom'] = 0.5
        
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
# ID COLUMN DETECTION
# =============================================================================

def _is_likely_id_column(series, col_name):
    """
    Detect ID-like columns that should be excluded from feature engineering.
    
    Uses a score-based approach (mirrors DataCollector3._is_likely_id_or_constant):
    - High unique ratio â†’ strong ID signal
    - Name contains "id" â†’ strong signal
    - Integer-like â†’ weak signal
    - Float with fractional values â†’ anti-signal (unlikely to be ID)
    
    Returns: (is_id: bool, reason: str)
    """
    clean = series.dropna()
    if len(clean) == 0:
        return False, "empty"
    
    n_unique = series.nunique(dropna=True)
    ur = n_unique / max(len(clean), 1)
    
    # Constants are not IDs but should also be skipped
    if n_unique <= 1:
        return True, "constant"
    
    score = 0
    
    # Unique ratio scoring
    if ur > 0.98:
        score += 60
    elif ur > 0.85:
        score += 40
    elif ur > 0.50:
        score += 10
    
    # Name-based detection Ã¢â‚¬â€ "id", "ID", "_id", "Id", etc.
    if re.search(r'(?i)(\b|_)id(\b|_|$)', col_name):
        score += 30
    # Also catch common patterns: "index", "key", "code" with high uniqueness
    if re.search(r'(?i)(\b|_)(index|key|uuid|guid|pk)(\b|_|$)', col_name) and ur > 0.85:
        score += 20
    
    # Type-based adjustments
    if pd.api.types.is_float_dtype(series):
        try:
            if not np.all(clean % 1 == 0):
                score -= 40  # Fractional floats are very unlikely to be IDs
            else:
                score -= 10  # Integer-valued floats are slightly less likely
        except:
            pass
    elif pd.api.types.is_integer_dtype(series):
        score += 10
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # String columns with very high uniqueness are likely IDs (e.g., "ORD-12345")
        if ur > 0.90:
            score += 15
    
    is_id = score > 50
    reason = f"id_detected (score={score}, ur={ur:.2f})" if is_id else "normal"
    return is_id, reason


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

class RecommendationEngine:
    NUMERIC_METHODS = ['log_transform', 'sqrt_transform', 'quantile_binning',
                       'polynomial_square', 'impute_median', 'missing_indicator']
    CATEGORICAL_METHODS = ['frequency_encoding', 'target_encoding', 'onehot_encoding',
                          'hashing_encoding', 'missing_indicator']
    TEMPORAL_METHODS = ['cyclical_encode', 'impute_median', 'missing_indicator']
    
    def __init__(self, preparator, trainer, gate_trainer=None, ranker_trainer=None,
                 upside_trainer=None, ensemble_weights=None):
        self.preparator = preparator
        self.trainer = trainer
        self.gate_trainer = gate_trainer
        self.ranker_trainer = ranker_trainer
        self.upside_trainer = upside_trainer
        self._has_gate = gate_trainer is not None and gate_trainer.model is not None
        self._has_ranker = ranker_trainer is not None and ranker_trainer.model is not None
        self._has_upside = upside_trainer is not None and getattr(upside_trainer, 'model', None) is not None
        self.ensemble_weights = ensemble_weights or {'regressor': 0.5, 'gate': 0.3, 'upside': 0.2}
        self.method_priors = {}  # Loaded after init from method_priors.json
        self._priors_loaded = False
        # Build family encoding lookup from preparator
        self._family_to_int = {}
        family_classes = getattr(preparator, '_family_encoder_classes', [])
        for i, fam in enumerate(family_classes):
            self._family_to_int[fam] = i
    
    def recommend(self, X, y, task_type='classification', top_k=30,
                  additive_boost=0.0, method_boosts= None):
        fc = FeatureComputer(task_type=task_type)
        ds_meta = fc.compute_dataset_meta(X, y)
        baseline = self._quick_baseline(X, y, task_type, fc)
        ds_meta['baseline_score'] = baseline['score']
        ds_meta['baseline_std'] = baseline['std']
        
        # Phase 2.1: Importance aggregate features
        imp_dict = baseline.get('importances', {})
        if imp_dict:
            imp_vals = np.array(list(imp_dict.values()))
            ds_meta['std_feature_importance'] = float(np.std(imp_vals))
            ds_meta['max_minus_min_importance'] = float(imp_vals.max() - imp_vals.min())
            median_imp = np.median(imp_vals)
            ds_meta['pct_features_above_median_importance'] = float((imp_vals > median_imp).mean())
        
        # Phase 2.5: Headroom
        # Headroom = room for improvement. Both AUC and RÂ² are higher-is-better,
        # so headroom = 1.0 - score for both task types.
        ds_meta['relative_headroom'] = max(1.0 - baseline['score'], 0.001)
        
        if task_type in self.preparator.task_type_encoder.classes_:
            task_encoded = int(self.preparator.task_type_encoder.transform([task_type])[0])
        else:
            task_encoded = 0
        
        # --- ID column detection: skip ID-like columns from all FE ---
        id_columns = set()
        for col in X.columns:
            is_id, reason = _is_likely_id_column(X[col], col)
            if is_id:
                id_columns.add(col)
        if id_columns:
            st.info(f"Ã°Å¸â€Â Detected {len(id_columns)} ID/constant column(s) Ã¢â‚¬â€ excluded from FE: {', '.join(sorted(id_columns))}")
        
        y_num = fc._ensure_numeric(y)
        mi_scores = {}
        for col in X.columns:
            if col in id_columns:
                continue
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
        
        # Single-column candidates (skip ID columns)
        for col in X.columns:
            if col in id_columns:
                continue
            col_meta = fc.compute_column_meta(X[col], y)
            # Use actual importance from baseline model
            col_meta['baseline_feature_importance'] = imp_dict.get(col, 0.0)
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
                # Fix 3: Skip methods empirically proven harmful
                if self._priors_loaded and method in self.method_priors:
                    mp = self.method_priors[method]
                    # Skip if mean score < 46 (consistently harmful) AND enough samples
                    if mp['mean_score'] < 46 and mp['count'] >= 10:
                        continue
                    # Skip pure-noise methods (mean Ã¢â€°Ë† 50, never helpful)
                    if mp['pct_helpful'] < 0.02 and mp['count'] >= 20:
                        continue
                method_encoded = int(self.preparator.method_encoder.transform([method])[0])
                features = {}
                features.update(ds_meta)
                features.update(col_meta)
                features['method_encoded'] = method_encoded
                features['task_type_encoded'] = task_encoded
                features['is_interaction'] = 0
                features['near_ceiling_flag'] = 0
                # Fix 2: Explicit sentinel for "not an interaction"
                # Without this, these are NaN â†’ median-imputed to look like
                # a moderately correlated pair, biasing gate predictions.
                features['pairwise_corr_ab'] = -1.0
                features['pairwise_spearman_ab'] = -1.0
                features['pairwise_mi_ab'] = -1.0
                features['interaction_scale_ratio'] = -1.0
                features['tree_pair_score'] = -1.0
                # v6: sentinel for col_b/col_c (no partner columns)
                for _prefix in ('col_b_', 'col_c_'):
                    for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                 'outlier_ratio', 'baseline_importance', 'entropy',
                                 'composite_predictive_score'):
                        features[f'{_prefix}{_suf}'] = -1.0
                candidates.append({
                    'column': col, 'method': method,
                    'is_interaction': False, 'features': features,
                })
        
        # Interaction candidates (exclude temporal from arithmetic, exclude IDs)
        numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns.tolist() if c not in id_columns]
        cat_cols = [c for c in X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() if c not in id_columns]
        arithmetic_nums = [c for c in numeric_cols if detect_temporal_component(X[c], c)[0] is None]
        top_nums = sorted(arithmetic_nums, key=lambda c: mi_scores.get(c, 0), reverse=True)[:8]
        top_cats = sorted(cat_cols, key=lambda c: mi_scores.get(c, 0), reverse=True)[:5]
        
        for i, col_a in enumerate(top_nums):
            for col_b in top_nums[i+1:]:
                pair_metrics = self._compute_pair_metrics(X, col_a, col_b)
                for method in ['product_interaction', 'division_interaction',
                               'addition_interaction', 'subtraction_interaction', 'abs_diff_interaction']:
                    # Fix 3: Skip empirically harmful interaction methods
                    if self._priors_loaded and method in self.method_priors:
                        mp = self.method_priors[method]
                        if mp['mean_score'] < 46 and mp['count'] >= 10:
                            continue
                    if method in ('addition_interaction', 'subtraction_interaction', 'abs_diff_interaction'):
                        if pair_metrics.get('interaction_scale_ratio', 0) > 10:
                            continue
                    if method in self.preparator.method_encoder.classes_:
                        col_meta_a = fc.compute_column_meta(X[col_a], y)
                        col_meta_b_raw = fc.compute_column_meta(X[col_b], y)
                        col_meta_b_raw['baseline_feature_importance'] = imp_dict.get(col_b, 0.0)
                        features = {}
                        features.update(ds_meta)
                        features.update(col_meta_a)
                        features['method_encoded'] = int(self.preparator.method_encoder.transform([method])[0])
                        features['task_type_encoded'] = task_encoded
                        features['is_interaction'] = 1
                        features['near_ceiling_flag'] = 0
                        features.update(pair_metrics)
                        # v6: col_b partner metadata
                        features['col_b_is_numeric'] = col_meta_b_raw.get('is_numeric', 0)
                        features['col_b_skewness'] = col_meta_b_raw.get('skewness', 0)
                        features['col_b_unique_ratio'] = col_meta_b_raw.get('unique_ratio', 0)
                        features['col_b_null_pct'] = col_meta_b_raw.get('null_pct', 0)
                        features['col_b_outlier_ratio'] = col_meta_b_raw.get('outlier_ratio', 0)
                        features['col_b_baseline_importance'] = col_meta_b_raw.get('baseline_feature_importance', 0)
                        features['col_b_entropy'] = col_meta_b_raw.get('entropy', 0)
                        features['col_b_composite_predictive_score'] = col_meta_b_raw.get('composite_predictive_score', 0)
                        # No col_c for 2-way interactions
                        for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                     'outlier_ratio', 'baseline_importance', 'entropy',
                                     'composite_predictive_score'):
                            features[f'col_c_{_suf}'] = -1.0
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
        
        # 3-way interactions (top 5 numerics)
        for i, col_a in enumerate(top_nums[:5]):
            for j, col_b in enumerate(top_nums[i+1:6], start=i+1):
                for col_c in top_nums[j+1:6]:
                    pair_metrics_ab = self._compute_pair_metrics(X, col_a, col_b)
                    for method in ['three_way_interaction', 'three_way_addition', 'three_way_ratio']:
                        # Fix 3: Skip empirically harmful 3-way methods
                        if self._priors_loaded and method in self.method_priors:
                            mp = self.method_priors[method]
                            if mp['mean_score'] < 46 and mp['count'] >= 10:
                                continue
                        if method == 'three_way_addition':
                            pair_ac = self._compute_pair_metrics(X, col_a, col_c)
                            pair_bc = self._compute_pair_metrics(X, col_b, col_c)
                            if any(m.get('interaction_scale_ratio', 0) > 10
                                   for m in [pair_metrics_ab, pair_ac, pair_bc]):
                                continue
                        if method in self.preparator.method_encoder.classes_:
                            col_meta_a = fc.compute_column_meta(X[col_a], y)
                            col_meta_b_raw = fc.compute_column_meta(X[col_b], y)
                            col_meta_b_raw['baseline_feature_importance'] = imp_dict.get(col_b, 0.0)
                            col_meta_c_raw = fc.compute_column_meta(X[col_c], y)
                            col_meta_c_raw['baseline_feature_importance'] = imp_dict.get(col_c, 0.0)
                            features = {}
                            features.update(ds_meta)
                            features.update(col_meta_a)
                            features['method_encoded'] = int(self.preparator.method_encoder.transform([method])[0])
                            features['task_type_encoded'] = task_encoded
                            features['is_interaction'] = 1
                            features['near_ceiling_flag'] = 0
                            features.update(pair_metrics_ab)
                            # v6: col_b partner metadata
                            for _src_key, _dst_key in [
                                ('is_numeric', 'col_b_is_numeric'), ('skewness', 'col_b_skewness'),
                                ('unique_ratio', 'col_b_unique_ratio'), ('null_pct', 'col_b_null_pct'),
                                ('outlier_ratio', 'col_b_outlier_ratio'),
                                ('baseline_feature_importance', 'col_b_baseline_importance'),
                                ('entropy', 'col_b_entropy'),
                                ('composite_predictive_score', 'col_b_composite_predictive_score'),
                            ]:
                                features[_dst_key] = col_meta_b_raw.get(_src_key, 0)
                            # v6: col_c partner metadata
                            for _src_key, _dst_key in [
                                ('is_numeric', 'col_c_is_numeric'), ('skewness', 'col_c_skewness'),
                                ('unique_ratio', 'col_c_unique_ratio'), ('null_pct', 'col_c_null_pct'),
                                ('outlier_ratio', 'col_c_outlier_ratio'),
                                ('baseline_feature_importance', 'col_c_baseline_importance'),
                                ('entropy', 'col_c_entropy'),
                                ('composite_predictive_score', 'col_c_composite_predictive_score'),
                            ]:
                                features[_dst_key] = col_meta_c_raw.get(_src_key, 0)
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
        
        # Group-by interactions (top 5 cats Ã— top 5 nums)
        for cat in top_cats[:5]:
            for num in top_nums[:5]:
                if 'group_mean' in self.preparator.method_encoder.classes_:
                    col_meta_cat = fc.compute_column_meta(X[cat], y)
                    col_meta_num_raw = fc.compute_column_meta(X[num], y)
                    col_meta_num_raw['baseline_feature_importance'] = imp_dict.get(num, 0.0)
                    pair_metrics = self._compute_pair_metrics(X, cat, num)
                    features = {}
                    features.update(ds_meta)
                    features.update(col_meta_cat)
                    features['method_encoded'] = int(self.preparator.method_encoder.transform(['group_mean'])[0])
                    features['task_type_encoded'] = task_encoded
                    features['is_interaction'] = 1
                    features['near_ceiling_flag'] = 0
                    features.update(pair_metrics)
                    # v6: col_b = numeric aggregation target
                    features['col_b_is_numeric'] = col_meta_num_raw.get('is_numeric', 0)
                    features['col_b_skewness'] = col_meta_num_raw.get('skewness', 0)
                    features['col_b_unique_ratio'] = col_meta_num_raw.get('unique_ratio', 0)
                    features['col_b_null_pct'] = col_meta_num_raw.get('null_pct', 0)
                    features['col_b_outlier_ratio'] = col_meta_num_raw.get('outlier_ratio', 0)
                    features['col_b_baseline_importance'] = col_meta_num_raw.get('baseline_feature_importance', 0)
                    features['col_b_entropy'] = col_meta_num_raw.get('entropy', 0)
                    features['col_b_composite_predictive_score'] = col_meta_num_raw.get('composite_predictive_score', 0)
                    for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                 'outlier_ratio', 'baseline_importance', 'entropy',
                                 'composite_predictive_score'):
                        features[f'col_c_{_suf}'] = -1.0
                    candidates.append({
                        'column': f"group_mean_{num}_by_{cat}",
                        'method': 'group_mean', 'col_a': cat, 'col_b': num,
                        'is_interaction': True, 'features': features,
                    })
                if 'group_std' in self.preparator.method_encoder.classes_:
                    col_meta_cat = fc.compute_column_meta(X[cat], y)
                    col_meta_num_raw = fc.compute_column_meta(X[num], y)
                    col_meta_num_raw['baseline_feature_importance'] = imp_dict.get(num, 0.0)
                    pair_metrics = self._compute_pair_metrics(X, cat, num)
                    features = {}
                    features.update(ds_meta)
                    features.update(col_meta_cat)
                    features['method_encoded'] = int(self.preparator.method_encoder.transform(['group_std'])[0])
                    features['task_type_encoded'] = task_encoded
                    features['is_interaction'] = 1
                    features['near_ceiling_flag'] = 0
                    features.update(pair_metrics)
                    # v6: col_b = numeric aggregation target
                    features['col_b_is_numeric'] = col_meta_num_raw.get('is_numeric', 0)
                    features['col_b_skewness'] = col_meta_num_raw.get('skewness', 0)
                    features['col_b_unique_ratio'] = col_meta_num_raw.get('unique_ratio', 0)
                    features['col_b_null_pct'] = col_meta_num_raw.get('null_pct', 0)
                    features['col_b_outlier_ratio'] = col_meta_num_raw.get('outlier_ratio', 0)
                    features['col_b_baseline_importance'] = col_meta_num_raw.get('baseline_feature_importance', 0)
                    features['col_b_entropy'] = col_meta_num_raw.get('entropy', 0)
                    features['col_b_composite_predictive_score'] = col_meta_num_raw.get('composite_predictive_score', 0)
                    for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                 'outlier_ratio', 'baseline_importance', 'entropy',
                                 'composite_predictive_score'):
                        features[f'col_c_{_suf}'] = -1.0
                    candidates.append({
                        'column': f"group_std_{num}_by_{cat}",
                        'method': 'group_std', 'col_a': cat, 'col_b': num,
                        'is_interaction': True, 'features': features,
                    })
        
        # CatÃ—Cat interactions (cat_concat)
        for i, cat_a in enumerate(top_cats[:5]):
            for cat_b in top_cats[i+1:5]:
                if 'cat_concat' in self.preparator.method_encoder.classes_:
                    # Fix 3: Check method prior before generating candidates
                    if self._priors_loaded and 'cat_concat' in self.method_priors:
                        mp = self.method_priors['cat_concat']
                        if mp['mean_score'] < 46 and mp['count'] >= 10:
                            continue
                    col_meta_a = fc.compute_column_meta(X[cat_a], y)
                    col_meta_b_raw = fc.compute_column_meta(X[cat_b], y)
                    col_meta_b_raw['baseline_feature_importance'] = imp_dict.get(cat_b, 0.0)
                    pair_metrics = self._compute_pair_metrics(X, cat_a, cat_b)
                    features = {}
                    features.update(ds_meta)
                    features.update(col_meta_a)
                    features['method_encoded'] = int(self.preparator.method_encoder.transform(['cat_concat'])[0])
                    features['task_type_encoded'] = task_encoded
                    features['is_interaction'] = 1
                    features['near_ceiling_flag'] = 0
                    features.update(pair_metrics)
                    # v6: col_b partner metadata
                    features['col_b_is_numeric'] = col_meta_b_raw.get('is_numeric', 0)
                    features['col_b_skewness'] = col_meta_b_raw.get('skewness', 0)
                    features['col_b_unique_ratio'] = col_meta_b_raw.get('unique_ratio', 0)
                    features['col_b_null_pct'] = col_meta_b_raw.get('null_pct', 0)
                    features['col_b_outlier_ratio'] = col_meta_b_raw.get('outlier_ratio', 0)
                    features['col_b_baseline_importance'] = col_meta_b_raw.get('baseline_feature_importance', 0)
                    features['col_b_entropy'] = col_meta_b_raw.get('entropy', 0)
                    features['col_b_composite_predictive_score'] = col_meta_b_raw.get('composite_predictive_score', 0)
                    for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                 'outlier_ratio', 'baseline_importance', 'entropy',
                                 'composite_predictive_score'):
                        features[f'col_c_{_suf}'] = -1.0
                    candidates.append({
                        'column': f"{cat_a}_concat_{cat_b}",
                        'method': 'cat_concat', 'col_a': cat_a, 'col_b': cat_b,
                        'is_interaction': True, 'features': features,
                    })
        
        # Row-statistics candidate (global, not tied to a single column)
        if 'row_stats' in self.preparator.method_encoder.classes_:
            row_stats_features = {}
            row_stats_features.update(ds_meta)
            # Column-level features are meaningless for row_stats Ã¢â‚¬â€ fill with zeros/NaN
            # (the preparator's fill_values will median-impute these, same as during training)
            for col_feat in self.preparator.feature_columns:
                if col_feat not in row_stats_features:
                    row_stats_features[col_feat] = np.nan
            row_stats_features['method_encoded'] = int(self.preparator.method_encoder.transform(['row_stats'])[0])
            row_stats_features['task_type_encoded'] = task_encoded
            row_stats_features['is_interaction'] = 0
            row_stats_features['near_ceiling_flag'] = 0
            # Fix 2: Sentinel for non-interaction
            row_stats_features['pairwise_corr_ab'] = -1.0
            row_stats_features['pairwise_spearman_ab'] = -1.0
            row_stats_features['pairwise_mi_ab'] = -1.0
            row_stats_features['interaction_scale_ratio'] = -1.0
            row_stats_features['tree_pair_score'] = -1.0
            # v6: sentinel for col_b/col_c (no partner columns)
            for _prefix in ('col_b_', 'col_c_'):
                for _suf in ('is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                             'outlier_ratio', 'baseline_importance', 'entropy',
                             'composite_predictive_score'):
                    row_stats_features[f'{_prefix}{_suf}'] = -1.0
            candidates.append({
                'column': 'GLOBAL_ROW_STATS', 'method': 'row_stats',
                'is_interaction': False, 'features': row_stats_features,
            })
        
        if not candidates:
            return [], baseline, ds_meta
        
        # Log filtered methods for user awareness
        if self._priors_loaded:
            filtered_methods = []
            for method, mp in self.method_priors.items():
                if mp['mean_score'] < 46 and mp['count'] >= 10:
                    filtered_methods.append(f"{method} (avg={mp['mean_score']:.0f})")
                elif mp['pct_helpful'] < 0.02 and mp['count'] >= 20:
                    filtered_methods.append(f"{method} (helpful={mp['pct_helpful']*100:.0f}%)")
            if filtered_methods:
                self._filtered_methods = filtered_methods
        
        # Enrich candidate features with method family and interactions
        for c in candidates:
            method = c['method']
            # Method family encoding
            family = METHOD_TO_FAMILY.get(method, 'other')
            c['features']['method_family_encoded'] = self._family_to_int.get(family, 0)
            # Meta-feature interactions
            for col_a, col_b in META_FEATURE_INTERACTION_PAIRS:
                ix_name = f'ix_{col_a}_x_{col_b}'
                val_a = c['features'].get(col_a, 0) or 0
                val_b = c['features'].get(col_b, 0) or 0
                c['features'][ix_name] = float(val_a) * float(val_b)
        
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
            # Phase 4.3: Tag impact type and apply additive boost
            c['impact_type'] = METHOD_IMPACT_TYPE.get(c['method'], 'encoding')
            if additive_boost > 0 and c['impact_type'] == 'additive':
                c['predicted_score'] += additive_boost
            
            if method_boosts and c['method'] in method_boosts:
                archetype_boost = method_boosts[c['method']]
                c['predicted_score'] += archetype_boost
                c['archetype_boost'] = archetype_boost  # track for UI
            else:
                c['archetype_boost'] = 0.0

            # Fix 1: Method prior adjustment
            # Blends empirical method performance into the model's prediction.
            # Weight 0.3 = priors nudge but don't dominate the model.
            # This ensures methods like group_mean (54.6) get boosted and
            # product_interaction (48.7) gets penalized, even when the model
            # can't distinguish them based on column features alone.
            if self._priors_loaded and c['method'] in self.method_priors:
                mp = self.method_priors[c['method']]
                prior_adj = mp['prior_adjustment']  # deviation from 50
                c['predicted_score'] += 0.3 * prior_adj
                c['method_prior'] = mp['mean_score']
            else:
                c['method_prior'] = 50.0
        
        # Upside predictions: predict 90th percentile effect (potential best case)
        if self._has_upside:
            upside_effects = self.upside_trainer.predict(X_cand)
            for i, c in enumerate(candidates):
                c['upside_effect'] = float(upside_effects[i])
                # Map effect to score scale for display
                c['upside_score'] = float(np.clip(50 + upside_effects[i] * 10, 0, 100))
                # Blend upside into predicted_score using ensemble weights
                w_up = self.ensemble_weights.get('upside', 0.2)
                if w_up > 0:
                    c['predicted_score'] = (1 - w_up) * c['predicted_score'] + w_up * c['upside_score']
        
        # Phase 1.4: Gate â†’ Ranker pipeline
        if self._has_gate:
            # Step 1: Gate Ã¢â‚¬â€ predict P(helpful) and filter
            gate_probs = self.gate_trainer.predict_proba(X_cand)
            for i, c in enumerate(candidates):
                c['gate_prob'] = float(gate_probs[i])
            # Filter by calibrated threshold
            gate_threshold = self.gate_trainer.threshold
            gated = [c for c in candidates if c['gate_prob'] >= gate_threshold]
            
            # Minimum recommendation guarantee: if the gate is too strict,
            # supplement with top candidates by gate probability.
            # This ensures we always provide enough recommendations to be useful,
            # especially for interactions which may have borderline gate scores.
            min_recs = max(top_k // 2, 5)
            if len(gated) < min_recs:
                # Sort all candidates by gate_prob and take the best ones
                remaining = [c for c in candidates if c['gate_prob'] < gate_threshold]
                remaining.sort(key=lambda x: x['gate_prob'], reverse=True)
                n_needed = min_recs - len(gated)
                gated.extend(remaining[:n_needed])
        else:
            # No gate model Ã¢â‚¬â€ use all candidates, sorted by predicted score
            gated = list(candidates)
            for c in gated:
                c['gate_prob'] = None
        
        if self._has_ranker and len(gated) > 0:
            # Step 2: Rank Ã¢â‚¬â€ predict relative ordering among gated candidates
            gated_rows = []
            for c in gated:
                row = {fc_name: c['features'].get(fc_name, np.nan) for fc_name in self.preparator.feature_columns}
                gated_rows.append(row)
            X_gated = pd.DataFrame(gated_rows, columns=self.preparator.feature_columns)
            for col in X_gated.columns:
                X_gated[col] = pd.to_numeric(X_gated[col], errors='coerce')
            X_gated = X_gated.replace([np.inf, -np.inf], np.nan).fillna(self.preparator.fill_values).fillna(0)
            rank_scores = self.ranker_trainer.predict(X_gated)
            for i, c in enumerate(gated):
                c['rank_score'] = float(rank_scores[i])
            # Sort by ranking score (primary) with gate_prob as tiebreaker
            gated.sort(key=lambda x: (x.get('rank_score', 0), x.get('gate_prob', 0)), reverse=True)
        else:
            gated.sort(key=lambda x: x['predicted_score'], reverse=True)
            for c in gated:
                c['rank_score'] = None
        
        # Phase 0.2: Deduplicate and diversify before stacking
        filtered = deduplicate_and_diversify(gated)
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
        
        # FIX 1: Use StratifiedKFold for classification to ensure classes are distributed
        if task_type == 'classification':
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            split_iter = kf.split(X_p, y_num)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iter = kf.split(X_p)

        scores = []
        last_model = None
        
        for tr, vl in split_iter:
            if task_type == 'classification':
                m = lgb.LGBMClassifier(n_estimators=100, verbosity=-1, random_state=42)
                m.fit(X_p.iloc[tr], y_num.iloc[tr])
                probs = m.predict_proba(X_p.iloc[vl])
                
                try:
                    if probs.shape[1] == 2:
                        scores.append(roc_auc_score(y_num.iloc[vl], probs[:, 1]))
                    else:
                        # FIX 2: Explicitly pass labels to handle missing classes in validation fold
                        scores.append(roc_auc_score(
                            y_num.iloc[vl], 
                            probs, 
                            multi_class='ovr', 
                            labels=list(range(probs.shape[1]))
                        ))
                except ValueError:
                    # Fallback if scoring fails (e.g., only 1 class in validation split)
                    scores.append(0.5)
            else:
                m = lgb.LGBMRegressor(n_estimators=100, verbosity=-1, random_state=42)
                m.fit(X_p.iloc[tr], y_num.iloc[tr])
                scores.append(mean_squared_error(y_num.iloc[vl], m.predict(X_p.iloc[vl])))
            last_model = m

        # Extract feature importances from the last fold's model
        importances = {}
        if last_model is not None and hasattr(last_model, 'feature_importances_'):
            imp = last_model.feature_importances_
            imp_max = imp.max() if imp.max() > 0 else 1.0
            imp_norm = imp / imp_max
            importances = dict(zip(X_p.columns, imp_norm))
            
        return {'score': float(np.mean(scores)), 'std': float(np.std(scores)),
                'importances': importances}
    
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
            # For interactions, 'column' is the output name Ã¢â‚¬â€ use col_a as source
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
        elif method == 'group_std':
            if col_b is None: return None
            cat_str = X[col].astype(str)
            grp_std = X[col_b].groupby(cat_str).std()
            # Fill NaN std (single-member groups) with 0
            spec['grp_map'] = grp_std.fillna(0).to_dict()
            spec['fill_val'] = float(X[col_b].std())
            spec['new_col'] = f"group_std_{col_b}_by_{col}"
        elif method == 'cat_concat':
            if col_b is None: return None
            # Store frequency map for the concatenated categories
            concat_vals = X[col].astype(str) + "_" + X[col_b].astype(str)
            freq_map = concat_vals.value_counts(normalize=True).to_dict()
            spec['freq_map'] = freq_map
            spec['new_col'] = f"{col}_concat_{col_b}"
        elif method == 'text_stats':
            spec['new_col_len'] = f"{col}_len"
            spec['new_col_wc'] = f"{col}_word_count"
            spec['new_col_dc'] = f"{col}_digit_count"
        elif method == 'row_stats':
            # Global row statistics Ã¢â‚¬â€ computed across all numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return None
            spec['numeric_cols'] = numeric_cols
            # Store per-column medians for imputation during transform
            spec['col_medians'] = {c: float(X[c].median()) for c in numeric_cols}
            # Phase 0.3: Determine which stats are actually informative
            active_stats = ['row_mean', 'row_std', 'row_sum', 'row_min', 'row_max', 'row_range']
            # Only include zeros_count if any numeric column actually has zeros
            has_zeros = any((X[c] == 0).any() for c in numeric_cols)
            if has_zeros:
                active_stats.append('row_zeros_count')
            # Only include missing_ratio if dataset actually has missing values
            has_missing = any(X[c].isnull().any() for c in numeric_cols)
            if has_missing:
                active_stats.append('row_missing_ratio')
            spec['active_stats'] = active_stats
        elif method == 'date_extract_basic':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            spec['has_hour'] = bool(dt.dt.hour.notna().any() and (dt.dt.hour != 0).any())
            spec['new_cols'] = [f"{col}_year", f"{col}_month", f"{col}_day",
                                f"{col}_dayofweek", f"{col}_is_weekend"]
            if spec['has_hour']:
                spec['new_cols'].append(f"{col}_hour")
        elif method == 'date_cyclical_month':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            spec['new_col_sin'] = f"{col}_month_sin"
            spec['new_col_cos'] = f"{col}_month_cos"
        elif method == 'date_cyclical_dow':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            spec['new_col_sin'] = f"{col}_dow_sin"
            spec['new_col_cos'] = f"{col}_dow_cos"
        elif method == 'date_cyclical_hour':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            if not (dt.dt.hour.notna().any() and (dt.dt.hour != 0).any()):
                return None
            spec['new_col_sin'] = f"{col}_hour_sin"
            spec['new_col_cos'] = f"{col}_hour_cos"
        elif method == 'date_cyclical_day':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            spec['new_col_sin'] = f"{col}_day_sin"
            spec['new_col_cos'] = f"{col}_day_cos"
        elif method == 'date_elapsed_days':
            dt = pd.to_datetime(X[col], errors='coerce')
            if dt.notna().sum() < 0.5 * len(dt):
                return None
            min_date = dt.min()
            if pd.isna(min_date):
                return None
            spec['min_date'] = min_date
            spec['new_col'] = f"{col}_days_elapsed"
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
        elif method == 'date_extract_basic':
            dt = pd.to_datetime(X[col], errors='coerce')
            X[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
            X[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
            X[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
            X[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
            X[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            if spec.get('has_hour'):
                X[f"{col}_hour"] = dt.dt.hour.fillna(-1).astype(int)
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'date_cyclical_month':
            dt = pd.to_datetime(X[col], errors='coerce')
            m = dt.dt.month.fillna(0)
            X[spec['new_col_sin']] = np.sin(2 * np.pi * m / 12)
            X[spec['new_col_cos']] = np.cos(2 * np.pi * m / 12)
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'date_cyclical_dow':
            dt = pd.to_datetime(X[col], errors='coerce')
            d = dt.dt.dayofweek.fillna(0)
            X[spec['new_col_sin']] = np.sin(2 * np.pi * d / 7)
            X[spec['new_col_cos']] = np.cos(2 * np.pi * d / 7)
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'date_cyclical_hour':
            dt = pd.to_datetime(X[col], errors='coerce')
            h = dt.dt.hour.fillna(0)
            X[spec['new_col_sin']] = np.sin(2 * np.pi * h / 24)
            X[spec['new_col_cos']] = np.cos(2 * np.pi * h / 24)
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'date_cyclical_day':
            dt = pd.to_datetime(X[col], errors='coerce')
            d = dt.dt.day.fillna(0)
            X[spec['new_col_sin']] = np.sin(2 * np.pi * d / 31)
            X[spec['new_col_cos']] = np.cos(2 * np.pi * d / 31)
            X.drop(columns=[col], inplace=True, errors='ignore')
        elif method == 'date_elapsed_days':
            dt = pd.to_datetime(X[col], errors='coerce')
            min_date = spec['min_date']
            X[spec['new_col']] = (dt - min_date).dt.total_seconds().fillna(-86400) / 86400
            X.drop(columns=[col], inplace=True, errors='ignore')
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
        elif method == 'group_std':
            X[spec['new_col']] = X[col].astype(str).map(spec['grp_map']).fillna(spec['fill_val']).astype(float)
        elif method == 'cat_concat':
            concat_vals = X[col].astype(str) + "_" + X[spec['col_b']].astype(str)
            X[spec['new_col']] = concat_vals.map(spec['freq_map']).fillna(0).astype(float)
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
            # Phase 0.3: Only compute stats that were deemed informative at fit time
            active_stats = spec.get('active_stats', 
                ['row_mean', 'row_std', 'row_sum', 'row_min', 'row_max', 'row_range', 'row_zeros_count', 'row_missing_ratio'])
            if 'row_mean' in active_stats:
                X['row_mean'] = num_df.mean(axis=1)
            if 'row_std' in active_stats:
                X['row_std'] = num_df.std(axis=1).fillna(0)
            if 'row_sum' in active_stats:
                X['row_sum'] = num_df.sum(axis=1)
            if 'row_min' in active_stats:
                X['row_min'] = num_df.min(axis=1)
            if 'row_max' in active_stats:
                X['row_max'] = num_df.max(axis=1)
            if 'row_range' in active_stats:
                X['row_range'] = num_df.max(axis=1) - num_df.min(axis=1)
            if 'row_zeros_count' in active_stats:
                X['row_zeros_count'] = (num_df == 0).sum(axis=1)
            if 'row_missing_ratio' in active_stats:
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
    st.set_page_config(page_title="Auto Feature Engineering", layout="wide", page_icon="âš™ï¸Â")
    st.title("âš™ï¸Â Auto Feature Engineering Pipeline")
    st.caption("Upload data â†’ Get recommendations â†’ Compare Baseline vs. Enhanced model")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("âš™ï¸Â Settings")
        model_dir = st.text_input("Meta-model directory", value=r"FE_PP_MetaModel\meta_model")
        task_type = st.selectbox("Task type", ['classification', 'regression'])

        pipeline_model_dir = st.text_input(
            "Pipeline meta-model directory", 
            value=r"Pipeline_MetaModel\pipeline_meta_model")

        # Phase 4.1: Adaptive recommendation count
        # Smart default computed from n_cols after data upload, user can override
        smart_k = st.session_state.get('smart_top_k', 20)
        top_k = st.slider("Max recommendations", 5, 50, smart_k,
                           help=f"Smart default: {smart_k} (30% of features, clamped to [5, 30])")
        if 'smart_top_k' in st.session_state:
            st.caption(f"ðŸ’¡ Recommended: {smart_k} transforms for {st.session_state.get('n_upload_cols', '?')} features")
        prioritize_additive = st.checkbox("Prioritize additive transforms",
                                           value=True,
                                           help="Boost score of additive transforms (interactions, row_stats, etc.) by +2 over encoding transforms")
        st.divider()
        st.caption("Meta-model must be trained first via `train_meta_model_3.py`")
    
    # --- Load meta-model ---
    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        st.error(f"Meta-model not found in `{model_dir}`. Run `train_meta_model.py` first.")
        st.code(f"python train_meta_model_3.py --meta-db ./meta_learning_output/meta_learning_db.csv --output-dir {model_dir}")
        return
    
    if 'engine' not in st.session_state:
        from FE_PP_MetaModel.train_meta_model_3 import MetaDBPreparator as TrainPreparator, MetaModelTrainer as TrainTrainer
        from FE_PP_MetaModel.train_meta_model_3 import GateModelTrainer, RankingModelTrainer, UpsideModelTrainer
        prep = TrainPreparator()
        prep.load(os.path.join(model_dir, 'preparator.pkl'))
        trainer = TrainTrainer()
        trainer.load(os.path.join(model_dir, 'meta_model.pkl'))
        
        # Phase 1.4: Load gate + ranker (optional Ã¢â‚¬â€ backward compatible)
        gate_trainer = None
        ranker_trainer = None
        gate_path = os.path.join(model_dir, 'gate_model.pkl')
        ranker_path = os.path.join(model_dir, 'ranker_model.pkl')
        if os.path.exists(gate_path):
            gate_trainer = GateModelTrainer()
            gate_trainer.load(gate_path)
        if os.path.exists(ranker_path):
            ranker_trainer = RankingModelTrainer()
            ranker_trainer.load(ranker_path)
        
        # Load upside model (optional â€” backward compatible)
        upside_trainer = None
        upside_path = os.path.join(model_dir, 'upside_model.pkl')
        if os.path.exists(upside_path):
            upside_trainer = UpsideModelTrainer()
            upside_trainer.load(upside_path)

        # --- Load Pipeline Guidance Models ---
        if 'archetype_guidance' not in st.session_state:
            guidance = load_pipeline_guidance_models(pipeline_model_dir)
            st.session_state['archetype_guidance'] = guidance
            st.session_state['pipeline_guidance_loaded'] = guidance is not None

        if st.session_state.get('pipeline_guidance_loaded'):
            st.sidebar.success("✓ Pipeline guidance loaded")
        else:
            st.sidebar.info("ℹ Pipeline guidance not available (optional)")
        
        # Load ensemble weights (optional â€” falls back to defaults)
        ensemble_weights = None
        ew_path = os.path.join(model_dir, 'ensemble_weights.json')
        if os.path.exists(ew_path):
            with open(ew_path) as f:
                ensemble_weights = json.load(f)
        
        st.session_state['engine'] = RecommendationEngine(
            prep, trainer, gate_trainer, ranker_trainer,
            upside_trainer=upside_trainer, ensemble_weights=ensemble_weights)
        st.session_state['preparator'] = prep
        st.session_state['trainer'] = trainer
        
        # Phase 5: Load HP meta-models (optional -- falls back to rule-based archetypes)
        hp_search_dirs = [
            os.path.join(model_dir, '..', 'hp_meta_model'),
            os.path.join(model_dir, 'hp_meta_model'),
            './hp_meta_model', r'Hyperparameter_MetaModel\hp_meta_model'
        ]
        hp_dir_found = None
        for d in hp_search_dirs:
            if os.path.exists(os.path.join(d, 'hp_predictor.pkl')):
                hp_dir_found = d
                break
        
        if hp_dir_found:
            try:
                from Hyperparameter_MetaModel.train_hp_meta_model import (
                    HPDataPreparator, DirectHPPredictor, HPScorerTrainer, HPRankerTrainer
                )
                hp_prep = HPDataPreparator()
                hp_prep.load(os.path.join(hp_dir_found, 'hp_preparator.pkl'))
                hp_predictor = DirectHPPredictor()
                hp_predictor.load(os.path.join(hp_dir_found, 'hp_predictor.pkl'))
                
                hp_scorer = None
                hp_scorer_path = os.path.join(hp_dir_found, 'hp_scorer.pkl')
                if os.path.exists(hp_scorer_path):
                    hp_scorer = HPScorerTrainer()
                    hp_scorer.load(hp_scorer_path)
                
                hp_ranker = None
                hp_ranker_path = os.path.join(hp_dir_found, 'hp_ranker.pkl')
                if os.path.exists(hp_ranker_path):
                    hp_ranker = HPRankerTrainer()
                    hp_ranker.load(hp_ranker_path)
                
                hp_archetypes = None
                archetype_json_path = os.path.join(hp_dir_found, 'hp_archetypes.json')
                if os.path.exists(archetype_json_path):
                    with open(archetype_json_path) as f:
                        hp_archetypes = json.load(f)
                
                st.session_state['hp_preparator'] = hp_prep
                st.session_state['hp_predictor'] = hp_predictor
                st.session_state['hp_scorer'] = hp_scorer
                st.session_state['hp_ranker'] = hp_ranker
                st.session_state['hp_archetypes'] = hp_archetypes
                st.session_state['hp_models_loaded'] = True
                
                hp_parts = [f"Predictor ({len(hp_predictor.models)} params)"]
                if hp_scorer and hp_scorer.model is not None:
                    r2 = hp_scorer.cv_scores.get('r2_mean', 0) if hp_scorer.cv_scores else 0
                    hp_parts.append(f"Scorer RÂ²={r2:.3f}")
                if hp_ranker and hp_ranker.model is not None:
                    hp_parts.append("Ranker âœ“")
                if hp_archetypes:
                    hp_parts.append(f"{len(hp_archetypes)} archetypes")
                st.sidebar.success(f"HP meta-model loaded ({' | '.join(hp_parts)})")
            except Exception as e:
                st.session_state['hp_models_loaded'] = False
                st.sidebar.warning(f"HP meta-model load failed: {e}")
        else:
            st.session_state['hp_models_loaded'] = False
        
        # Load method priors (empirical performance from training data)
        priors_path = os.path.join(model_dir, 'method_priors.json')
        if os.path.exists(priors_path):
            with open(priors_path) as f:
                method_priors = json.load(f)
            st.session_state['engine'].method_priors = method_priors
            n_blacklisted = sum(1 for v in method_priors.values() if v['mean_score'] < 45)
            st.session_state['engine']._priors_loaded = True
        else:
            st.session_state['engine'].method_priors = {}
            st.session_state['engine']._priors_loaded = False
        
        status_parts = []
        if trainer.cv_scores:
            status_parts.append(f"Regressor MAE: {trainer.cv_scores['mae_mean']:.2f}")
        if gate_trainer and gate_trainer.cv_scores:
            auc = gate_trainer.cv_scores.get('auc_mean', gate_trainer.cv_scores.get('f1_mean', 0))
            status_parts.append(f"Gate AUC: {auc:.2f}")
        if ranker_trainer and ranker_trainer.model is not None:
            status_parts.append("Ranker âœ“")
        if status_parts:
            st.sidebar.success(f"FE meta-model loaded ({' | '.join(status_parts)})")
    
    engine = st.session_state['engine']
    
    # =====================================================================
    # STEP 1: UPLOAD
    # =====================================================================
    st.header("ðŸ“‚ Step 1: Upload Training Data")
    train_file = st.file_uploader("Drop your training CSV here", type=['csv'], key='train')
    if train_file is None:
        st.info("Upload a CSV file to get started.")
        return
    
    df_train = pd.read_csv(train_file)
    # Phase 4.1: Compute adaptive top-k based on feature count
    n_feature_cols = df_train.shape[1] - 1  # minus target
    smart_k = int(min(max(n_feature_cols * 0.3, 5), 30))
    if st.session_state.get('smart_top_k') != smart_k:
        st.session_state['smart_top_k'] = smart_k
        st.session_state['n_upload_cols'] = n_feature_cols
    c1, c2 = st.columns(2)
    c1.write(f"**Shape:** {df_train.shape[0]:,} rows Ã— {df_train.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df_train.head(20), use_container_width=True)
    target_col = c2.selectbox("Target column", df_train.columns.tolist())
    
    # =====================================================================
    # STEP 2: GENERATE RECOMMENDATIONS
    # =====================================================================
    if st.button("Ã°Å¸â€Â Analyze Dataset & Generate Recommendations", type="primary"):
        y_train = df_train[target_col]
        X_train = df_train.drop(columns=[target_col])
        label_enc = None
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_train):
            label_enc = LabelEncoder()
            y_train = pd.Series(label_enc.fit_transform(y_train.astype(str)),
                               index=y_train.index, name=y_train.name)
        
            

        with st.spinner("Analyzing dataset and generating recommendations..."):
            boost = 2.0 if prioritize_additive else 0.0
            
            # --- Archetype Guidance ---
            method_boosts = None
            archetype_guidance_result = None
            
            if st.session_state.get('pipeline_guidance_loaded'):
                guidance = st.session_state['archetype_guidance']
                # We need ds_meta first — compute it via the engine's FeatureComputer
                # Actually, recommend() computes ds_meta internally and returns it.
                # So we do a two-pass approach: first get ds_meta, then compute guidance,
                # then pass boosts to recommend.
                # 
                # Simpler approach: compute guidance from a quick ds_meta first,
                # then pass the boosts to recommend().
                fc_temp = FeatureComputer(task_type=task_type)
                ds_meta_temp = fc_temp.compute_dataset_meta(X_train, y_train)
                
                # Quick baseline for headroom estimate
                ds_meta_temp.setdefault('baseline_score', 0.5)
                ds_meta_temp.setdefault('baseline_std', 0.05)
                ds_meta_temp.setdefault('relative_headroom', 0.5)
                
                try:
                    archetype_guidance_result = guidance.compute_guidance(
                        ds_meta_temp, task_type)
                    method_boosts = archetype_guidance_result['method_boosts']
                except Exception as e:
                    st.warning(f"Archetype guidance failed (non-critical): {e}")
                    archetype_guidance_result = None
            
            recs, baseline, ds_meta = engine.recommend(
                X_train, y_train, task_type=task_type,
                top_k=top_k,
                additive_boost=boost,
                method_boosts=method_boosts)
            
            # Update guidance with actual baseline if we got it
            if archetype_guidance_result and 'baseline_score' in ds_meta:
                ds_meta_temp['baseline_score'] = ds_meta['baseline_score']
                ds_meta_temp['baseline_std'] = ds_meta.get('baseline_std', 0.05)
                ds_meta_temp['relative_headroom'] = ds_meta.get('relative_headroom', 0.5)
                try:
                    archetype_guidance_result = guidance.compute_guidance(
                        ds_meta_temp, task_type)
                except:
                    pass  # Keep the initial guidance
            
            st.session_state['archetype_guidance_result'] = archetype_guidance_result

        n_classes = int(y_train.nunique()) if task_type == 'classification' else 2
        
        # Phase 5: Use HP meta-model if available, else fall back to rule-based archetypes
        if st.session_state.get('hp_models_loaded', False):
            suggested_params, hp_explanation = compute_lgbm_params_meta(
                ds_meta=ds_meta, task_type=task_type, n_classes=n_classes,
                class_imbalance_ratio=ds_meta.get('class_imbalance_ratio', 1.0),
                hp_preparator=st.session_state['hp_preparator'],
                hp_predictor=st.session_state['hp_predictor'],
                hp_scorer=st.session_state.get('hp_scorer'),
                hp_ranker=st.session_state.get('hp_ranker'),
                n_candidates=30)
            hp_explanation['hp_source'] = 'meta-model'
        else:
            suggested_params, hp_explanation = compute_lgbm_params(
                n_rows=X_train.shape[0], n_cols=X_train.shape[1],
                task_type=task_type, n_classes=n_classes,
                missing_ratio=ds_meta.get('missing_ratio', 0),
                cat_ratio=ds_meta.get('cat_ratio', 0),
                avg_feature_corr=ds_meta.get('avg_feature_corr', 0),
                class_imbalance_ratio=ds_meta.get('class_imbalance_ratio', 1.0))
            hp_explanation['hp_source'] = 'rule-based'
        
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
    # STEP 2b: ARCHETYPE STRATEGY (from pipeline meta-model)
    # =====================================================================
    archetype_result = st.session_state.get('archetype_guidance_result')
    if archetype_result:
        
        st.header("🧭 FE Strategy Recommendation")
        
        expl = archetype_result['explanation']
        emoji = expl.get('top_archetype_emoji', '🔧')
        
        # Main recommendation
        fe_sens = archetype_result['fe_sensitivity']
        if not archetype_result['fe_worth_trying']:
            st.warning(
                f"⚠️ **Low FE sensitivity** (predicted max Δ: {fe_sens:.4f}). "
                f"Feature engineering may have limited impact on this dataset. "
                f"Consider focusing on hyperparameter tuning instead."
            )
        else:
            st.success(
                f"{emoji} **Recommended strategy: {expl['top_archetype']}** "
                f"({expl['top_archetype_prob']:.0%} confidence)\n\n"
                f"*{expl['top_archetype_desc']}*  •  "
                f"Predicted FE sensitivity: {fe_sens:.4f}  •  "
                f"{expl['confidence_note']}"
            )
        
        # Archetype probability breakdown
        with st.expander("📊 Strategy details", expanded=False):
            col_strategy, col_boosts = st.columns(2)
            
            with col_strategy:
                st.write("**Archetype Probabilities:**")
                for arch_name, arch_prob in expl['archetype_ranking']:
                    arch_emoji = expl['archetype_emoji'].get(arch_name, '')
                    desc = expl['archetype_descriptions'].get(arch_name, '')
                    bar = "█" * int(arch_prob * 20)
                    st.write(f"{arch_emoji} **{arch_name}**: {arch_prob:.0%} {bar}")
                    if desc:
                        st.caption(f"   {desc}")
            
            with col_boosts:
                st.write("**Top Method Boosts:**")
                for method, boost in expl['top_boosted_methods']:
                    if boost > 0:
                        st.write(f"  `{method}`: +{boost:.2f} pts")
                
                # Show profile stats if available
                imp_rate = expl.get('archetype_improvement_rate')
                avg_delta = expl.get('archetype_avg_delta')
                if imp_rate is not None:
                    st.write(f"\n**Historical performance:**")
                    st.write(f"  Improvement rate: {imp_rate:.1%}")
                    if avg_delta is not None:
                        st.write(f"  Avg delta: {avg_delta:+.5f}")



    # =====================================================================
    # STEP 3: REVIEW RECOMMENDATIONS (checkboxes)
    # =====================================================================
    st.header("ðŸ“‹ Step 2: Review Recommendations")
    quick_metric = "AUC" if task_type == 'classification' else "MSE"
    st.write(f"**Quick baseline ({quick_metric}):** {baseline['score']:.5f} Ã‚Â± {baseline['std']:.5f}")
    
    if not recs:
        st.warning("No recommendations generated. The dataset may be too small or already well-optimized.")
    else:
        st.write(f"**{len(recs)} recommendations found.** Deselect any you want to skip:")
        
        # Show which methods were auto-filtered based on empirical data
        filtered = getattr(engine, '_filtered_methods', [])
        if filtered:
            with st.expander(f"ðŸš« {len(filtered)} method(s) auto-excluded (empirically harmful)"):
                for fm in filtered:
                    st.write(f"- {fm}")
        
        if 'rec_selection' not in st.session_state or len(st.session_state.get('rec_selection', [])) != len(recs):
            st.session_state['rec_selection'] = [True] * len(recs)
        
        # Phase 1.4: Detect if gate/ranker data is available
        has_gate_data = len(recs) > 0 and recs[0].get('gate_prob') is not None
        has_rank_data = len(recs) > 0 and recs[0].get('rank_score') is not None
        has_prior_data = len(recs) > 0 and recs[0].get('method_prior', 50.0) != 50.0
        
        # Header row Ã¢â‚¬â€ adaptive columns based on available data
        if has_gate_data and has_rank_data:
            widths = [0.4, 0.6, 0.6, 0.6, 0.8, 1.8, 2.5, 1.0]
            hdr = st.columns(widths)
            hdr[0].write("**Use**"); hdr[1].write("**Gate P**"); hdr[2].write("**Rank**")
            hdr[3].write("**Prior**"); hdr[4].write("**Impact**"); hdr[5].write("**Method**")
            hdr[6].write("**Column**"); hdr[7].write("**Type**")
        elif has_gate_data:
            widths = [0.4, 0.6, 0.6, 0.6, 0.8, 1.8, 2.7, 1.0]
            hdr = st.columns(widths)
            hdr[0].write("**Use**"); hdr[1].write("**Gate P**"); hdr[2].write("**Score**")
            hdr[3].write("**Prior**"); hdr[4].write("**Impact**"); hdr[5].write("**Method**")
            hdr[6].write("**Column**"); hdr[7].write("**Type**")
        else:
            widths = [0.5, 1, 1, 2.5, 3, 1.5]
            hdr = st.columns(widths)
            hdr[0].write("**Use**"); hdr[1].write("**Score**"); hdr[2].write("**Impact**")
            hdr[3].write("**Method**"); hdr[4].write("**Column**"); hdr[5].write("**Type**")
        
        for i, rec in enumerate(recs):
            cols = st.columns(widths)
            st.session_state['rec_selection'][i] = cols[0].checkbox(
                f"r{i}", value=st.session_state['rec_selection'][i],
                label_visibility="collapsed", key=f"rec_check_{i}")
            
            # Phase 4.3: Impact type
            impact = rec.get('impact_type', METHOD_IMPACT_TYPE.get(rec['method'], 'encoding'))
            emoji = IMPACT_TYPE_EMOJI.get(impact, '')
            
            if has_gate_data and has_rank_data:
                cols[1].write(f"**{rec.get('gate_prob', 0):.2f}**")
                cols[2].write(f"{rec.get('rank_score', 0):.1f}")
                # Prior: color-code based on empirical method performance
                prior = rec.get('method_prior', 50.0)
                if prior >= 53:
                    cols[3].write(f"ðŸŸ¢ {prior:.0f}")
                elif prior >= 49:
                    cols[3].write(f"ðŸŸ¡ {prior:.0f}")
                else:
                    cols[3].write(f"ðŸ”´ {prior:.0f}")
                cols[4].write(f"{emoji} {impact}")
                method_col, column_col, type_col = cols[5], cols[6], cols[7]
            elif has_gate_data:
                cols[1].write(f"**{rec.get('gate_prob', 0):.2f}**")
                cols[2].write(f"{rec['predicted_score']:.1f}")
                prior = rec.get('method_prior', 50.0)
                if prior >= 53:
                    cols[3].write(f"ðŸŸ¢ {prior:.0f}")
                elif prior >= 49:
                    cols[3].write(f"ðŸŸ¡ {prior:.0f}")
                else:
                    cols[3].write(f"ðŸ”´ {prior:.0f}")
                cols[4].write(f"{emoji} {impact}")
                method_col, column_col, type_col = cols[5], cols[6], cols[7]
            else:
                cols[1].write(f"**{rec['predicted_score']:.1f}**")
                cols[2].write(f"{emoji} {impact}")
                method_col, column_col, type_col = cols[3], cols[4], cols[5]
            
            method_col.write(rec['method'])
            # Show descriptive column info
            if rec['method'] == 'row_stats':
                column_col.write("All numeric columns (row-level features)")
                type_col.write("ðŸŒÂ Global")
            elif rec.get('is_interaction'):
                col_a = rec.get('col_a', '')
                col_b = rec.get('col_b', '')
                if rec['method'] == 'group_mean':
                    column_col.write(f"mean(`{col_b}`) by `{col_a}`")
                else:
                    column_col.write(rec['column'])
                type_col.write("ðŸ”— Inter.")
            else:
                column_col.write(rec['column'])
                type_col.write("ðŸ“Š Single")
    
    # =====================================================================
    # STEP 3b: MANUAL TRANSFORMS (user domain knowledge)
    # =====================================================================
    st.divider()
    with st.expander("âž• Add Manual Transforms (optional Ã¢â‚¬â€ use domain knowledge)", expanded=False):
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
                preview_text += f"\n\n  Sample values: {sample.tolist()} â†’ log1p(x + offset)"
            elif m_method == 'polynomial_square':
                preview_text += f"\n\n  Creates: `{m_col}_squared` = xÃ‚Â²"
            elif m_method == 'cyclical_encode':
                temporal_type, period = detect_temporal_component(X_train[m_col], m_col)
                if temporal_type:
                    preview_text += f"\n\n  Detected: **{temporal_type}** (period={period}) â†’ sin/cos encoding"
                else:
                    preview_text += "\n\n  âš ï¸Â No temporal pattern detected Ã¢â‚¬â€ cyclical encoding may not be useful"
            elif m_method == 'missing_indicator':
                miss_pct = X_train[m_col].isnull().mean() * 100
                preview_text += f"\n\n  Creates: `{m_col}_is_na` (missing: {miss_pct:.1f}%)"
            elif m_method in ('frequency_encoding', 'target_encoding'):
                preview_text += f"\n\n  Replaces categories with {'frequency ratios' if m_method == 'frequency_encoding' else 'smoothed target means'}"
            preview_col.info(preview_text)
            
            if st.button("âž• Add transform", key="add_manual"):
                new_rec = {
                    'column': m_col, 'method': m_method,
                    'is_interaction': False, 'predicted_score': -1,
                    'manual': True,
                }
                # Phase 0.2: Check for duplicate manual transform
                existing = st.session_state['manual_transforms']
                is_dup = any(e.get('method') == m_method and e.get('column') == m_col 
                             for e in existing)
                if is_dup:
                    st.warning(f"âš ï¸Â `{m_method}` on `{m_col}` already exists in manual transforms.")
                else:
                    st.session_state['manual_transforms'].append(new_rec)
                    st.rerun()
        
        elif transform_type == "Interaction (2 columns)":
            # Choose interaction sub-type first to constrain column selection
            interaction_kind = st.radio("Interaction kind",
                                         ["Numeric Ã— Numeric (arithmetic)", "Categorical â†’ Numeric (group-by)"],
                                         horizontal=True, key="int_kind")
            
            if interaction_kind == "Numeric Ã— Numeric (arithmetic)":
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for arithmetic interactions.")
                else:
                    mc1, mc2, mc3 = st.columns(3)
                    m_col_a = mc1.selectbox("Column A (numeric)", numeric_cols, key="manual_col_a")
                    remaining = [c for c in numeric_cols if c != m_col_a]
                    m_col_b = mc2.selectbox("Column B (numeric)", remaining, key="manual_col_b")
                    m_method = mc3.selectbox("Operation", NUMERIC_INTERACTION_METHODS, key="manual_int_method")
                    
                    # Preview
                    op_symbols = {'product_interaction': 'Ã—', 'division_interaction': 'Ã·',
                                  'addition_interaction': '+', 'subtraction_interaction': 'âˆ’',
                                  'abs_diff_interaction': '|Aâˆ’B|'}
                    name_map = {
                        'product_interaction': f"{m_col_a}_x_{m_col_b}",
                        'division_interaction': f"{m_col_a}_div_{m_col_b}",
                        'addition_interaction': f"{m_col_a}_plus_{m_col_b}",
                        'subtraction_interaction': f"{m_col_a}_minus_{m_col_b}",
                        'abs_diff_interaction': f"{m_col_a}_absdiff_{m_col_b}",
                    }
                    st.info(f"**Preview:** `{name_map[m_method]}` = `{m_col_a}` {op_symbols[m_method]} `{m_col_b}`")
                    
                    if st.button("âž• Add interaction", key="add_manual_int"):
                        new_rec = {
                            'column': name_map[m_method],
                            'method': m_method,
                            'col_a': m_col_a, 'col_b': m_col_b,
                            'is_interaction': True, 'predicted_score': -1,
                            'manual': True,
                        }
                        # Phase 0.2: Check for duplicate interaction
                        existing = st.session_state['manual_transforms']
                        is_dup = any(e.get('method') == m_method and 
                                     set([e.get('col_a',''), e.get('col_b','')]) == set([m_col_a, m_col_b])
                                     for e in existing)
                        if is_dup:
                            st.warning(f"âš ï¸Â `{m_method}` on `{m_col_a}` Ã— `{m_col_b}` already exists.")
                        else:
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
                    
                    if st.button("âž• Add group-by", key="add_manual_grp"):
                        new_rec = {
                            'column': f"group_mean_{m_num}_by_{m_cat}",
                            'method': m_agg,
                            'col_a': m_cat, 'col_b': m_num,
                            'is_interaction': True, 'predicted_score': -1,
                            'manual': True,
                        }
                        # Phase 0.2: Check for duplicate group-by
                        existing = st.session_state['manual_transforms']
                        is_dup = any(e.get('method') == m_agg and 
                                     e.get('col_a') == m_cat and e.get('col_b') == m_num
                                     for e in existing)
                        if is_dup:
                            st.warning(f"âš ï¸Â `{m_agg}` of `{m_num}` by `{m_cat}` already exists.")
                        else:
                            st.session_state['manual_transforms'].append(new_rec)
                            st.rerun()
        
        else:  # Global (row statistics)
            n_num = len(numeric_cols)
            if n_num < 2:
                st.warning("Need at least 2 numeric columns for row statistics.")
            else:
                st.info(f"**Preview:** Adds row-level features across {n_num} numeric columns:\n\n"
                        f"  Always: `row_mean`, `row_std`, `row_sum`, `row_min`, `row_max`, `row_range`\n\n"
                        f"  Conditional: `row_zeros_count` (if zeros exist), `row_missing_ratio` (if missing values exist)")
                # Check if already added
                already_has = any(mt['method'] == 'row_stats' for mt in st.session_state['manual_transforms'])
                if already_has:
                    st.caption("âœ… Row statistics already added.")
                elif st.button("âž• Add row statistics", key="add_manual_rowstats"):
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
                    mc_disp.write(f"â€¢ ðŸŒÂ `row_stats` Ã¢â‚¬â€ 8 global row-level features")
                elif mt.get('is_interaction'):
                    col_a = mt.get('col_a', '?')
                    col_b = mt.get('col_b', '?')
                    if mt['method'] == 'group_mean':
                        mc_disp.write(f"â€¢ ðŸ”— `group_mean` Ã¢â‚¬â€ mean of `{col_b}` grouped by `{col_a}`")
                    else:
                        mc_disp.write(f"â€¢ ðŸ”— `{mt['method']}` Ã¢â‚¬â€ `{col_a}` â†” `{col_b}`")
                else:
                    mc_disp.write(f"â€¢ ðŸ“Š `{mt['method']}` on `{mt['column']}`")
                if mc_del.button("ðŸ—‘ï¸Â", key=f"del_manual_{idx}"):
                    st.session_state['manual_transforms'].pop(idx)
                    st.rerun()
        else:
            st.caption("No manual transforms added yet.")
    
    # =====================================================================
    # STEP 4: HP CONFIGURATION
    # =====================================================================
    st.divider()
    st.header("ðŸŽ›ï¸Â Step 3: LightGBM Hyperparameters")
    
    suggested_params = st.session_state['suggested_params']
    hp_explanation = st.session_state['hp_explanation']
    
    hp_mode = st.radio("Hyperparameter mode",
                       ["ðŸ¤– Use suggested (data-driven)", "Ã¢Å“ÂÃ¯Â¸Â Custom override"],
                       horizontal=True)
    
    if hp_mode == "ðŸ¤– Use suggested (data-driven)":
        # Show HP source info (meta-model vs rule-based)
        hp_source = hp_explanation.get('hp_source', 'rule-based')
        archetype_info = hp_explanation.get('archetype', '')
        match_reason = hp_explanation.get('match_reason', '')
        
        if hp_source == 'meta-model':
            # Phase 5: Meta-model HP explanation
            strategy = hp_explanation.get('strategy', 'meta-model')
            refinement = hp_explanation.get('refinement', 'none')
            search_result = hp_explanation.get('search_result', '')
            
            st.success(f"**HP Strategy:** {archetype_info}\n\n"
                       f"*{match_reason}*")
            
            with st.expander("HP Meta-Model Details", expanded=False):
                detail_c1, detail_c2 = st.columns(2)
                with detail_c1:
                    st.write("**Direct Prediction (Path A):**")
                    direct_pred = hp_explanation.get('direct_prediction', {})
                    if direct_pred:
                        for k, v in direct_pred.items():
                            st.write(f"- `{k}`: {v}")
                with detail_c2:
                    st.write(f"**Refinement:** {refinement}")
                    n_cands = hp_explanation.get('n_candidates_scored')
                    if n_cands:
                        st.write(f"- Candidates scored: {n_cands}")
                    pred_score = hp_explanation.get('predicted_primary_score')
                    if pred_score is not None:
                        st.write(f"- Predicted primary score: {pred_score:.5f}")
                    if search_result:
                        st.write(f"- {search_result}")
                    pred_r2 = hp_explanation.get('predictor_avg_r2')
                    scorer_r2 = hp_explanation.get('scorer_r2')
                    if pred_r2 is not None:
                        st.write(f"- Predictor avg RÂ²: {pred_r2:.3f}")
                    if scorer_r2 is not None:
                        st.write(f"- Scorer RÂ²: {scorer_r2:.3f}")
                
                # Show learned archetypes if available
                hp_archetypes = st.session_state.get('hp_archetypes')
                if hp_archetypes:
                    st.write("**Learned Archetypes** (data-driven clusters):")
                    arch_rows = []
                    for name, info in hp_archetypes.items():
                        profile = info.get('dataset_profile', {})
                        arch_rows.append({
                            'Archetype': name,
                            'Datasets': info.get('n_datasets', 0),
                            'Avg Rows': f"{profile.get('avg_n_rows', 0):,.0f}",
                            'Avg Cols': f"{profile.get('avg_n_cols', 0):.0f}",
                            'Avg Score': f"{info.get('avg_primary_score', 0):.5f}",
                        })
                    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)
        else:
            # Rule-based archetype (fallback)
            if archetype_info:
                st.info(f"**Matched archetype:** {archetype_info}\n\n*Based on:* {match_reason}")
        
        exp_data = []
        for key in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                     'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'max_bin', 'max_depth']:
            exp_data.append({'Parameter': key, 'Value': suggested_params.get(key, ''),
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
        c9, c10 = st.columns(2)
        custom_reg_lambda = c9.number_input("reg_lambda", 0.0, 10.0, suggested_params['reg_lambda'], 0.05)
        custom_max_bin = c10.number_input("max_bin", 31, 511, suggested_params.get('max_bin', 255), 32)
        active_params = suggested_params.copy()
        active_params.update({
            'n_estimators': custom_n_est, 'learning_rate': custom_lr,
            'num_leaves': custom_leaves, 'max_depth': custom_depth,
            'min_child_samples': custom_min_child, 'subsample': custom_subsample,
            'colsample_bytree': custom_colsample, 'reg_alpha': custom_reg_alpha,
            'reg_lambda': custom_reg_lambda, 'max_bin': custom_max_bin})
    
    # =====================================================================
    # STEP 5: TRAIN THREE MODELS
    # =====================================================================
    st.divider()
    st.header("ðŸš€ Step 4: Train & Compare")
    
    if st.button("ðŸš€ Train Baseline + HP-Only + Enhanced Models", type="primary"):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        n_classes = st.session_state['n_classes']
        selected_recs = [r for r, sel in zip(recs, st.session_state.get('rec_selection', [True]*len(recs))) if sel]
        # Append manual transforms from user domain knowledge
        manual_transforms = st.session_state.get('manual_transforms', [])
        if manual_transforms:
            selected_recs = selected_recs + manual_transforms
        
        # --- MODEL 1: BASELINE Ã¢â‚¬â€ raw data, simple default HPs ---
        progress = st.progress(0, text="Training baseline model (raw data, simple HPs)...")
        X_base = final_encode(X_train.copy())
        # Phase 0.4: Defensible simple baseline
        base_params = {
            'n_estimators': 300,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_bin': 63,
            'min_child_samples': 20,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42, 
            'verbosity': -1, 
            'n_jobs': -1,
        }
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
        
        # --- MODEL 2: HP-ONLY Ã¢â‚¬â€ raw data + smart HPs (isolates HP effect) ---
        progress.progress(33, text="Training HP-only model (raw data + smart HPs)...")
        hp_only_model, hp_only_cv, _ = train_lgbm_cv(X_base, y_train, active_params, task_type)
        st.session_state['hp_only_model'] = hp_only_model
        st.session_state['hp_only_cv'] = hp_only_cv
        st.session_state['hp_only_params'] = active_params
        
        # --- MODEL 3: FULL ENHANCED Ã¢â‚¬â€ auto-FE + smart HPs ---
        progress.progress(66, text="Training enhanced model (auto-FE + smart HPs)...")
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
        hp_only_cv = st.session_state.get('hp_only_cv', baseline_cv)
        enhanced_cv = st.session_state['enhanced_cv']
        metric_name = "AUC" if task_type == 'classification' else "MSE"
        base_mean, base_std = np.mean(baseline_cv), np.std(baseline_cv)
        hp_mean, hp_std = np.mean(hp_only_cv), np.std(hp_only_cv)
        enh_mean, enh_std = np.mean(enhanced_cv), np.std(enhanced_cv)
        
        if task_type == 'classification':
            delta_total = enh_mean - base_mean
            delta_hp = hp_mean - base_mean
            delta_fe = enh_mean - hp_mean
            improved = delta_total > 0
        else:
            delta_total = base_mean - enh_mean
            delta_hp = base_mean - hp_mean
            delta_fe = hp_mean - enh_mean
            improved = delta_total > 0
        
        st.subheader(f"Cross-Validation Results ({metric_name})")
        # Phase 4.2: 3-model comparison
        c1, c2, c3 = st.columns(3)
        c1.metric("ðŸ”µ Baseline", f"{base_mean:.5f} Ã‚Â± {base_std:.5f}")
        c2.metric("ðŸŸ¡ HP-Only", f"{hp_mean:.5f} Ã‚Â± {hp_std:.5f}",
                  delta=f"{hp_mean - base_mean:+.5f}")
        c3.metric("ðŸŸ¢ Full Enhanced", f"{enh_mean:.5f} Ã‚Â± {enh_std:.5f}",
                  delta=f"{enh_mean - base_mean:+.5f}")
        
        # Contribution breakdown
        pct_total = abs(delta_total / base_mean * 100) if base_mean != 0 else 0
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Improvement", f"{pct_total:.2f}%",
                  delta="âœ… Better" if improved else "âš ï¸Â No improvement")
        hp_sign = "+" if (delta_hp > 0 if task_type == 'classification' else delta_hp > 0) else ""
        fe_sign = "+" if (delta_fe > 0 if task_type == 'classification' else delta_fe > 0) else ""
        d2.metric("HP Contribution", f"{hp_sign}{delta_hp:.5f}",
                  delta="HPs vs baseline")
        d3.metric("FE Contribution", f"{fe_sign}{delta_fe:.5f}",
                  delta="FE vs HP-only")
        
        # Clear verdict on FE impact
        if task_type == 'classification':
            fe_helps = delta_fe > 0
        else:
            fe_helps = delta_fe > 0  # lower MSE is better, delta_fe = hp - enh
        
        if not fe_helps:
            st.warning(f"âš ï¸Â **Feature engineering is hurting performance** (FE contribution: {delta_fe:+.5f}). "
                      "The HP-only model outperforms the enhanced model. Consider deselecting "
                      "low-confidence transforms and retraining, or using HP-only predictions.")
        elif abs(delta_fe) < 0.001:
            st.info("â„¹ï¸Â Feature engineering has negligible impact. The improvement comes mainly from hyperparameters.")
        else:
            st.success(f"âœ… Feature engineering is contributing positively (FE contribution: {delta_fe:+.5f}).")
        
        selected_recs = st.session_state.get('selected_recs', [])
        with st.expander(f"ðŸ“‹ {len(selected_recs)} transforms applied + HP comparison"):
            tc1, tc2 = st.columns(2)
            with tc1:
                st.write("**Applied Transforms:**")
                for r in selected_recs:
                    impact = r.get('impact_type', METHOD_IMPACT_TYPE.get(r['method'], 'encoding'))
                    emoji = IMPACT_TYPE_EMOJI.get(impact, '')
                    if r['method'] == 'row_stats':
                        st.write(f"- {emoji} ðŸŒÂ `row_stats` (row-level features)")
                    elif r.get('is_interaction') and r['method'] == 'group_mean':
                        st.write(f"- {emoji} ðŸ”— `group_mean` Ã¢â‚¬â€ mean(`{r.get('col_b')}`) by `{r.get('col_a')}`")
                    elif r.get('is_interaction'):
                        st.write(f"- {emoji} ðŸ”— `{r['method']}` â†’ `{r['column']}`")
                    else:
                        st.write(f"- {emoji} `{r['method']}` on `{r['column']}`")
            with tc2:
                st.write("**ðŸ”µ Baseline HPs:** Simple defaults (n_est=300, lr=0.1, leaves=31, max_bin=63)")
                hp_src = st.session_state.get('hp_explanation', {}).get('hp_source', 'rule-based')
                hp_label = "meta-model predicted" if hp_src == 'meta-model' else "archetype-selected"
                st.write(f"**HP-Only / Enhanced HPs** ({hp_label}):")
                ep = st.session_state.get('enhanced_params', {})
                for k in ['num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples',
                           'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree', 'max_bin']:
                    if k in ep:
                        st.write(f"- `{k}`: {ep[k]}")
                st.caption("HP-Only uses the same smart HPs on raw data. Enhanced adds FE transforms on top.")
        
        # --- FE Feature Impact Analysis ---
        with st.expander("ðŸ”¬ FE Feature Impact Analysis"):
            enhanced_model = st.session_state.get('enhanced_model')
            enhanced_cols = st.session_state.get('enhanced_train_cols', [])
            baseline_cols = st.session_state.get('baseline_train_cols', [])
            selected_recs = st.session_state.get('selected_recs', [])
            
            if enhanced_model and hasattr(enhanced_model, 'feature_importances_'):
                imp = pd.Series(enhanced_model.feature_importances_, index=enhanced_cols)
                imp_pct = imp / imp.sum() * 100 if imp.sum() > 0 else imp
                
                # Separate FE features from original features
                original_set = set(baseline_cols)
                fe_features = {col: imp_pct[col] for col in enhanced_cols if col not in original_set and col in imp_pct.index}
                orig_features_imp = sum(imp_pct[col] for col in enhanced_cols if col in original_set and col in imp_pct.index)
                fe_features_imp = sum(fe_features.values())
                
                c1, c2 = st.columns(2)
                c1.metric("Original features", f"{orig_features_imp:.1f}% importance", 
                         delta=f"{len(original_set)} features")
                c2.metric("FE features", f"{fe_features_imp:.1f}% importance",
                         delta=f"{len(fe_features)} features")
                
                if fe_features:
                    fe_df = pd.DataFrame([
                        {'Feature': feat, 'Importance %': f"{imp:.2f}%", 
                         'Status': 'âœ… Used' if imp > 0.5 else ('âš ï¸Â Low' if imp > 0 else 'Ã¢ÂÅ’ Unused')}
                        for feat, imp in sorted(fe_features.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(fe_df, use_container_width=True, hide_index=True)
                    
                    n_used = sum(1 for v in fe_features.values() if v > 0.5)
                    n_unused = sum(1 for v in fe_features.values() if v <= 0)
                    if n_unused > n_used:
                        st.warning(f"âš ï¸Â {n_unused}/{len(fe_features)} FE features have zero importance. "
                                  f"Consider deselecting unused transforms and retraining.")
                    elif fe_features_imp > 5:
                        st.success(f"âœ… FE features contribute {fe_features_imp:.1f}% of model signal. "
                                  f"{n_used} features actively used.")
            else:
                st.info("Train the enhanced model first to see feature impact analysis.")
        
        # --- Download preprocessed training data ---
        with st.expander("ðŸ“¥ Download Preprocessed Training Data"):
            dl1, dl2 = st.columns(2)
            # Baseline preprocessed
            X_base_dl = final_encode(st.session_state['X_train'].copy())
            base_train_csv = BytesIO()
            X_base_dl.to_csv(base_train_csv, index=False)
            dl1.download_button("ðŸ“¥ Baseline Train (preprocessed)",
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
                dl2.download_button("ðŸ“¥ Enhanced Train (preprocessed)",
                                   enh_train_csv.getvalue(),
                                   "train_enhanced_preprocessed.csv", "text/csv",
                                   key="dl_enh_train")
    
    # =====================================================================
    # STEP 6: TEST SET EVALUATION
    # =====================================================================
    st.divider()
    st.header("ðŸ§ª Step 5: Test Set Evaluation")
    
    if 'baseline_model' not in st.session_state:
        st.info("â¬†ï¸Â Train both models first (Step 4)")
        return
    
    test_file = st.file_uploader("Drop your test CSV here", type=['csv'], key='test')
    if test_file is None:
        st.info("Upload a test CSV to evaluate both models side-by-side.")
        return
    
    df_test = pd.read_csv(test_file)
    st.write(f"**Test shape:** {df_test.shape[0]:,} rows Ã— {df_test.shape[1]} columns")
    target_col = st.session_state.get('target_col')
    has_target = target_col in df_test.columns
    if not has_target:
        st.warning(f"Target column '{target_col}' not found. Predictions only.")
    
    if st.button("ðŸ”® Evaluate All Models on Test Set", type="primary"):
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
        
        # Baseline + HP-only test (same raw features, different HPs)
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
        hp_only_model = st.session_state.get('hp_only_model')
        enhanced_model = st.session_state['enhanced_model']
        
        if has_target and y_test is not None:
            metrics_base = evaluate_on_test(baseline_model, X_test_base, y_test, task_type)
            metrics_hp = evaluate_on_test(hp_only_model, X_test_base, y_test, task_type) if hp_only_model else metrics_base
            metrics_enh = evaluate_on_test(enhanced_model, X_test_enh, y_test, task_type)
            st.session_state['test_metrics_baseline'] = metrics_base
            st.session_state['test_metrics_hp_only'] = metrics_hp
            st.session_state['test_metrics_enhanced'] = metrics_enh
            
            st.subheader("ðŸ“Š Test Set Comparison")
            higher_better = {'AUC', 'Accuracy', 'F1 (macro)', 'Precision (macro)', 'Recall (macro)', 'R2'}
            results = {'Metric': [], 'ðŸ”µ Baseline': [], 'ðŸŸ¡ HP-Only': [], 'ðŸŸ¢ Enhanced': [],
                       'Î” (HP)': [], 'Î” (FE)': [], '': []}
            for metric in metrics_base:
                vb, vh, ve = metrics_base[metric], metrics_hp[metric], metrics_enh[metric]
                if np.isnan(vb) or np.isnan(ve): continue
                diff_hp = vh - vb
                diff_fe = ve - vh
                is_hb = metric in higher_better
                ok = ((ve - vb) > 0) == is_hb
                results['Metric'].append(metric)
                results['ðŸ”µ Baseline'].append(f"{vb:.5f}")
                results['ðŸŸ¡ HP-Only'].append(f"{vh:.5f}")
                results['ðŸŸ¢ Enhanced'].append(f"{ve:.5f}")
                results['Î” (HP)'].append(f"{diff_hp:+.5f}")
                results['Î” (FE)'].append(f"{diff_fe:+.5f}")
                results[''].append("âœ…" if ok else ("âž–" if ve == vb else "âš ï¸Â"))
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            primary = 'AUC' if task_type == 'classification' else 'MSE'
            if primary in metrics_base and primary in metrics_enh:
                diff_total = metrics_enh[primary] - metrics_base[primary]
                diff_hp = metrics_hp[primary] - metrics_base[primary]
                diff_fe = metrics_enh[primary] - metrics_hp[primary]
                st.success(f"**{primary}: {metrics_base[primary]:.5f} â†’ {metrics_enh[primary]:.5f} ({diff_total:+.5f} total | HP: {diff_hp:+.5f} | FE: {diff_fe:+.5f})**")
            
            # Download preprocessed test data
            with st.expander("ðŸ“¥ Download Preprocessed Test Data"):
                tdl1, tdl2 = st.columns(2)
                base_test_csv = BytesIO()
                X_test_base.to_csv(base_test_csv, index=False)
                tdl1.download_button("ðŸ“¥ Baseline Test (preprocessed)",
                                    base_test_csv.getvalue(),
                                    "test_baseline_preprocessed.csv", "text/csv",
                                    key="dl_base_test")
                enh_test_csv = BytesIO()
                X_test_enh.to_csv(enh_test_csv, index=False)
                tdl2.download_button("ðŸ“¥ Enhanced Test (preprocessed)",
                                    enh_test_csv.getvalue(),
                                    "test_enhanced_preprocessed.csv", "text/csv",
                                    key="dl_enh_test")
        else:
            base_preds = baseline_model.predict(X_test_base)
            hp_preds = hp_only_model.predict(X_test_base) if hp_only_model else base_preds
            enh_preds = enhanced_model.predict(X_test_enh)
            pred_df = pd.DataFrame({'baseline_pred': base_preds, 'hp_only_pred': hp_preds, 'enhanced_pred': enh_preds})
            st.dataframe(pred_df.head(20), use_container_width=True)
            csv_buf = BytesIO()
            pred_df.to_csv(csv_buf, index=False)
            st.download_button("ðŸ“¥ Download Predictions", csv_buf.getvalue(), "predictions.csv", "text/csv")
    
    # --- Save pipeline ---
    if 'test_metrics_baseline' in st.session_state:
        st.divider()
        if st.button("ðŸ’¾ Save Full Pipeline"):
            save_path = os.path.join(".", "fitted_pipeline.pkl")
            state = {
                'enhanced_applicator': st.session_state.get('enhanced_applicator'),
                'enhanced_model': st.session_state.get('enhanced_model'),
                'hp_only_model': st.session_state.get('hp_only_model'),
                'baseline_model': st.session_state.get('baseline_model'),
                'enhanced_params': st.session_state.get('enhanced_params'),
                'hp_only_params': st.session_state.get('hp_only_params'),
                'baseline_params': st.session_state.get('baseline_params'),
                'label_enc': st.session_state.get('label_enc'),
                'task_type': task_type,
                'enhanced_train_cols': st.session_state.get('enhanced_train_cols'),
                'baseline_train_cols': st.session_state.get('baseline_train_cols'),
                'selected_recs': st.session_state.get('selected_recs'),
                'test_metrics_baseline': st.session_state.get('test_metrics_baseline'),
                'test_metrics_hp_only': st.session_state.get('test_metrics_hp_only'),
                'test_metrics_enhanced': st.session_state.get('test_metrics_enhanced'),
            }
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            st.success(f"Pipeline saved to `{save_path}`")


if __name__ == "__main__":
    main()