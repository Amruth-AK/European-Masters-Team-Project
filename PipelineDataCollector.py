"""
PipelineDataCollector.py  â€”  Whole-Pipeline FE Evaluation
=========================================================
Evaluates entire FE pipelines (combinations of transforms applied together)
end-to-end, producing a dataset for training a Pipeline Meta-Model.

Separate from DataCollector_3.py which evaluates individual transforms.
Reuses MetaDataCollector as a transform engine for shared functionality.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, ttest_rel
import warnings
import csv
import os
import json
import time
import gc
import random
import traceback
from datetime import datetime
from collections import defaultdict
import math
import sys

warnings.filterwarnings('ignore')

# Define the project root (the '000' folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from FE_PP_MetaModel.DataCollector_3 import (
        MetaDataCollector,
        _detect_openml_task_type,
        _infer_task_type,
        _smart_size_reduction,
    )
except ImportError as e:
    print(f"Import failed. Current sys.path: {sys.path}")
    raise e

# =========================================================================
# CSV SCHEMA  â€”  one row per (dataset, pipeline)
# =========================================================================
PIPELINE_CSV_SCHEMA = [
    # --- Dataset metadata (29 fields) ---
    'n_rows', 'n_cols', 'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'class_imbalance_ratio', 'n_classes', 'target_std', 'target_skew',
    'landmarking_score', 'landmarking_score_norm',
    'avg_feature_corr', 'max_feature_corr', 'avg_target_corr', 'max_target_corr',
    'avg_numeric_sparsity', 'linearity_gap',
    'n_numeric_cols', 'n_cat_cols',
    'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
    'matrix_rank_ratio',
    'std_feature_importance', 'max_minus_min_importance',
    'pct_features_above_median_importance',
    'relative_headroom', 'baseline_score', 'baseline_std',
    # --- Pipeline configuration features (23 fields) ---
    'pipeline_id', 'pipeline_archetype',
    'n_transforms_total', 'n_single_col_transforms',
    'n_interactions_2way', 'n_interactions_3way',
    'n_group_by_transforms', 'n_encoding_transforms',
    'n_log_sqrt_transforms', 'has_row_stats',
    'has_missing_indicators', 'has_polynomial',
    'estimated_features_added', 'feature_expansion_ratio',
    'pct_numeric_cols_touched', 'pct_cat_cols_touched',
    'avg_touched_col_importance', 'max_touched_col_importance',
    'min_touched_col_importance',
    'importance_coverage_top5', 'importance_coverage_top10',
    'n_unique_methods', 'method_diversity_ratio',
    # --- Pipeline evaluation results (8 fields) ---
    'pipeline_score', 'pipeline_std',
    'pipeline_delta', 'pipeline_delta_pct', 'pipeline_improved',
    'pipeline_p_value', 'pipeline_is_significant',
    'pipeline_fold_scores',
    # --- Pipeline content (1 field, JSON) ---
    'pipeline_transforms_json',
    # --- OpenML identifiers (4 fields) ---
    'openml_task_id', 'dataset_name', 'dataset_id', 'task_type',
]

# Transform application order (lower = applied first)
TRANSFORM_ORDER = {
    'impute_median': 10, 'missing_indicator': 10,
    'date_extract_basic': 20, 'date_cyclical_month': 20, 'date_cyclical_dow': 20,
    'date_cyclical_hour': 20, 'date_cyclical_day': 20, 'date_elapsed_days': 20,
    'frequency_encoding': 30, 'target_encoding': 30, 'hashing_encoding': 30,
    'log_transform': 40, 'sqrt_transform': 40, 'quantile_binning': 40,
    'polynomial_square': 50, 'cyclical_encode': 50,
    'onehot_encoding': 55, 'text_stats': 55,
    'product_interaction': 60, 'division_interaction': 60,
    'addition_interaction': 60, 'subtraction_interaction': 60,
    'abs_diff_interaction': 60,
    'group_mean': 60, 'group_std': 60, 'cat_concat': 60,
    'three_way_interaction': 70, 'three_way_addition': 70,
    'three_way_ratio': 70, 'three_way_normalized_diff': 70,
    'row_stats': 80,
}

# How many features each transform adds (0 = in-place modification)
FEATURES_ADDED = {
    'impute_median': 0, 'missing_indicator': 1,
    'log_transform': 0, 'sqrt_transform': 0, 'quantile_binning': 0,
    'frequency_encoding': 0, 'target_encoding': 0, 'hashing_encoding': 0,
    'polynomial_square': 1, 'cyclical_encode': 2,
    'onehot_encoding': -1,  # dynamic: nunique - 1
    'date_extract_basic': 5, 'date_cyclical_month': 2, 'date_cyclical_dow': 2,
    'date_cyclical_hour': 2, 'date_cyclical_day': 2, 'date_elapsed_days': 1,
    'text_stats': 3,
    'product_interaction': 1, 'division_interaction': 1,
    'addition_interaction': 1, 'subtraction_interaction': 1,
    'abs_diff_interaction': 1,
    'group_mean': 1, 'group_std': 1, 'cat_concat': 1,
    'three_way_interaction': 1, 'three_way_addition': 1,
    'three_way_ratio': 1, 'three_way_normalized_diff': 1,
    'row_stats': 10,
}

# Methods that transform a column in-place (only one allowed per column)
IN_PLACE_METHODS = {
    'log_transform', 'sqrt_transform', 'quantile_binning',
    'impute_median', 'frequency_encoding', 'target_encoding', 'hashing_encoding',
}

# Methods that replace/drop the original column
REPLACE_METHODS = {'onehot_encoding', 'date_extract_basic', 'date_cyclical_month',
                   'date_cyclical_dow', 'date_cyclical_hour', 'date_cyclical_day',
                   'date_elapsed_days', 'text_stats'}

INTERACTION_METHODS_2WAY = {
    'product_interaction', 'division_interaction', 'addition_interaction',
    'subtraction_interaction', 'abs_diff_interaction',
    'group_mean', 'group_std', 'cat_concat',
}

INTERACTION_METHODS_3WAY = {
    'three_way_interaction', 'three_way_addition',
    'three_way_ratio', 'three_way_normalized_diff',
}


class PipelineDataCollector:
    """
    Generates and evaluates diverse FE pipelines on a dataset.

    For each dataset:
    1. Data cleanup (ID removal, pruning, leakage detection)
    2. Baseline CV evaluation
    3. Column profiling (types, importance, properties)
    4. Generate 10-15 diverse pipelines from 7 archetypes
    5. Evaluate each pipeline end-to-end with repeated K-fold CV
    6. Output one CSV row per pipeline with dataset + pipeline features + score
    """

    def __init__(self, task_type='classification', n_folds=5, n_repeats=3,
                 output_dir='./pipeline_meta_output', time_budget_seconds=7200,
                 n_jobs=-1):
        self.task_type = task_type
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.output_dir = output_dir
        self.time_budget = time_budget_seconds
        self.n_jobs = n_jobs
        self._start_time = None
        self._results = []

        os.makedirs(output_dir, exist_ok=True)

        # Reuse MetaDataCollector as transform engine
        self._engine = MetaDataCollector(
            task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
            output_dir=output_dir, time_budget_seconds=time_budget_seconds,
            n_jobs=n_jobs)

        self.eval_params = {
            'n_estimators': 150, 'learning_rate': 0.1,
            'num_leaves': 31, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': 42, 'verbosity': -1, 'n_jobs': n_jobs,
        }
        self.log_file = os.path.join(output_dir, 'pipeline_log.txt')

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    def log(self, msg, level='INFO'):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}"
        print(line)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass

    def _time_remaining(self):
        if self._start_time is None:
            return float('inf')
        return self.time_budget - (time.time() - self._start_time)

    def _is_over_budget(self):
        return self._time_remaining() <= 0

    # -----------------------------------------------------------------
    # Dynamic constraints (scale with dataset size)
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_constraints(n_rows, n_cols, n_numeric, n_cat):
        """All limits are relative to dataset dimensions."""
        return {
            'max_features_added': max(3, int(n_cols * (0.5 if n_cols < 30 else 0.3))),
            'max_ohe_columns': min(3, n_cat),
            'max_ohe_cardinality': min(10, max(5, n_cols // 5)),
            'max_ohe_total_dummies': max(10, n_cols // 3),
            'max_interactions': max(2, min(15, n_numeric // 2)),
            'allow_3way': n_numeric >= 6 and n_rows >= 500,
            'max_3way': min(3, max(1, n_numeric // 4)),
            'allow_row_stats': n_numeric >= 4,
            'n_pipelines': 15 if n_rows * n_cols < 5_000_000 else 10,
            'max_single_col_transforms': max(3, min(20, n_cols // 2)),
            'max_encoding_cols': max(2, min(8, n_cat)),
        }

    # -----------------------------------------------------------------
    # Column profiling
    # -----------------------------------------------------------------
    def _profile_columns(self, X, y, importances):
        """Profile each column for pipeline generation.
        Returns dict: col_name -> {type, importance, ...}"""
        profiles = {}
        for col in X.columns:
            is_num = pd.api.types.is_numeric_dtype(X[col])
            is_date = self._engine._is_likely_date(X[col], col)
            temp_type, temp_period = self._engine._detect_temporal_component(X[col], col)

            p = {
                'type': 'date' if is_date else ('temporal' if temp_type else
                        ('numeric' if is_num else 'categorical')),
                'importance': float(importances.get(col, 0)),
                'nunique': int(X[col].nunique()),
                'has_nulls': bool(X[col].isnull().any()),
                'is_positive': bool(is_num and X[col].min() >= 0) if is_num else False,
                'temporal_period': temp_period,
            }
            if is_num:
                vals = X[col].dropna()
                p['skewness'] = float(skew(vals)) if len(vals) > 10 else 0.0
                p['outlier_ratio'] = 0.0
                if len(vals) > 4:
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        p['outlier_ratio'] = float(((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).mean())
            else:
                p['skewness'] = 0.0
                p['outlier_ratio'] = 0.0
            profiles[col] = p
        return profiles

    def _get_applicable_single_transforms(self, col, profile):
        """Return list of applicable single-column methods for a column."""
        t = profile['type']
        methods = []
        if t == 'date':
            methods = ['date_extract_basic']
        elif t == 'temporal':
            methods = ['cyclical_encode']
            if profile['has_nulls']:
                methods.append('impute_median')
        elif t == 'numeric':
            if profile['has_nulls']:
                methods.extend(['impute_median', 'missing_indicator'])
            if abs(profile['skewness']) > 2 and profile['is_positive']:
                methods.append('log_transform')
            if profile['skewness'] > 1.5 and profile['is_positive']:
                methods.append('sqrt_transform')
            methods.append('polynomial_square')
        elif t == 'categorical':
            nu = profile['nunique']
            methods.append('target_encoding')
            methods.append('frequency_encoding')
            if 2 < nu <= 10:
                methods.append('onehot_encoding')
            if nu > 20:
                methods.append('hashing_encoding')
            if profile['has_nulls']:
                methods.append('missing_indicator')
        return methods

    # -----------------------------------------------------------------
    # Pipeline sorting and validation
    # -----------------------------------------------------------------
    @staticmethod
    def _sort_transforms(transforms):
        """Sort transforms by application order for safe sequencing."""
        return sorted(transforms, key=lambda t: TRANSFORM_ORDER.get(t['method'], 50))

    def _estimate_features_added(self, transforms, profiles):
        """Estimate how many new columns a pipeline will add."""
        total = 0
        for t in transforms:
            method = t['method']
            fa = FEATURES_ADDED.get(method, 0)
            if fa == -1:  # OHE: dynamic
                col = t.get('column', '')
                nu = profiles.get(col, {}).get('nunique', 5)
                total += max(0, min(nu - 1, 10))
            else:
                total += fa
        return total

    def _validate_pipeline(self, transforms, profiles, constraints):
        """Remove conflicts: max 1 in-place per col, respect constraints.
        When trimming for feature budget, remove highest-cost transforms first
        rather than blindly popping from the end (which always kills interactions)."""
        seen_inplace = {}
        seen_replace = set()
        valid = []

        for t in transforms:
            col = t.get('column', '')
            method = t['method']

            # Skip if column was replaced by an earlier transform
            if col in seen_replace and method not in ('row_stats',):
                continue

            # Enforce max 1 in-place transform per column
            if method in IN_PLACE_METHODS:
                if col in seen_inplace:
                    continue
                seen_inplace[col] = method

            if method in REPLACE_METHODS:
                seen_replace.add(col)

            valid.append(t)

        # Enforce feature budget: remove highest feature-cost transforms first
        est = self._estimate_features_added(valid, profiles)
        max_feat = constraints['max_features_added']
        while est > max_feat and valid:
            # Find the transform that adds the most features
            worst_idx = -1
            worst_cost = -1
            for i, t in enumerate(valid):
                cost = FEATURES_ADDED.get(t['method'], 0)
                if cost == -1:  # OHE: dynamic
                    col = t.get('column', '')
                    cost = max(0, min(profiles.get(col, {}).get('nunique', 5) - 1, 10))
                if cost > worst_cost:
                    worst_cost = cost
                    worst_idx = i
            if worst_idx >= 0 and worst_cost > 0:
                valid.pop(worst_idx)
            else:
                # All remaining transforms are zero-cost, can't trim further
                break
            est = self._estimate_features_added(valid, profiles)

        return valid

    # -----------------------------------------------------------------
    # Pipeline archetype generators
    # -----------------------------------------------------------------
    def _gen_minimal_surgical(self, profiles, ranked_cols, constraints, rng):
        """Archetype 1: Only top-2 highest-confidence transforms."""
        pipelines = []
        eligible = [c for c in ranked_cols if profiles[c]['type'] in ('numeric', 'categorical')]
        if len(eligible) < 2:
            return pipelines
        for variant in range(2):
            transforms = []
            if variant == 0:
                cols = eligible[:2]
            else:
                # Randomly sample 2 from top-6 for diversity
                pool = eligible[:min(6, len(eligible))]
                cols = rng.sample(pool, min(2, len(pool)))
            for col in cols:
                methods = self._get_applicable_single_transforms(col, profiles[col])
                if methods:
                    m = rng.choice(methods[:2]) if len(methods) > 1 and variant == 1 else methods[0]
                    transforms.append({'method': m, 'column': col})
            if transforms:
                pipelines.append(('minimal_surgical', transforms))
        return pipelines

    def _gen_encoding_focused(self, profiles, ranked_cols, constraints, rng):
        """Archetype 2: Focus on categorical encoding."""
        pipelines = []
        cat_cols = [c for c in ranked_cols if profiles[c]['type'] == 'categorical']
        max_enc = constraints['max_encoding_cols']
        for variant in range(min(2, max(1, len(cat_cols) // 2))):
            transforms = []
            if variant == 0:
                cols = cat_cols[:max_enc]
            else:
                # Shuffle categorical columns and pick a different subset
                pool = cat_cols[:max_enc + 3] if len(cat_cols) > max_enc else cat_cols
                cols = rng.sample(pool, min(max_enc, len(pool)))
            for col in cols:
                nu = profiles[col]['nunique']
                # Variant 1: randomize encoding method choice for mid-cardinality
                if 2 < nu <= 10:
                    if variant == 1 and rng.random() < 0.4:
                        transforms.append({'method': 'target_encoding', 'column': col})
                    else:
                        transforms.append({'method': 'onehot_encoding', 'column': col})
                elif nu <= 50:
                    if variant == 1 and rng.random() < 0.3:
                        transforms.append({'method': 'frequency_encoding', 'column': col})
                    else:
                        transforms.append({'method': 'target_encoding', 'column': col})
                else:
                    if variant == 1 and rng.random() < 0.3:
                        transforms.append({'method': 'hashing_encoding', 'column': col})
                    else:
                        transforms.append({'method': 'frequency_encoding', 'column': col})
                if profiles[col]['has_nulls']:
                    transforms.append({'method': 'missing_indicator', 'column': col})
            if transforms:
                pipelines.append(('encoding_focused', transforms))
        return pipelines

    def _gen_interaction_heavy(self, profiles, ranked_cols, constraints, rng):
        """Archetype 3: Primarily 2-way interactions."""
        pipelines = []
        num_cols = [c for c in ranked_cols if profiles[c]['type'] == 'numeric']
        cat_cols = [c for c in ranked_cols if profiles[c]['type'] == 'categorical']
        max_int = constraints['max_interactions']
        all_int_methods = ['product_interaction', 'division_interaction',
                           'addition_interaction', 'subtraction_interaction',
                           'abs_diff_interaction']

        for variant in range(2):
            transforms = []
            # Numeric x numeric interactions
            pairs_added = 0
            if variant == 0:
                pool = num_cols[:8]
            else:
                pool_size = min(8, len(num_cols))
                pool = rng.sample(num_cols[:min(12, len(num_cols))], pool_size) if pool_size > 0 else []
            for i in range(len(pool)):
                if pairs_added >= max_int:
                    break
                for j in range(i + 1, len(pool)):
                    if pairs_added >= max_int:
                        break
                    m = rng.choice(all_int_methods) if variant == 1 else all_int_methods[pairs_added % 2]
                    transforms.append({
                        'method': m, 'column': pool[i], 'col_b': pool[j]})
                    pairs_added += 1
            # Add 1-2 group_mean if categorical cols available
            if cat_cols and num_cols:
                gc = rng.choice(cat_cols[:3]) if variant == 1 and len(cat_cols) > 1 else cat_cols[0]
                gn = rng.choice(num_cols[:3]) if variant == 1 and len(num_cols) > 1 else num_cols[0]
                transforms.append({
                    'method': 'group_mean', 'column': gc, 'col_b': gn})
            if transforms:
                pipelines.append(('interaction_heavy', transforms))
        return pipelines

    def _gen_single_col_heavy(self, profiles, ranked_cols, constraints, rng):
        """Archetype 4: Many single-column transforms."""
        pipelines = []
        max_sc = constraints['max_single_col_transforms']
        applicable = []
        for col in ranked_cols:
            methods = self._get_applicable_single_transforms(col, profiles[col])
            for m in methods:
                applicable.append((col, m))
        for variant in range(2):
            if variant == 0:
                selected = applicable[:max_sc]
            else:
                # Shuffle and sample for diversity
                pool = applicable[:max_sc * 2] if len(applicable) > max_sc else applicable
                selected = rng.sample(pool, min(max_sc, len(pool)))
            transforms = [{'method': m, 'column': c} for c, m in selected]
            if transforms:
                pipelines.append(('single_col_heavy', transforms))
        return pipelines

    def _gen_balanced_mix(self, profiles, ranked_cols, constraints, rng):
        """Archetype 5: balanced mix of single-col + interactions + row_stats."""
        pipelines = []
        num_cols = [c for c in ranked_cols if profiles[c]['type'] == 'numeric']
        cat_cols = [c for c in ranked_cols if profiles[c]['type'] == 'categorical']
        int_methods = ['product_interaction', 'division_interaction',
                       'addition_interaction', 'subtraction_interaction']

        for variant in range(2):
            transforms = []
            # Single-col: top 2-4 columns
            if variant == 0:
                sc_pool = ranked_cols[:6]
            else:
                sc_pool = rng.sample(ranked_cols[:10], min(6, len(ranked_cols)))
            sc_count = 0
            for col in sc_pool:
                if sc_count >= 4:
                    break
                methods = self._get_applicable_single_transforms(col, profiles[col])
                if methods:
                    m = rng.choice(methods[:2]) if variant == 1 and len(methods) > 1 else methods[0]
                    transforms.append({'method': m, 'column': col})
                    sc_count += 1
            # Interactions: 2-3
            int_count = 0
            int_pool = num_cols[:4] if variant == 0 else rng.sample(
                num_cols[:8], min(4, len(num_cols))) if len(num_cols) > 0 else []
            for i in range(len(int_pool)):
                for j in range(i + 1, len(int_pool)):
                    if int_count >= 3:
                        break
                    m = rng.choice(int_methods) if variant == 1 else 'product_interaction'
                    transforms.append({
                        'method': m,
                        'column': int_pool[i], 'col_b': int_pool[j]})
                    int_count += 1
            # Group-by if possible
            if cat_cols and num_cols:
                gc = rng.choice(cat_cols[:3]) if variant == 1 and len(cat_cols) > 1 else cat_cols[0]
                gn = rng.choice(num_cols[:3]) if variant == 1 and len(num_cols) > 1 else num_cols[0]
                transforms.append({
                    'method': 'group_mean', 'column': gc, 'col_b': gn})
            # Row stats
            if constraints['allow_row_stats']:
                transforms.append({'method': 'row_stats', 'column': None})
            if transforms:
                pipelines.append(('balanced_mix', transforms))
        return pipelines

    def _gen_kitchen_sink(self, profiles, ranked_cols, constraints, rng):
        """Archetype 6: Large pipeline -- broad coverage (2 variants)."""
        pipelines = []
        num_cols = [c for c in ranked_cols if profiles[c]['type'] == 'numeric']
        cat_cols = [c for c in ranked_cols if profiles[c]['type'] == 'categorical']

        for variant in range(2):
            transforms = []
            # Single-col on many columns
            if variant == 0:
                sc_cols = ranked_cols[:min(10, len(ranked_cols))]
            else:
                sc_cols = rng.sample(ranked_cols[:min(15, len(ranked_cols))],
                                     min(10, len(ranked_cols)))
            for col in sc_cols:
                methods = self._get_applicable_single_transforms(col, profiles[col])
                if methods:
                    m = rng.choice(methods[:2]) if variant == 1 and len(methods) > 1 else methods[0]
                    transforms.append({'method': m, 'column': col})
            # Interactions: up to max
            int_count = 0
            int_pool = num_cols[:6] if variant == 0 else rng.sample(
                num_cols[:10], min(6, len(num_cols))) if len(num_cols) > 0 else []
            for i in range(len(int_pool)):
                if int_count >= constraints['max_interactions']:
                    break
                for j in range(i + 1, len(int_pool)):
                    if int_count >= constraints['max_interactions']:
                        break
                    all_int_m = ['product_interaction', 'division_interaction',
                                 'addition_interaction', 'subtraction_interaction']
                    m = all_int_m[int_count % 2] if variant == 0 else rng.choice(all_int_m)
                    transforms.append({'method': m, 'column': int_pool[i], 'col_b': int_pool[j]})
                    int_count += 1
            # Group-by
            gb_cats = cat_cols[:3] if variant == 0 else rng.sample(
                cat_cols[:5], min(3, len(cat_cols))) if len(cat_cols) > 0 else []
            for ci, cat in enumerate(gb_cats):
                if num_cols:
                    gn = num_cols[min(ci, len(num_cols) - 1)] if variant == 0 else rng.choice(num_cols[:5])
                    transforms.append({
                        'method': 'group_mean', 'column': cat, 'col_b': gn})
            # 3-way if allowed
            if constraints['allow_3way'] and len(num_cols) >= 3:
                if variant == 0:
                    c0, c1, c2 = num_cols[0], num_cols[1], num_cols[2]
                else:
                    trip = rng.sample(num_cols[:6], 3)
                    c0, c1, c2 = trip
                three_way_m = ['three_way_interaction', 'three_way_addition',
                               'three_way_ratio', 'three_way_normalized_diff']
                m3 = 'three_way_interaction' if variant == 0 else rng.choice(three_way_m)
                transforms.append({
                    'method': m3, 'column': c0, 'col_b': c1, 'col_c': c2})
            # Row stats
            if constraints['allow_row_stats']:
                transforms.append({'method': 'row_stats', 'column': None})
            if transforms:
                pipelines.append(('kitchen_sink', transforms))
        return pipelines

    def _gen_row_stats_plus(self, profiles, ranked_cols, constraints, rng):
        """Archetype 7: row_stats + top-K single-col transforms."""
        if not constraints['allow_row_stats']:
            return []
        pipelines = []
        for variant in range(2):
            transforms = [{'method': 'row_stats', 'column': None}]
            if variant == 0:
                pool = ranked_cols[:4]
            else:
                pool = rng.sample(ranked_cols[:min(8, len(ranked_cols))],
                                  min(4, len(ranked_cols)))
            for col in pool:
                methods = self._get_applicable_single_transforms(col, profiles[col])
                if methods:
                    m = rng.choice(methods[:2]) if variant == 1 and len(methods) > 1 else methods[0]
                    transforms.append({'method': m, 'column': col})
            pipelines.append(('row_stats_plus', transforms))
        return pipelines

    @staticmethod
    def _pipeline_hash(transforms):
        """Content hash for deduplication: canonical sorted representation."""
        canonical = []
        for t in transforms:
            key = (t['method'], t.get('column', ''), t.get('col_b', ''), t.get('col_c', ''))
            canonical.append(key)
        return tuple(sorted(canonical))

    def _generate_all_pipelines(self, profiles, ranked_cols, constraints, rng):
        """Generate all candidate pipelines from all archetypes (deduplicated)."""
        all_pipelines = []
        seen_hashes = set()
        generators = [
            self._gen_minimal_surgical,
            self._gen_encoding_focused,
            self._gen_interaction_heavy,
            self._gen_single_col_heavy,
            self._gen_balanced_mix,
            self._gen_kitchen_sink,
            self._gen_row_stats_plus,
        ]
        n_dupes = 0
        for gen_fn in generators:
            try:
                for archetype, transforms in gen_fn(profiles, ranked_cols, constraints, rng):
                    sorted_t = self._sort_transforms(transforms)
                    valid_t = self._validate_pipeline(sorted_t, profiles, constraints)
                    if valid_t:
                        h = self._pipeline_hash(valid_t)
                        if h in seen_hashes:
                            n_dupes += 1
                            continue
                        seen_hashes.add(h)
                        all_pipelines.append((archetype, valid_t))
            except Exception as e:
                self.log(f"Pipeline generation error in {gen_fn.__name__}: {e}", "WARNING")

        if n_dupes > 0:
            self.log(f"Deduplicated {n_dupes} duplicate pipelines")

        # Cap to budget
        max_p = constraints['n_pipelines']
        if len(all_pipelines) > max_p:
            all_pipelines = all_pipelines[:max_p]

        self.log(f"Generated {len(all_pipelines)} candidate pipelines")
        return all_pipelines

    # -----------------------------------------------------------------
    # Pipeline evaluation (multi-transform CV)
    # -----------------------------------------------------------------
    def _apply_pipeline(self, X_train, X_val, y_train, transforms):
        """Apply all transforms in a pipeline sequentially to train/val."""
        for t in transforms:
            method = t['method']
            col = t.get('column')
            col_b = t.get('col_b')
            col_c = t.get('col_c')
            try:
                success, X_train, X_val = self._engine._apply_intervention(
                    X_train, X_val, y_train, col, col_b, col_c, method)
                if not success:
                    return False, X_train, X_val
            except Exception:
                return False, X_train, X_val
        return True, X_train, X_val

    def evaluate_pipeline(self, X, y, transforms):
        """Run K-fold CV with entire pipeline applied.
        Returns: (mean_score, std_score, fold_scores) or (None, None, None)."""
        scores = []
        y = self._engine._ensure_numeric_target(y)
        n_classes = y.nunique()

        if self.n_repeats > 1:
            cv = RepeatedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=42)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Apply ALL transforms in pipeline
            success, X_train, X_val = self._apply_pipeline(X_train, X_val, y_train, transforms)
            if not success:
                return None, None, None

            # Prepare for LightGBM (encode categoricals, handle dtypes)
            X_train, X_val = self._engine._prepare_data_for_model(X_train, X_val, y_train)


            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_val = X_val.replace([np.inf, -np.inf], np.nan)

            # Drop columns that are entirely NaN (transform failed completely)
            all_nan_cols = X_train.columns[X_train.isna().all()]
            if len(all_nan_cols) > 0:
                X_train = X_train.drop(columns=all_nan_cols)
                X_val = X_val.drop(columns=all_nan_cols)

            # Fill remaining NaN with column median (safe fallback)
            fill_vals = X_train.median()
            X_train = X_train.fillna(fill_vals)
            X_val = X_val.fillna(fill_vals)
                        try:
                params = self.eval_params.copy()
                if self.task_type == 'classification':
                    if n_classes > 2:
                        params.update({'objective': 'multiclass', 'num_class': n_classes,
                                       'metric': 'multi_logloss'})
                    else:
                        params.update({'objective': 'binary', 'metric': 'auc'})
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    probs = model.predict_proba(X_val, num_iteration=model.best_iteration_)
                    if n_classes > 2:
                        try:
                            scores.append(metrics.roc_auc_score(
                                y_val, probs, multi_class='ovr',
                                labels=list(range(n_classes))))
                        except ValueError:
                            scores.append(np.nan)
                    else:
                        scores.append(metrics.roc_auc_score(y_val, probs[:, 1]))
                else:
                    params.update({'objective': 'regression', 'metric': 'l2'})
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    preds = model.predict(X_val, num_iteration=model.best_iteration_)
                    scores.append(metrics.r2_score(y_val, preds))
            except Exception as e:
                self.log(f"  CV fold failed: {e}", "WARNING")
                return None, None, None

        valid_scores = [s for s in scores if not np.isnan(s)]
        if not valid_scores:
            return None, None, None
        return float(np.mean(valid_scores)), float(np.std(valid_scores)), valid_scores

    # -----------------------------------------------------------------
    # Pipeline feature extraction
    # -----------------------------------------------------------------
    def _extract_pipeline_features(self, transforms, profiles, n_cols, ranked_cols):
        """Compute pipeline configuration feature dict."""
        methods = [t['method'] for t in transforms]
        touched_cols = set()
        for t in transforms:
            if t.get('column'):
                touched_cols.add(t['column'])
            if t.get('col_b'):
                touched_cols.add(t['col_b'])
            if t.get('col_c'):
                touched_cols.add(t['col_c'])

        num_cols_set = {c for c, p in profiles.items() if p['type'] == 'numeric'}
        cat_cols_set = {c for c, p in profiles.items() if p['type'] == 'categorical'}
        touched_importances = [profiles[c]['importance'] for c in touched_cols if c in profiles]

        top5 = set(ranked_cols[:5])
        top10 = set(ranked_cols[:10])
        n_unique_methods = len(set(methods))
        n_total = len(transforms)

        est_feat = self._estimate_features_added(transforms, profiles)

        return {
            'n_transforms_total': n_total,
            'n_single_col_transforms': sum(
                1 for t in transforms
                if t['method'] not in INTERACTION_METHODS_2WAY
                   and t['method'] not in INTERACTION_METHODS_3WAY
                   and t['method'] != 'row_stats'),
            'n_interactions_2way': sum(1 for m in methods if m in INTERACTION_METHODS_2WAY),
            'n_interactions_3way': sum(1 for m in methods if m in INTERACTION_METHODS_3WAY),
            'n_group_by_transforms': sum(1 for m in methods if m in ('group_mean', 'group_std')),
            'n_encoding_transforms': sum(1 for m in methods if m in (
                'target_encoding', 'frequency_encoding', 'onehot_encoding', 'hashing_encoding')),
            'n_log_sqrt_transforms': sum(1 for m in methods if m in ('log_transform', 'sqrt_transform')),
            'has_row_stats': int('row_stats' in methods),
            'has_missing_indicators': int('missing_indicator' in methods),
            'has_polynomial': int('polynomial_square' in methods),
            'estimated_features_added': est_feat,
            'feature_expansion_ratio': est_feat / max(n_cols, 1),
            'pct_numeric_cols_touched': (
                len(touched_cols & num_cols_set) / max(len(num_cols_set), 1)),
            'pct_cat_cols_touched': (
                len(touched_cols & cat_cols_set) / max(len(cat_cols_set), 1)),
            'avg_touched_col_importance': (
                float(np.mean(touched_importances)) if touched_importances else 0.0),
            'max_touched_col_importance': (
                float(max(touched_importances)) if touched_importances else 0.0),
            'min_touched_col_importance': (
                float(min(touched_importances)) if touched_importances else 0.0),
            'importance_coverage_top5': (
                len(touched_cols & top5) / max(len(top5), 1)),
            'importance_coverage_top10': (
                len(touched_cols & top10) / max(len(top10), 1)),
            'n_unique_methods': n_unique_methods,
            'method_diversity_ratio': n_unique_methods / max(n_total, 1),
        }

    # -----------------------------------------------------------------
    # Main collection method
    # -----------------------------------------------------------------
    def collect(self, df, target_col, dataset_name="unknown"):
        """
        Main pipeline data collection.

        1. Data cleanup â†’ baseline â†’ column profiling
        2. Generate diverse pipelines from 7 archetypes
        3. Evaluate each pipeline end-to-end
        4. Return DataFrame with one row per pipeline
        """
        self.log(f"{'=' * 70}")
        self.log(f"PIPELINE COLLECTOR: {dataset_name}")
        self.log(f"{'=' * 70}")

        self._start_time = time.time()
        self._results = []

        # --- Separate target ---
        y = df[target_col]
        X_raw = df.drop(columns=[target_col])
        y = self._engine._ensure_numeric_target(y)

        self.log(f"Raw shape: {X_raw.shape}")

        # --- ID detection ---
        drop_cols = []
        for col in X_raw.columns:
            is_id, is_const, reason = self._engine._is_likely_id_or_constant(X_raw[col], col, y)
            if is_id or is_const:
                drop_cols.append(col)
        X = X_raw.drop(columns=drop_cols) if drop_cols else X_raw.copy()

        # --- Strategic pruning ---
        prune_list = self._engine._strategic_column_pruning(X, y)
        if prune_list:
            X = X.drop(columns=[c for c, _ in prune_list])

        # --- Target leakage detection ---
        leaky = self._engine._detect_target_leakage(X, y)
        if leaky:
            X = X.drop(columns=[c for c, _, _ in leaky])

        if X.shape[1] == 0:
            self.log("No features left after cleanup", "ERROR")
            return pd.DataFrame()

        self.log(f"Shape after cleanup: {X.shape}")

        # --- Baseline evaluation ---
        self.log("Running baseline evaluation...")
        base_score, base_std, base_folds, base_importances = \
            self._engine.evaluate_with_intervention(X, y, return_importance=True)

        if base_score is None:
            self.log("Baseline failed", "ERROR")
            return pd.DataFrame()

        self.log(f"Baseline: {base_score:.5f} (std: {base_std:.5f})")

        # --- Dataset metadata ---
        ds_meta = self._engine.get_dataset_meta(X, y)
        ds_meta['baseline_score'] = base_score
        ds_meta['baseline_std'] = base_std
        # Headroom: for AUC (classification), 1-score is natural headroom.
        # For R² (regression), scores can be negative; use abs(1-score) clamped.
        if self.task_type == 'classification':
            ds_meta['relative_headroom'] = max(1.0 - base_score, 0.001)
        else:
            ds_meta['relative_headroom'] = max(abs(1.0 - base_score), 0.001)

        # Aggregate importance features
        imp_series = pd.Series(base_importances, index=X.columns)
        imp_vals = imp_series.values
        if len(imp_vals) > 0 and imp_vals.max() > 0:
            imp_norm = imp_vals / imp_vals.max()
            ds_meta['std_feature_importance'] = float(np.std(imp_norm))
            ds_meta['max_minus_min_importance'] = float(imp_norm.max() - imp_norm.min())
            ds_meta['pct_features_above_median_importance'] = float(
                (imp_norm > np.median(imp_norm)).mean())
        else:
            ds_meta['std_feature_importance'] = 0.0
            ds_meta['max_minus_min_importance'] = 0.0
            ds_meta['pct_features_above_median_importance'] = 0.5

        # --- Column profiling ---
        self.log("Profiling columns...")
        profiles = self._profile_columns(X, y, imp_series)
        ranked_cols = sorted(profiles.keys(),
                             key=lambda c: profiles[c]['importance'], reverse=True)

        n_numeric = sum(1 for p in profiles.values() if p['type'] == 'numeric')
        n_cat = sum(1 for p in profiles.values() if p['type'] == 'categorical')

        # --- Dynamic constraints ---
        constraints = self._compute_constraints(X.shape[0], X.shape[1], n_numeric, n_cat)
        self.log(f"Constraints: {constraints}")

        # --- Generate pipelines ---
        rng = random.Random(hash(dataset_name) % (2**31))
        all_pipelines = self._generate_all_pipelines(profiles, ranked_cols, constraints, rng)

        if not all_pipelines:
            self.log("No valid pipelines generated", "WARNING")
            return pd.DataFrame()

        # --- Evaluate each pipeline ---
        for p_idx, (archetype, transforms) in enumerate(all_pipelines):
            if self._is_over_budget():
                self.log(f"Time budget exceeded after {p_idx} pipelines", "WARNING")
                break

            t_desc = [f"{t['method']}({t.get('column', 'global')})" for t in transforms]
            self.log(f"  Pipeline {p_idx + 1}/{len(all_pipelines)} [{archetype}]: "
                     f"{len(transforms)} transforms")

            p_score, p_std, p_folds = self.evaluate_pipeline(X, y, transforms)
            if p_score is None:
                self.log(f"    FAILED", "WARNING")
                continue

            delta = p_score - base_score
            delta_pct = (delta / abs(base_score) * 100) if base_score != 0 else 0.0

            # Statistical significance via paired t-test on fold scores
            p_value = np.nan
            is_significant = False
            if (base_folds is not None and p_folds is not None
                    and len(base_folds) == len(p_folds) and len(p_folds) >= 3
                    and not np.allclose(base_folds, p_folds)):
                try:
                    _, p_value = ttest_rel(p_folds, base_folds)
                    if not np.isnan(p_value):
                        is_significant = (p_value < 0.05) and (delta > 0)
                except Exception:
                    pass
            # improved = significant positive delta (not just any noise)
            improved = int(is_significant)

            self.log(f"    Score: {p_score:.5f} (delta: {delta:+.5f}, {delta_pct:+.2f}%)")

            # Extract pipeline features
            p_features = self._extract_pipeline_features(
                transforms, profiles, X.shape[1], ranked_cols)

            # Build row
            row = {}
            # Dataset metadata
            for key in PIPELINE_CSV_SCHEMA:
                if key in ds_meta:
                    row[key] = ds_meta[key]

            # Pipeline config features
            row.update(p_features)
            row['pipeline_id'] = p_idx
            row['pipeline_archetype'] = archetype

            # Evaluation results
            row['pipeline_score'] = p_score
            row['pipeline_std'] = p_std
            row['pipeline_delta'] = delta
            row['pipeline_delta_pct'] = delta_pct
            row['pipeline_improved'] = improved
            row['pipeline_p_value'] = p_value
            row['pipeline_is_significant'] = is_significant
            row['pipeline_fold_scores'] = json.dumps(
                [round(s, 6) for s in p_folds]) if p_folds else '[]'
            row['pipeline_transforms_json'] = json.dumps(transforms, default=str)

            self._results.append(row)

        elapsed = time.time() - self._start_time
        self.log(f"Done: {len(self._results)} pipelines evaluated in {elapsed:.1f}s")

        if not self._results:
            return pd.DataFrame()

        result_df = pd.DataFrame(self._results)
        # Ensure schema alignment
        for col in PIPELINE_CSV_SCHEMA:
            if col not in result_df.columns:
                result_df[col] = np.nan
        result_df = result_df[PIPELINE_CSV_SCHEMA]
        return result_df

    # -----------------------------------------------------------------
    # CSV writer
    # -----------------------------------------------------------------
    @classmethod
    def write_csv(cls, df_result, csv_file):
        """Write results to CSV with strict schema enforcement."""
        file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
        mode = 'a' if file_exists else 'w'
        write_header = not file_exists

        # Ensure columns match schema
        for col in PIPELINE_CSV_SCHEMA:
            if col not in df_result.columns:
                df_result[col] = np.nan
        df_result = df_result[PIPELINE_CSV_SCHEMA]

        with open(csv_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=PIPELINE_CSV_SCHEMA,
                                    extrasaction='ignore')
            if write_header:
                writer.writeheader()
            for _, row in df_result.iterrows():
                clean = {}
                for k, v in row.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        clean[k] = ''
                    else:
                        clean[k] = v
                writer.writerow(clean)


# =====================================================================
# RUNNER: Process OpenML datasets
# =====================================================================
def run_pipeline_openml(output_dir='./pipeline_meta_output',
                        n_folds=5, n_repeats=3,
                        time_budget=7200, task_list_file=None):
    """Run the pipeline data collection on OpenML datasets.

    Reads the same task_list.json format as DataCollector_3.
    Outputs to pipeline_meta_learning_db.csv.
    """
    import openml

    os.makedirs(output_dir, exist_ok=True)

    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    if not os.path.exists(task_list_file):
        raise FileNotFoundError(
            f"Task list not found at {task_list_file}. "
            "Run generate_task_list.py first.")

    with open(task_list_file, 'r') as f:
        task_data = json.load(f)
    task_list = task_data['tasks']
    print(f"Loaded {len(task_list)} tasks from {task_list_file}")

    env_info = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': n_folds, 'n_repeats': n_repeats,
        'n_tasks': len(task_list),
        'version': 'pipeline_v1',
    }
    with open(os.path.join(output_dir, 'environment.json'), 'w') as f:
        json.dump(env_info, f, indent=2)

    print("=" * 70)
    print("PIPELINE META-LEARNING DATA COLLECTOR v1")
    print("=" * 70)
    print(f"  Output:      {output_dir}")
    print(f"  Tasks:       {len(task_list)}")
    print(f"  CV:          {n_folds}-fold x {n_repeats} repeats")
    print(f"  Time budget: {time_budget}s per dataset")
    print("=" * 70)

    csv_file = os.path.join(output_dir, 'pipeline_meta_learning_db.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')

    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed = set(json.load(f).get('processed_tasks', []))
            print(f"Checkpoint: {len(processed)} already done")

    ok, skip, fail = 0, 0, 0

    for entry in task_list:
        task_id = entry['task_id']
        dataset_id = entry.get('dataset_id')
        forced_type = entry.get('task_type')
        ckpt_key = str(task_id) if task_id and task_id > 0 else f"ds_{dataset_id}"

        if ckpt_key in processed:
            continue

        try:
            print(f"\n{'=' * 70}")
            print(f"[{ok + skip + fail + 1}/{len(task_list)}] "
                  f"task={task_id}  dataset={dataset_id}  "
                  f"({entry.get('dataset_name', '?')})")
            print("=" * 70)

            dataset = None
            target_col = None
            task_type = forced_type

            if task_id and task_id > 0:
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()
                target_col = dataset.default_target_attribute
                if task_type is None:
                    task_type = _detect_openml_task_type(task)
            elif dataset_id:
                dataset = openml.datasets.get_dataset(dataset_id)
                target_col = dataset.default_target_attribute
            else:
                print("  No valid task_id or dataset_id â€” skipping")
                skip += 1
                continue

            if not target_col:
                print(f"  No default target â€” skipping")
                skip += 1
                processed.add(ckpt_key)
                continue

            if task_type is None:
                _y_tmp = dataset.get_data(target=target_col, dataset_format='dataframe')[1]
                task_type = _infer_task_type(_y_tmp)

            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            if task_type is None:
                task_type = _infer_task_type(y)

            # Smart size reduction for very large datasets
            raw_cells = X.shape[0] * X.shape[1]
            if raw_cells > 100_000_000:
                print(f"  Large dataset ({X.shape}, {raw_cells:,} cells) â€” reducing...")
                X, y, msg = _smart_size_reduction(X, y, task_type, cell_limit=100_000_000)
                print(f"  -> {msg}")
                if X.shape[1] == 0:
                    processed.add(ckpt_key)
                    skip += 1
                    continue

            print(f"  Type: {task_type} | Shape: {X.shape}")

            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

            # --- Run pipeline collector ---
            t0 = time.time()
            collector = PipelineDataCollector(
                task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
                output_dir=output_dir, time_budget_seconds=time_budget,
                n_jobs=-1)

            full_df = pd.concat([X, y], axis=1)
            df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
            elapsed = time.time() - t0

            if df_result.empty:
                print(f"  Empty result â€” skipped ({elapsed:.1f}s)")
                skip += 1
            else:
                df_result['openml_task_id'] = task_id if (task_id and task_id > 0) else f"ds_{dataset_id}"
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                df_result['task_type'] = task_type

                PipelineDataCollector.write_csv(df_result, csv_file)
                print(f"  Done: {len(df_result)} pipelines in {elapsed:.1f}s")
                ok += 1

            processed.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed_tasks': list(processed)}, f)

            del collector, full_df
            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            fail += 1
            processed.add(ckpt_key)
            continue

    print(f"\n{'=' * 70}")
    print(f"DONE: {ok} ok, {skip} skipped, {fail} failed")
    if os.path.exists(csv_file):
        try:
            db = pd.read_csv(csv_file)
            print(f"Total rows: {len(db)} | Datasets: {db['dataset_name'].nunique()}")
            imp = db['pipeline_improved'].sum()
            print(f"Pipelines that improved: {imp}/{len(db)}")
        except Exception:
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline Meta-Learning Data Collector')
    parser.add_argument('--output-dir', default='./pipeline_meta_output')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--time-budget', type=int, default=7200)
    parser.add_argument('--task-list', default=None)
    args = parser.parse_args()

    run_pipeline_openml(
        output_dir=args.output_dir,
        n_folds=args.n_folds, n_repeats=args.n_repeats,
        time_budget=args.time_budget, task_list_file=args.task_list)