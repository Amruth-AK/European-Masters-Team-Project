import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import metrics
from scipy.stats import skew, kurtosis
import warnings
import csv
import re
import os
import json
import traceback
from datetime import datetime
from scipy.stats import f_oneway
from scipy.stats import shapiro, pearsonr, spearmanr, ttest_rel
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from diptest import diptest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import networkx as nx
import pyarrow
import openml
import psutil
import math
import time
import gc
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================================================
# OPTIONAL DEPENDENCIES: AutoFeat + FeatureWiz
# =============================================================================
# These are used for enhanced interaction discovery and column pre-filtering.
# If not installed, the collector falls back to the existing tree-guided +
# importance-based interaction strategy with zero behavior change.

_HAS_AUTOFEAT = False
try:
    from autofeat import AutoFeatClassifier, AutoFeatRegressor
    _HAS_AUTOFEAT = True
except ImportError:
    pass

_HAS_FEATUREWIZ = False
try:
    import featurewiz as fwiz
    _HAS_FEATUREWIZ = True
except ImportError:
    pass


class MetaDataCollector:
    """
    Meta-Learning Data Collector v8 ("Fix: null-calibration, Bonferroni, cv_strategy, primary_metric")
    
    v7 changes over v5/v6:
    - FeatureWiz integration: SULOV + XGBoost column ranking before column selection.
      Blends FeatureWiz signal (20%) into composite_predictive_score to improve
      column selection and interaction pool prioritization. Adds featurewiz_selected
      and featurewiz_importance columns to CSV for meta-model training.
    - AutoFeat integration: Polynomial feature generation + L1 selection to discover
      non-obvious interaction pairs. AutoFeat-discovered pairs are tested alongside
      tree-guided and importance-based pairs with 'autofeat_discovered' source label.
    - Both integrations are optional: graceful fallback if libraries not installed.
    - New __init__ flags: use_autofeat, use_featurewiz, autofeat_max_gb,
      autofeat_feateng_steps, featurewiz_corr_limit

    v5 changes over v4:
    - CV strategy: 3x5-fold repeated CV (was 10-fold) for more stable estimates
    - Tree-guided interaction selection: fits shallow decision trees to identify
      conditional dependencies, then tests THOSE pairs instead of brute-force top-K
    - Noise-floor calibration: null intervention variance used to compute calibrated
      deltas and Cohen's d effect sizes for cross-dataset comparability
    - Time budget: per-dataset time limit prevents one slow dataset from stalling run
    - Memory management: explicit GC between datasets for overnight stability
    - New CSV fields: null_delta, null_std, cohens_d, individual_cohens_d,
      calibrated_delta, individual_calibrated_delta, tree_pair_score
    """
    
    
    # =========================================================================
    # FIXED CSV SCHEMA -- every row MUST have exactly these columns in this order.
    # This prevents column-shift bugs when appending datasets to the same CSV.
    # =========================================================================
    CSV_SCHEMA = [
        # --- Dataset-level metadata (21 fields) ---
        'n_rows', 'n_cols', 'cat_ratio', 'missing_ratio', 'row_col_ratio',
        'class_imbalance_ratio', 'n_classes', 'target_std', 'target_skew',
        'landmarking_score', 'landmarking_score_norm',
        'avg_feature_corr', 'max_feature_corr', 'avg_target_corr', 'max_target_corr',
        'avg_numeric_sparsity', 'linearity_gap',
        'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
        'matrix_rank_ratio',
        # --- Phase 2.1: Aggregate dataset features (5 fields) ---
        'n_numeric_cols', 'n_cat_cols',
        'std_feature_importance', 'max_minus_min_importance',
        'pct_features_above_median_importance',
        # --- Phase 2.5: Headroom normalization (1 field) ---
        'relative_headroom',
        # --- Dataset-level evaluation context (4 fields) ---
        'near_ceiling_flag', 'baseline_score', 'baseline_std',
        'n_cols_before_selection', 'n_cols_selected', 'column_selection_method',
        # --- Column-level metadata (32 fields) ---
        'null_pct', 'unique_ratio', 'is_numeric', 'outlier_ratio', 'entropy',
        'baseline_feature_importance', 'skewness', 'kurtosis', 'coeff_variation',
        'zeros_ratio', 'shapiro_p_value', 'has_multiple_modes',
        'bimodality_proxy_heuristic', 'range_iqr_ratio', 'dominant_quartile_pct',
        'pct_in_0_1_range', 'spearman_corr_target', 'hartigan_dip_pval',
        'is_multimodal', 'top_category_dominance', 'normalized_entropy',
        'is_binary', 'is_low_cardinality', 'is_high_cardinality',
        'top3_category_concentration', 'rare_category_pct', 'conditional_entropy',
        'pps_score', 'mutual_information_score',
        'vif', 'composite_predictive_score',
        'is_temporal_component', 'temporal_period',
        # --- v7: External library discovery metadata (2 fields) ---
        'featurewiz_selected', 'featurewiz_importance',
        # --- Intervention identification (3 fields) ---
        'column_name', 'method', 'is_interaction',
        # --- Full-model evaluation (8 fields) ---
        'delta', 'delta_normalized', 'absolute_score', 'absolute_score_mse',
        't_statistic', 'p_value', 'p_value_bonferroni',
        'is_significant', 'is_significant_bonferroni',
        # --- Individual evaluation (8 fields) ---
        'individual_baseline_score', 'individual_intervention_score',
        'individual_delta', 'individual_delta_normalized',
        'individual_p_value', 'individual_p_value_bonferroni',
        'individual_is_significant', 'individual_is_significant_bonferroni',
        'individual_skip_reason',
        # --- Interaction metadata (8 fields) ---
        'interaction_col_a', 'interaction_col_b', 'interaction_col_c',
        'pairwise_corr_ab', 'pairwise_spearman_ab', 'pairwise_mi_ab',
        'interaction_scale_ratio', 'interaction_source',
        # --- Optional fold scores (2 fields) ---
        'baseline_fold_scores', 'intervention_fold_scores',
        # --- v5: Noise-floor calibration and effect sizes (6 fields) ---
        'null_delta', 'null_std',
        'null_std_was_clamped',
        'cohens_d', 'individual_cohens_d',
        'calibrated_delta', 'individual_calibrated_delta',
        # --- v5: Tree-guided interaction metadata (1 field) ---
        'tree_pair_score',
        # --- v6: col_b features for 2-way & 3-way interactions (8 fields) ---
        'col_b_is_numeric', 'col_b_skewness', 'col_b_unique_ratio',
        'col_b_null_pct', 'col_b_outlier_ratio', 'col_b_baseline_importance',
        'col_b_entropy', 'col_b_composite_predictive_score',
        # --- v6: col_c features for 3-way interactions (8 fields) ---
        'col_c_is_numeric', 'col_c_skewness', 'col_c_unique_ratio',
        'col_c_null_pct', 'col_c_outlier_ratio', 'col_c_baseline_importance',
        'col_c_entropy', 'col_c_composite_predictive_score',
        # --- OpenML identifiers (added by pipeline runner) ---
        'openml_task_id', 'dataset_name', 'dataset_id', 'task_type',
        # --- Fix 4, 5 & 6: model complexity, CV strategy, and metric context ---
        # individual_model_complexity: 'simplified' when individual_params were used
        #   (100 trees, 15 leaves, depth 4) vs 'standard' (full base_params).
        #   Needed because individual_delta and delta use different model budgets.
        # cv_strategy: 'repeated_NfoldxR' | 'chunked_2' | 'chunked_3'.
        #   Chunked CV trains on a fraction of data; scores not comparable across
        #   strategies without conditioning on n_rows.
        # primary_metric: 'roc_auc' (classification) | 'r2' (regression).
        #   AUC is bounded [~0.5, 1.0]; R^2 can be strongly negative.
        #   Meta-model MUST condition on task_type/primary_metric, or train separate models.
        'individual_model_complexity', 'cv_strategy', 'primary_metric',
    ]
    
    def __init__(self, task_type='classification', n_folds=5, n_repeats=3, 
                 store_fold_scores=False, output_dir='./meta_learning_output',
                 time_budget_seconds=14400, n_jobs=-1,
                 use_autofeat=True, use_featurewiz=True,
                 autofeat_max_gb=0.5, autofeat_feateng_steps=1,
                 featurewiz_corr_limit=0.7):  
        self.task_type = task_type
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.store_fold_scores = store_fold_scores
        self.output_dir = output_dir
        self.meta_data_log = []
        self.n_tests_performed = 0
        self.time_budget_seconds = time_budget_seconds  # v5: per-dataset time limit
        self._start_time = None  # Set in collect()
        
        # v7: External library integration flags
        self.use_autofeat = use_autofeat and _HAS_AUTOFEAT
        self.use_featurewiz = use_featurewiz and _HAS_FEATUREWIZ
        self.autofeat_max_gb = autofeat_max_gb
        self.autofeat_feateng_steps = autofeat_feateng_steps
        self.featurewiz_corr_limit = featurewiz_corr_limit
        # v7: FeatureWiz column ranking cache (populated in collect if enabled)
        self._featurewiz_selected = set()   # columns FeatureWiz kept
        self._featurewiz_importance = {}     # col -> relative importance (0-1)
        
        # Cache for individual column baselines: cache_key -> (score, std, fold_scores)
        self._individual_baseline_cache = {}
        self._last_X_ref = pd.DataFrame()  # Set properly in collect()
        self._pps_cache = {}
        self._mi_cache = {}
        # Adaptive method tracking: {method: {'tested': N, 'improved': N, 'disabled': bool}}
        # Reset per dataset in collect()
        self._method_tracker = {}
        
        # v5/v8: Noise-floor calibration (set during collect from baseline fold std)
        self._null_delta = 0.0
        self._null_std = 0.005  # Default; set in collect() from baseline fold variance
        self._null_std_was_clamped = False
        # Fix 5: CV strategy label, set in collect() after baseline evaluation
        self._cv_strategy = 'unknown'
        
        # v6: Column metadata cache for col_b/col_c features in interactions
        # Populated during univariate loop, keyed by column name
        self._column_meta_cache = {}
        self._composite_scores_cache = {}  # col -> composite predictive score
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.base_params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': n_jobs
        }
        
        # Simpler params for individual column evaluation (few features -> simpler model)
        self.individual_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': n_jobs
        }
        
        self.log_file = os.path.join(output_dir, 'pipeline_log.txt')
        self.error_log_file = os.path.join(output_dir, 'error_log.txt')
        
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
        log_message = f"[{timestamp}] [{level}] [MEM: {mem_usage:.1f}MB] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def log_error(self, message, exception=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_message = f"[{timestamp}] ERROR: {message}\n"
        if exception:
            error_message += f"Exception: {str(exception)}\n"
            error_message += traceback.format_exc() + '\n'
        error_message += '-' * 80 + '\n'
        print(f"-> {message}")
        if exception:
            print(f"   Exception: {str(exception)}")
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write(error_message)

    

    # =========================================================================
    # ROW BUILDING, VALIDATION, AND CSV WRITING
    # =========================================================================




    @staticmethod
    def _sanitize_string(val):
        """Sanitize a string value for CSV: strip commas, quotes, newlines."""
        if val is None:
            return ''
        s = str(val)
        # Remove characters that break CSV parsing
        s = s.replace(',', '_').replace('"', '').replace("'", '').replace('\n', ' ').replace('\r', '')
        return s.strip()

    def _build_row(self, ds_meta=None, col_meta=None, **kwargs):
        """
        Build a row dict with EXACTLY the columns in CSV_SCHEMA.
        
        Any key not in the schema is silently dropped.
        Any schema key not provided defaults to np.nan.
        String fields (column_name, method, etc.) are sanitized.
        """
        row = {}
        
        # Merge all sources into a flat dict
        merged = {}
        if ds_meta:
            merged.update(ds_meta)
        if col_meta:
            merged.update(col_meta)
        merged.update(kwargs)
        
        # Build row with exactly the schema columns
        STRING_FIELDS = {'column_name', 'method', 'column_selection_method',
                         'individual_skip_reason', 'interaction_source',
                         'interaction_col_a', 'interaction_col_b', 'interaction_col_c',
                         'dataset_name', 'task_type'}
        
        for col in self.CSV_SCHEMA:
            val = merged.get(col, np.nan)
            if col in STRING_FIELDS and val is not None and not (isinstance(val, float) and np.isnan(val)):
                val = self._sanitize_string(val)
            row[col] = val
        
        return row

    def _validate_and_append(self, ds_meta=None, col_meta=None, **kwargs):
        """
        Build, validate, and append a row to meta_data_log.
        
        Raises ValueError if the row doesn't match the schema length.
        """
        row = self._build_row(ds_meta=ds_meta, col_meta=col_meta, **kwargs)
        
        if len(row) != len(self.CSV_SCHEMA):
            raise ValueError(
                f"Row has {len(row)} columns, expected {len(self.CSV_SCHEMA)}. "
                f"Extra: {set(row.keys()) - set(self.CSV_SCHEMA)}, "
                f"Missing: {set(self.CSV_SCHEMA) - set(row.keys())}"
            )
        
        self.meta_data_log.append(row)

    @classmethod
    def write_csv(cls, df_result, csv_file):
        """
        Write results to CSV with strict schema enforcement.
        
        - First write: includes header
        - Subsequent writes: appends without header, columns aligned to schema
        
        Uses csv.DictWriter for guaranteed column alignment.
        """
        # Only keep schema columns (drop any extras like temp columns)
        # Add any schema columns missing from df_result
        schema_cols = [c for c in cls.CSV_SCHEMA if c in df_result.columns]
        for c in cls.CSV_SCHEMA:
            if c not in df_result.columns:
                df_result[c] = np.nan
        
        df_out = df_result[cls.CSV_SCHEMA]
        
        write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        
        # Use DictWriter for strict alignment
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cls.CSV_SCHEMA, extrasaction='ignore',
                                    quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writeheader()
            for _, row in df_out.iterrows():
                # Convert to dict, replace NaN with empty string for clean CSV
                # NOTE: Do NOT re-sanitize strings here. _build_row() already
                # sanitized STRING_FIELDS on input. csv.DictWriter handles
                # quoting of fields that contain commas (e.g. JSON fold scores).
                row_dict = {}
                for col in cls.CSV_SCHEMA:
                    val = row[col]
                    # Catch all flavors of missing: NaN, None, pd.NA
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row_dict[col] = ''
                    else:
                        try:
                            if pd.isna(val):
                                row_dict[col] = ''
                                continue
                        except (TypeError, ValueError):
                            pass
                        row_dict[col] = val
                writer.writerow(row_dict)

    # =========================================================================
    # COLUMN TYPE CLASSIFICATION
    # =========================================================================

    def _classify_column_type(self, series, col_name):
        """
        Classify a column into one of: 'numeric', 'categorical', 'date', 'text'
        Used for stratified selection.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'date'
        
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check for date-like strings
        if self._is_likely_date(series, col_name):
            return 'date'
        
        # Check for text-like (high cardinality string with long average length)
        # We are already in the non-numeric, non-date branch, so just check content
        nunique = series.nunique()
        avg_len = series.astype(str).str.len().mean()
        if nunique > 50 and avg_len > 10:
            return 'text'
        
        return 'categorical'

    # =========================================================================
    # TARGET LEAKAGE DETECTION
    # =========================================================================

    def _detect_target_leakage(self, X, y):
        """
        Detect columns that are suspiciously predictive of the target (>0.95).
        
        For numeric: abs(Pearson correlation) > 0.95
        For categorical: normalized MI > 0.95 or PPS > 0.95
        
        Returns: list of (col_name, reason, score) tuples
        """
        leaky_cols = []
        y_numeric = self._ensure_numeric_target(y)
        
        for col in X.columns:
            is_num = pd.api.types.is_numeric_dtype(X[col])
            
            if is_num:
                # Pearson correlation
                try:
                    clean_idx = X[col].notna() & y_numeric.notna()
                    if clean_idx.sum() > 10:
                        corr = abs(X[col][clean_idx].corr(y_numeric[clean_idx]))
                        if corr > 0.95:
                            leaky_cols.append((col, f"pearson_corr={corr:.4f}", corr))
                            continue
                except:
                    pass
                
                # Spearman as backup (catches monotonic non-linear relationships)
                try:
                    clean_idx = X[col].notna()
                    if clean_idx.sum() > 10:
                        sp_corr, _ = spearmanr(X[col][clean_idx], y_numeric[clean_idx])
                        if abs(sp_corr) > 0.95:
                            leaky_cols.append((col, f"spearman_corr={abs(sp_corr):.4f}", abs(sp_corr)))
                            continue
                except:
                    pass
            
            # For ALL column types: PPS check
            pps = self._pps_cache.get(col, self._calculate_pps(X[col], y_numeric))
            if pps > 0.95:
                leaky_cols.append((col, f"pps={pps:.4f}", pps))
                continue
            
            # For categorical: normalized MI check
            if not is_num:
                try:
                    le = LabelEncoder()
                    encoded = pd.Series(le.fit_transform(X[col].astype(str).fillna('NaN')), index=X[col].index)
                    if self.task_type == 'classification':
                        mi = mutual_info_classif(encoded.to_frame(), y_numeric, random_state=42)[0]
                    else:
                        mi = mutual_info_regression(encoded.to_frame(), y_numeric, random_state=42)[0]
                    
                    # Normalize MI by entropy of target
                    target_entropy = self._calculate_entropy(y_numeric)
                    if target_entropy > 0:
                        normalized_mi = mi / target_entropy
                        if normalized_mi > 0.95:
                            leaky_cols.append((col, f"normalized_mi={normalized_mi:.4f}", normalized_mi))
                            continue
                except:
                    pass
        
        return leaky_cols

    # =========================================================================
    # COMPOSITE PREDICTIVE SCORE
    # =========================================================================

    def _compute_column_predictive_scores(self, X, y, base_feature_importance):
        """
        Compute a composite predictive score for each column using 3 signals:
        1. Baseline LightGBM feature importance (normalized to [0, 1])
        2. Mutual Information (normalized to [0, 1])  
        3. Predictive Power Score (already in [0, 1])
        
        All three work for ALL column types (numeric, categorical, text, date)
        because:
        - LightGBM importance: categoricals are encoded via .cat.codes before training
        - MI: categoricals are label-encoded, numerics used directly
        - PPS: same encoding as MI
        
        Final score = weighted average: 40% importance + 30% MI + 30% PPS
        
        Returns: dict of {col_name: composite_score}
        """
        y_numeric = self._ensure_numeric_target(y)
        scores = {}
        
        # 1. Normalize baseline importance to [0, 1]
        imp_max = base_feature_importance.max() if base_feature_importance.max() > 0 else 1.0
        imp_normalized = base_feature_importance / imp_max
        
        # 2. Compute MI in one batched call (much faster than N individual calls)
        try:
            X_enc = pd.DataFrame(index=X.index)
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_enc[col] = X[col].fillna(X[col].median())
                else:
                    le = LabelEncoder()
                    X_enc[col] = le.fit_transform(X[col].astype(str).fillna('NaN'))
            if self.task_type == 'classification':
                mi_vals = mutual_info_classif(X_enc, y_numeric, random_state=42)
            else:
                mi_vals = mutual_info_regression(X_enc, y_numeric, random_state=42)
            mi_scores = dict(zip(X.columns, mi_vals))
        except:
            mi_scores = {col: 0.0 for col in X.columns}
        self._mi_cache = mi_scores  # save for get_column_meta
        
        mi_max = max(mi_scores.values()) if mi_scores and max(mi_scores.values()) > 0 else 1.0
        
        # 3. Compute PPS for all columns and cache
        self._pps_cache = {}
        for col in X.columns:
            self._pps_cache[col] = self._calculate_pps(X[col], y_numeric)
        
        # 4. Composite score
        for col in X.columns:
            imp_score = imp_normalized.get(col, 0.0)
            mi_score = mi_scores.get(col, 0.0) / mi_max
            pps_score = self._pps_cache.get(col, 0.0)
            scores[col] = 0.4 * imp_score + 0.3 * mi_score + 0.3 * pps_score
        
        return scores

    # =========================================================================
    # STRATIFIED COLUMN SELECTION
    # =========================================================================

    def _select_columns_stratified(self, X, y, base_feature_importance):
        """
        Select columns with cap and stratified type distribution.
        
        Cap formula: min(n_cols, 500 + ceil(0.5 * n_cols))
        
        Strategy:
        - 70% of slots allocated proportionally by type (numeric/categorical/date/text)
        - 30% of slots filled by best available regardless of type
        - Within each type, ranked by composite predictive score
        
        Returns: (selected_columns, selection_report)
        """
        n_cols = X.shape[1]
        cap = min(n_cols, 250 + math.ceil(0.25 * n_cols))
        
        # If under cap, use all columns
        if n_cols <= cap:
            return X.columns.tolist(), {
                'cap': cap, 'n_original': n_cols, 'n_selected': n_cols,
                'method': 'all_columns (under cap)'
            }
        
        self.log(f"Column selection: {n_cols} columns -> cap at {cap}")
        
        # 1. Compute composite scores
        composite_scores = self._compute_column_predictive_scores(X, y, base_feature_importance)
        
        # 2. Classify columns by type
        col_types = {}
        type_groups = {'numeric': [], 'categorical': [], 'date': [], 'text': []}
        
        for col in X.columns:
            ctype = self._classify_column_type(X[col], col)
            col_types[col] = ctype
            type_groups[ctype].append(col)
        
        # Log type distribution
        type_dist = {t: len(cols) for t, cols in type_groups.items() if cols}
        self.log(f"  Type distribution: {type_dist}")
        
        # 3. Sort each type group by composite score (descending)
        for t in type_groups:
            type_groups[t].sort(key=lambda c: composite_scores.get(c, 0), reverse=True)
        
        # 4. Allocate 70% proportionally, 30% best-available
        proportional_slots = int(cap * 0.7)
        free_slots = cap - proportional_slots
        
        # Proportional allocation (only for types that have columns)
        active_types = {t: len(cols) for t, cols in type_groups.items() if cols}
        total_active = sum(active_types.values())
        
        type_allocations = {}
        allocated_so_far = 0
        for t, count in active_types.items():
            proportion = count / total_active
            alloc = max(1, int(proportional_slots * proportion))  # At least 1 per type
            alloc = min(alloc, count)  # Can't allocate more than available
            type_allocations[t] = alloc
            allocated_so_far += alloc
        
        # If we over-allocated due to min(1) constraints, trim from largest
        while allocated_so_far > proportional_slots:
            largest_type = max(type_allocations, key=type_allocations.get)
            if type_allocations[largest_type] > 1:
                type_allocations[largest_type] -= 1
                allocated_so_far -= 1
        
        # If under-allocated, add remainder to free slots
        free_slots += (proportional_slots - allocated_so_far)
        
        self.log(f"  Proportional allocation: {type_allocations}")
        self.log(f"  Free (best-available) slots: {free_slots}")
        
        # 5. Select from each type
        selected = set()
        for t, alloc in type_allocations.items():
            for col in type_groups[t][:alloc]:
                selected.add(col)
        
        # 6. Fill free slots with best remaining columns (any type)
        remaining = [(col, composite_scores.get(col, 0)) for col in X.columns if col not in selected]
        remaining.sort(key=lambda x: x[1], reverse=True)
        
        for col, score in remaining[:free_slots]:
            selected.add(col)
        
        selected_list = [col for col in X.columns if col in selected]  # Preserve original order
        
        # Report
        selected_type_dist = {}
        for col in selected_list:
            t = col_types[col]
            selected_type_dist[t] = selected_type_dist.get(t, 0) + 1
        
        report = {
            'cap': cap,
            'n_original': n_cols,
            'n_selected': len(selected_list),
            'method': 'stratified_selection',
            'original_type_dist': type_dist,
            'selected_type_dist': selected_type_dist,
            'type_allocations': type_allocations,
            'free_slots_used': min(free_slots, len(remaining)),
            'min_composite_score_selected': min(composite_scores[c] for c in selected_list),
            'max_composite_score_dropped': max(
                (composite_scores[c] for c in X.columns if c not in selected), default=0
            )
        }
        
        self.log(f"  Selected type distribution: {selected_type_dist}")
        self.log(f"  Min score selected: {report['min_composite_score_selected']:.4f}")
        self.log(f"  Max score dropped: {report['max_composite_score_dropped']:.4f}")
        
        return selected_list, report

    # =========================================================================
    # STRATEGIC COLUMN PRUNING (pre-selection cleanup)
    # =========================================================================

    def _strategic_column_pruning(self, X, y):
        """
        Remove redundant columns before selection:
        1. Constant (1 unique value)
        2. Near-constant (>99% same value)
        3. Near-zero variance (std < 1e-8)
        4. High inter-feature correlation (r > 0.99, keep the one more correlated with target)
        """
        drop_cols = {}
        
        for col in X.columns:
            nunique = X[col].nunique(dropna=True)
            if nunique <= 1:
                drop_cols[col] = "constant"
                continue
            top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > 0.99:
                drop_cols[col] = f"near_constant (top_freq={top_freq:.4f})"
                continue
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in drop_cols:
                continue
            if X[col].std() < 1e-8:
                drop_cols[col] = "near_zero_variance"
        
        numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in drop_cols]
        if len(numeric_cols) > 1:
            y_numeric = self._ensure_numeric_target(y)
            if pd.api.types.is_numeric_dtype(y_numeric):
                target_corrs = {col: abs(X[col].corr(y_numeric)) for col in numeric_cols}
            else:
                target_corrs = {col: 0.0 for col in numeric_cols}
            
            X_sample = X[numeric_cols]
            if len(X_sample) > 100000:
                X_sample = X_sample.sample(100000, random_state=42)
            
            corr_matrix = X_sample.corr().abs()
            for i, col_a in enumerate(numeric_cols):
                if col_a in drop_cols:
                    continue
                for j in range(i + 1, len(numeric_cols)):
                    col_b = numeric_cols[j]
                    if col_b in drop_cols:
                        continue
                    if corr_matrix.loc[col_a, col_b] > 0.99:
                        to_drop = col_b if target_corrs.get(col_a, 0) >= target_corrs.get(col_b, 0) else col_a
                        kept = col_a if to_drop == col_b else col_b
                        drop_cols[to_drop] = f"high_corr_with_{kept} (r={corr_matrix.loc[col_a, col_b]:.4f})"
        
        return list(drop_cols.items())

    # =========================================================================
    # DATASET-LEVEL METADATA
    # =========================================================================

    def get_dataset_meta(self, X, y):
        """Extended statistical fingerprint of the dataset."""
        ds_meta = {
            'n_rows': X.shape[0],
            'n_cols': X.shape[1],
            'cat_ratio': len(X.select_dtypes(include=['object', 'category', 'bool']).columns) / max(X.shape[1], 1),
            'missing_ratio': X.isnull().sum().sum() / max(X.shape[0] * X.shape[1], 1),
            'row_col_ratio': X.shape[0] / max(X.shape[1], 1),
            # Phase 2.1: Aggregate column type counts
            'n_numeric_cols': len(X.select_dtypes(include=[np.number]).columns),
            'n_cat_cols': len(X.select_dtypes(include=['object', 'category', 'bool']).columns),
        }
        
        if self.task_type == 'classification':
            vc = y.value_counts()
            ds_meta['class_imbalance_ratio'] = vc.min() / vc.max() if len(vc) > 1 else 1.0
            ds_meta['n_classes'] = len(vc)
            ds_meta['target_std'] = np.nan
            ds_meta['target_skew'] = np.nan
        else:
            ds_meta['class_imbalance_ratio'] = -1.0
            ds_meta['n_classes'] = -1
            ds_meta['target_std'] = y.std()
            ds_meta['target_skew'] = skew(y.dropna())

        ds_meta['landmarking_score'] = self._run_landmarking_model(X, y)
        if self.task_type == 'classification':
            ds_meta['landmarking_score_norm'] = (ds_meta['landmarking_score'] - 0.5) * 2
        else:
            ds_meta['landmarking_score_norm'] = ds_meta['landmarking_score']

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # For very wide datasets, sample columns for correlation
            if len(numeric_cols) > 500:
                sample_cols = numeric_cols[:500]  # Use first 500 for speed
            else:
                sample_cols = numeric_cols
            corr_matrix = X[sample_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            ds_meta['avg_feature_corr'] = upper_tri.stack().mean()
            ds_meta['max_feature_corr'] = upper_tri.stack().max()
        else:
            ds_meta['avg_feature_corr'] = 0.0
            ds_meta['max_feature_corr'] = 0.0

        if len(numeric_cols) > 0:
            y_num = self._ensure_numeric_target(y)
            target_corrs = []
            for col in numeric_cols:
                if not X[col].isnull().all():
                    try:
                        target_corrs.append(abs(X[col].corr(y_num)))
                    except:
                        pass
            ds_meta['avg_target_corr'] = np.mean(target_corrs) if target_corrs else 0.0
            ds_meta['max_target_corr'] = np.max(target_corrs) if target_corrs else 0.0
        else:
            ds_meta['avg_target_corr'] = 0.0
            ds_meta['max_target_corr'] = 0.0

        if len(numeric_cols) > 0:
            zero_ratios = [(X[col] == 0).mean() for col in numeric_cols]
            ds_meta['avg_numeric_sparsity'] = np.mean(zero_ratios)
        else:
            ds_meta['avg_numeric_sparsity'] = 0.0

        ds_meta['linearity_gap'] = self._calculate_linearity_gap(X, y)
        
        if len(numeric_cols) > 1:
            sample_cols = numeric_cols[:500] if len(numeric_cols) > 500 else numeric_cols
            graph_metrics = self._calculate_correlation_graph_metrics(X[sample_cols])
            ds_meta.update(graph_metrics)
        else:
            ds_meta['corr_graph_components'] = 0
            ds_meta['corr_graph_clustering'] = 0.0
            ds_meta['corr_graph_density'] = 0.0
        
        if len(numeric_cols) > 1:
            ds_meta['matrix_rank_ratio'] = self._calculate_matrix_rank_ratio(X[numeric_cols])
        else:
            ds_meta['matrix_rank_ratio'] = 0.0

        return ds_meta

    # =========================================================================
    # COLUMN-LEVEL METADATA
    # =========================================================================

    def get_column_meta(self, col_series, y_target, baseline_importance=None):
        """Extended statistical fingerprint of a column."""
        is_num = pd.api.types.is_numeric_dtype(col_series)
        clean_col = col_series.dropna()
        if is_num and len(clean_col) > 0:
            clean_col = clean_col.astype(float)

        outlier_ratio = 0
        iqr_val = 0
        if is_num and len(clean_col) > 0:
            q1, q3 = np.percentile(clean_col, [25, 75])
            iqr_val = q3 - q1
            if iqr_val > 0:
                outliers = clean_col[(clean_col < (q1 - 1.5 * iqr_val)) | (clean_col > (q3 + 1.5 * iqr_val))]
                outlier_ratio = len(outliers) / len(clean_col)

        res = {
            'dtype': str(col_series.dtype),
            'null_pct': col_series.isnull().mean(),
            'unique_ratio': col_series.nunique() / max(len(col_series), 1),
            'is_numeric': int(is_num),
            'outlier_ratio': outlier_ratio,
            'entropy': self._calculate_entropy(clean_col) if not is_num else 0,
            'baseline_feature_importance': baseline_importance if baseline_importance is not None else 0.0
        }
        
        y_numeric = self._ensure_numeric_target(y_target)
        
        if is_num:
            res.update({
                'skewness': skew(clean_col) if len(clean_col) > 2 else 0,
                'kurtosis': kurtosis(clean_col) if len(clean_col) > 2 else 0,
                'coeff_variation': (clean_col.std() / clean_col.mean()) if clean_col.mean() != 0 else 0,
                'zeros_ratio': (clean_col == 0).mean()
            })

            if len(clean_col) >= 3:
                try:
                    sample = clean_col.sample(min(5000, len(clean_col)), random_state=42)
                    res['shapiro_p_value'] = shapiro(sample)[1]
                except:
                    res['shapiro_p_value'] = 1.0
            else:
                res['shapiro_p_value'] = np.nan

            modes = clean_col.mode()
            res['has_multiple_modes'] = int(len(modes) > 1)
            res['bimodality_proxy_heuristic'] = int(res['kurtosis'] < -0.5 and abs(res['skewness']) < 0.5)
            res['range_iqr_ratio'] = (clean_col.max() - clean_col.min()) / iqr_val if iqr_val > 0 else 0.0

            if len(clean_col) > 0:
                q1v, q2v, q3v = np.percentile(clean_col, [25, 50, 75])
                q_intervals = [(clean_col.min(), q1v), (q1v, q2v), (q2v, q3v), (q3v, clean_col.max())]
                counts = [len(clean_col[(clean_col >= lo) & (clean_col <= hi)]) for lo, hi in q_intervals]
                res['dominant_quartile_pct'] = np.max(counts) / len(clean_col)
            else:
                res['dominant_quartile_pct'] = 0.0

            res['min_val'] = clean_col.min() if len(clean_col) > 0 else np.nan
            res['max_val'] = clean_col.max() if len(clean_col) > 0 else np.nan
            res['mean_val'] = clean_col.mean() if len(clean_col) > 0 else np.nan
            res['mode_val'] = modes.iloc[0] if len(modes) > 0 else np.nan
            res['pct_in_0_1_range'] = ((clean_col >= 0) & (clean_col <= 1)).mean() if len(clean_col) > 0 else 0.0

            if len(clean_col) > 0:
                try:
                    res['spearman_corr_target'] = spearmanr(clean_col, y_numeric.loc[clean_col.index]).correlation
                except:
                    res['spearman_corr_target'] = np.nan
            else:
                res['spearman_corr_target'] = np.nan

            if len(clean_col) >= 10:
                try:
                    _, dip_pval = diptest(clean_col.values)
                    res['hartigan_dip_pval'] = dip_pval
                    res['is_multimodal'] = int(dip_pval < 0.05)
                except:
                    res['hartigan_dip_pval'] = 1.0
                    res['is_multimodal'] = 0
            else:
                res['hartigan_dip_pval'] = 1.0
                res['is_multimodal'] = 0
            
            res['top_category_dominance'] = np.nan
            res['normalized_entropy'] = np.nan
            res['is_binary'] = 0
            res['is_low_cardinality'] = 0
            res['is_high_cardinality'] = 0
            res['top3_category_concentration'] = np.nan
            res['rare_category_pct'] = np.nan
            res['conditional_entropy'] = np.nan
        else:
            # Categorical
            if len(clean_col) > 0:
                res['top_category_dominance'] = clean_col.value_counts(normalize=True).iloc[0]
            else:
                res['top_category_dominance'] = 0.0

            nunique = col_series.nunique()
            if nunique > 1:
                res['normalized_entropy'] = res['entropy'] / np.log2(nunique)
            else:
                res['normalized_entropy'] = 0.0
            
            res['is_binary'] = int(nunique == 2)
            res['is_low_cardinality'] = int(2 < nunique < 10)
            res['is_high_cardinality'] = int(nunique >= 50 and nunique < len(col_series) * 0.5)

            if len(clean_col) > 0:
                res['top3_category_concentration'] = clean_col.value_counts(normalize=True).head(3).sum()
                freqs = clean_col.value_counts(normalize=True)
                res['rare_category_pct'] = freqs[freqs < 0.01].sum()
            else:
                res['top3_category_concentration'] = 0.0
                res['rare_category_pct'] = 0.0

            res['conditional_entropy'] = self._calculate_conditional_entropy(col_series, y_target)

            # Numeric-only -> NaN
            for k in ['shapiro_p_value', 'range_iqr_ratio', 'dominant_quartile_pct',
                       'min_val', 'max_val', 'mean_val', 'mode_val', 'pct_in_0_1_range',
                       'spearman_corr_target', 'hartigan_dip_pval', 'skewness', 'kurtosis',
                       'coeff_variation', 'zeros_ratio']:
                res[k] = np.nan
            res['has_multiple_modes'] = 0
            res['bimodality_proxy_heuristic'] = 0
            res['is_multimodal'] = 0
        
        col_name = col_series.name
        res['pps_score'] = self._pps_cache.get(col_name, self._calculate_pps(col_series, y_numeric))

        # MI
        if pd.api.types.is_numeric_dtype(col_series):
            mi_series = col_series.fillna(col_series.median()).to_frame()
        else:
            le = LabelEncoder()
            mi_series = pd.DataFrame(
                le.fit_transform(col_series.astype(str).fillna('NaN')),
                index=col_series.index, columns=[col_series.name or 'col']
            )
        try:
            if self.task_type == 'classification':
                res['mutual_information_score'] = mutual_info_classif(mi_series, y_numeric, random_state=42)[0]
            else:
                res['mutual_information_score'] = mutual_info_regression(mi_series, y_numeric, random_state=42)[0]
        except:
            res['mutual_information_score'] = 0.0

        return res

    # =========================================================================
    # EVALUATION METHODS
    # =========================================================================

    def _ensure_numeric_target(self, y):
        if pd.api.types.is_numeric_dtype(y):
            return y
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)

    def _prepare_data_for_model(self, X_train, X_val, y_train):
        """Convert all columns to model-ready dtypes after interventions."""
        for df_tmp in [X_train, X_val]:
            for c in df_tmp.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                df_tmp[c] = df_tmp[c].astype('int64') / 10**9
                df_tmp[c] = df_tmp[c].fillna(-1)
            for c in df_tmp.select_dtypes(include=['object', 'category', 'bool']).columns:
                df_tmp[c] = df_tmp[c].astype('category').cat.codes
        return X_train, X_val

    def _apply_intervention(self, X_train, X_val, y_train, col_to_transform, 
                            col_second=None, col_third=None, method=None):
        """
        Apply a single intervention to train/val splits.
        Operates on RAW data (before categorical encoding).
        
        Returns: (success: bool, X_train: DataFrame, X_val: DataFrame)
        Always returns the (possibly modified) DataFrames so that methods
        using pd.concat/drop (onehot, date_*, text_stats) propagate correctly.
        """
        if not method:
            return True, X_train, X_val
        
        # Methods that don't need a specific column
        if not col_to_transform and method not in ['row_stats', 'null_intervention']:
            return True, X_train, X_val
        
        y_train_numeric = self._ensure_numeric_target(y_train)
        
        try:
            if method == 'impute_median':
                median_val = X_train[col_to_transform].median()
                X_train[col_to_transform] = X_train[col_to_transform].fillna(median_val)
                X_val[col_to_transform] = X_val[col_to_transform].fillna(median_val)

            elif method == 'missing_indicator':
                col_na = f"{col_to_transform}_is_na"
                X_train[col_na] = X_train[col_to_transform].isnull().astype(int)
                X_val[col_na] = X_val[col_to_transform].isnull().astype(int)

            elif method == 'log_transform':
                fill_val = X_train[col_to_transform].min()
                temp_train = X_train[col_to_transform].fillna(fill_val)
                temp_val = X_val[col_to_transform].fillna(fill_val)
                offset = abs(temp_train.min()) + 1 if temp_train.min() <= 0 else 0
                X_train[col_to_transform] = np.log1p(temp_train + offset)
                X_val[col_to_transform] = np.log1p(temp_val + offset)

            elif method == 'frequency_encoding':
                str_train = X_train[col_to_transform].astype(str)
                str_val = X_val[col_to_transform].astype(str)
                freq = str_train.value_counts(normalize=True)
                X_train[col_to_transform] = str_train.map(freq).fillna(0).astype(float)
                X_val[col_to_transform] = str_val.map(freq).fillna(0).astype(float)
            
            elif method == 'target_encoding':
                str_train = X_train[col_to_transform].astype(str)
                str_val = X_val[col_to_transform].astype(str)
                global_mean = float(y_train_numeric.mean())
                agg = y_train_numeric.groupby(str_train).agg(['count', 'mean'])
                smooth = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
                X_train[col_to_transform] = str_train.map(smooth).fillna(global_mean).astype(float)
                X_val[col_to_transform] = str_val.map(smooth).fillna(global_mean).astype(float)
            
            elif method == 'product_interaction' and col_second:
                med_a = X_train[col_to_transform].median()
                med_b = X_train[col_second].median()
                new_col = f"{col_to_transform}_x_{col_second}"
                X_train[new_col] = X_train[col_to_transform].fillna(med_a) * X_train[col_second].fillna(med_b)
                X_val[new_col] = X_val[col_to_transform].fillna(med_a) * X_val[col_second].fillna(med_b)

            elif method == 'division_interaction' and col_second:
                eps = 1e-5
                med_a = X_train[col_to_transform].median()
                med_b = X_train[col_second].median()
                new_col = f"{col_to_transform}_div_{col_second}"
                X_train[new_col] = X_train[col_to_transform].fillna(med_a) / (X_train[col_second].fillna(med_b) + eps)
                X_val[new_col] = X_val[col_to_transform].fillna(med_a) / (X_val[col_second].fillna(med_b) + eps)

            elif method == 'addition_interaction' and col_second:
                med_a = X_train[col_to_transform].median()
                med_b = X_train[col_second].median()
                new_col = f"{col_to_transform}_plus_{col_second}"
                X_train[new_col] = X_train[col_to_transform].fillna(med_a) + X_train[col_second].fillna(med_b)
                X_val[new_col] = X_val[col_to_transform].fillna(med_a) + X_val[col_second].fillna(med_b)

            elif method == 'subtraction_interaction' and col_second:
                med_a = X_train[col_to_transform].median()
                med_b = X_train[col_second].median()
                new_col = f"{col_to_transform}_minus_{col_second}"
                X_train[new_col] = X_train[col_to_transform].fillna(med_a) - X_train[col_second].fillna(med_b)
                X_val[new_col] = X_val[col_to_transform].fillna(med_a) - X_val[col_second].fillna(med_b)

            elif method == 'abs_diff_interaction' and col_second:
                med_a = X_train[col_to_transform].median()
                med_b = X_train[col_second].median()
                new_col = f"{col_to_transform}_absdiff_{col_second}"
                X_train[new_col] = (X_train[col_to_transform].fillna(med_a) - X_train[col_second].fillna(med_b)).abs()
                X_val[new_col] = (X_val[col_to_transform].fillna(med_a) - X_val[col_second].fillna(med_b)).abs()

            elif method in ['group_mean', 'group_std']:
                cat_str_train = X_train[col_to_transform].astype(str)
                cat_str_val = X_val[col_to_transform].astype(str)
                
                if col_second:
                    num_series = X_train[col_second]
                    if method == 'group_mean':
                        grp_map = num_series.groupby(cat_str_train).mean()
                        fill_val = float(num_series.mean())
                    else:
                        grp_map = num_series.groupby(cat_str_train).std()
                        fill_val = float(num_series.std())
                else:
                    if method == 'group_mean':
                        grp_map = y_train_numeric.groupby(cat_str_train).mean()
                        fill_val = float(y_train_numeric.mean())
                    else:
                        grp_map = y_train_numeric.groupby(cat_str_train).std()
                        fill_val = float(y_train_numeric.std())
                
                new_col = f"{method}_{col_second or 'target'}_by_{col_to_transform}"
                X_train[new_col] = cat_str_train.map(grp_map).fillna(fill_val).astype(float)
                X_val[new_col] = cat_str_val.map(grp_map).fillna(fill_val).astype(float)

            elif method == 'onehot_encoding':
                if X_train[col_to_transform].nunique() <= 10:
                    str_train = X_train[col_to_transform].astype(str)
                    str_val = X_val[col_to_transform].astype(str)
                    train_dummies = pd.get_dummies(str_train, prefix=col_to_transform, drop_first=True)
                    val_dummies = pd.get_dummies(str_val, prefix=col_to_transform, drop_first=True)
                    val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
                    X_train = X_train.drop(columns=[col_to_transform])
                    X_val = X_val.drop(columns=[col_to_transform])
                    X_train = pd.concat([X_train, train_dummies], axis=1)
                    X_val = pd.concat([X_val, val_dummies], axis=1)

            elif method == 'quantile_binning':
                n_bins = 5
                _, bin_edges = pd.qcut(X_train[col_to_transform].dropna(), q=n_bins, retbins=True, duplicates='drop')
                X_train[col_to_transform] = pd.cut(X_train[col_to_transform], bins=bin_edges, labels=False, include_lowest=True)
                X_val[col_to_transform] = pd.cut(X_val[col_to_transform], bins=bin_edges, labels=False, include_lowest=True)
                X_train[col_to_transform] = X_train[col_to_transform].fillna(n_bins // 2).astype(float)
                X_val[col_to_transform] = X_val[col_to_transform].fillna(n_bins // 2).astype(float)

            elif method == 'polynomial_square':
                med_val = X_train[col_to_transform].median()
                new_col = f"{col_to_transform}_squared"
                X_train[new_col] = X_train[col_to_transform].fillna(med_val) ** 2
                X_val[new_col] = X_val[col_to_transform].fillna(med_val) ** 2


            elif method == 'polynomial_cube':
                med_val = X_train[col_to_transform].median()
                new_col = f"{col_to_transform}_cubed"
                filled_train = X_train[col_to_transform].fillna(med_val)
                filled_val = X_val[col_to_transform].fillna(med_val)
                X_train[new_col] = filled_train ** 3
                X_val[new_col] = filled_val ** 3

            elif method == 'abs_transform':
                med_val = X_train[col_to_transform].median()
                new_col = f"{col_to_transform}_abs"
                X_train[new_col] = X_train[col_to_transform].fillna(med_val).abs()
                X_val[new_col] = X_val[col_to_transform].fillna(med_val).abs()

            elif method == 'exp_transform':
                # Standardize-then-exp: self-calibrating, no magic clip constants
                filled_train = X_train[col_to_transform].fillna(X_train[col_to_transform].median())
                train_mean = filled_train.mean()
                train_std = filled_train.std()
                if train_std < 1e-10:
                    train_std = 1.0  # constant column fallback
                z_train = (filled_train - train_mean) / train_std
                X_train[col_to_transform] = np.exp(z_train)
                
                filled_val = X_val[col_to_transform].fillna(X_train[col_to_transform].median())
                z_val = (filled_val - train_mean) / train_std  # use TRAIN stats
                X_val[col_to_transform] = np.exp(z_val)

            elif method == 'reciprocal_transform':
                filled_train = X_train[col_to_transform].fillna(X_train[col_to_transform].median())
                filled_val = X_val[col_to_transform].fillna(X_train[col_to_transform].median())
                # Sign-aware epsilon: preserves sign, avoids division by zero
                eps = 1e-6
                X_train[col_to_transform] = 1.0 / (filled_train + np.sign(filled_train) * eps)
                X_val[col_to_transform] = 1.0 / (filled_val + np.sign(filled_val) * eps)
                # Handle exact zeros: map to 0 (neutral, doesn't blow up)
                X_train[col_to_transform] = X_train[col_to_transform].replace([np.inf, -np.inf], 0.0)
                X_val[col_to_transform] = X_val[col_to_transform].replace([np.inf, -np.inf], 0.0)

            elif method == 'sqrt_transform':
                temp_train = X_train[col_to_transform].fillna(X_train[col_to_transform].min())
                temp_val = X_val[col_to_transform].fillna(X_train[col_to_transform].min())
                offset = abs(temp_train.min()) if temp_train.min() < 0 else 0
                X_train[col_to_transform] = np.sqrt(temp_train + offset)
                X_val[col_to_transform] = np.sqrt(temp_val + offset)

            elif method == 'cat_concat' and col_second:
                new_col = f"{col_to_transform}_concat_{col_second}"
                X_train[new_col] = X_train[col_to_transform].astype(str) + "_" + X_train[col_second].astype(str)
                X_val[new_col] = X_val[col_to_transform].astype(str) + "_" + X_val[col_second].astype(str)
                global_mean = float(y_train_numeric.mean())
                agg = y_train_numeric.groupby(X_train[new_col]).agg(['count', 'mean'])
                smooth = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
                X_train[new_col] = X_train[new_col].map(smooth).fillna(global_mean).astype(float)
                X_val[new_col] = X_val[new_col].map(smooth).fillna(global_mean).astype(float)

            elif method == 'date_extract_basic':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform].astype(str), errors='coerce'))]:
                    df_x[f"{col_to_transform}_year"] = src.dt.year.fillna(-1).astype(int)
                    df_x[f"{col_to_transform}_month"] = src.dt.month.fillna(-1).astype(int)
                    df_x[f"{col_to_transform}_day"] = src.dt.day.fillna(-1).astype(int)
                    df_x[f"{col_to_transform}_dayofweek"] = src.dt.dayofweek.fillna(-1).astype(int)
                    df_x[f"{col_to_transform}_is_weekend"] = (src.dt.dayofweek >= 5).astype(int)
                    if src.dt.hour.notna().any():
                        df_x[f"{col_to_transform}_hour"] = src.dt.hour.fillna(-1).astype(int)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_month':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')),
                                   (X_val, pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce'))]:
                    m = src.dt.month.fillna(0)
                    df_x[f"{col_to_transform}_month_sin"] = np.sin(2 * np.pi * m / 12)
                    df_x[f"{col_to_transform}_month_cos"] = np.cos(2 * np.pi * m / 12)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_dow':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform].astype(str), errors='coerce'))]:
                    d = src.dt.dayofweek.fillna(0)
                    df_x[f"{col_to_transform}_dow_sin"] = np.sin(2 * np.pi * d / 7)
                    df_x[f"{col_to_transform}_dow_cos"] = np.cos(2 * np.pi * d / 7)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_hour':
                dt_train = pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')
                dt_val = pd.to_datetime(X_val[col_to_transform].astype(str), errors='coerce')
                if dt_train.dt.hour.notna().any():
                    for df_x, src in [(X_train, dt_train), (X_val, dt_val)]:
                        h = src.dt.hour.fillna(0)
                        df_x[f"{col_to_transform}_hour_sin"] = np.sin(2 * np.pi * h / 24)
                        df_x[f"{col_to_transform}_hour_cos"] = np.cos(2 * np.pi * h / 24)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_elapsed_days':
                dt_train = pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')
                dt_val = pd.to_datetime(X_val[col_to_transform].astype(str), errors='coerce')
                min_date = dt_train.min()
                if pd.notna(min_date):
                    X_train[f"{col_to_transform}_days_elapsed"] = (dt_train - min_date).dt.total_seconds().fillna(-86400) / 86400
                    X_val[f"{col_to_transform}_days_elapsed"] = (dt_val - min_date).dt.total_seconds().fillna(-86400) / 86400
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_day':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform].astype(str), errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform].astype(str), errors='coerce'))]:
                    d = src.dt.day.fillna(0)
                    df_x[f"{col_to_transform}_day_sin"] = np.sin(2 * np.pi * d / 31)
                    df_x[f"{col_to_transform}_day_cos"] = np.cos(2 * np.pi * d / 31)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'cyclical_encode':
                # Generic cyclical encoding for standalone temporal columns
                # (month, hour, day_of_week, etc.). Detects period from column metadata.
                # ADDS sin/cos columns -- keeps original for tree histogram splits.
                temporal_type, period = self._detect_temporal_component(
                    X_train[col_to_transform], col_to_transform)
                if period is None:
                    return False, X_train, X_val
                # Coerce to numeric (handles string-encoded temporal columns like "1","2",...)
                num_train = pd.to_numeric(X_train[col_to_transform], errors='coerce')
                num_val = pd.to_numeric(X_val[col_to_transform], errors='coerce')
                med_val = num_train.median()
                vals_train = num_train.fillna(med_val)
                vals_val = num_val.fillna(med_val)
                X_train[f"{col_to_transform}_sin"] = np.sin(2 * np.pi * vals_train / period)
                X_train[f"{col_to_transform}_cos"] = np.cos(2 * np.pi * vals_train / period)
                X_val[f"{col_to_transform}_sin"] = np.sin(2 * np.pi * vals_val / period)
                X_val[f"{col_to_transform}_cos"] = np.cos(2 * np.pi * vals_val / period)

            elif method == 'hashing_encoding':
                X_train[col_to_transform] = X_train[col_to_transform].astype(str).apply(lambda x: hash(x) % 100)
                X_val[col_to_transform] = X_val[col_to_transform].astype(str).apply(lambda x: hash(x) % 100)

            elif method == 'text_stats':
                for df_x in [X_train, X_val]:
                    s = df_x[col_to_transform].astype(str).replace("nan", "").fillna("")
                    df_x[f"{col_to_transform}_len"] = s.str.len()
                    df_x[f"{col_to_transform}_word_count"] = s.apply(lambda x: len(x.split()))
                    df_x[f"{col_to_transform}_digit_count"] = s.str.count(r'\d')
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'row_stats':
                num_cols = X_train.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    for df_x in [X_train, X_val]:
                        df_x['row_missing_count'] = df_x.isnull().sum(axis=1)
                        df_x['row_missing_ratio'] = df_x.isnull().mean(axis=1)
                        df_x['row_mean'] = df_x[num_cols].mean(axis=1).fillna(0)
                        df_x['row_std'] = df_x[num_cols].std(axis=1).fillna(0)
                        df_x['row_sum'] = df_x[num_cols].sum(axis=1).fillna(0)
                        df_x['row_min'] = df_x[num_cols].min(axis=1).fillna(0)
                        df_x['row_max'] = df_x[num_cols].max(axis=1).fillna(0)
                        df_x['row_range'] = df_x['row_max'] - df_x['row_min']
                        df_x['row_zeros_count'] = (df_x[num_cols] == 0).sum(axis=1)
                        df_x['row_zeros_ratio'] = (df_x[num_cols] == 0).mean(axis=1)

            elif method == 'three_way_interaction' and col_second and col_third:
                new_col = f"{col_to_transform}_x_{col_second}_x_{col_third}"
                m1 = X_train[col_to_transform].median()
                m2 = X_train[col_second].median()
                m3 = X_train[col_third].median()
                X_train[new_col] = (X_train[col_to_transform].fillna(m1) * 
                                    X_train[col_second].fillna(m2) * 
                                    X_train[col_third].fillna(m3))
                X_val[new_col] = (X_val[col_to_transform].fillna(m1) * 
                                  X_val[col_second].fillna(m2) * 
                                  X_val[col_third].fillna(m3))

            elif method == 'three_way_addition' and col_second and col_third:
                new_col = f"{col_to_transform}_plus_{col_second}_plus_{col_third}"
                m1 = X_train[col_to_transform].median()
                m2 = X_train[col_second].median()
                m3 = X_train[col_third].median()
                X_train[new_col] = (X_train[col_to_transform].fillna(m1) + 
                                    X_train[col_second].fillna(m2) + 
                                    X_train[col_third].fillna(m3))
                X_val[new_col] = (X_val[col_to_transform].fillna(m1) + 
                                  X_val[col_second].fillna(m2) + 
                                  X_val[col_third].fillna(m3))

            elif method == 'three_way_ratio' and col_second and col_third:
                # (a * b) / c -- useful for rate-like derived features
                eps = 1e-5
                new_col = f"{col_to_transform}_x_{col_second}_div_{col_third}"
                m1 = X_train[col_to_transform].median()
                m2 = X_train[col_second].median()
                m3 = X_train[col_third].median()
                X_train[new_col] = ((X_train[col_to_transform].fillna(m1) * 
                                     X_train[col_second].fillna(m2)) /
                                    (X_train[col_third].fillna(m3) + eps))
                X_val[new_col] = ((X_val[col_to_transform].fillna(m1) * 
                                   X_val[col_second].fillna(m2)) /
                                  (X_val[col_third].fillna(m3) + eps))

            elif method == 'three_way_normalized_diff' and col_second and col_third:
                # (a - b) / c -- normalized difference, e.g. (income - expenses) / household_size
                # Directional: order of a, b, c matters
                eps = 1e-5
                new_col = f"{col_to_transform}_minus_{col_second}_div_{col_third}"
                m1 = X_train[col_to_transform].median()
                m2 = X_train[col_second].median()
                m3 = X_train[col_third].median()
                X_train[new_col] = ((X_train[col_to_transform].fillna(m1) - 
                                     X_train[col_second].fillna(m2)) /
                                    (X_train[col_third].fillna(m3) + eps))
                X_val[new_col] = ((X_val[col_to_transform].fillna(m1) - 
                                   X_val[col_second].fillna(m2)) /
                                  (X_val[col_third].fillna(m3) + eps))

            elif method == 'null_intervention':
                pass

            return True, X_train, X_val
        except Exception as e:
            self.log_error(f"Intervention {method} on {col_to_transform} failed", e)
            return False, X_train, X_val

    def _get_cv_chunks(self, n_rows):
        """
        Determine chunked CV strategy based on dataset size.
        
        For large datasets, instead of 3x5-fold repeated CV on all rows,
        we split data into disjoint chunks and run 5-fold CV on each chunk.
        This gives the same number of fold estimates but each fold trains
        on ~1/n_chunks of the data, yielding significant speedup.
        
        Returns:
            n_chunks: 0 = use standard RepeatedKFold; >0 = use disjoint chunks
        """
        if n_rows > 500_000:
            return 3  # 3 chunks x 5-fold = 15 folds, each on ~167K rows
        elif n_rows > 100_000:
            return 2  # 2 chunks x 5-fold = 10 folds, each on ~50K+ rows
        else:
            return 0  # Standard 3x5-fold repeated CV on all data

    def evaluate_with_intervention(self, X, y, col_to_transform=None, col_second=None, 
                                   col_third=None, method=None, return_importance=False,
                                   use_individual_params=False):
        """Runs K-Fold CV with optional repeated CV or chunked CV for large datasets."""
        scores = []
        params = self.individual_params if use_individual_params else self.base_params
        feat_imp_accum = None
        y = self._ensure_numeric_target(y)

        # FIX: Calculate n_classes globally from the full target, not the fold
        n_classes_global = y.nunique()

        # v8: Chunked CV strategy for large datasets
        n_chunks = self._get_cv_chunks(len(X))
        
        if n_chunks > 0:
            # Disjoint chunk strategy: split data into n_chunks disjoint subsets,
            # run 5-fold CV on each. Gives same number of estimates but faster.
            chunk_size = len(X) // n_chunks
            shuffled_indices = np.random.RandomState(42).permutation(len(X))
            cv_splits = []
            for chunk_i in range(n_chunks):
                start = chunk_i * chunk_size
                end = start + chunk_size if chunk_i < n_chunks - 1 else len(X)
                chunk_idx = shuffled_indices[start:end]
                X_chunk = X.iloc[chunk_idx]
                cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42 + chunk_i)
                for train_rel, val_rel in cv.split(X_chunk):
                    # Map relative indices back to absolute indices
                    train_abs = chunk_idx[train_rel]
                    val_abs = chunk_idx[val_rel]
                    cv_splits.append((train_abs, val_abs))
        elif self.n_repeats > 1:
            cv = RepeatedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=42)
            cv_splits = list(cv.split(X))
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X))

        fold_idx = 0
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # For classification: ensure y_val only contains labels seen in y_train
            if self.task_type == 'classification':
                train_classes = set(y_train.unique())
                unseen_mask = ~y_val.isin(train_classes)
                if unseen_mask.any():
                    # Drop unseen-class rows from validation
                    keep = ~unseen_mask
                    X_val = X_val.loc[keep]
                    y_val = y_val.loc[keep]
                    if len(y_val) < 2:
                        scores.append(np.nan)
                        fold_idx += 1
                        continue

            # Apply intervention -- unpack (success, X_train, X_val) tuple
            # so that methods using pd.concat/drop (OHE, dates, text_stats)
            # correctly propagate their modified DataFrames
            if col_to_transform and method:
                success, X_train, X_val = self._apply_intervention(X_train, X_val, y_train, col_to_transform, col_second, col_third, method)
                if not success:
                    return (None, None, None, None) if return_importance else (None, None, None)
            elif method in ['row_stats']:
                _, X_train, X_val = self._apply_intervention(X_train, X_val, y_train, col_to_transform=None, method=method)
            elif method == 'null_intervention':
                pass  # Do nothing

            X_train, X_val = self._prepare_data_for_model(X_train, X_val, y_train)
            
            if fold_idx == 0 and return_importance:
                feat_imp_accum = np.zeros(X_train.shape[1])

            try:
                if self.task_type == 'classification':
                    
                    p = params.copy()
                    if n_classes_global > 2:
                        p.update({'objective': 'multiclass', 'num_class': n_classes_global, 'metric': 'multi_logloss'})
                        metric_name = 'multi_logloss'
                    else:
                        p.update({'objective': 'binary', 'metric': 'auc'})
                        metric_name = 'auc'

                    model = lgb.LGBMClassifier(**p)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              eval_metric=metric_name,
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    probs = model.predict_proba(X_val, num_iteration=model.best_iteration_)
                    if n_classes_global > 2:
                        # FIX: Handle cases where fold might miss a class in y_val, 
                        # but we still need robustness. 
                        # Ideally, ensure y is 0..N-1 encoded globally before this method.
                        try:
                            scores.append(metrics.roc_auc_score(y_val, probs, multi_class='ovr', labels=list(range(n_classes_global))))
                        except ValueError:
                             # Fallback for extreme edge cases where y_val only has 1 class
                            scores.append(np.nan)
                    else:
                        scores.append(metrics.roc_auc_score(y_val, probs[:, 1]))
                else:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              eval_metric='l2',
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    preds = model.predict(X_val, num_iteration=model.best_iteration_)
                    # v7: Use R^2 as primary regression score (scale-invariant, higher=better)
                    scores.append(metrics.r2_score(y_val, preds))
                
                if return_importance and feat_imp_accum is not None:
                    imp = model.feature_importances_
                    if len(imp) == len(feat_imp_accum):
                        feat_imp_accum += imp
            except Exception as e:
                self.log_error(f"Model training failed fold {fold_idx}", e)
                return (None, None, None, None) if return_importance else (None, None, None)
            
            fold_idx += 1

        n_total = self.n_folds * self.n_repeats
        if return_importance:
            return np.mean(scores), np.std(scores), scores, feat_imp_accum / n_total
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        if not valid_scores:
            return (None, None, None, None) if return_importance else (None, None, None)
        return np.mean(valid_scores), np.std(valid_scores), scores

    def evaluate_individual(self, X_subset, y, col_to_transform=None, col_second=None,
                           col_third=None, method=None):
        """Evaluate using ONLY the specified column subset with simpler params."""
        return self.evaluate_with_intervention(
            X_subset, y, col_to_transform=col_to_transform,
            col_second=col_second, col_third=col_third,
            method=method, use_individual_params=True)

    def _get_individual_baseline(self, X, y, cols, cache_key=None):
        """Get individual baseline for column(s), with caching."""
        if len(self._individual_baseline_cache) > 256:
            keys = list(self._individual_baseline_cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._individual_baseline_cache[k]
        
        if cache_key and cache_key in self._individual_baseline_cache:
            return self._individual_baseline_cache[cache_key]
        score, std, fold_scores = self.evaluate_individual(X[cols], y)
        if cache_key and score is not None:
            self._individual_baseline_cache[cache_key] = (score, std, fold_scores)
        return score, std, fold_scores

    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================

    def _paired_ttest_with_bonferroni(self, baseline_scores, intervention_scores, n_tests):
        if baseline_scores is None or intervention_scores is None:
            return np.nan, np.nan, np.nan, False, False
        if np.allclose(baseline_scores, intervention_scores):
            return 0.0, 1.0, 1.0, False, False
        try:
            t_stat, p_value = ttest_rel(intervention_scores, baseline_scores)
            if np.isnan(p_value):
                return 0.0, 1.0, 1.0, False, False
            p_bonf = min(p_value * n_tests, 1.0)
            return t_stat, p_value, p_bonf, p_value < 0.05, p_bonf < 0.05
        except:
            return np.nan, np.nan, np.nan, False, False

    def _get_delta(self, base_score, new_score):
        # Both AUC (classification) and R^2 (regression) are higher-is-better
        return new_score - base_score

    def _normalize_delta(self, delta, base_score):
        # FIX 3 (naming clarification): Despite the name "normalized", this simply
        # converts the raw delta to percentage points (x100). It is NOT normalized by
        # the headroom or any other reference. The properly calibrated signal is
        # calibrated_delta (= delta / null_std), which is the preferred meta-model target.
        # delta_normalized is retained for schema compatibility with existing CSV files.
        return delta * 100

    def _check_ceiling_effect(self, baseline_score):
        if self.task_type == 'classification':
            if baseline_score >= 0.97:
                return True, False, f"ceiling_hard (AUC={baseline_score:.5f} >= 0.97)"
            elif baseline_score >= 0.93:
                return False, True, f"ceiling_soft (AUC={baseline_score:.5f} >= 0.93)"
        else:
            # R^2 is primary regression metric: 1.0 = perfect, higher = better
            if baseline_score >= 0.995:
                return True, False, f"ceiling_hard (R^2={baseline_score:.5f} >= 0.995)"
            elif baseline_score >= 0.98:
                return False, True, f"ceiling_soft (R^2={baseline_score:.5f} >= 0.98)"
        return False, False, "no_ceiling"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_linearity_gap(self, X, y):
        X_proc = X.copy()
        for col in X_proc.columns:
            if pd.api.types.is_numeric_dtype(X_proc[col]):
                X_proc[col] = X_proc[col].fillna(X_proc[col].median())
            else:
                X_proc[col] = X_proc[col].astype('category').cat.codes
        if len(X_proc) > 100000:
            np.random.seed(42)
            idx = np.random.choice(len(X_proc), 100000, replace=False)
            X_s, y_s = X_proc.iloc[idx], y.iloc[idx]
        else:
            X_s, y_s = X_proc, y
        try:
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            lin_scores, tree_scores = [], []
            for tr, vl in kf.split(X_s):
                Xtr, Xvl = X_s.iloc[tr], X_s.iloc[vl]
                ytr, yvl = y_s.iloc[tr], y_s.iloc[vl]
                if self.task_type == 'classification' and len(np.unique(ytr)) == 2:
                    lin = LogisticRegression(max_iter=100, random_state=42); lin.fit(Xtr, ytr)
                    lin_scores.append(roc_auc_score(yvl, lin.predict_proba(Xvl)[:, 1]))
                    tree = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42); tree.fit(Xtr, ytr)
                    tree_scores.append(roc_auc_score(yvl, tree.predict_proba(Xvl)[:, 1]))
                elif self.task_type == 'regression':
                    from sklearn.metrics import r2_score
                    lin = LinearRegression(); lin.fit(Xtr, ytr)
                    lin_scores.append(r2_score(yvl, lin.predict(Xvl)))
                    tree = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42); tree.fit(Xtr, ytr)
                    tree_scores.append(r2_score(yvl, tree.predict(Xvl)))
                else:
                    return 0.0
            # Both metrics: positive gap = tree beats linear = non-linear dataset
            return (np.mean(tree_scores) - np.mean(lin_scores)) 
        except:
            return 0.0

    def _calculate_correlation_graph_metrics(self, X_numeric):
        try:
            corr = X_numeric.corr().abs()
            G = nx.Graph(); G.add_nodes_from(corr.columns)
            for i, a in enumerate(corr.columns):
                for j, b in enumerate(corr.columns):
                    if i < j and corr.iloc[i, j] > 0.5:
                        G.add_edge(a, b)
            return {
                'corr_graph_components': nx.number_connected_components(G),
                'corr_graph_clustering': nx.average_clustering(G) if len(G.nodes) > 0 else 0.0,
                'corr_graph_density': nx.density(G) if len(G.nodes) > 1 else 0.0
            }
        except:
            return {'corr_graph_components': 0, 'corr_graph_clustering': 0.0, 'corr_graph_density': 0.0}

    def _calculate_matrix_rank_ratio(self, X_numeric):
        try:
            X_f = X_numeric.fillna(X_numeric.median())
            if X_f.shape[0] > 100000: X_f = X_f.sample(100000, random_state=42)
            if X_f.shape[1] > 1000: X_f = X_f.iloc[:, :10000]  # Cap columns for speed
            return np.linalg.matrix_rank(X_f.values) / min(X_f.shape) if min(X_f.shape) > 0 else 0.0
        except:
            return 0.0

    def _calculate_entropy(self, series):
        p = series.value_counts(normalize=True)
        return -(p * np.log2(p)).sum()

    def _calculate_pps(self, feature, target):
        try:
            if pd.api.types.is_numeric_dtype(feature):
                X = feature.fillna(feature.median()).values.reshape(-1, 1)
            else:
                le = LabelEncoder()
                X = le.fit_transform(feature.astype(str).fillna('NaN')).reshape(-1, 1)
            valid = feature.notna()
            X, y = X[valid], target[valid]
            if len(X) < 10: return 0.0    
            if self.task_type == 'classification':
                scores = cross_val_score(
                    DecisionTreeClassifier(max_depth=3, random_state=42),
                    X, y, cv=3, scoring='roc_auc', error_score=0.0
                )
                return max(0.0, float(np.mean(scores) * 2 - 1))
            else:
                scores = cross_val_score(
                    DecisionTreeRegressor(max_depth=3, random_state=42),
                    X, y, cv=3, scoring='r2', error_score=0.0
                )
                return max(0.0, float(np.mean(scores)))
        except:
            return 0.0

    def _calculate_conditional_entropy(self, feature, target):
        try:
            if pd.api.types.is_numeric_dtype(feature):
                fb = pd.qcut(feature.dropna(), q=10, duplicates='drop', labels=False)
            else:
                fb = feature.dropna()
            ci = fb.index.intersection(target.index)
            fb, ta = fb.loc[ci], target.loc[ci]
            if len(fb) < 10: return 0.0
            joint = pd.crosstab(fb, ta)
            jp = joint / joint.sum().sum()
            je = -(jp * np.log2(jp + 1e-10)).sum().sum()
            tp = ta.value_counts(normalize=True)
            te = -(tp * np.log2(tp + 1e-10)).sum()
            return max(0, je - te)
        except:
            return 0.0

    def _calculate_vif_for_features(self, X_numeric):
        vif = {}
        try:
            X_f = X_numeric.fillna(X_numeric.median())
            if len(X_f) > 100000: X_f = X_f.sample(100000, random_state=42)
            for i, col in enumerate(X_f.columns):
                try:
                    v = variance_inflation_factor(X_f.values, i)
                    vif[col] = v if not np.isinf(v) else 999.0
                except:
                    vif[col] = 0.0
        except:
            pass
        return vif

    def _is_likely_date(self, series, col_name):
        if pd.api.types.is_datetime64_any_dtype(series): return True
        if pd.api.types.is_numeric_dtype(series): return False
        date_kw = ['date', 'time', 'datetime', 'timestamp', 'dt']
        if any(kw in col_name.lower() for kw in date_kw):
            try:
                sample = series.dropna().head(100)
                if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().sum() / len(sample) > 0.8:
                    return True
            except: pass
        return False

    # Patterns for detecting standalone temporal component columns.
    # Each entry: (regex_pattern, value_range_check, temporal_type, cyclical_period)
    TEMPORAL_PATTERNS = [
        # Month: 1-12
        (r'(?i)(?:^|[_\W])(?:month|mon(?:th)?|mo)(?:$|[_\W\d])',
         lambda mn, mx: 1 <= mn and mx <= 12, 'month', 12),
        # Day of week: 0-6 or 1-7
        (r'(?i)(?:^|[_\W])(?:dayofweek|day_of_week|dow|weekday|wday|day_of_wk)(?:$|[_\W\d])',
         lambda mn, mx: (0 <= mn and mx <= 6) or (1 <= mn and mx <= 7), 'day_of_week', 7),
        # Hour: 0-23
        (r'(?i)(?:^|[_\W])(?:hour|hr|hh)(?:$|[_\W\d])',
         lambda mn, mx: 0 <= mn and mx <= 23, 'hour', 24),
        # Minute: 0-59 (require full word 'minute' -- 'min' is too ambiguous: min_temperature, min_val)
        (r'(?i)(?:^|[_\W])(?:minute)(?:$|[_\W\d])',
         lambda mn, mx: 0 <= mn and mx <= 59, 'minute', 60),
        # Second: 0-59 (require full word 'second' -- 'sec' is too ambiguous: section, security)
        (r'(?i)(?:^|[_\W])(?:second)(?:$|[_\W\d])',
         lambda mn, mx: 0 <= mn and mx <= 59, 'second', 60),
        # Day of month: 1-31
        (r'(?i)(?:^|[_\W])(?:day(?:ofmonth|_of_month)?)(?:$|[_\W\d])',
         lambda mn, mx: 1 <= mn and mx <= 31, 'day_of_month', 31),
        # Quarter: 1-4
        (r'(?i)(?:^|[_\W])(?:quarter|qtr|q)(?:$|[_\W\d])',
         lambda mn, mx: 1 <= mn and mx <= 4, 'quarter', 4),
        # Week of year: 1-53
        (r'(?i)(?:^|[_\W])(?:week(?:ofyear|_of_year)?|wk|isoweek)(?:$|[_\W\d])',
         lambda mn, mx: 1 <= mn and mx <= 53, 'week', 53),
    ]

    def _detect_temporal_component(self, series, col_name):
        """
        Detect if a numeric column is a standalone temporal component
        (e.g., 'month', 'hour', 'day_of_week') that should use cyclical encoding.
        
        Requirements:
        1. Column is numeric
        2. Column name matches a temporal keyword pattern
        3. Value range is consistent with the temporal type
        4. Values are integer-like (no fractional months/hours)
        
        Returns: (temporal_type: str, period: int) or (None, None)
        
        temporal_type values: 'month', 'day_of_week', 'hour', 'minute', 'second',
                              'day_of_month', 'quarter', 'week'
        period: cyclical period for sin/cos encoding (e.g., 12 for month)
        """
        # If not already numeric, try coercing (handles "1","2","3" strings)
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors='coerce')
            if coerced.notna().sum() < 0.9 * series.notna().sum():
                return None, None
            series = coerced
        
        clean = series.dropna()
        if len(clean) < 10:
            return None, None
        
        # Must be integer-valued (no fractional months)
        try:
            if not (clean % 1 == 0).all():
                return None, None
        except:
            return None, None
        
        min_v, max_v = float(clean.min()), float(clean.max())
        
        for pattern, range_check, temp_type, period in self.TEMPORAL_PATTERNS:
            if re.search(pattern, col_name) and range_check(min_v, max_v):
                return temp_type, period
        
        return None, None

    def _run_landmarking_model(self, X, y):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        X_p = X.copy()
        for col in X_p.columns:
            if pd.api.types.is_numeric_dtype(X_p[col]):
                X_p[col] = X_p[col].fillna(X_p[col].median())
            else:
                X_p[col] = X_p[col].astype('category').cat.codes
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
        return np.mean(scores)

    def _is_likely_id_or_constant(self, series, col_name, target):
        clean = series.dropna()
        if len(clean) == 0: return False, True, "all_null"
        if series.nunique(dropna=True) <= 1: return False, True, "constant"
        if pd.api.types.is_datetime64_any_dtype(series): return False, False, "datetime"
        score = 0
        ur = series.nunique(dropna=True) / len(clean)
        if ur > 0.98: score += 60
        elif ur > 0.85: score += 40
        elif ur > 0.50: score += 10
        if re.search(r'(?i)(\b|_)id(\b|_)', col_name): score += 30
        if pd.api.types.is_float_dtype(series):
            score += -40 if not np.all(clean % 1 == 0) else -10
        elif pd.api.types.is_integer_dtype(series): score += 10
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
            try:
                c = abs(series.corr(target))
                if c > 0.05: score -= 50
                if c > 0.20: score -= 100
            except: pass
        return (score > 50, False, "id_detected") if score > 50 else (False, False, "normal")

    # =========================================================================
    # PROPERTY-GATED METHOD SELECTION + ADAPTIVE EARLY STOP
    # =========================================================================

    # Methods that have property gates for LightGBM
    # LightGBM is invariant to monotonic transforms, so log/sqrt/square only help when
    # they improve histogram bin resolution (extreme distributions) or add non-trivial signal.
    GATED_METHODS = {'log_transform', 'sqrt_transform', 'quantile_binning', 'polynomial_square',
                     'polynomial_cube', 'abs_transform', 'exp_transform', 'reciprocal_transform'}

    # Methods where individual (single-column) evaluation is structurally uninformative.
    # 
    # For a single feature, LightGBM's histogram-based splitting evaluates ALL possible 
    # threshold splits. Any injective re-encoding of categories (frequency, target, hash) 
    # or monotonic transform of numerics (log, sqrt) produces the SAME optimal tree because 
    # the row-to-bin mapping is equivalent -- just with different split thresholds.
    #
    # These methods DO help in full-model evaluation because the value ordering affects 
    # how the column interacts with splits on OTHER features in the tree.
    #
    # We skip individual eval for these to avoid:
    # 1. Wasting compute on guaranteed p=1.0 results  
    # 2. Polluting the meta-DB with meaningless zeros
    #
    # Methods that ADD columns (polynomial_square, missing_indicator, interactions) are NOT 
    # in this set because the individual model gains an extra feature, which IS meaningful.
    INDIVIDUAL_INVARIANT_METHODS = {
        'frequency_encoding',   # re-encodes cat -> frequency floats (injective)
        'target_encoding',      # re-encodes cat -> target-smoothed floats (injective with smoothing)
        'hashing_encoding',     # re-encodes cat -> hash integers (injective mod collisions)
        'log_transform',        # monotonic transform -> same histogram partitions
        'sqrt_transform',       # monotonic transform -> same histogram partitions
        'quantile_binning',     # reduces unique values, but LightGBM already bins optimally
        'exp_transform',        #  monotonic â†’ same histogram partitions
        'reciprocal_transform', # monotonic per sign â†’ same partitions
    }

    @staticmethod
    def _safe_get(meta, key, default=0.0):
        """Get a value from meta dict, replacing None/NaN with default."""
        val = meta.get(key)
        if val is None:
            return default
        try:
            if np.isnan(val):
                return default
        except (TypeError, ValueError):
            pass
        return val

    def _should_test_method(self, method, col_meta):
        """
        Property-gated check: should we test this method on this column?
        
        v4 tightened gates -- log/sqrt/poly/binning rarely help with LightGBM:
        - log_transform: only when |skewness| > 3 AND (outlier_ratio > 0.20 OR range_iqr > 12)
          (need BOTH extreme shape AND extreme range for histogram resolution gain)
        - sqrt_transform: only when skewness > 2.5 AND min_val >= 0 AND outlier > 0.10
        - quantile_binning: only when outlier > 0.30 AND range_iqr > 15 AND is_multimodal
          (almost never helps -- only test with very strong evidence of pathological distribution)
        - polynomial_square: only when |spearman_corr_target| > 0.25 AND (is_multimodal OR kurtosis < -1)
          (need evidence of non-linear relationship worth capturing)
        
        Returns: (should_test: bool, reason: str)
        """
        # Check adaptive early-stop first (cheapest check)
        if method in self._method_tracker:
            if self._method_tracker[method].get('disabled', False) and np.random.random() > 0.10:
                return False, "adaptive_early_stop"
        
        # Property gates (only for gated methods)
        if method in self.GATED_METHODS:
            skew = self._safe_get(col_meta, 'skewness', 0.0)
            outlier = self._safe_get(col_meta, 'outlier_ratio', 0.0)
            min_val = self._safe_get(col_meta, 'min_val', 0.0)
            range_iqr = self._safe_get(col_meta, 'range_iqr_ratio', 0.0)
            spearman = self._safe_get(col_meta, 'spearman_corr_target', 0.0)
            is_multimodal = self._safe_get(col_meta, 'is_multimodal', 0)
            kurtosis_val = self._safe_get(col_meta, 'kurtosis', 0.0)
            
            if method == 'log_transform':
                # v4: require BOTH extreme skew AND outlier/range evidence
                # Also require non-negative values -- log of a shifted negative column
                # changes the semantic meaning and is never what we want.
                if not (abs(skew) > 3 and (outlier > 0.20 or range_iqr > 12) and min_val >= 0):
                    return False, "property_gate: |skew|<=3 or (outlier<=0.20 and range_iqr<=12) or min<0"
                    
            elif method == 'sqrt_transform':
                # v4: stricter -- need strong skew + non-negative + some outliers
                if not (skew > 2.5 and min_val >= 0 and outlier > 0.10):
                    return False, "property_gate: skew<=2.5 or min<0 or outlier<=0.10"
                    
            elif method == 'quantile_binning':
                # v4: extremely strict -- almost never helps with LightGBM
                # Only test for pathological distributions with all three indicators
                if not (outlier > 0.30 and range_iqr > 15 and is_multimodal):
                    return False, "property_gate: need outlier>0.30 AND range_iqr>15 AND multimodal"
                    
            elif method == 'polynomial_square':
                # v4: need evidence of non-linear relationship worth capturing
                if not (abs(spearman) > 0.25 and (is_multimodal or kurtosis_val < -1)):
                    return False, "property_gate: |spearman|<=0.25 or no non-linearity evidence"

            elif method == 'polynomial_cube':
                # Cube is useful when there's a strong non-linear (odd-powered) relationship
                # More aggressive than square -- need strong evidence
                if not (abs(spearman) > 0.30 and kurtosis_val > 2):
                    return False, "property_gate: |spearman|<=0.30 or kurtosis<=2 (need strong non-linearity)"

            elif method == 'abs_transform':
                # Abs is only useful when the column has negative values -- otherwise it's a no-op.
                # min_val is already extracted above from col_meta via _safe_get.
                if min_val >= 0:
                    return False, "property_gate: min_val >= 0, abs_transform is a no-op on non-negative columns"

            elif method == 'exp_transform':
                # Exp amplifies differences in the upper range.
                # Useful when: distribution is left-skewed or compressed (low kurtosis)
                # and there's correlation signal to amplify
                if not (abs(spearman) > 0.15 and skew < 1.5):
                    return False, "property_gate: |spearman|<=0.15 or skew>=1.5 (exp would amplify already-skewed)"

            elif method == 'reciprocal_transform':
                # Reciprocal flips the ordering and is useful for rate-like features
                # (e.g., time_to_X -> speed). Need positive values and some correlation.
                # Use _safe_get so NaN zeros_ratio does not silently block the method.
                zeros_ratio = self._safe_get(col_meta, 'zeros_ratio', 0.0)
                if not (abs(spearman) > 0.15 and zeros_ratio < 0.05):
                    return False, "property_gate: |spearman|<=0.15 or too many zeros for reciprocal"
        
        return True, "ok"

    def _update_method_tracker(self, method, improved, n_selected_cols):
        """
        Update tracking for adaptive early-stop.
        
        v4: Differentiated thresholds:
        - Rarely-useful methods (log, sqrt, poly_square, quantile_binning):
          After 5 columns with 0 improvements -> disable.
        - Other methods: After 20% of columns (min 10) with 0 improvements -> disable.
        """
        if method not in self._method_tracker:
            self._method_tracker[method] = {'tested': 0, 'improved': 0, 'disabled': False}
        
        tracker = self._method_tracker[method]
        tracker['tested'] += 1
        if improved:
            tracker['improved'] += 1
        
        # Aggressive early-stop for methods that rarely help
        FAST_STOP_METHODS = {'log_transform', 'sqrt_transform', 'polynomial_square', 'quantile_binning',
                             'polynomial_cube', 'abs_transform', 'exp_transform', 'reciprocal_transform'}
        if method in FAST_STOP_METHODS:
            min_trials = 5
        else:
            min_trials = max(10, int(n_selected_cols * 0.2))
        
        if tracker['tested'] >= min_trials and tracker['improved'] == 0:
            tracker['disabled'] = True
            self.log(f"    [!] Adaptive early-stop: {method} disabled after {tracker['tested']} "
                     f"columns with 0 improvements", "INFO")

    # =========================================================================
    # v5: TIME BUDGET CHECK
    # =========================================================================

    def _time_remaining(self):
        """Check if we still have time budget remaining. Returns seconds left."""
        if self._start_time is None:
            return float('inf')
        elapsed = time.time() - self._start_time
        return max(0, self.time_budget_seconds - elapsed)

    def _is_over_budget(self):
        """Returns True if we've exceeded the time budget for this dataset."""
        return self._time_remaining() <= 0

    # =========================================================================
    # v5: NOISE-FLOOR CALIBRATION HELPERS
    # =========================================================================

    def _compute_cohens_d(self, baseline_folds, intervention_folds):
        """
        Compute Cohen's d effect size between two sets of fold scores.
        
        Cohen's d = (mean_intervention - mean_baseline) / pooled_std
        
        Both AUC (classification) and R^2 (regression) are higher-is-better,
        so positive d always means improvement.
        """
        if baseline_folds is None or intervention_folds is None:
            return np.nan
        b = np.array(baseline_folds)
        i = np.array(intervention_folds)
        if len(b) < 2 or len(i) < 2:
            return np.nan
        
        pooled_std = np.sqrt((np.var(b, ddof=1) + np.var(i, ddof=1)) / 2)
        if pooled_std < 1e-15:
            return 0.0
        
        raw_d = (np.mean(i) - np.mean(b)) / pooled_std
        return float(raw_d)

    def _compute_calibrated_delta(self, delta):
        """
        Calibrate a raw delta against the CV noise floor (null_std = std of baseline folds).

        calibrated_delta = (delta - null_delta) / null_std

        Interpretation: how many CV-standard-deviations above the baseline variance
        the observed delta is. 2.0 means the intervention gained 2x the natural fold
        noise of this dataset.

        null_delta is always 0.0 (v8: null = baseline itself, no systematic bias).
        null_std = std(baseline_fold_scores), clamped at 0.001 minimum.

        Returns np.nan only if delta is NaN or null_std is effectively zero (< 1e-6).
        The clamped minimum of 0.001 means calibrated_delta is almost always finite.
        """
        if np.isnan(delta) or self._null_std < 1e-6:
            return np.nan
        return float((delta - self._null_delta) / self._null_std)

    # =========================================================================
    # v5: TREE-GUIDED INTERACTION PAIR SELECTION
    # =========================================================================

    def _get_tree_guided_pairs(self, X, y, max_depth=4, n_trees=5):
        """
        Fit multiple shallow decision trees and extract parent->child split pairs
        as interaction candidates. This identifies CONDITIONAL DEPENDENCIES between
        features, which is exactly what interaction features capture.
        
        How it works:
        - A decision tree split on column A at the root, then column B in a child,
          means "B's optimal split depends on where you are in A's range"
        - This is a textbook case for an A*B or A-B interaction feature
        - We fit multiple trees with different seeds for robustness
        - Pairs are scored by sum of impurity decrease at the child node
        
        Returns: list of (col_a, col_b, score) sorted by score descending.
                 Also returns three-way candidates from grandparent->parent->child chains.
        """
        pair_scores = defaultdict(float)
        triple_scores = defaultdict(float)
        
        # Prepare data: encode categoricals, fill NaN
        X_enc = X.copy()
        col_names = list(X_enc.columns)
        for c in X_enc.columns:
            if pd.api.types.is_numeric_dtype(X_enc[c]):
                X_enc[c] = X_enc[c].fillna(X_enc[c].median())
            else:
                X_enc[c] = X_enc[c].astype('category').cat.codes
        
        # Subsample for speed on large datasets
        if len(X_enc) > 100000:
            X_enc = X_enc.sample(100000, random_state=42)
            y_sample = y.loc[X_enc.index]
        else:
            y_sample = y
        
        y_enc = self._ensure_numeric_target(y_sample)
        
        for seed in range(n_trees):
            try:
                if self.task_type == 'classification':
                    tree = DecisionTreeClassifier(
                        max_depth=max_depth, random_state=seed,
                        min_samples_leaf=max(20, len(X_enc) // 100))
                else:
                    tree = DecisionTreeRegressor(
                        max_depth=max_depth, random_state=seed,
                        min_samples_leaf=max(20, len(X_enc) // 100))
                
                tree.fit(X_enc, y_enc)
                
                # Extract tree structure arrays
                t = tree.tree_
                n_nodes = t.node_count
                feature_idx = t.feature       # feature index at each node (-2 for leaf)
                children_left = t.children_left
                children_right = t.children_right
                impurity = t.impurity
                n_samples = t.n_node_samples
                
                # Build parent map
                parent = np.full(n_nodes, -1, dtype=int)
                for node_id in range(n_nodes):
                    left = children_left[node_id]
                    right = children_right[node_id]
                    if left != -1:  # Not a leaf
                        parent[left] = node_id
                        parent[right] = node_id
                
                # Walk all internal nodes and record parent->child pairs
                for node_id in range(n_nodes):
                    if feature_idx[node_id] < 0:
                        continue  # Leaf node, skip
                    
                    parent_id = parent[node_id]
                    if parent_id < 0:
                        continue  # Root node has no parent
                    if feature_idx[parent_id] < 0:
                        continue  # Parent is leaf (shouldn't happen, but safety)
                    
                    parent_feat = col_names[feature_idx[parent_id]]
                    child_feat = col_names[feature_idx[node_id]]
                    
                    if parent_feat == child_feat:
                        continue  # Same feature split at different thresholds
                    
                    # Score by impurity decrease at child node, weighted by samples
                    # Impurity decrease ~ parent_impurity * parent_samples - sum(child_impurity * child_samples)
                    sample_weight = n_samples[node_id] / n_samples[0]  # Fraction of data reaching this node
                    gain = impurity[parent_id] - impurity[node_id]
                    score = max(0, gain) * sample_weight
                    
                    pair_scores[(parent_feat, child_feat)] += score
                    
                    # Also check grandparent for three-way candidates
                    grandparent_id = parent[parent_id]
                    if grandparent_id >= 0 and feature_idx[grandparent_id] >= 0:
                        gp_feat = col_names[feature_idx[grandparent_id]]
                        if gp_feat != parent_feat and gp_feat != child_feat:
                            triple_scores[(gp_feat, parent_feat, child_feat)] += score * 0.5
                
            except Exception as e:
                self.log(f"  Tree-guided pair extraction (seed={seed}) failed: {e}", "WARNING")
                continue
        
        # Merge symmetric pairs: (A,B) and (B,A) -> combined score
        merged_pairs = defaultdict(float)
        for (a, b), score in pair_scores.items():
            key = tuple(sorted([a, b]))
            merged_pairs[key] += score
        
        # Sort by score
        ranked_pairs = sorted(merged_pairs.items(), key=lambda x: x[1], reverse=True)
        two_way = [(a, b, s) for (a, b), s in ranked_pairs]
        
        # Merge symmetric triples
        merged_triples = defaultdict(float)
        for (a, b, c), score in triple_scores.items():
            key = tuple(sorted([a, b, c]))
            merged_triples[key] += score
        ranked_triples = sorted(merged_triples.items(), key=lambda x: x[1], reverse=True)
        three_way = [(a, b, c, s) for (a, b, c), s in ranked_triples]
        
        return two_way, three_way

    # =========================================================================
    # v7: FEATUREWIZ COLUMN RANKING
    # =========================================================================

    def _featurewiz_rank_columns(self, X, y, max_rows=10000):
        """
        Use FeatureWiz's SULOV (Searching for Uncorrelated List Of Variables) +
        recursive XGBoost to identify which features carry independent predictive
        signal.

        Returns:
            selected_set: set of column names FeatureWiz kept
            importance_dict: {col_name: relative_importance (0-1)} for ALL columns
                             (selected columns get their rank-based importance,
                              dropped columns get 0.0)

        This is used to:
        1. Boost composite_predictive_score for selected columns -> better
           column selection and interaction pool prioritization
        2. Populate featurewiz_selected / featurewiz_importance in CSV rows
           so the meta-model can learn "FWiz-approved columns benefit more from X"
        3. Pre-filter columns: skip expensive transforms on columns FeatureWiz
           confidently dropped

        Runtime: ~2-15s on typical datasets (fast enough for data collection).
        """
        if not _HAS_FEATUREWIZ:
            return set(), {}

        self.log("Running FeatureWiz column ranking (SULOV + XGBoost)...")
        t0 = time.time()

        try:
            # Prepare a clean copy for FeatureWiz
            X_fw = X.copy()

            # FeatureWiz needs string column names
            X_fw.columns = [str(c) for c in X_fw.columns]

            # Subsample large datasets for speed
            if len(X_fw) > max_rows:
                sample_idx = X_fw.sample(max_rows, random_state=42).index
                X_fw = X_fw.loc[sample_idx]
                y_fw = y.loc[sample_idx]
            else:
                y_fw = y.copy()

            # Encode non-numeric columns so FeatureWiz can handle them
            for col in X_fw.columns:
                if not pd.api.types.is_numeric_dtype(X_fw[col]):
                    X_fw[col] = X_fw[col].astype(str).astype('category').cat.codes

            # Fill NaN (FeatureWiz may choke on NaN in some paths)
            X_fw = X_fw.fillna(X_fw.median(numeric_only=True))

            # Cast all columns to float32 to avoid dtype issues
            # (e.g. uint8 columns reject fill_value=-99 inside lazytransform)
            for col in X_fw.columns:
                if pd.api.types.is_numeric_dtype(X_fw[col]):
                    X_fw[col] = X_fw[col].astype(np.float32)

            # Run FeatureWiz
            # feature_engg='' means no feature generation -- just selection
            out = fwiz.FeatureWiz(
                corr_limit=self.featurewiz_corr_limit,
                feature_engg='',
                nrows=len(X_fw),
                verbose=0,
            )
            X_selected = out.fit_transform(X_fw, y_fw)


            # Some FeatureWiz versions return (X, y) tuple from fit_transform
            if isinstance(X_selected, tuple):
                X_selected = X_selected[0]

            selected_cols = set(X_selected.columns)
            n_selected = len(selected_cols)

            # Build importance dict: selected columns get rank-based score
            # Rank by position in FeatureWiz output (earlier = more important)
            importance = {}
            for i, col in enumerate(X_selected.columns):
                importance[col] = max(0.0, 1.0 - (i / max(n_selected, 1)))

            # Non-selected columns get 0.0
            for col in X.columns:
                col_str = str(col)
                if col_str not in importance:
                    importance[col_str] = 0.0

            elapsed = time.time() - t0
            self.log(f"  FeatureWiz selected {n_selected}/{X.shape[1]} columns in {elapsed:.1f}s")
            if n_selected > 0:
                top5 = list(X_selected.columns[:5])
                self.log(f"  Top 5 selected: {top5}")

            return selected_cols, importance

        except Exception as e:
            self.log(f"  FeatureWiz failed: {e}", "WARNING")
            self.log_error("FeatureWiz column ranking failed", exception=e)
            return set(), {}

    # =========================================================================
    # v7: AUTOFEAT INTERACTION PAIR DISCOVERY
    # =========================================================================

    @staticmethod
    def _parse_autofeat_feature_name(feat_name, known_columns):
        """
        Parse an AutoFeat-generated feature name to extract the source columns
        and operation type.

        AutoFeat uses symbolic names like:
            'col_a**2'           -> single column, polynomial
            'col_a * col_b'     -> two columns, product
            'log(col_a)'        -> single column, log
            'exp(col_a)'        -> single column, exp
            'col_a / col_b'     -> two columns, division
            'col_a + col_b'     -> two columns, addition
            'col_a - col_b'     -> two columns, subtraction
            'abs(col_a)'        -> single column, abs
            'sqrt(col_a)'       -> single column, sqrt
            '1/(col_a)'         -> single column, reciprocal

        Returns:
            dict with keys:
                'columns': list of column names involved
                'operation': str describing the operation
                'n_columns': number of columns involved
            or None if parsing fails.
        """
        known_set = set(str(c) for c in known_columns)
        # Sort by length descending so longer column names match first
        # (prevents partial matches like 'col' matching inside 'col_extended')
        sorted_cols = sorted(known_set, key=len, reverse=True)

        # Find all known column names present in the feature string
        found_cols = []
        remaining = feat_name
        for col in sorted_cols:
            if col in remaining:
                found_cols.append(col)
                # Replace to avoid double-matching
                remaining = remaining.replace(col, '@', 1)

        if not found_cols:
            return None

        # Determine operation from remaining symbols
        op = 'unknown'
        if '**2' in feat_name or '^2 ' in feat_name:
            op = 'polynomial_square'
        elif '**3' in feat_name:
            op = 'polynomial_cube'
        elif ' * ' in feat_name or '*' in remaining:
            op = 'product_interaction'
        elif ' / ' in feat_name or '/' in remaining:
            op = 'division_interaction'
        elif ' + ' in feat_name or '+' in remaining:
            op = 'addition_interaction'
        elif ' - ' in feat_name or '-' in remaining:
            op = 'subtraction_interaction'
        elif feat_name.startswith('log(') or 'log(' in feat_name:
            op = 'log_transform'
        elif feat_name.startswith('sqrt(') or 'sqrt(' in feat_name:
            op = 'sqrt_transform'
        elif feat_name.startswith('abs(') or 'abs(' in feat_name:
            op = 'abs_transform'

        return {
            'columns': found_cols,
            'operation': op,
            'n_columns': len(found_cols),
        }

    def _autofeat_discover_pairs(self, X, y, max_rows=5000, max_cols=30):
        """
        Use AutoFeat to discover which feature pairs (and singles) have
        predictive polynomial/interaction signal.

        AutoFeat generates O(n^2) candidate features from polynomial and
        interaction terms, then uses L1 (Lasso) selection to prune down to
        only the predictive ones. We parse the surviving feature names back
        into column pairs.

        Returns:
            discovered_pairs: list of (col_a, col_b, operation) tuples
                              for 2-column interactions
            discovered_singles: list of (col, operation) tuples
                                for single-column transforms AutoFeat found useful

        These are fed into the interaction testing pipeline as additional
        candidates alongside tree-guided and importance-based pairs.

        Runtime: ~10-60s depending on dataset size. Capped by max_rows and
        max_cols to stay within the time budget on large datasets.
        """
        if not _HAS_AUTOFEAT:
            return [], []

        self.log("Running AutoFeat interaction discovery...")
        t0 = time.time()

        try:
            # Select only numeric columns (AutoFeat operates on numerics)
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] < 2:
                self.log("  AutoFeat: fewer than 2 numeric columns, skipping")
                return [], []

            # Cap columns to top-N by importance for tractability
            if X_numeric.shape[1] > max_cols:
                # Use existing composite scores if available, else variance
                if self._composite_scores_cache:
                    col_scores = {c: self._composite_scores_cache.get(str(c), 0)
                                  for c in X_numeric.columns}
                else:
                    col_scores = {c: X_numeric[c].var() for c in X_numeric.columns}
                top_cols = sorted(col_scores, key=col_scores.get, reverse=True)[:max_cols]
                X_numeric = X_numeric[top_cols]
                self.log(f"  AutoFeat: capped to top {max_cols} numeric columns")

            # Subsample rows for speed
            if len(X_numeric) > max_rows:
                sample_idx = X_numeric.sample(max_rows, random_state=42).index
                X_af = X_numeric.loc[sample_idx].copy()
                y_af = y.loc[sample_idx].copy()
            else:
                X_af = X_numeric.copy()
                y_af = y.copy()

            # Fill NaN
            X_af = X_af.fillna(X_af.median())

            # Ensure clean column names (AutoFeat can struggle with special chars)
            col_name_map = {}
            clean_cols = []
            for c in X_af.columns:
                clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(c))
                if clean in clean_cols:
                    clean = f"{clean}_{len(clean_cols)}"
                col_name_map[clean] = str(c)
                clean_cols.append(clean)
            X_af.columns = clean_cols
            known_clean_cols = set(clean_cols)

            # Run AutoFeat
            if self.task_type == 'classification':
                af = AutoFeatClassifier(
                    feateng_steps=self.autofeat_feateng_steps,
                    max_gb=self.autofeat_max_gb,
                    n_jobs=1,  # Avoid nested parallelism with LightGBM
                    verbose=0,
                )
            else:
                af = AutoFeatRegressor(
                    feateng_steps=self.autofeat_feateng_steps,
                    max_gb=self.autofeat_max_gb,
                    n_jobs=1,
                    verbose=0,
                )

            af.fit(X_af, y_af)

            # Parse discovered features
            new_feat_cols = getattr(af, 'new_feat_cols_', [])
            if not new_feat_cols:
                elapsed = time.time() - t0
                self.log(f"  AutoFeat: no new features discovered ({elapsed:.1f}s)")
                return [], []

            discovered_pairs = []
            discovered_singles = []

            for feat_name in new_feat_cols:
                parsed = self._parse_autofeat_feature_name(feat_name, known_clean_cols)
                if parsed is None:
                    continue

                # Map clean names back to original column names
                orig_cols = []
                for c in parsed['columns']:
                    orig = col_name_map.get(c, c)
                    orig_cols.append(orig)

                if parsed['n_columns'] == 2:
                    discovered_pairs.append((orig_cols[0], orig_cols[1], parsed['operation']))
                elif parsed['n_columns'] == 1:
                    discovered_singles.append((orig_cols[0], parsed['operation']))

            # Deduplicate pairs (treat commutative ops as same)
            seen_pairs = set()
            unique_pairs = []
            COMMUTATIVE_OPS = {'product_interaction', 'addition_interaction'}
            for a, b, op in discovered_pairs:
                if op in COMMUTATIVE_OPS:
                    key = (op, tuple(sorted([a, b])))
                else:
                    key = (op, a, b)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    unique_pairs.append((a, b, op))

            elapsed = time.time() - t0
            self.log(f"  AutoFeat discovered {len(unique_pairs)} pairs + "
                     f"{len(discovered_singles)} singles in {elapsed:.1f}s")
            if unique_pairs:
                self.log(f"  Top pairs: {unique_pairs[:5]}")
            if discovered_singles:
                self.log(f"  Singles: {discovered_singles[:5]}")

            return unique_pairs, discovered_singles

        except Exception as e:
            elapsed = time.time() - t0
            self.log(f"  AutoFeat failed after {elapsed:.1f}s: {e}", "WARNING")
            self.log_error("AutoFeat interaction discovery failed", exception=e)
            return [], []

    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================

    def collect(self, df, target_col, dataset_name="unknown"):
        """
        Main collection method (v4).
        
        Pipeline:
        1. ID detection -> drop
        2. Strategic pruning (constant, near-constant, zero-var, r>0.99) -> drop
        3. Target leakage detection (corr/MI/PPS > 0.95 with target) -> drop
        4. Baseline evaluation on full remaining set
        5. Ceiling check
        6. Composite predictive scoring (importance + MI + PPS)
        7. Stratified column selection (cap = 100 + 10%)
        8. For each selected column: univariate interventions + default interaction with top columns
           Both full-model AND individual evaluation with p-values for each
        9. Cross-column interactions (product, division, addition, subtraction, abs_diff,
           3-way multiply/add/ratio, group-by, cat-concat)
           Individual evaluation = model on raw column pair/triple vs. pair/triple + interaction
        10. Row-level statistics (mean, std, sum, min, max, range, zeros, missing)
        """
        self.log(f"{'=' * 80}")
        self.log(f"Starting analysis: {dataset_name}")
        self.log(f"{'=' * 80}")
        
        self.n_tests_performed = 0
        self._individual_baseline_cache = {}
        self._method_tracker = {}  # Reset adaptive early-stop per dataset
        self._start_time = time.time()  # v5: time budget tracking
        self._null_delta = 0.0  # v5: reset noise floor
        self._null_std = 1e-4
        self._cv_strategy = 'unknown'  # Fix 5: set after baseline eval
        # v6: Reset column metadata caches per dataset
        self._column_meta_cache = {}
        self._composite_scores_cache = {}
        # v7: Reset FeatureWiz caches per dataset
        self._featurewiz_selected = set()
        self._featurewiz_importance = {}
        # Phase 0.1: Deduplication -- track (method, col_a, col_b) tuples already tested
        # For commutative ops (product, addition, abs_diff), sort col names so AxB == BxA
        self._tested_interactions = set()
        
        # 1. Separate target
        y = df[target_col]
        X_raw = df.drop(columns=[target_col])
        y = self._ensure_numeric_target(y)
        
        nan_mask = y.isna()
        if nan_mask.any():
            n_nan = nan_mask.sum()
            self.log(f"Dropping {n_nan} rows with NaN target ({n_nan/len(y)*100:.1f}%)")
            X_raw = X_raw.loc[~nan_mask]
            y = y.loc[~nan_mask]
            if len(y) < 50:
                self.log("ERROR: Too few rows after dropping NaN targets!", "ERROR")
                return pd.DataFrame()

        self.log(f"Raw shape: {X_raw.shape}")

        # 2. ID detection
        drop_cols = []
        for col in X_raw.columns:
            is_id, is_const, reason = self._is_likely_id_or_constant(X_raw[col], col, y)
            if is_id or is_const:
                drop_cols.append((col, reason))
        if drop_cols:
            self.log(f"Dropping {len(drop_cols)} ID/constant columns")
            X = X_raw.drop(columns=[c for c, _ in drop_cols])
        else:
            X = X_raw.copy()

        # 3. Strategic pruning
        prune_list = self._strategic_column_pruning(X, y)
        if prune_list:
            self.log(f"Strategic pruning: {len(prune_list)} columns")
            X = X.drop(columns=[c for c, _ in prune_list])

        # 4. Target leakage detection
        leaky = self._detect_target_leakage(X, y)
        if leaky:
            self.log(f"[!]   Target leakage detected in {len(leaky)} columns:")
            for col, reason, score in leaky:
                self.log(f"  - {col}: {reason}")
            X = X.drop(columns=[c for c, _, _ in leaky])

        if X.shape[1] == 0:
            self.log("ERROR: No features left!", "ERROR")
            return pd.DataFrame()

        self.log(f"Shape after cleanup: {X.shape}")

        # 5. Baseline evaluation (on ALL remaining columns)
        self.log("Running baseline evaluation (full model)...")
        base_score, base_std, baseline_fold_scores, baseline_importances = \
            self.evaluate_with_intervention(X, y, return_importance=True)
        

        if base_score is None or np.isnan(base_score):
            self.log("ERROR: Baseline failed (score is None or NaN)!", "ERROR")
            return pd.DataFrame()
        
        self.log(f"Baseline: {base_score:.5f} (std: {base_std:.5f})")
        
        # Fix 5: record CV strategy used for this dataset so the meta-model can
        # condition on it (chunked CV trains on fewer rows -> lower scores).
        _n_chunks = self._get_cv_chunks(len(X))
        if _n_chunks > 0:
            _cv_strategy = f"chunked_{_n_chunks}"
        elif self.n_repeats > 1:
            _cv_strategy = f"repeated_{self.n_folds}fold_x{self.n_repeats}"
        else:
            _cv_strategy = f"kfold_{self.n_folds}"
        self._cv_strategy = _cv_strategy  # Store for use in ds_meta below

        # 6. Ceiling check
        should_skip, is_near_ceiling, ceiling_reason = self._check_ceiling_effect(base_score)
        if should_skip:
            self.log(f"[!]   SKIPPING: {ceiling_reason}", "WARNING")
            return pd.DataFrame()
        
        # 7. Dataset metadata (computed on full set BEFORE column selection)
        ds_meta = self.get_dataset_meta(X, y)
        ds_meta['near_ceiling_flag'] = is_near_ceiling
        ds_meta['baseline_score'] = base_score
        ds_meta['baseline_std'] = base_std
        # v5: Initialize noise floor defaults (updated by null intervention if successful)
        ds_meta['null_delta'] = 0.0
        ds_meta['null_std'] = 1e-8

        base_feature_importance = pd.Series(baseline_importances, index=X.columns)

        # Phase 2.1: Aggregate importance features
        # These give the meta-model context about how "spread" the importance is,
        # which helps predict when row_stats and interactions will help.
        imp_vals = base_feature_importance.values
        if len(imp_vals) > 0 and imp_vals.max() > 0:
            imp_norm = imp_vals / imp_vals.max()
            ds_meta['std_feature_importance'] = float(np.std(imp_norm))
            ds_meta['max_minus_min_importance'] = float(imp_norm.max() - imp_norm.min())
            median_imp = np.median(imp_norm)
            ds_meta['pct_features_above_median_importance'] = float((imp_norm > median_imp).mean())
        else:
            ds_meta['std_feature_importance'] = 0.0
            ds_meta['max_minus_min_importance'] = 0.0
            ds_meta['pct_features_above_median_importance'] = 0.5
        
        # Phase 2.5: relative_headroom -- how much room for improvement exists
        # Both AUC (classification) and R^2 (regression) are bounded at 1.0
        ds_meta['relative_headroom'] = max(1.0 - base_score, 0.001)

        # Fix 4 & 5: record CV strategy and model complexity context in every row
        ds_meta['cv_strategy'] = self._cv_strategy
        # individual_model_complexity is set per-row in _validate_and_append calls;
        # we set a dataset-level default on ds_meta here so rows that don't override
        # it (e.g. null calibration, row_stats) get 'standard'.
        ds_meta['individual_model_complexity'] = 'standard'
        # Fix 6: record the primary metric so the meta-model can always condition on it.
        # AUC and R^2 have fundamentally different distributions and ranges.
        ds_meta['primary_metric'] = 'roc_auc' if self.task_type == 'classification' else 'r2'

        # =====================================================================
        # v7: FEATUREWIZ COLUMN PRE-RANKING
        # =====================================================================
        # Run FeatureWiz SULOV before column selection to identify which columns
        # carry independent predictive signal. Results are blended into
        # composite_predictive_score to improve column selection + interaction
        # pool prioritization. Also stored per-column for meta-model training.
        if self.use_featurewiz and not self._is_over_budget():
            fwiz_selected, fwiz_importance = self._featurewiz_rank_columns(X, y)
            self._featurewiz_selected = fwiz_selected
            self._featurewiz_importance = fwiz_importance
        else:
            if self.use_featurewiz:
                self.log("  FeatureWiz skipped (time budget)", "WARNING")
            self._featurewiz_selected = set()
            self._featurewiz_importance = {}

        # 8. Stratified column selection
        selected_cols, selection_report = self._select_columns_stratified(X, y, base_feature_importance)
        ds_meta['n_cols_before_selection'] = X.shape[1]
        ds_meta['n_cols_selected'] = len(selected_cols)
        ds_meta['column_selection_method'] = selection_report['method']
        
        self.log(f"Selected {len(selected_cols)} / {X.shape[1]} columns for analysis")
        
        # The full model still uses ALL columns for evaluation
        # Column selection only determines WHICH columns we test interventions on
        X_full = X  # Full feature set for full-model evaluation
        
        self._X_full_ref = X_full  # for binary detection in _filter_methods_for_binary

        # Store reference for pairwise correlation lookups in _log_interaction_result
        self._last_X_ref = X_full
        
        # 9. VIF (only on selected numeric columns, up to 100 for speed)
        selected_numeric = [c for c in selected_cols if pd.api.types.is_numeric_dtype(X[c])]
        if len(selected_numeric) > 1:
            vif_cols = selected_numeric[:100]
            self.log(f"Calculating VIF for {len(vif_cols)} numeric columns...")
            vif_scores = self._calculate_vif_for_features(X[vif_cols])
        else:
            vif_scores = {}

        # 10. Identify TOP-2 globally most predictive columns (for default interactions)
        composite_scores = self._compute_column_predictive_scores(X, y, base_feature_importance)
        
        # v7: Blend FeatureWiz signal into composite scores.
        # FeatureWiz uses a fundamentally different methodology (SULOV + recursive
        # XGBoost) which captures complementary signal to the existing importance +
        # MI + PPS composite. Blending gives us a 4th perspective.
        # Weight: 20% FeatureWiz, 80% original composite. This is additive --
        # FeatureWiz can only boost a column, never suppress it below the original.
        if self._featurewiz_importance:
            fwiz_blend_weight = 0.20
            n_blended = 0
            for col in composite_scores:
                col_str = str(col)
                fwiz_imp = self._featurewiz_importance.get(col_str, 0.0)
                if fwiz_imp > 0:
                    original = composite_scores[col]
                    composite_scores[col] = (1 - fwiz_blend_weight) * original + fwiz_blend_weight * fwiz_imp
                    n_blended += 1
            if n_blended > 0:
                self.log(f"  Blended FeatureWiz signal into {n_blended} column scores")
        
        all_cols_ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top columns: prefer numeric for interactions, but include any type
        top_global_cols = [col for col, _ in all_cols_ranked[:2]]
        self.log(f"Top global columns for default interactions: {top_global_cols}")

        # =====================================================================
        # v8: COLUMN SAMPLING -- reduce per-column individual analysis
        # =====================================================================
        # For datasets with many columns, we don't need to individually test
        # every selected column. We guarantee the top-K by composite score and
        # sample the rest, weighted by importance. Unsampled columns still
        # participate in the full model and as interaction partners -- they just
        # don't get their own univariate transform tests.
        MAX_INDIVIDUAL_COLS = 60  # hard cap for per-column individual analysis
        ALWAYS_KEEP = 20  # top-N by composite score are always analyzed
        
        if len(selected_cols) > MAX_INDIVIDUAL_COLS:
            ranked_cols = sorted(selected_cols,
                                 key=lambda c: composite_scores.get(c, 0), reverse=True)
            guaranteed = ranked_cols[:ALWAYS_KEEP]
            remaining_pool = ranked_cols[ALWAYS_KEEP:]
            
            n_sample = MAX_INDIVIDUAL_COLS - ALWAYS_KEEP
            if remaining_pool and n_sample > 0:
                weights = np.array([max(composite_scores.get(c, 0.01), 0.01) 
                                    for c in remaining_pool])
                weights = weights / weights.sum()
                sampled = list(np.random.RandomState(42).choice(
                    remaining_pool, size=min(n_sample, len(remaining_pool)),
                    replace=False, p=weights))
            else:
                sampled = []
            
            cols_for_individual = set(guaranteed) | set(sampled)
            self.log(f"Column sampling: {len(selected_cols)} selected, "
                     f"{len(cols_for_individual)} for individual analysis "
                     f"(top-{ALWAYS_KEEP} guaranteed + {len(sampled)} sampled)")
        else:
            cols_for_individual = set(selected_cols)

        # 11. NOISE FLOOR CALIBRATION (v8: derived from baseline fold variance)
        #
        # Previous approach: re-run the model with method='null_intervention' (which does nothing)
        # and compare fold scores. Because LightGBM is fully deterministic with random_state=42
        # and identical CV splits, the "null" scores are bit-for-bit identical to baseline, so
        # fold_diffs = 0 and null_std is always clamped to the 0.005 constant -- making
        # calibrated_delta and cohens_d effectively meaningless (just delta / 0.005).
        #
        # v8 fix: use the std of the baseline fold scores directly as the noise floor.
        # std(baseline_fold_scores) is the natural CV variance of this model on this dataset
        # -- exactly what we want: calibrated_delta = delta / (expected fold-to-fold noise).
        # A calibrated_delta of 2.0 means "2 CV-std-deviations above the baseline variance".
        # null_delta = 0.0 by definition (the null model is the baseline itself).
        self.log("Computing noise floor from baseline fold variance (v8)...")
        if baseline_fold_scores is not None and len(baseline_fold_scores) >= 2:
            valid_bl_folds = [s for s in baseline_fold_scores if s is not None and not np.isnan(s)]
            if len(valid_bl_folds) >= 2:
                raw_std = float(np.std(valid_bl_folds, ddof=1))
                # Clamp minimum to prevent division explosion on near-deterministic datasets.
                # 0.001 is chosen as ~0.1 pp (AUC) or 0.1 pp (R²), a reasonable resolution floor.
                self._null_delta = 0.0
                self._null_std = max(raw_std, 0.001)
                self._null_std_was_clamped = (raw_std < 0.001)
                ds_meta['null_delta'] = self._null_delta
                ds_meta['null_std'] = self._null_std
                ds_meta['null_std_was_clamped'] = int(self._null_std_was_clamped)
                self.log(f"  Noise floor: null_std={self._null_std:.6f} (raw={raw_std:.6f}, "
                         f"clamped={self._null_std_was_clamped}, n_folds={len(valid_bl_folds)})")
            else:
                self.log("  Noise floor: insufficient valid baseline folds, using fallback", "WARNING")
                self._null_delta = 0.0
                self._null_std = 0.005
                self._null_std_was_clamped = True
                ds_meta['null_delta'] = 0.0
                ds_meta['null_std'] = 0.005
                ds_meta['null_std_was_clamped'] = 1
        else:
            self.log("  Noise floor: no baseline fold scores, using fallback", "WARNING")
            self._null_delta = 0.0
            self._null_std = 0.005
            self._null_std_was_clamped = True
            ds_meta['null_delta'] = 0.0
            ds_meta['null_std'] = 0.005
            ds_meta['null_std_was_clamped'] = 1

        # Log a sentinel row so the null_intervention method is still present in the CSV
        # for schema compatibility, but now reflects the calibration approach honestly.
        self._validate_and_append(
            ds_meta=ds_meta,
            column_name='NULL_CALIBRATION', method='null_intervention',
            delta=0.0, delta_normalized=0.0,
            absolute_score=base_score, is_interaction=False,
            t_statistic=0.0, p_value=1.0, p_value_bonferroni=1.0,
            is_significant=False, is_significant_bonferroni=False,
            null_delta=self._null_delta, null_std=self._null_std,
            cohens_d=0.0, calibrated_delta=0.0,
        )

        # =====================================================================
        # 12. UNIVARIATE INTERVENTIONS + DEFAULT INTERACTIONS
        # =====================================================================
        self.log(f"Testing interventions on {len(selected_cols)} selected columns...")
        
        for col_idx, col in enumerate(selected_cols):
            # v5: Time budget check
            if self._is_over_budget():
                self.log(f"  Time budget exceeded after {col_idx}/{len(selected_cols)} columns. Wrapping up.", "WARNING")
                break
            
            # v8: Column sampling -- skip individual analysis for unsampled columns.
            # Still cache metadata for interaction partner features (col_b/col_c).
            if col not in cols_for_individual:
                # Lightweight: cache column metadata without running any evaluations
                col_meta = self.get_column_meta(X[col], y, baseline_importance=base_feature_importance.get(col, 0.0))
                col_meta['vif'] = vif_scores.get(col, -1.0)
                col_meta['composite_predictive_score'] = composite_scores.get(col, 0.0)
                col_str = str(col)
                col_meta['featurewiz_selected'] = int(col_str in self._featurewiz_selected) if self._featurewiz_selected else np.nan
                col_meta['featurewiz_importance'] = self._featurewiz_importance.get(col_str, np.nan) if self._featurewiz_importance else np.nan
                temporal_type, temporal_period = self._detect_temporal_component(X[col], col)
                col_meta['is_temporal_component'] = int(temporal_type is not None)
                col_meta['temporal_period'] = temporal_period if temporal_period is not None else np.nan
                self._column_meta_cache[col] = col_meta.copy()
                self._composite_scores_cache[col] = composite_scores.get(col, 0.0)
                continue
            
            self.log(f"[{col_idx+1}/{len(selected_cols)}] Column: {col}")
            
            col_meta = self.get_column_meta(X[col], y, baseline_importance=base_feature_importance.get(col, 0.0))
            col_meta['vif'] = vif_scores.get(col, -1.0)
            col_meta['composite_predictive_score'] = composite_scores.get(col, 0.0)
            
            # v7: FeatureWiz provenance metadata
            col_str = str(col)
            col_meta['featurewiz_selected'] = int(col_str in self._featurewiz_selected) if self._featurewiz_selected else np.nan
            col_meta['featurewiz_importance'] = self._featurewiz_importance.get(col_str, np.nan) if self._featurewiz_importance else np.nan
            
            # v6: Cache col_meta for lookup when this column is col_b or col_c in interactions
            self._column_meta_cache[col] = col_meta.copy()
            self._composite_scores_cache[col] = composite_scores.get(col, 0.0)
            
            is_num = pd.api.types.is_numeric_dtype(X[col])
            is_date = self._is_likely_date(X[col], col)
            
            # Detect standalone temporal components (month, hour, day_of_week, etc.)
            temporal_type, temporal_period = self._detect_temporal_component(X[col], col)
            col_meta['is_temporal_component'] = int(temporal_type is not None)
            col_meta['temporal_period'] = temporal_period if temporal_period is not None else np.nan
            
            # Select methods
            if is_date:
                methods = ['date_extract_basic', 'date_cyclical_month', 'date_cyclical_dow',
                          'date_cyclical_hour', 'date_cyclical_day', 'date_elapsed_days']
            elif temporal_type is not None:
                # Standalone temporal column: cyclical encoding is the primary method.
                # Skip log/sqrt/poly/binning -- meaningless for cyclical data.
                # Keep impute_median/missing_indicator for null handling.
                methods = ['cyclical_encode', 'impute_median', 'missing_indicator']
                self.log(f"    [*] Detected temporal component: {temporal_type} (period={temporal_period})")
                # Coerce string-encoded temporal columns to numeric so downstream
                # methods (median, sin/cos) work correctly
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if col in X_full.columns:
                        X_full[col] = pd.to_numeric(X_full[col], errors='coerce')
                    is_num = True
            elif is_num:
                methods = ['log_transform', 'impute_median', 'missing_indicator',
                          'quantile_binning', 'polynomial_square', 'sqrt_transform',
                          'polynomial_cube', 'abs_transform',         
                          'exp_transform', 'reciprocal_transform']   
            else:
                nunique = X[col].nunique()
                methods = ['frequency_encoding', 'target_encoding', 'missing_indicator']
                if nunique > 10: methods.append('hashing_encoding')
                # We are already in the 'else' block (non-numeric), so just check content properties
                if nunique > 20 and X[col].astype(str).str.len().mean() > 5:
                    methods.append('text_stats')
                if 2 < nunique <= 10: methods.append('onehot_encoding')

            # Individual baseline for this column (cached)
            cache_key = f"single_{col}"
            indiv_base_score, indiv_base_std, indiv_base_folds = \
                self._get_individual_baseline(X_full, y, [col], cache_key=cache_key)

            # Track skips for summary logging
            skipped_methods = []
            
            for method in methods:
                if method in ['impute_median', 'missing_indicator'] and col_meta['null_pct'] == 0:
                    continue
                if method == 'sqrt_transform' and is_num and X[col].min() < 0:
                    continue

                # PROPERTY GATE: skip methods that can't theoretically help this column
                should_test, gate_reason = self._should_test_method(method, col_meta)
                if not should_test:
                    skipped_methods.append(f"{method}({gate_reason.split(':')[0]})")
                    continue

                # FULL MODEL evaluation
                full_score, full_std, full_folds = self.evaluate_with_intervention(
                    X_full, y, col_to_transform=col, method=method)
                if full_score is None:
                    self.log(f"    {method}: FAILED (full)", "WARNING")
                    continue

                # INDIVIDUAL evaluation
                # Skip for methods that are structurally invariant on a single feature
                if method in self.INDIVIDUAL_INVARIANT_METHODS:
                    indiv_score = None
                    indiv_std = None
                    indiv_folds = None
                    individual_skip_reason = 'encoding_invariant'
                else:
                    indiv_score, indiv_std, indiv_folds = self.evaluate_individual(
                        X_full[[col]], y, col_to_transform=col, method=method)
                    individual_skip_reason = None

                self.n_tests_performed += 1
                
                # Full-model stats
                delta_full = self._get_delta(base_score, full_score)
                delta_full_norm = self._normalize_delta(delta_full, base_score)
                t_full, p_full, p_full_bonf, sig_full, sig_full_bonf = \
                    self._paired_ttest_with_bonferroni(baseline_fold_scores, full_folds, self.n_tests_performed)

                # Individual stats
                indiv_delta = np.nan
                indiv_delta_norm = np.nan
                t_indiv = np.nan
                p_indiv = np.nan
                p_indiv_bonf = np.nan
                sig_indiv = False
                sig_indiv_bonf = False
                
                if indiv_base_score is not None and indiv_score is not None:
                    indiv_delta = self._get_delta(indiv_base_score, indiv_score)
                    indiv_delta_norm = self._normalize_delta(indiv_delta, indiv_base_score)
                    if indiv_base_folds is not None and indiv_folds is not None:
                        t_indiv, p_indiv, p_indiv_bonf, sig_indiv, sig_indiv_bonf = \
                            self._paired_ttest_with_bonferroni(indiv_base_folds, indiv_folds, self.n_tests_performed)

                # ADAPTIVE EARLY-STOP: track whether this method improved anything
                # "Improved" = positive delta on EITHER full or individual (at p < 0.10 for leniency)
                improved = False
                if delta_full > 0 and (p_full is not None and not np.isnan(p_full) and p_full < 0.10):
                    improved = True
                if not improved and indiv_delta is not None and not np.isnan(indiv_delta):
                    if indiv_delta > 0 and (p_indiv is not None and not np.isnan(p_indiv) and p_indiv < 0.10):
                        improved = True
                self._update_method_tracker(method, improved, len(selected_cols))

                self._validate_and_append(
                    ds_meta=ds_meta, col_meta=col_meta,
                    column_name=col, method=method,
                    delta=delta_full, delta_normalized=delta_full_norm,
                    absolute_score=full_score, is_interaction=False,
                    t_statistic=t_full, p_value=p_full,
                    p_value_bonferroni=p_full_bonf,
                    is_significant=sig_full, is_significant_bonferroni=sig_full_bonf,
                    individual_baseline_score=indiv_base_score,
                    individual_intervention_score=indiv_score,
                    individual_delta=indiv_delta,
                    individual_delta_normalized=indiv_delta_norm,
                    individual_p_value=p_indiv,
                    individual_p_value_bonferroni=p_indiv_bonf,
                    individual_is_significant=sig_indiv,
                    individual_is_significant_bonferroni=sig_indiv_bonf,
                    individual_skip_reason=individual_skip_reason,
                    # Fix 4: tag individual model complexity so meta-model can condition on it.
                    # Rows with individual evaluation used self.individual_params (simplified);
                    # those that skipped it (encoding_invariant) still used standard for full.
                    individual_model_complexity=(
                        'simplified' if individual_skip_reason != 'encoding_invariant' else 'standard'
                    ),
                    baseline_fold_scores=json.dumps(baseline_fold_scores) if self.store_fold_scores else None,
                    intervention_fold_scores=json.dumps(full_folds) if self.store_fold_scores else None,
                    # v5: noise-floor calibration and effect sizes
                    null_delta=self._null_delta,
                    null_std=self._null_std,
                    cohens_d=self._compute_cohens_d(baseline_fold_scores, full_folds),
                    individual_cohens_d=self._compute_cohens_d(indiv_base_folds, indiv_folds) if indiv_folds is not None else np.nan,
                    calibrated_delta=self._compute_calibrated_delta(delta_full),
                    individual_calibrated_delta=self._compute_calibrated_delta(indiv_delta) if not np.isnan(indiv_delta) else np.nan,
                )
                
                sig_str = "[OK]" if sig_full else "[X]"
                if individual_skip_reason:
                    indiv_str = f"indiv=SKIP({individual_skip_reason})"
                elif not np.isnan(indiv_delta_norm):
                    isig_str = "[OK]" if sig_indiv else "[X]"
                    indiv_str = f"indiv={indiv_delta_norm:+.2f}% p={p_indiv:.4f} {isig_str}"
                else:
                    indiv_str = "indiv=FAILED"
                self.log(f"    {method}: full={delta_full_norm:+.2f}% p={p_full:.4f} {sig_str} | {indiv_str}")

            if skipped_methods:
                self.log(f"    Skipped {len(skipped_methods)}: {', '.join(skipped_methods)}")

            # DEFAULT INTERACTION: test this column with top-1 and top-2 global columns
            # DEFAULT INTERACTION: test this column with top-1 and top-2 global columns
            for top_col in top_global_cols:
                if top_col == col:
                    continue  # Skip self-interaction
                
                both_numeric = is_num and pd.api.types.is_numeric_dtype(X[top_col])
                col_is_cat = not is_num and not is_date
                top_is_cat = not pd.api.types.is_numeric_dtype(X[top_col])
                top_is_num = pd.api.types.is_numeric_dtype(X[top_col])
                
                # Decide interaction type
                if both_numeric:
                    # GATE: If either column is a temporal component, skip raw arithmetic
                    col_is_temp, _ = self._detect_temporal_component(X[col], col)
                    top_is_temp, _ = self._detect_temporal_component(X[top_col], top_col)
                    
                    if col_is_temp or top_is_temp:
                        interaction_methods = []
                    else:
                        interaction_methods = ['product_interaction', 'division_interaction']
                        if self._is_scale_compatible(col, top_col):
                            interaction_methods.append('subtraction_interaction')
                        # v8: Skip degenerate methods when either column is binary
                        interaction_methods = self._filter_methods_for_binary(
                            interaction_methods, col, top_col)
                elif col_is_cat and top_is_num:
                    # Case 1: Grouping by selected (Cat), aggregating Top (Num)
                    # FIX: Ensure the numeric value (top_col) is not temporal
                    top_is_temp, _ = self._detect_temporal_component(X[top_col], top_col)
                    if top_is_temp:
                        interaction_methods = []
                    else:
                        interaction_methods = ['group_mean']

                elif is_num and top_is_cat:
                    # Case 2: Grouping by Top (Cat), aggregating selected (Num)
                    # FIX: Ensure the numeric value (col) is not temporal
                    col_is_temp, _ = self._detect_temporal_component(X[col], col)
                    if col_is_temp:
                        continue

                    # Reverse: top_col is cat, col is num -> group_mean with top as grouper
                    interaction_methods = []  # Will handle below
                    # Test: group_mean of col by top_col
                    if self._is_interaction_duplicate('group_mean', top_col, col):
                        continue
                    cache_key = f"pair_{top_col}_{col}"
                    indiv_pair_base = self._get_individual_baseline(X_full, y, [top_col, col], cache_key=cache_key)
                    
                    full_s, full_sd, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=top_col, col_second=col, method='group_mean')
                    if full_s is not None:
                        indiv_pair_new = self.evaluate_individual(
                            X_full[[top_col, col]], y, col_to_transform=top_col, col_second=col, method='group_mean')
                        self._log_interaction_result(
                            ds_meta, baseline_fold_scores, base_score,
                            f"group_mean_{col}_by_{top_col}", 'group_mean',
                            full_s, full_fs, indiv_pair_base, indiv_pair_new,
                            top_col, col, interaction_source='default_interaction')
                    continue
                
                elif col_is_cat and top_is_cat:
                    interaction_methods = ['cat_concat']
                else:
                    continue
                
                for int_method in interaction_methods:
                    if self._is_interaction_duplicate(int_method, col, top_col):
                        continue
                    cache_key = f"pair_{col}_{top_col}"
                    indiv_pair_base = self._get_individual_baseline(X_full, y, [col, top_col], cache_key=cache_key)
                    
                    full_s, full_sd, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col, col_second=top_col, method=int_method)
                    if full_s is None:
                        continue
                    
                    indiv_pair_new = self.evaluate_individual(
                        X_full[[col, top_col]], y, col_to_transform=col, col_second=top_col, method=int_method)
                    
                    if int_method == 'product_interaction':
                        col_name = f"{col}_x_{top_col}"
                    elif int_method == 'division_interaction':
                        col_name = f"{col}_div_{top_col}"
                    elif int_method == 'subtraction_interaction':
                        col_name = f"{col}_minus_{top_col}"
                    elif int_method == 'group_mean':
                        col_name = f"group_mean_{top_col}_by_{col}"
                    elif int_method == 'cat_concat':
                        col_name = f"{col}_concat_{top_col}"
                    else:
                        col_name = f"{col}_{int_method}_{top_col}"
                    
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        col_name, int_method,
                        full_s, full_fs, indiv_pair_base, indiv_pair_new,
                        col, top_col, interaction_source='default_interaction')

        



        # 13. ROW STATS
        self.log("Evaluating Row-Level Statistics...")
        row_s, row_sd, row_fs = self.evaluate_with_intervention(X_full, y, method='row_stats')
        if row_s is not None:
            self.n_tests_performed += 1
            delta = self._get_delta(base_score, row_s)
            t, p, pb, sig, sigb = self._paired_ttest_with_bonferroni(baseline_fold_scores, row_fs, self.n_tests_performed)
            self._validate_and_append(
                ds_meta=ds_meta,
                column_name='GLOBAL_ROW_STATS', method='row_stats',
                delta=delta, delta_normalized=self._normalize_delta(delta, base_score),
                absolute_score=row_s, is_interaction=False,
                t_statistic=t, p_value=p, p_value_bonferroni=pb,
                is_significant=sig, is_significant_bonferroni=sigb,
                # v5: noise-floor calibration
                null_delta=self._null_delta,
                null_std=self._null_std,
                cohens_d=self._compute_cohens_d(baseline_fold_scores, row_fs),
                calibrated_delta=self._compute_calibrated_delta(delta),
            )

        # =====================================================================
        # 14. CROSS-COLUMN INTERACTIONS (top features x top features)
        # =====================================================================
        self.log("Selecting features for cross-column interactions...")
        
        num_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(X[c]) and not self._is_likely_date(X[c], c)]
        
        # Sort numeric columns by importance
        sorted_nums = sorted(num_cols, key=lambda c: composite_scores.get(c, 0), reverse=True)
        
        # --- REFINED POOLING LOGIC ---
        # 1. Identify Temporal Columns strictly
        temp_cols = []
        for c in selected_cols:
            is_t, _ = self._detect_temporal_component(X[c], c)
            if is_t: temp_cols.append(c)
        
        # 2. Arithmetic Candidates: Numeric but NOT temporal
        arith_cols = [c for c in num_cols if c not in temp_cols]
        
        # 3. Grouper Candidates: Anything categorical OR temporal
        # These are columns that represent "buckets"
        grouper_pool = sorted(
            [c for c in selected_cols if (c in cat_cols or c in temp_cols)],
            key=lambda c: composite_scores.get(c, 0), reverse=True
        )
        
        # 4. Value Pool: True continuous numerics only
        value_pool = sorted(arith_cols, key=lambda c: composite_scores.get(c, 0), reverse=True)

        # Dynamic Budgets: Use up to 8 groupers and 8 value columns
        # Phase 2.3: Wider scope for more interaction diversity
        top_groupers = grouper_pool[:4] 
        top_values = value_pool[:4]


        # SPLIT: True Arithmetic Numerics vs. Temporal Component Labels
        # We check the temporal flag we detected earlier
        arithmetic_candidates = []
        temporal_candidates = []
        
        for c in sorted_nums:
            is_temp, _ = self._detect_temporal_component(X[c], c)
            if is_temp:
                temporal_candidates.append(c)
            else:
                arithmetic_candidates.append(c)
        
        # Cap the lists for performance
        top_arithmetic = arithmetic_candidates[:8]
        top_temporal = temporal_candidates[:4]
        top_cats = sorted(cat_cols, key=lambda c: composite_scores.get(c, 0), reverse=True)[:6]

        # =====================================================================
        # v5: TREE-GUIDED INTERACTION PAIR SELECTION
        # =====================================================================
        # Instead of brute-force testing all pairs of top-K columns, we fit 
        # shallow decision trees and extract parent->child split pairs.
        # These pairs represent conditional dependencies = ideal interaction targets.
        
        tree_pair_lookup = {}  # (col_a, col_b) -> score for metadata
        tree_triple_lookup = {}
        
        if len(arith_cols) >= 2 and not self._is_over_budget():
            self.log("Running tree-guided interaction pair selection...")
            tree_num_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(X[c])]
            if len(tree_num_cols) >= 2:
                tree_pairs_raw, tree_triples_raw = self._get_tree_guided_pairs(
                    X_full[tree_num_cols], y, max_depth=4, n_trees=5)
                
                # Filter to arithmetic-only pairs (no temporal columns)
                tree_pairs = [(a, b, s) for a, b, s in tree_pairs_raw 
                              if a in arith_cols and b in arith_cols]
                tree_triples = [(a, b, c, s) for a, b, c, s in tree_triples_raw
                                if a in arith_cols and b in arith_cols and c in arith_cols]
                
                for a, b, s in tree_pairs:
                    tree_pair_lookup[tuple(sorted([a, b]))] = s
                for a, b, c, s in tree_triples:
                    tree_triple_lookup[tuple(sorted([a, b, c]))] = s
                
                self.log(f"  Tree-guided: {len(tree_pairs)} numeric pairs, {len(tree_triples)} triples")
                if tree_pairs:
                    self.log(f"  Top 5 pairs: {[(a, b, round(s, 4)) for a, b, s in tree_pairs[:5]]}")
            else:
                tree_pairs = []
                tree_triples = []
        else:
            tree_pairs = []
            tree_triples = []

    

        # =====================================================================
        # v7: AUTOFEAT INTERACTION PAIR DISCOVERY
        # =====================================================================
        autofeat_pairs = []
        autofeat_singles = []
        if self.use_autofeat and not self._is_over_budget():
            autofeat_pairs, autofeat_singles = self._autofeat_discover_pairs(X_full, y)
            # Filter to arithmetic-only pairs (no temporal)
            autofeat_pairs = [
                (a, b, op) for a, b, op in autofeat_pairs
                if a in arith_cols and b in arith_cols
            ]
        elif self.use_autofeat:
            self.log("  AutoFeat skipped (time budget)", "WARNING")

        # =====================================================================
        # 12b. AUTOFEAT-GUIDED SINGLE-COLUMN TRANSFORMS (gate bypass)
        # =====================================================================
        # AutoFeat's L1 selection identified these (column, transform) pairs as
        # predictive. We test them even if property gates or adaptive early-stop
        # would have blocked them in step 12. This catches cases where:
        #   - Property heuristics are too conservative for this specific column
        #   - Adaptive early-stop killed a method globally before reaching this col
        #
        # For EXISTING methods (log, sqrt, square): use existing _apply_intervention
        # For NEW methods (cube, abs, exp, reciprocal): requires Tier 2 additions

        AF_SINGLE_TO_METHOD = {
            'polynomial_square': 'polynomial_square',
            'log_transform': 'log_transform',
            'sqrt_transform': 'sqrt_transform',
            'polynomial_cube': 'polynomial_cube',
            'abs_transform': 'abs_transform',
            'exp_transform': 'exp_transform',
            # 'reciprocal_transform': 'reciprocal_transform',  # deferred
        }

        af_singles_tested = 0
        af_singles_budget = 30  # cap to avoid blowing time budget

        if autofeat_singles and not self._is_over_budget():
            self.log(f"AutoFeat-guided single-column transforms "
                     f"({len(autofeat_singles)} candidates, budget={af_singles_budget})...")

            for af_col, af_op in autofeat_singles:
                if af_singles_tested >= af_singles_budget or self._is_over_budget():
                    break

                method = AF_SINGLE_TO_METHOD.get(af_op)
                if method is None:
                    self.log(f"  AutoFeat single: unknown op '{af_op}' on '{af_col}', skipping")
                    continue

                if af_col not in X_full.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(X_full[af_col]):
                    continue

                if self._is_interaction_duplicate(method, af_col, None):
                    self.log(f"  AutoFeat single: {method} on {af_col} already tested, skipping")
                    continue

                if af_col in self._column_meta_cache:
                    col_meta = self._column_meta_cache[af_col]
                else:
                    col_meta = self.get_column_meta(
                        X_full[af_col], y,
                        baseline_importance=base_feature_importance.get(af_col, 0.0))
                    col_meta['vif'] = vif_scores.get(af_col, -1.0)
                    col_meta['composite_predictive_score'] = composite_scores.get(af_col, 0.0)
                    col_str = str(af_col)
                    col_meta['featurewiz_selected'] = int(col_str in self._featurewiz_selected) if self._featurewiz_selected else np.nan
                    col_meta['featurewiz_importance'] = self._featurewiz_importance.get(col_str, np.nan) if self._featurewiz_importance else np.nan
                    temporal_type, temporal_period = self._detect_temporal_component(X_full[af_col], af_col)
                    col_meta['is_temporal_component'] = int(temporal_type is not None)
                    col_meta['temporal_period'] = temporal_period if temporal_period is not None else np.nan
                    self._column_meta_cache[af_col] = col_meta.copy()

                self.log(f"  AutoFeat-guided: {method} on {af_col} (gate bypass)")

                full_score, full_std, full_folds = self.evaluate_with_intervention(
                    X_full, y, col_to_transform=af_col, method=method)
                if full_score is None:
                    self.log(f"    {method}: FAILED (full)", "WARNING")
                    continue

                if method in self.INDIVIDUAL_INVARIANT_METHODS:
                    indiv_score = None
                    indiv_std = None
                    indiv_folds = None
                    individual_skip_reason = 'encoding_invariant'
                else:
                    indiv_score, indiv_std, indiv_folds = self.evaluate_individual(
                        X_full[[af_col]], y, col_to_transform=af_col, method=method)
                    individual_skip_reason = None

                self.n_tests_performed += 1
                af_singles_tested += 1

                delta_full = self._get_delta(base_score, full_score)
                delta_full_norm = self._normalize_delta(delta_full, base_score)
                t_full, p_full, p_full_bonf, sig_full, sig_full_bonf = \
                    self._paired_ttest_with_bonferroni(baseline_fold_scores, full_folds, self.n_tests_performed)

                cache_key = f"single_{af_col}"
                indiv_base_score, indiv_base_std, indiv_base_folds = \
                    self._get_individual_baseline(X_full, y, [af_col], cache_key=cache_key)

                indiv_delta = np.nan
                indiv_delta_norm = np.nan
                t_indiv = np.nan
                p_indiv = np.nan
                p_indiv_bonf = np.nan
                sig_indiv = False
                sig_indiv_bonf = False

                if indiv_base_score is not None and indiv_score is not None:
                    indiv_delta = self._get_delta(indiv_base_score, indiv_score)
                    indiv_delta_norm = self._normalize_delta(indiv_delta, indiv_base_score)
                    if indiv_base_folds is not None and indiv_folds is not None:
                        t_indiv, p_indiv, p_indiv_bonf, sig_indiv, sig_indiv_bonf = \
                            self._paired_ttest_with_bonferroni(
                                indiv_base_folds, indiv_folds, self.n_tests_performed)

                self._validate_and_append(
                    ds_meta=ds_meta, col_meta=col_meta,
                    column_name=af_col, method=method,
                    delta=delta_full, delta_normalized=delta_full_norm,
                    absolute_score=full_score, is_interaction=False,
                    t_statistic=t_full, p_value=p_full,
                    p_value_bonferroni=p_full_bonf,
                    is_significant=sig_full, is_significant_bonferroni=sig_full_bonf,
                    individual_baseline_score=indiv_base_score,
                    individual_intervention_score=indiv_score,
                    individual_delta=indiv_delta,
                    individual_delta_normalized=indiv_delta_norm,
                    individual_p_value=p_indiv,
                    individual_p_value_bonferroni=p_indiv_bonf,
                    individual_is_significant=sig_indiv,
                    individual_is_significant_bonferroni=sig_indiv_bonf,
                    individual_skip_reason=individual_skip_reason,
                    baseline_fold_scores=json.dumps(baseline_fold_scores) if self.store_fold_scores else None,
                    intervention_fold_scores=json.dumps(full_folds) if self.store_fold_scores else None,
                    null_delta=self._null_delta,
                    null_std=self._null_std,
                    cohens_d=self._compute_cohens_d(baseline_fold_scores, full_folds),
                    individual_cohens_d=self._compute_cohens_d(indiv_base_folds, indiv_folds) if indiv_folds else np.nan,
                    calibrated_delta=self._compute_calibrated_delta(delta_full),
                    individual_calibrated_delta=self._compute_calibrated_delta(indiv_delta) if not np.isnan(indiv_delta) else np.nan,
                    interaction_source='autofeat_single_guided',
                )

                if delta_full > 0:
                    self.log(f"    {method}: delta={delta_full:.6f} (p={p_full:.4f}) *** AutoFeat override ***")
                else:
                    self.log(f"    {method}: delta={delta_full:.6f} (p={p_full:.4f})")

            self.log(f"  AutoFeat singles: tested {af_singles_tested}/{len(autofeat_singles)}")

     

        # =====================================================================
        # A. NUMERIC INTERACTIONS (tree-guided + fallback to importance-based)
        # =====================================================================
        # v8: Scale interaction budget based on dataset size.
        # With chunked CV, per-test time is already reduced, so we can be generous.
        n_rows_full = len(X_full)
        if n_rows_full > 500_000:
            MAX_NUMERIC_PAIR_TESTS = 32
        elif n_rows_full > 100_000:
            MAX_NUMERIC_PAIR_TESTS = 48
        else:
            MAX_NUMERIC_PAIR_TESTS = 64
        
        # v8: For large datasets, skip addition and abs_diff (subtraction is enough)
        _is_large_dataset = n_rows_full > 100_000
        
        METHODS_PER_PAIR = ['product_interaction', 'division_interaction']
        
        tested_numeric_pairs = set()
        pair_test_count = 0
        
        # A.1: Tree-guided pairs (priority)
        if tree_pairs and not self._is_over_budget():
            self.log(f"Cross-column: Tree-Guided Numeric Interactions (up to {MAX_NUMERIC_PAIR_TESTS} pairs)...")
            for col_a, col_b, t_score in tree_pairs:
                if pair_test_count >= MAX_NUMERIC_PAIR_TESTS or self._is_over_budget():
                    break
                
                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key in tested_numeric_pairs:
                    continue
                tested_numeric_pairs.add(pair_key)
                
                cache_key = f"pair_{col_a}_{col_b}"
                indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                
                pair_methods = list(METHODS_PER_PAIR)
                if self._is_scale_compatible(col_a, col_b):
                    if _is_large_dataset:
                        pair_methods.append('subtraction_interaction')
                    else:
                        pair_methods.extend(['subtraction_interaction', 'addition_interaction', 'abs_diff_interaction'])
                # v8: Skip degenerate methods when either column is binary
                pair_methods = self._filter_methods_for_binary(pair_methods, col_a, col_b)
                if not pair_methods:
                    continue
                for int_method in pair_methods:  
                    if self._is_interaction_duplicate(int_method, col_a, col_b):
                        continue
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method=int_method)
                    if full_s is None:
                        continue
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method=int_method)
                    
                    name_map = {
                        'product_interaction': f"{col_a}_x_{col_b}",
                        'division_interaction': f"{col_a}_div_{col_b}",
                        'subtraction_interaction': f"{col_a}_minus_{col_b}",
                    }
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        name_map.get(int_method, f"{col_a}_{int_method}_{col_b}"),
                        int_method, full_s, full_fs, indiv_base, indiv_new,
                        col_a, col_b, interaction_source='tree_guided',
                        tree_pair_score=t_score)
                
                pair_test_count += 1
        
        # A.1b: AutoFeat-discovered pairs (complementary to tree-guided)
        # AutoFeat uses polynomial L1 selection -- a fundamentally different approach
        # from tree-guided splitting, so it often finds pairs that trees miss.
        if autofeat_pairs and not self._is_over_budget():
            af_budget = max(16, MAX_NUMERIC_PAIR_TESTS - pair_test_count)
            self.log(f"Cross-column: AutoFeat-Discovered Pairs (up to {af_budget}, "
                     f"{len(autofeat_pairs)} candidates)...")
            af_tested = 0
            for col_a, col_b, af_operation in autofeat_pairs:
                if af_tested >= af_budget or self._is_over_budget():
                    break

                # Validate columns still exist in X_full
                if col_a not in X_full.columns or col_b not in X_full.columns:
                    continue

                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key in tested_numeric_pairs:
                    continue
                tested_numeric_pairs.add(pair_key)

                cache_key = f"pair_{col_a}_{col_b}"
                indiv_base = self._get_individual_baseline(
                    X_full, y, [col_a, col_b], cache_key=cache_key)

                # Use the operation AutoFeat found, plus standard methods
                pair_methods = list(METHODS_PER_PAIR)
                # If AutoFeat found a specific op, prioritize it
                if af_operation in ('addition_interaction', 'subtraction_interaction',
                                    'abs_diff_interaction'):
                    if self._is_scale_compatible(col_a, col_b):
                        pair_methods = [af_operation] + [m for m in pair_methods if m != af_operation]
                elif af_operation not in pair_methods:
                    pair_methods.append(af_operation)

                if self._is_scale_compatible(col_a, col_b):
                    if _is_large_dataset:
                        # Only add subtraction for large datasets
                        if 'subtraction_interaction' not in pair_methods:
                            pair_methods.append('subtraction_interaction')
                    else:
                        for extra in ('addition_interaction', 'abs_diff_interaction'):
                            if extra not in pair_methods:
                                pair_methods.append(extra)

                pair_methods = self._filter_methods_for_binary(pair_methods, col_a, col_b)
                if not pair_methods:
                    af_tested += 1  # still count toward budget to avoid infinite loop
                    continue
                for int_method in pair_methods:
                    if self._is_interaction_duplicate(int_method, col_a, col_b):
                        continue
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b,
                        method=int_method)
                    if full_s is None:
                        continue
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a,
                        col_second=col_b, method=int_method)
                    name_map = {
                        'product_interaction': f"{col_a}_x_{col_b}",
                        'division_interaction': f"{col_a}_div_{col_b}",
                        'addition_interaction': f"{col_a}_plus_{col_b}",
                        'subtraction_interaction': f"{col_a}_minus_{col_b}",
                        'abs_diff_interaction': f"{col_a}_absdiff_{col_b}",
                    }
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        name_map.get(int_method, f"{col_a}_{int_method}_{col_b}"),
                        int_method, full_s, full_fs, indiv_base, indiv_new,
                        col_a, col_b, interaction_source='autofeat_discovered')

                af_tested += 1
                pair_test_count += 1

            self.log(f"  AutoFeat pairs tested: {af_tested}")

        # A.2: Fallback importance-based pairs (fill remaining budget)
        remaining_budget = MAX_NUMERIC_PAIR_TESTS - pair_test_count
        if remaining_budget > 0 and len(top_arithmetic) >= 2 and not self._is_over_budget():
            self.log(f"Cross-column: Importance-Based Fallback ({remaining_budget} remaining)...")
            fallback_count = 0
            for i, col_a in enumerate(top_arithmetic):
                if fallback_count >= remaining_budget or self._is_over_budget():
                    break
                for col_b in top_arithmetic[i+1:]:
                    if fallback_count >= remaining_budget or self._is_over_budget():
                        break
                    
                    pair_key = tuple(sorted([col_a, col_b]))
                    if pair_key in tested_numeric_pairs:
                        continue
                    tested_numeric_pairs.add(pair_key)
                    
                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    
                    pair_methods = ['product_interaction', 'division_interaction']
                    if self._is_scale_compatible(col_a, col_b):
                        if _is_large_dataset:
                            pair_methods.append('subtraction_interaction')
                        else:
                            pair_methods.extend(['addition_interaction', 'abs_diff_interaction'])
                    # v8: Skip degenerate methods when either column is binary
                    pair_methods = self._filter_methods_for_binary(pair_methods, col_a, col_b)
                    if not pair_methods:
                        continue
                    for int_method in pair_methods:
                        if self._is_interaction_duplicate(int_method, col_a, col_b):
                            continue
                        full_s, _, full_fs = self.evaluate_with_intervention(
                            X_full, y, col_to_transform=col_a, col_second=col_b, method=int_method)
                        if full_s is None:
                            continue
                        indiv_new = self.evaluate_individual(
                            X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method=int_method)
                        name_map = {
                            'product_interaction': f"{col_a}_x_{col_b}",
                            'division_interaction': f"{col_a}_div_{col_b}",
                            'addition_interaction': f"{col_a}_plus_{col_b}",
                            'abs_diff_interaction': f"{col_a}_absdiff_{col_b}",
                        }
                        self._log_interaction_result(
                            ds_meta, baseline_fold_scores, base_score,
                            name_map.get(int_method, f"{col_a}_{int_method}_{col_b}"),
                            int_method, full_s, full_fs, indiv_base, indiv_new,
                            col_a, col_b, interaction_source='importance_fallback')
                    
                    fallback_count += 1
        
        self.log(f"  Total numeric pairs tested: {len(tested_numeric_pairs)}")

        # B. 3-Way interactions (multiplication, addition, ratio)
        if len(top_arithmetic) >= 3 and not self._is_over_budget():
            self.log("Cross-column: 3-Way Interactions...")
            # v4: expanded to top 5 columns, multiple 3-way methods
            three_way_methods = ['three_way_interaction', 'three_way_addition', 'three_way_ratio', 'three_way_normalized_diff']
            n_3way = min(4, len(top_arithmetic))
            for i in range(n_3way):
                for j in range(i+1, n_3way):
                    for k in range(j+1, n_3way):
                        a, b, c = top_arithmetic[i], top_arithmetic[j], top_arithmetic[k]
                        cache_key = f"triple_{a}_{b}_{c}"
                        indiv_base = self._get_individual_baseline(X_full, y, [a, b, c], cache_key=cache_key)
                        
                        for tw_method in three_way_methods:
                            if self._is_interaction_duplicate(tw_method, a, b, c):
                                continue
                            # Scale gate: three_way_addition needs all three columns compatible
                            if tw_method == 'three_way_addition':
                                if not (self._is_scale_compatible(a, b) and 
                                        self._is_scale_compatible(a, c) and
                                        self._is_scale_compatible(b, c)):
                                    continue
                            
                            full_s, _, full_fs = self.evaluate_with_intervention(
                                X_full, y, col_to_transform=a, col_second=b, col_third=c, method=tw_method)
                            if full_s is None: continue
                            indiv_new = self.evaluate_individual(
                                X_full[[a, b, c]], y, col_to_transform=a, col_second=b, col_third=c, method=tw_method)
                            
                            if tw_method == 'three_way_interaction':
                                name = f"{a}_x_{b}_x_{c}"
                            elif tw_method == 'three_way_addition':
                                name = f"{a}_plus_{b}_plus_{c}"
                            else:
                                name = f"{a}_x_{b}_div_{c}"
                            
                            self._log_interaction_result(
                                ds_meta, baseline_fold_scores, base_score,
                                name, tw_method,
                                full_s, full_fs, indiv_base, indiv_new, a, b, col_c=c)

        # C. Division interactions (expanded scope -- tests both A/B and B/A directions)
        # v8: Cap scope for large datasets (6 cols -> max 30 pairs vs 8 -> 56 pairs)
        div_cols = top_arithmetic[:6] if _is_large_dataset else top_arithmetic
        if len(div_cols) >= 2 and not self._is_over_budget():
            self.log(f"Cross-column: Division Interactions ({len(div_cols)} cols)...")
            for col_a in div_cols:
                for col_b in div_cols:
                    if col_a == col_b: continue
                    if self._is_interaction_duplicate('division_interaction', col_a, col_b):
                        continue
                    # v8: Division by binary is degenerate (A/0=inf, A/1=A)
                    b_is_binary = self._column_meta_cache.get(col_b, {}).get('is_binary', False)
                    if not b_is_binary and col_b in X_full.columns:
                        b_is_binary = X_full[col_b].nunique() <= 2
                    if b_is_binary:
                        continue
                    # Also skip dividing a binary by anything (result is 0/x or 1/x â€” trivial)
                    a_is_binary = self._column_meta_cache.get(col_a, {}).get('is_binary', False)
                    if not a_is_binary and col_a in X_full.columns:
                        a_is_binary = X_full[col_a].nunique() <= 2
                    if a_is_binary:
                        continue
                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method='division_interaction')
                    if full_s is None: continue
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method='division_interaction')
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{col_a}_div_{col_b}", 'division_interaction',
                        full_s, full_fs, indiv_base, indiv_new, col_a, col_b)

        
       
        # D. Group-by interactions
        if top_groupers and top_values and not self._is_over_budget():
            self.log(f"Cross-column: Group-By Interactions ({len(top_groupers)} groupers x {len(top_values)} values)...")
            for grouper_col in top_groupers:
                for value_col in top_values:
                    # Semantic Check: Never use a column as its own grouper
                    if grouper_col == value_col:
                        continue
                        
                    for method in ['group_mean', 'group_std']:
                        # Logic: grouper_col is the "By", value_col is the "Value"
                        # Example: group_mean of 'Temperature' by 'Hour'
                        
                        # High-cardinality check for std
                        nunique_grouper = X[grouper_col].nunique()
                        if method == 'group_std' and (len(X) / max(nunique_grouper, 1) < 10):
                            continue
                        if self._is_interaction_duplicate(method, grouper_col, value_col):
                            continue
                            
                        cache_key = f"pair_{grouper_col}_{value_col}"
                        indiv_base = self._get_individual_baseline(X_full, y, [grouper_col, value_col], cache_key=cache_key)
                        
                        # Important: col_to_transform is the Grouper, col_second is the Value
                        full_s, _, full_fs = self.evaluate_with_intervention(
                            X_full, y, col_to_transform=grouper_col, col_second=value_col, method=method)
                        
                        if full_s is None: continue
                        
                        indiv_new = self.evaluate_individual(
                            X_full[[grouper_col, value_col]], y, 
                            col_to_transform=grouper_col, col_second=value_col, method=method)
                            
                        self._log_interaction_result(
                            ds_meta, baseline_fold_scores, base_score,
                            f"{method}_{value_col}_by_{grouper_col}", method,
                            full_s, full_fs, indiv_base, indiv_new, grouper_col, value_col)

        # E. Cat concat
        if len(top_cats) >= 2 and not self._is_over_budget():
            self.log("Cross-column: Cat Concat...")
            for i, cat_a in enumerate(top_cats):
                for cat_b in top_cats[i+1:]:
                    if X[cat_a].nunique() > len(X)*0.5 or X[cat_b].nunique() > len(X)*0.5:
                        continue
                    if self._is_interaction_duplicate('cat_concat', cat_a, cat_b):
                        continue
                    cache_key = f"pair_{cat_a}_{cat_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [cat_a, cat_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=cat_a, col_second=cat_b, method='cat_concat')
                    if full_s is None: continue
                    indiv_new = self.evaluate_individual(
                        X_full[[cat_a, cat_b]], y, col_to_transform=cat_a, col_second=cat_b, method='cat_concat')
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{cat_a}_concat_{cat_b}", 'cat_concat',
                        full_s, full_fs, indiv_base, indiv_new, cat_a, cat_b)

        # Summary
        elapsed = time.time() - self._start_time
        self.log(f"Analysis complete. Total tests: {self.n_tests_performed} in {elapsed:.1f}s")
        self.log(f"Unique interactions tracked: {len(self._tested_interactions)}")
        if self._is_over_budget():
            self.log(f"  (Time budget of {self.time_budget_seconds}s was exceeded)", "WARNING")
        if self.n_tests_performed > 0:
            self.log(f"Bonferroni threshold: {0.05/self.n_tests_performed:.6f}")
        
        # Method tracker summary
        if self._method_tracker:
            self.log("Method effectiveness summary:")
            for method, tracker in sorted(self._method_tracker.items()):
                status = "DISABLED" if tracker.get('disabled') else "active"
                self.log(f"  {method}: {tracker['improved']}/{tracker['tested']} improved [{status}]")
        
        # v7: External library integration summary
        if self._featurewiz_selected:
            self.log(f"FeatureWiz: {len(self._featurewiz_selected)}/{X.shape[1]} columns selected")
        if autofeat_pairs:
            self.log(f"AutoFeat: {len(autofeat_pairs)} interaction pairs discovered")
            if autofeat_singles:
                self.log(f"AutoFeat: {len(autofeat_singles)} single-column transforms identified, "
                         f"{af_singles_tested} tested with gate bypass")
        
        self._X_full_ref = None

        # =====================================================================
        # FIX 2: Post-hoc Bonferroni correction using the FINAL n_tests_performed
        #
        # During the test loop, each call to _paired_ttest_with_bonferroni() passes
        # self.n_tests_performed at the time of the call -- which is the running count
        # up to that test, not the total. This makes early-tested columns (the most
        # important ones, due to ranked ordering) use a much smaller multiplier than
        # late-tested ones, creating a systematic bias.
        #
        # Fix: build the DataFrame, then overwrite the Bonferroni columns with the
        # correct values computed from the final total test count.
        # =====================================================================
        df = pd.DataFrame(self.meta_data_log)
        if not df.empty and self.n_tests_performed > 0:
            n_total = self.n_tests_performed
            self.log(f"Post-hoc Bonferroni correction: n_total_tests={n_total}, "
                     f"alpha_corrected={0.05/n_total:.6f}")

            # Full-model Bonferroni
            if 'p_value' in df.columns:
                df['p_value_bonferroni'] = (df['p_value'].astype(float) * n_total).clip(upper=1.0)
                df['is_significant_bonferroni'] = df['p_value_bonferroni'] < 0.05

            # Individual Bonferroni
            if 'individual_p_value' in df.columns:
                df['individual_p_value_bonferroni'] = (
                    df['individual_p_value'].astype(float) * n_total
                ).clip(upper=1.0)
                df['individual_is_significant_bonferroni'] = df['individual_p_value_bonferroni'] < 0.05

        return df

    def _compute_interaction_pairwise_metrics(self, col_a, col_b):
        """
        Compute pairwise relationship metrics between two interacting columns.
        Returns dict with: pearson, spearman, MI, scale_ratio.
        
        Scale ratio = max(std_a, std_b) / min(std_a, std_b).
        When this is large (>100), additive interactions are meaningless because
        the smaller column becomes noise relative to the larger one.
        """
        X_ref = self._last_X_ref
        metrics = {
            'pairwise_corr_ab': np.nan,
            'pairwise_spearman_ab': np.nan,
            'pairwise_mi_ab': np.nan,
            'interaction_scale_ratio': np.nan,
        }
        
        if col_a not in X_ref.columns or col_b not in X_ref.columns:
            return metrics
        
        a_is_num = pd.api.types.is_numeric_dtype(X_ref[col_a])
        b_is_num = pd.api.types.is_numeric_dtype(X_ref[col_b])
        
        try:
            if a_is_num and b_is_num:
                clean_idx = X_ref[col_a].notna() & X_ref[col_b].notna()
                a_clean = X_ref[col_a][clean_idx]
                b_clean = X_ref[col_b][clean_idx]
                
                if len(a_clean) > 10:
                    # Pearson
                    metrics['pairwise_corr_ab'] = abs(a_clean.corr(b_clean))
                    
                    # Spearman
                    sp_corr, _ = spearmanr(a_clean, b_clean)
                    metrics['pairwise_spearman_ab'] = abs(sp_corr) if not np.isnan(sp_corr) else np.nan
                    
                    # Scale ratio (std-based)
                    std_a = a_clean.std()
                    std_b = b_clean.std()
                    if min(std_a, std_b) > 1e-10:
                        metrics['interaction_scale_ratio'] = max(std_a, std_b) / min(std_a, std_b)
                    else:
                        metrics['interaction_scale_ratio'] = np.inf
            
            # MI: works for all column type combinations
            if a_is_num:
                enc_a = X_ref[col_a].fillna(X_ref[col_a].median()).values.reshape(-1, 1)
            else:
                le_a = LabelEncoder()
                enc_a = le_a.fit_transform(X_ref[col_a].astype(str).fillna('NaN')).reshape(-1, 1)
            
            if b_is_num:
                target_for_mi = X_ref[col_b].fillna(X_ref[col_b].median())
            else:
                le_b = LabelEncoder()
                target_for_mi = pd.Series(
                    le_b.fit_transform(X_ref[col_b].astype(str).fillna('NaN')),
                    index=X_ref[col_b].index
                )
            
            # MI between the two columns (treat col_b as "target")
            mi_val = mutual_info_regression(enc_a, target_for_mi, random_state=42)[0]
            metrics['pairwise_mi_ab'] = float(mi_val)
        except:
            pass
        
        return metrics

    
    # Methods that are mathematically meaningful with binary columns
    BINARY_SAFE_METHODS = {
        'product_interaction',   # A Ã— binary = conditional masking (marginal value)
        'group_mean',            # mean(A) by binary = genuine conditional mean
        'group_std',             # std(A) by binary = genuine conditional spread
    }

    def _filter_methods_for_binary(self, pair_methods, col_a, col_b):
        """
        Filter interaction methods when one or both columns are binary.

        Binary columns (0/1 or 2-unique) make most arithmetic interactions
        degenerate:
          - division by binary: A/0=inf, A/1=A â†’ useless
          - addition/subtraction of binary: shifts A by 0 or 1 â†’ same splits
          - abs_diff with binary: equivalent to subtraction â†’ same problem

        Only product (conditional masking) and group aggregations remain useful.
        Returns the filtered method list, or the original if neither column is binary.
        """
        a_is_binary = self._column_meta_cache.get(col_a, {}).get('is_binary', False)
        b_is_binary = self._column_meta_cache.get(col_b, {}).get('is_binary', False)

        # Fallback: check nunique directly if not in cache
        if not a_is_binary and col_a in self._X_full_ref.columns:
            a_is_binary = self._X_full_ref[col_a].nunique() <= 2
        if not b_is_binary and col_b in self._X_full_ref.columns:
            b_is_binary = self._X_full_ref[col_b].nunique() <= 2

        if a_is_binary or b_is_binary:
            filtered = [m for m in pair_methods if m in self.BINARY_SAFE_METHODS]
            if len(filtered) < len(pair_methods):
                self.log(f"    Binary filter: {col_a if a_is_binary else col_b} is binary, "
                         f"reduced methods {len(pair_methods)}â†’{len(filtered)}")
            return filtered

        return pair_methods

    def _is_scale_compatible(self, col_a, col_b, threshold=100.0):
        """
        Check if two numeric columns are scale-compatible for additive interactions.
        
        Returns True if addition/subtraction is meaningful (scale ratio < threshold).
        Non-numeric columns always return True (scale doesn't apply to encodings).
        """
        X_ref = self._last_X_ref
        if col_a not in X_ref.columns or col_b not in X_ref.columns:
            return True
        if not pd.api.types.is_numeric_dtype(X_ref[col_a]) or not pd.api.types.is_numeric_dtype(X_ref[col_b]):
            return True
        
        try:
            std_a = X_ref[col_a].std()
            std_b = X_ref[col_b].std()
            if min(std_a, std_b) < 1e-10:
                return False  # One column is near-constant
            ratio = max(std_a, std_b) / min(std_a, std_b)
            return ratio < threshold
        except:
            return True

    def _is_interaction_duplicate(self, method, col_a, col_b, col_c=None):
        """
        Phase 0.1: Check if this (method, col_a, col_b[, col_c]) has already been tested.
        For commutative operations, sort column names so AxB == BxA.
        Returns True if duplicate (should skip), False if new (registers it).
        """
        # Commutative operations: order doesn't matter
        COMMUTATIVE = {'product_interaction', 'addition_interaction', 'abs_diff_interaction',
                       'three_way_interaction', 'three_way_addition', 'cat_concat'}
        
        if method in COMMUTATIVE:
            if col_c:
                key = (method, tuple(sorted([col_a, col_b, col_c])))
            else:
                key = (method, tuple(sorted([col_a, col_b])))
        else:
            # Directional: division, subtraction, group_mean, group_std, three_way_ratio
            if col_c:
                key = (method, col_a, col_b, col_c)
            else:
                key = (method, col_a, col_b)
        
        if key in self._tested_interactions:
            return True  # duplicate -- skip
        self._tested_interactions.add(key)
        return False  # new -- proceed

    def _get_col_interaction_features(self, col_name, prefix='col_b'):
        """
        v6: Extract 8 curated metadata fields from the column cache for use
        as col_b or col_c features in interaction rows.
        
        Returns a dict like {'col_b_is_numeric': ..., 'col_b_skewness': ..., ...}
        Falls back to np.nan for all fields if column wasn't cached (not in selected_cols).
        """
        FEATURE_KEYS = [
            'is_numeric', 'skewness', 'unique_ratio', 'null_pct',
            'outlier_ratio', 'baseline_feature_importance', 'entropy',
            'composite_predictive_score',
        ]
        result = {}
        meta = self._column_meta_cache.get(col_name, {})
        for key in FEATURE_KEYS:
            result[f'{prefix}_{key}'] = meta.get(key, np.nan)
        # Rename baseline_feature_importance to baseline_importance for schema match
        result[f'{prefix}_baseline_importance'] = result.pop(f'{prefix}_baseline_feature_importance', np.nan)
        return result

    def _log_interaction_result(self, ds_meta, baseline_fold_scores, base_score,
                                col_name, method, full_score, full_folds,
                                indiv_base, indiv_new, col_a, col_b, col_c=None,
                                interaction_source='cross_column', tree_pair_score=np.nan):
        """Helper to log an interaction result with both full and individual stats."""
        self.n_tests_performed += 1
        
        # Full-model stats
        delta_full = self._get_delta(base_score, full_score)
        delta_full_norm = self._normalize_delta(delta_full, base_score)
        t_full, p_full, p_full_bonf, sig_full, sig_full_bonf = \
            self._paired_ttest_with_bonferroni(baseline_fold_scores, full_folds, self.n_tests_performed)
        
        # Individual stats
        indiv_base_score = indiv_base[0] if indiv_base and indiv_base[0] is not None else np.nan
        indiv_base_folds = indiv_base[2] if indiv_base and indiv_base[2] is not None else None
        indiv_new_score = indiv_new[0] if indiv_new and indiv_new[0] is not None else np.nan
        indiv_new_folds = indiv_new[2] if indiv_new and indiv_new[2] is not None else None
        
        indiv_delta = np.nan
        indiv_delta_norm = np.nan
        t_indiv, p_indiv, p_indiv_bonf = np.nan, np.nan, np.nan
        sig_indiv, sig_indiv_bonf = False, False
        
        if not np.isnan(indiv_base_score) and not np.isnan(indiv_new_score):
            indiv_delta = self._get_delta(indiv_base_score, indiv_new_score)
            indiv_delta_norm = self._normalize_delta(indiv_delta, indiv_base_score)
            if indiv_base_folds is not None and indiv_new_folds is not None:
                t_indiv, p_indiv, p_indiv_bonf, sig_indiv, sig_indiv_bonf = \
                    self._paired_ttest_with_bonferroni(indiv_base_folds, indiv_new_folds, self.n_tests_performed)

        # Pairwise metrics (correlation, spearman, MI, scale ratio)
        pairwise_metrics = self._compute_interaction_pairwise_metrics(col_a, col_b)
        
        # v6: Col_b and col_c features from cache
        col_b_features = self._get_col_interaction_features(col_b, prefix='col_b')
        col_c_features = self._get_col_interaction_features(col_c, prefix='col_c') if col_c else {
            f'col_c_{k}': np.nan for k in ['is_numeric', 'skewness', 'unique_ratio', 'null_pct',
                                             'outlier_ratio', 'baseline_importance', 'entropy',
                                             'composite_predictive_score']
        }
        
        self._validate_and_append(
            ds_meta=ds_meta,
            column_name=col_name, method=method,
            delta=delta_full, delta_normalized=delta_full_norm,
            absolute_score=full_score, is_interaction=True,
            t_statistic=t_full, p_value=p_full,
            p_value_bonferroni=p_full_bonf,
            is_significant=sig_full, is_significant_bonferroni=sig_full_bonf,
            individual_baseline_score=indiv_base_score,
            individual_intervention_score=indiv_new_score,
            individual_delta=indiv_delta,
            individual_delta_normalized=indiv_delta_norm,
            individual_p_value=p_indiv,
            individual_p_value_bonferroni=p_indiv_bonf,
            individual_is_significant=sig_indiv,
            individual_is_significant_bonferroni=sig_indiv_bonf,
            interaction_col_a=col_a, interaction_col_b=col_b,
            interaction_col_c=col_c if col_c else np.nan,
            pairwise_corr_ab=pairwise_metrics['pairwise_corr_ab'],
            pairwise_spearman_ab=pairwise_metrics['pairwise_spearman_ab'],
            pairwise_mi_ab=pairwise_metrics['pairwise_mi_ab'],
            interaction_scale_ratio=pairwise_metrics['interaction_scale_ratio'],
            interaction_source=interaction_source,
            # v5: noise-floor calibration and effect sizes
            null_delta=self._null_delta,
            null_std=self._null_std,
            cohens_d=self._compute_cohens_d(baseline_fold_scores, full_folds),
            individual_cohens_d=self._compute_cohens_d(indiv_base_folds, indiv_new_folds),
            calibrated_delta=self._compute_calibrated_delta(delta_full),
            individual_calibrated_delta=self._compute_calibrated_delta(indiv_delta) if not np.isnan(indiv_delta) else np.nan,
            tree_pair_score=tree_pair_score,
            # v6: col_b and col_c features
            **col_b_features,
            **col_c_features,
        )
        
        sig_str = "[OK]" if sig_full else "[X]"
        isig_str = "[OK]" if sig_indiv else "[X]"
        indiv_str = f"indiv={indiv_delta_norm:+.2f}% p={p_indiv:.4f} {isig_str}" if not np.isnan(indiv_delta_norm) else "indiv=N/A"
        self.log(f"  {col_name}: full={delta_full_norm:+.2f}% p={p_full:.4f} {sig_str} | {indiv_str}")


# =============================================================================
# OPENML PIPELINE
# =============================================================================

# -------- Curated temporal/date-heavy datasets for testing temporal detection --------
# Each entry: (dataset_id, description, expected_temporal_features)
# These are processed AFTER CC18 to enrich the meta-learning DB with temporal signal.
TEMPORAL_EXTRA_DATASETS = [
    # -------- HIGH QUALITY: Verified Tasks & Temporal Columns --------
    
    # Bike Sharing Demand (Regression)
    # Features: year, month, hour, weekday
    (42713, 'Bike_Sharing_Demand_Verified', 
     'hour, month, weekday -> cyclical; regression'),

    # Seoul Bike Sharing (Regression)
    # Features: Hour, Date (string), Seasons
    (42736, 'Seoul_Bike_Sharing_Verified', 
     'Hour -> cyclical; Date -> extract; regression'),

    # Beijing PM2.5 Data (Regression)
    # Features: year, month, day, hour
    (42721, 'Beijing_PM25', 
     'month, day, hour -> cyclical; regression'),

    # Electricity (Classification)
    # Features: date (numeric/string), day (1-7), period (0-48)
    # Note: Using ID 151 usually works, but if it fails, try 1471
    (151, 'Electricity_Elec2', 
     'day, period -> cyclical; classification'),

    # Traffic Volume (Regression)
    # Features: date_time (string) -> needs parsing
    (42714, 'Metro_Interstate_Traffic_Volume', 
     'date_time -> date_extract; regression'),

    # Online Shoppers Intention (Classification)
    # Features: Month (string "Feb", "Mar"), Weekend (bool), SpecialDay
    (42730, 'Online_Shoppers_Intention', 
     'Month -> cyclical/cat; Weekend; classification'),

    # Avocados (Regression)
    # Features: Date (string)
    (42722, 'Avocado_Sales', 
     'Date -> date_extract; regression'),

    # Appliances Energy Prediction (Regression)
    # Features: date (string)
    (42727, 'Appliances_Energy', 
     'date -> date_extract; regression'),
     
    # Ozone Level Detection (Classification)
    # Features: Date (string)
    (1487, 'Ozone_Level_8hr', 
     'Date -> date_extract; classification'),

    # -------- EXTRA: DateTime String Heavy --------
    
    # Bitcoin Heist (Classification)
    # Features: only has 'year' and 'day' integers, but good for simple cyclical
    (42617, 'Bitcoin_Heist', 
     'day -> cyclical; classification'),

    # Walmart Store Sales (Regression)
    # Features: Date (string)
    (42732, 'Walmart_Store_Sales', 
     'Date -> date_extract; regression')
]

# Subset for quick temporal testing (smaller datasets only)
# TEMPORAL_QUICK_DATASETS = [42712, 151, 46297, 42041, 44226]


def _detect_openml_task_type(task):
    """
    Robustly extract task type from an OpenML task object.
    
    Handles:
    - Old API: task.task_type_id is int (1=classification, 2=regression)
    - New API: task.task_type_id is string ('Supervised Classification', etc.)
    - Enum API: task.task_type_id is an enum with .value attribute
    - Missing attribute: falls back to task.task_type string
    
    Returns: 'classification', 'regression', or None if detection fails.
    """
    CLASSIFICATION_SIGNALS = {'1', 'supervised classification', 'classification'}
    REGRESSION_SIGNALS = {'2', 'supervised regression', 'regression'}
    
    # Try task_type_id first (primary attribute)
    for attr_name in ('task_type_id', 'task_type'):
        if hasattr(task, attr_name):
            val = getattr(task, attr_name)
            
            # Handle enum objects (e.g. TaskType.SUPERVISED_CLASSIFICATION)
            if hasattr(val, 'value'):
                val = val.value
            
            val_str = str(val).strip().lower()
            
            if val_str in CLASSIFICATION_SIGNALS:
                return 'classification'
            if val_str in REGRESSION_SIGNALS:
                return 'regression'
    
    return None

def _resolve_task_for_dataset(dataset_id, preferred_type=None):
    """
    Find a suitable OpenML task for a dataset ID.
    Robustly handles missing columns and string-based task types.
    """
    try:
        tasks_df = openml.tasks.list_tasks(data_id=dataset_id, output_format='dataframe')
        
        if tasks_df.empty:
            # Fallback: If no tasks exist, we can't fetch a task ID. 
            # We return a specific flag to tell the pipeline to run in "Dataset Mode"
            return -1, 'dataset_mode'

        # 1. Standardize Task Type ID
        # Newer OpenML versions return 'task_type' (string) instead of 'task_type_id' (int)
        if 'task_type_id' not in tasks_df.columns and 'task_type' in tasks_df.columns:
            type_map = {
                'Supervised Classification': 1, 
                'Supervised Regression': 2,
                'Clustering': 3, 
                'Learning Curve': 4
            }
            tasks_df['task_type_id'] = tasks_df['task_type'].map(type_map).fillna(0).astype(int)
        
        # If we still don't have IDs, we can't filter safely
        if 'task_type_id' not in tasks_df.columns:
            # Last ditch: assume everything is valid if we can't check
            sup = tasks_df
        else:
            # Filter to Supervised Classification (1) or Regression (2)
            sup = tasks_df[tasks_df['task_type_id'].isin([1, 2])]

        if sup.empty:
            return -1, 'dataset_mode'

        # 2. Filter by Preferred Type
        if preferred_type == 'classification':
            typed = sup[sup['task_type_id'] == 1]
            if not typed.empty: sup = typed
        elif preferred_type == 'regression':
            typed = sup[sup['task_type_id'] == 2]
            if not typed.empty: sup = typed

        # 3. Filter by Estimation Procedure (Prefer 10-fold)
        # Check if column exists (sometimes it's 'estimation_procedure_id')
        if 'estimation_procedure' in sup.columns:
            cv10 = sup[sup['estimation_procedure'].str.contains('10-fold', case=False, na=False)]
            if not cv10.empty:
                sup = cv10

        # 4. Pick best (most runs or first available)
        if 'NumberOfRuns' in sup.columns:
            best = sup.sort_values('NumberOfRuns', ascending=False).iloc[0]
        else:
            best = sup.iloc[0]

        t_id = int(best['tid'])
        t_type = 'classification' if best.get('task_type_id') == 1 else 'regression'
        
        return t_id, t_type

    except Exception as e:
        print(f"  [!]   Error resolving task for dataset {dataset_id}: {e}")
        # Return fallback to try loading dataset directly
        return -1, 'dataset_mode'



# =========================================================================
    # Infering Task Type with a Robust Heuristic Ladder to Avoid Misclassification of High-Cardinality Datasets
    # =========================================================================

def _infer_task_type(y):
    """
    Robustly infer whether a target variable is classification or regression.
    
    Replaces the broken `nunique() > 20` heuristic which misclassifies datasets
    like letter (26 classes), cifar (100 classes), etc.
    
    Heuristic ladder:
    1. Non-numeric target -> classification (always)
    2. <= 2 unique values -> classification (binary)
    3. Float (fractional values) with many uniques -> regression
    4. Integer-like values: check unique_ratio (nunique / n_rows)
    - unique_ratio < 0.05 -> classification (few classes, many repeats)
    - nunique <= 200 AND unique_ratio < 0.10 -> classification (medium class count)
    - Otherwise -> regression
    """
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    
    y_clean = y.dropna()
    if len(y_clean) == 0:
        return 'classification'
    
    nunique = int(y_clean.nunique())
    n = len(y_clean)
    
    # Binary
    if nunique <= 2:
        return 'classification'
    
    # Check if values are integer-like (no fractional part)
    try:
        is_integer_like = (y_clean % 1 == 0).all()
    except (TypeError, ValueError):
        is_integer_like = False
    
    # Float values with many unique -> regression
    # (allows small float enumerations like {0.0, 1.0, 2.0} to still be classification)
    if not is_integer_like and nunique > 20:
        return 'regression'
    
    # Integer-like or few-unique float: use unique ratio
    unique_ratio = nunique / n
    
    # Very low unique ratio -> classification
    # Example: cifar has 100 classes / 60000 rows = 0.0017
    if unique_ratio < 0.05:
        return 'classification'
    
    # Moderate unique ratio but still manageable class count
    # Example: a 200-class problem with 2500 rows -> ratio=0.08
    if nunique <= 200 and unique_ratio < 0.10:
        return 'classification'
    
    return 'regression'

# =========================================================================
# Hanlding Oversized Datasets with Intelligent Reduction Instead of Skipping
# =========================================================================

def _smart_size_reduction(X, y, task_type, cell_limit=100_000_000):
    """
    Intelligently reduce dataset size instead of skipping it entirely.
    
    Strategy:
    1. Lightweight column pruning (constants, near-constants >99%, zero-variance)
    2. Re-check cell count with pruned columns
    3. If still over limit: stratified row sampling to bring under limit
    
    Returns: (X_reduced, y_reduced, reduction_log: str)
    """
    original_shape = X.shape
    log_parts = []
    
    # --- Step 1: Lightweight column pruning ---
    drop_cols = []
    for col in X.columns:
        nunique = X[col].nunique(dropna=True)
        if nunique <= 1:
            drop_cols.append(col)
            continue
        top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.99:
            drop_cols.append(col)
            continue
        # Zero-variance numeric columns
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].std() < 1e-8:
                drop_cols.append(col)

    if drop_cols:
        X = X.drop(columns=drop_cols)
        log_parts.append(f"pruned {len(drop_cols)} junk cols ({original_shape[1]}->{X.shape[1]})")
    
    if X.shape[1] == 0:
        return X, y, "all columns pruned -- skip"
    
    # --- Step 2: Estimate effective workload ---
    # The collector caps intervention testing at 100 + 10% of columns,
    # but the baseline evaluation still uses all remaining columns.
    # Use post-prune column count for the cell check.
    current_cells = X.shape[0] * X.shape[1]
    
    if current_cells <= cell_limit:
        msg = f"Under limit after column pruning ({current_cells:,} cells). " + "; ".join(log_parts) if log_parts else f"Already under limit ({current_cells:,} cells)"
        return X, y, msg
    
    # --- Step 3: Stratified row sampling ---
    target_rows = cell_limit // X.shape[1]
    # Keep at least 1000 rows for meaningful CV
    target_rows = max(target_rows, 1000)
    
    if target_rows >= X.shape[0]:
        msg = f"Under limit after pruning. " + "; ".join(log_parts) if log_parts else "No reduction needed"
        return X, y, msg
    
    original_rows = X.shape[0]
    
    if task_type == 'classification':
        # Stratified sampling: preserve class distribution
        from sklearn.model_selection import StratifiedShuffleSplit
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=target_rows, random_state=42)
            sample_idx, _ = next(sss.split(X, y))
            X = X.iloc[sample_idx].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)
        except ValueError:
            # Fallback if stratification fails (e.g., class with 1 sample)
            idx = np.random.RandomState(42).choice(len(X), size=target_rows, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
    else:
        # Regression: bin target into quantiles, then stratify on bins
        try:
            n_bins = min(10, len(y.unique()))
            y_bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=target_rows, random_state=42)
            sample_idx, _ = next(sss.split(X, y_bins))
            X = X.iloc[sample_idx].reset_index(drop=True)
            y = y.iloc[sample_idx].reset_index(drop=True)
        except (ValueError, TypeError):
            # Fallback: uniform random sample
            idx = np.random.RandomState(42).choice(len(X), size=target_rows, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
    
    log_parts.append(f"stratified sampled {original_rows:,}->{len(X):,} rows")
    final_cells = X.shape[0] * X.shape[1]
    log_parts.append(f"final: {X.shape} = {final_cells:,} cells")
    
    return X, y, "; ".join(log_parts)







def run_openml_pipeline(output_dir='./meta_learning_output',
                        n_folds=5, n_repeats=3, store_fold_scores=False,
                        time_budget=14400, task_list_file=None):
    """Run the meta-learning data collection pipeline locally (sequential).

    Reads a pre-generated task_list.json (from generate_task_list.py) and
    processes each task one by one.  For large-scale runs, use
    DataCollector_slurm.py with SLURM array jobs instead.

    Args:
        output_dir: Directory for output CSV, checkpoints, logs.
        n_folds: Number of CV folds.
        n_repeats: Number of CV repeats.
        store_fold_scores: Whether to store per-fold scores.
        time_budget: Maximum seconds per dataset.
        task_list_file: Path to task_list.json.  Defaults to
                        <output_dir>/task_list.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load task list ---
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

    # --- Environment metadata ---
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'lightgbm_version': lgb.__version__,
        'n_folds': n_folds, 'n_repeats': n_repeats,
        'n_tasks': len(task_list),
        'store_fold_scores': store_fold_scores,
        'version': 'v7',
    }
    with open(os.path.join(output_dir, 'environment.json'), 'w') as f:
        json.dump(env_info, f, indent=2)

    print("=" * 80)
    print("OPENML META-LEARNING PIPELINE v8 (Local Sequential)")
    print("=" * 80)
    print(f"  Output:      {output_dir}")
    print(f"  Tasks:       {len(task_list)}")
    print(f"  CV:          {n_folds}-fold x {n_repeats} repeats")
    print(f"  Time budget: {time_budget}s per dataset")
    print("=" * 80)

    csv_file = os.path.join(output_dir, 'meta_learning_db.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')

    processed_tasks = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_tasks = set(json.load(f).get('processed_tasks', []))
            print(f"Checkpoint: {len(processed_tasks)} already done")

    tasks_processed = 0
    tasks_skipped = 0
    tasks_failed = 0

    for entry in task_list:
        task_id = entry['task_id']
        dataset_id = entry.get('dataset_id')
        forced_type = entry.get('task_type')
        ckpt_key = str(task_id) if task_id and task_id > 0 else f"ds_{dataset_id}"

        if ckpt_key in processed_tasks:
            continue

        try:
            print(f"\n{'=' * 80}")
            print(f"[{tasks_processed + tasks_skipped + tasks_failed + 1}/{len(task_list)}] "
                  f"task={task_id}  dataset={dataset_id}  "
                  f"({entry.get('dataset_name', '?')})")
            print("=" * 80)

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
                print("  No valid task_id or dataset_id -- skipping")
                tasks_skipped += 1
                continue

            if not target_col:
                print(f"  No default target attribute -- skipping")
                tasks_skipped += 1
                processed_tasks.add(ckpt_key)
                continue

            # Infer task type from data if still unknown
            if task_type is None:
                _y_tmp = dataset.get_data(target=target_col, dataset_format='dataframe')[1]
                task_type = _infer_task_type(_y_tmp)

            # --- Load data ---
            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            if task_type is None:
                task_type = _infer_task_type(y)

            # --- Smart size reduction for very large datasets ---
            raw_cells = X.shape[0] * X.shape[1]
            if raw_cells > 100_000_000:
                print(f"  Large dataset ({X.shape}, {raw_cells:,} cells) -- applying smart reduction...")
                X, y, reduction_msg = _smart_size_reduction(X, y, task_type, cell_limit=100_000_000)
                print(f"   ' {reduction_msg}")
                if X.shape[1] == 0:
                    print("  No columns survived pruning -- skipping")
                    processed_tasks.add(ckpt_key)
                    tasks_skipped += 1
                    continue

            print(f"  Type: {task_type} | Shape: {X.shape}")

            # --- Encode classification targets ---
            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

            # --- Run collector ---
            dataset_start = time.time()
            collector = MetaDataCollector(
                task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
                store_fold_scores=store_fold_scores, output_dir=output_dir,
                time_budget_seconds=time_budget)

            full_df = pd.concat([X, y], axis=1)
            df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
            dataset_elapsed = time.time() - dataset_start

            # --- Save results ---
            if df_result.empty:
                print(f"  Empty result -- skipped ({dataset_elapsed:.1f}s)")
                tasks_skipped += 1
            else:
                df_result['openml_task_id'] = task_id if (task_id and task_id > 0) else f"ds_{dataset_id}"
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                df_result['task_type'] = task_type

                MetaDataCollector.write_csv(df_result, csv_file)
                print(f"  Done: {len(df_result)} rows in {dataset_elapsed:.1f}s")
                tasks_processed += 1

            processed_tasks.add(ckpt_key)
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed_tasks': list(processed_tasks)}, f)

            del collector, full_df
            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            tasks_failed += 1
            processed_tasks.add(ckpt_key)
            continue

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print(f"DONE: {tasks_processed} ok, {tasks_skipped} skipped, {tasks_failed} failed")
    if os.path.exists(csv_file):
        try:
            db = pd.read_csv(csv_file)
            print(f"Total rows: {len(db)} | Datasets: {db['dataset_name'].nunique()}")
            print(f"Significant full-model (p<0.05): {db['is_significant'].sum()}")
            print(f"Significant individual (p<0.05): {db['individual_is_significant'].sum()}")
            print(f"Bonferroni full: {db['is_significant_bonferroni'].sum()}")
            print(f"Bonferroni individual: {db['individual_is_significant_bonferroni'].sum()}")
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meta-Learning Pipeline v5 (local sequential)")
    parser.add_argument('--output_dir', default='./meta_learning_output')
    parser.add_argument('--task_list', default=None,
                        help='Path to task_list.json (default: <output_dir>/task_list.json)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--time_budget', type=int, default=14400)
    parser.add_argument('--store_fold_scores', action='store_true')
    args = parser.parse_args()

    run_openml_pipeline(
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        store_fold_scores=args.store_fold_scores,
        time_budget=args.time_budget,
        task_list_file=args.task_list,
    )