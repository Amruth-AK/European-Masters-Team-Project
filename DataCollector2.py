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
import networkx as nx
import pyarrow
import openml
import psutil
import math

warnings.filterwarnings('ignore')


class MetaDataCollector:
    """
    Meta-Learning Data Collector v4
    
    v4 changes over v3:
    - Tightened property gates: log/sqrt/poly/binning require much stronger indicators
    - Aggressive early-stop: monotonic-transform methods stop after 5 failures (was 10+)
    - New interaction types: addition, subtraction, abs_diff (2-way); addition & ratio (3-way)
    - Expanded interaction scope: top 8 numeric cols (was 5), top 5 3-way combos (was 3)
    - Default interactions for num-num pairs now test product + division + subtraction
    - Expanded row_stats: zeros_count, zeros_ratio, min, max, range, missing_ratio
    - Individual eval for interactions includes original columns + interaction feature
    
    v3 changes over v2:
    - Column cap: max columns = 100 + ceil(10% of total)
    - Stratified column selection: 70% proportional by type, 30% best-available
    - Target leakage detection (>0.95 corr/MI/PPS with target → drop)
    - Default interactions: every selected column tested with top-1 and top-2 global columns
    - Individual p-values (paired t-test on individual fold scores)
    - Composite predictive score for column ranking (importance + MI + PPS)
    """
    
    # =========================================================================
    # FIXED CSV SCHEMA — every row MUST have exactly these columns in this order.
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
        # --- Intervention identification (3 fields) ---
        'column_name', 'method', 'is_interaction',
        # --- Full-model evaluation (7 fields) ---
        'delta', 'delta_normalized', 'absolute_score',
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
        # --- OpenML identifiers (added by pipeline runner) ---
        'openml_task_id', 'dataset_name', 'dataset_id', 'task_type',
    ]
    
    def __init__(self, task_type='classification', n_folds=10, n_repeats=1, 
                 store_fold_scores=False, output_dir='./meta_learning_output'):
        self.task_type = task_type
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.store_fold_scores = store_fold_scores
        self.output_dir = output_dir
        self.meta_data_log = []
        self.n_tests_performed = 0
        
        # Cache for individual column baselines: cache_key -> (score, std, fold_scores)
        self._individual_baseline_cache = {}
        self._last_X_ref = pd.DataFrame()  # Set properly in collect()
        
        # Adaptive method tracking: {method: {'tested': N, 'improved': N, 'disabled': bool}}
        # Reset per dataset in collect()
        self._method_tracker = {}
        
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
            'n_jobs': -1
        }
        
        # Simpler params for individual column evaluation (few features → simpler model)
        self.individual_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
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
        print(f"❌ {message}")
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
            pps = self._calculate_pps(X[col], y_numeric)
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
        
        # 2. Compute MI for all columns
        mi_scores = {}
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    encoded = X[col].fillna(X[col].median()).to_frame()
                else:
                    le = LabelEncoder()
                    encoded = pd.DataFrame(
                        le.fit_transform(X[col].astype(str).fillna('NaN')),
                        index=X[col].index, columns=[col]
                    )
                
                if self.task_type == 'classification':
                    mi_scores[col] = mutual_info_classif(encoded, y_numeric, random_state=42)[0]
                else:
                    mi_scores[col] = mutual_info_regression(encoded, y_numeric, random_state=42)[0]
            except:
                mi_scores[col] = 0.0
        
        mi_max = max(mi_scores.values()) if mi_scores and max(mi_scores.values()) > 0 else 1.0
        
        # 3. Compute PPS for all columns
        pps_scores = {}
        for col in X.columns:
            pps_scores[col] = self._calculate_pps(X[col], y_numeric)
        
        # 4. Composite score
        for col in X.columns:
            imp_score = imp_normalized.get(col, 0.0)
            mi_score = mi_scores.get(col, 0.0) / mi_max
            pps_score = pps_scores.get(col, 0.0)
            
            scores[col] = 0.4 * imp_score + 0.3 * mi_score + 0.3 * pps_score
        
        return scores

    # =========================================================================
    # STRATIFIED COLUMN SELECTION
    # =========================================================================

    def _select_columns_stratified(self, X, y, base_feature_importance):
        """
        Select columns with cap and stratified type distribution.
        
        Cap formula: min(n_cols, 100 + ceil(0.1 * n_cols))
        
        Strategy:
        - 70% of slots allocated proportionally by type (numeric/categorical/date/text)
        - 30% of slots filled by best available regardless of type
        - Within each type, ranked by composite predictive score
        
        Returns: (selected_columns, selection_report)
        """
        n_cols = X.shape[1]
        cap = min(n_cols, 100 + math.ceil(0.1 * n_cols))
        
        # If under cap, use all columns
        if n_cols <= cap:
            return X.columns.tolist(), {
                'cap': cap, 'n_original': n_cols, 'n_selected': n_cols,
                'method': 'all_columns (under cap)'
            }
        
        self.log(f"Column selection: {n_cols} columns → cap at {cap}")
        
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
            if len(X_sample) > 5000:
                X_sample = X_sample.sample(5000, random_state=42)
            
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
            'row_col_ratio': X.shape[0] / max(X.shape[1], 1)
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
            if len(numeric_cols) > 200:
                sample_cols = numeric_cols[:200]  # Use first 200 for speed
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
            sample_cols = numeric_cols[:100] if len(numeric_cols) > 100 else numeric_cols
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

            # Numeric-only → NaN
            for k in ['shapiro_p_value', 'range_iqr_ratio', 'dominant_quartile_pct',
                       'min_val', 'max_val', 'mean_val', 'mode_val', 'pct_in_0_1_range',
                       'spearman_corr_target', 'hartigan_dip_pval', 'skewness', 'kurtosis',
                       'coeff_variation', 'zeros_ratio']:
                res[k] = np.nan
            res['has_multiple_modes'] = 0
            res['bimodality_proxy_heuristic'] = 0
            res['is_multimodal'] = 0
        
        res['pps_score'] = self._calculate_pps(col_series, y_numeric)

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
        """
        if not method:
            return True
        
        # Methods that don't need a specific column
        if not col_to_transform and method not in ['row_stats', 'null_intervention']:
            return True
        
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
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform], errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform], errors='coerce'))]:
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
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform], errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform], errors='coerce'))]:
                    m = src.dt.month.fillna(0)
                    df_x[f"{col_to_transform}_month_sin"] = np.sin(2 * np.pi * m / 12)
                    df_x[f"{col_to_transform}_month_cos"] = np.cos(2 * np.pi * m / 12)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_dow':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform], errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform], errors='coerce'))]:
                    d = src.dt.dayofweek.fillna(0)
                    df_x[f"{col_to_transform}_dow_sin"] = np.sin(2 * np.pi * d / 7)
                    df_x[f"{col_to_transform}_dow_cos"] = np.cos(2 * np.pi * d / 7)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_hour':
                dt_train = pd.to_datetime(X_train[col_to_transform], errors='coerce')
                dt_val = pd.to_datetime(X_val[col_to_transform], errors='coerce')
                if dt_train.dt.hour.notna().any():
                    for df_x, src in [(X_train, dt_train), (X_val, dt_val)]:
                        h = src.dt.hour.fillna(0)
                        df_x[f"{col_to_transform}_hour_sin"] = np.sin(2 * np.pi * h / 24)
                        df_x[f"{col_to_transform}_hour_cos"] = np.cos(2 * np.pi * h / 24)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_elapsed_days':
                dt_train = pd.to_datetime(X_train[col_to_transform], errors='coerce')
                dt_val = pd.to_datetime(X_val[col_to_transform], errors='coerce')
                min_date = dt_train.min()
                if pd.notna(min_date):
                    X_train[f"{col_to_transform}_days_elapsed"] = (dt_train - min_date).dt.total_seconds().fillna(-86400) / 86400
                    X_val[f"{col_to_transform}_days_elapsed"] = (dt_val - min_date).dt.total_seconds().fillna(-86400) / 86400
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'date_cyclical_day':
                for df_x, src in [(X_train, pd.to_datetime(X_train[col_to_transform], errors='coerce')),
                                   (X_val, pd.to_datetime(X_val[col_to_transform], errors='coerce'))]:
                    d = src.dt.day.fillna(0)
                    df_x[f"{col_to_transform}_day_sin"] = np.sin(2 * np.pi * d / 31)
                    df_x[f"{col_to_transform}_day_cos"] = np.cos(2 * np.pi * d / 31)
                X_train = X_train.drop(columns=[col_to_transform])
                X_val = X_val.drop(columns=[col_to_transform])

            elif method == 'cyclical_encode':
                # Generic cyclical encoding for standalone temporal columns
                # (month, hour, day_of_week, etc.). Detects period from column metadata.
                # ADDS sin/cos columns — keeps original for tree histogram splits.
                temporal_type, period = self._detect_temporal_component(
                    X_train[col_to_transform], col_to_transform)
                if period is None:
                    return False
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
                    s = df_x[col_to_transform].fillna("").astype(str)
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
                # (a * b) / c — useful for rate-like derived features
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

            elif method == 'null_intervention':
                pass

            return True
        except Exception as e:
            self.log_error(f"Intervention {method} on {col_to_transform} failed", e)
            return False

    def evaluate_with_intervention(self, X, y, col_to_transform=None, col_second=None, 
                                   col_third=None, method=None, return_importance=False,
                                   use_individual_params=False):
        """Runs K-Fold CV with optional repeated CV."""
        scores = []
        params = self.individual_params if use_individual_params else self.base_params
        feat_imp_accum = None
        y = self._ensure_numeric_target(y)

        if self.n_repeats > 1:
            cv = RepeatedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=42)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        fold_idx = 0
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Apply intervention
            if col_to_transform and method:
                if not self._apply_intervention(X_train, X_val, y_train, col_to_transform, col_second, col_third, method):
                    return (None, None, None, None) if return_importance else (None, None, None)
            elif method in ['row_stats']:
                self._apply_intervention(X_train, X_val, y_train, col_to_transform=None, method=method)
            elif method == 'null_intervention':
                pass  # Do nothing

            X_train, X_val = self._prepare_data_for_model(X_train, X_val, y_train)
            
            if fold_idx == 0 and return_importance:
                feat_imp_accum = np.zeros(X_train.shape[1])

            try:
                if self.task_type == 'classification':
                    n_classes = y_train.nunique()
                    p = params.copy()
                    if n_classes > 2:
                        p.update({'objective': 'multiclass', 'num_class': n_classes, 'metric': 'multi_logloss'})
                        metric_name = 'multi_logloss'
                    else:
                        p.update({'objective': 'binary', 'metric': 'auc'})
                        metric_name = 'auc'

                    model = lgb.LGBMClassifier(**p)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              eval_metric=metric_name,
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    probs = model.predict_proba(X_val, num_iteration=model.best_iteration_)
                    if n_classes > 2:
                        scores.append(metrics.roc_auc_score(y_val, probs, multi_class='ovr'))
                    else:
                        scores.append(metrics.roc_auc_score(y_val, probs[:, 1]))
                else:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              eval_metric='l2',
                              callbacks=[early_stopping(stopping_rounds=30, verbose=False)])
                    preds = model.predict(X_val, num_iteration=model.best_iteration_)
                    scores.append(metrics.mean_squared_error(y_val, preds))
                
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
        return np.mean(scores), np.std(scores), scores

    def evaluate_individual(self, X_subset, y, col_to_transform=None, col_second=None,
                           col_third=None, method=None):
        """Evaluate using ONLY the specified column subset with simpler params."""
        return self.evaluate_with_intervention(
            X_subset, y, col_to_transform=col_to_transform,
            col_second=col_second, col_third=col_third,
            method=method, use_individual_params=True)

    def _get_individual_baseline(self, X, y, cols, cache_key=None):
        """Get individual baseline for column(s), with caching."""
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
        if self.task_type == 'classification':
            return new_score - base_score
        return base_score - new_score

    def _normalize_delta(self, delta, base_score):
        if self.task_type == 'classification':
            return delta * 100
        return (delta / base_score) * 100 if base_score != 0 else 0.0

    def _check_ceiling_effect(self, baseline_score):
        if self.task_type == 'classification':
            if baseline_score >= 0.97:
                return True, False, f"ceiling_hard (AUC={baseline_score:.5f} >= 0.97)"
            elif baseline_score >= 0.93:
                return False, True, f"ceiling_soft (AUC={baseline_score:.5f} >= 0.93)"
        else:
            if baseline_score < 1e-6:
                return True, False, f"ceiling_hard (MSE near-zero)"
            elif baseline_score < 1e-4:
                return False, True, f"ceiling_soft (MSE very low)"
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
        if len(X_proc) > 5000:
            np.random.seed(42)
            idx = np.random.choice(len(X_proc), 5000, replace=False)
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
                    lin = LinearRegression(); lin.fit(Xtr, ytr)
                    lin_scores.append(mean_squared_error(yvl, lin.predict(Xvl)))
                    tree = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42); tree.fit(Xtr, ytr)
                    tree_scores.append(mean_squared_error(yvl, tree.predict(Xvl)))
                else:
                    return 0.0
            return (np.mean(tree_scores) - np.mean(lin_scores)) if self.task_type == 'classification' else \
                   (np.mean(lin_scores) - np.mean(tree_scores)) if np.mean(tree_scores) > 0 else 0.0
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
            if X_f.shape[0] > 2000: X_f = X_f.sample(2000, random_state=42)
            if X_f.shape[1] > 500: X_f = X_f.iloc[:, :500]  # Cap columns for speed
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
                m = DecisionTreeClassifier(max_depth=3, random_state=42); m.fit(X, y)
                proba = m.predict_proba(X)
                return (roc_auc_score(y, proba[:, 1]) - 0.5) * 2 if proba.shape[1] == 2 else 0.0
            else:
                m = DecisionTreeRegressor(max_depth=3, random_state=42); m.fit(X, y)
                mse = mean_squared_error(y, m.predict(X))
                base = np.var(y)
                return max(0, 1 - mse / base) if base > 0 else 0.0
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
            if len(X_f) > 5000: X_f = X_f.sample(5000, random_state=42)
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
        # Minute: 0-59 (require full word 'minute' — 'min' is too ambiguous: min_temperature, min_val)
        (r'(?i)(?:^|[_\W])(?:minute)(?:$|[_\W\d])',
         lambda mn, mx: 0 <= mn and mx <= 59, 'minute', 60),
        # Second: 0-59 (require full word 'second' — 'sec' is too ambiguous: section, security)
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
    GATED_METHODS = {'log_transform', 'sqrt_transform', 'quantile_binning', 'polynomial_square'}

    # Methods where individual (single-column) evaluation is structurally uninformative.
    # 
    # For a single feature, LightGBM's histogram-based splitting evaluates ALL possible 
    # threshold splits. Any injective re-encoding of categories (frequency, target, hash) 
    # or monotonic transform of numerics (log, sqrt) produces the SAME optimal tree because 
    # the row-to-bin mapping is equivalent — just with different split thresholds.
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
        'frequency_encoding',   # re-encodes cat → frequency floats (injective)
        'target_encoding',      # re-encodes cat → target-smoothed floats (injective with smoothing)
        'hashing_encoding',     # re-encodes cat → hash integers (injective mod collisions)
        'log_transform',        # monotonic transform → same histogram partitions
        'sqrt_transform',       # monotonic transform → same histogram partitions
        'quantile_binning',     # reduces unique values, but LightGBM already bins optimally
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
        
        v4 tightened gates — log/sqrt/poly/binning rarely help with LightGBM:
        - log_transform: only when |skewness| > 3 AND (outlier_ratio > 0.20 OR range_iqr > 12)
          (need BOTH extreme shape AND extreme range for histogram resolution gain)
        - sqrt_transform: only when skewness > 2.5 AND min_val >= 0 AND outlier > 0.10
        - quantile_binning: only when outlier > 0.30 AND range_iqr > 15 AND is_multimodal
          (almost never helps — only test with very strong evidence of pathological distribution)
        - polynomial_square: only when |spearman_corr_target| > 0.25 AND (is_multimodal OR kurtosis < -1)
          (need evidence of non-linear relationship worth capturing)
        
        Returns: (should_test: bool, reason: str)
        """
        # Check adaptive early-stop first (cheapest check)
        if method in self._method_tracker:
            if self._method_tracker[method].get('disabled', False):
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
                if not (abs(skew) > 3 and (outlier > 0.20 or range_iqr > 12)):
                    return False, "property_gate: |skew|<=3 or (outlier<=0.20 and range_iqr<=12)"
                    
            elif method == 'sqrt_transform':
                # v4: stricter — need strong skew + non-negative + some outliers
                if not (skew > 2.5 and min_val >= 0 and outlier > 0.10):
                    return False, "property_gate: skew<=2.5 or min<0 or outlier<=0.10"
                    
            elif method == 'quantile_binning':
                # v4: extremely strict — almost never helps with LightGBM
                # Only test for pathological distributions with all three indicators
                if not (outlier > 0.30 and range_iqr > 15 and is_multimodal):
                    return False, "property_gate: need outlier>0.30 AND range_iqr>15 AND multimodal"
                    
            elif method == 'polynomial_square':
                # v4: need evidence of non-linear relationship worth capturing
                if not (abs(spearman) > 0.25 and (is_multimodal or kurtosis_val < -1)):
                    return False, "property_gate: |spearman|<=0.25 or no non-linearity evidence"
        
        return True, "ok"

    def _update_method_tracker(self, method, improved, n_selected_cols):
        """
        Update tracking for adaptive early-stop.
        
        v4: Differentiated thresholds:
        - Rarely-useful methods (log, sqrt, poly_square, quantile_binning):
          After 5 columns with 0 improvements → disable.
        - Other methods: After 20% of columns (min 10) with 0 improvements → disable.
        """
        if method not in self._method_tracker:
            self._method_tracker[method] = {'tested': 0, 'improved': 0, 'disabled': False}
        
        tracker = self._method_tracker[method]
        tracker['tested'] += 1
        if improved:
            tracker['improved'] += 1
        
        # Aggressive early-stop for methods that rarely help
        FAST_STOP_METHODS = {'log_transform', 'sqrt_transform', 'polynomial_square', 'quantile_binning'}
        if method in FAST_STOP_METHODS:
            min_trials = 5
        else:
            min_trials = max(10, int(n_selected_cols * 0.2))
        
        if tracker['tested'] >= min_trials and tracker['improved'] == 0:
            tracker['disabled'] = True
            self.log(f"    ⚡ Adaptive early-stop: {method} disabled after {tracker['tested']} "
                     f"columns with 0 improvements", "INFO")

    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================

    def collect(self, df, target_col, dataset_name="unknown"):
        """
        Main collection method (v4).
        
        Pipeline:
        1. ID detection → drop
        2. Strategic pruning (constant, near-constant, zero-var, r>0.99) → drop
        3. Target leakage detection (corr/MI/PPS > 0.95 with target) → drop
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
        
        # 1. Separate target
        y = df[target_col]
        X_raw = df.drop(columns=[target_col])
        y = self._ensure_numeric_target(y)
        
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
            self.log(f"⚠️  Target leakage detected in {len(leaky)} columns:")
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
        
        if base_score is None:
            self.log("ERROR: Baseline failed!", "ERROR")
            return pd.DataFrame()
        
        self.log(f"Baseline: {base_score:.5f} (std: {base_std:.5f})")

        # 6. Ceiling check
        should_skip, is_near_ceiling, ceiling_reason = self._check_ceiling_effect(base_score)
        if should_skip:
            self.log(f"⚠️  SKIPPING: {ceiling_reason}", "WARNING")
            return pd.DataFrame()
        
        # 7. Dataset metadata (computed on full set BEFORE column selection)
        ds_meta = self.get_dataset_meta(X, y)
        ds_meta['near_ceiling_flag'] = is_near_ceiling
        ds_meta['baseline_score'] = base_score
        ds_meta['baseline_std'] = base_std

        base_feature_importance = pd.Series(baseline_importances, index=X.columns)

        # 8. Stratified column selection
        selected_cols, selection_report = self._select_columns_stratified(X, y, base_feature_importance)
        ds_meta['n_cols_before_selection'] = X.shape[1]
        ds_meta['n_cols_selected'] = len(selected_cols)
        ds_meta['column_selection_method'] = selection_report['method']
        
        self.log(f"Selected {len(selected_cols)} / {X.shape[1]} columns for analysis")
        
        # The full model still uses ALL columns for evaluation
        # Column selection only determines WHICH columns we test interventions on
        X_full = X  # Full feature set for full-model evaluation
        
        # Store reference for pairwise correlation lookups in _log_interaction_result
        self._last_X_ref = X_full
        
        # 9. VIF (only on selected numeric columns, up to 50 for speed)
        selected_numeric = [c for c in selected_cols if pd.api.types.is_numeric_dtype(X[c])]
        if len(selected_numeric) > 1:
            vif_cols = selected_numeric[:50]
            self.log(f"Calculating VIF for {len(vif_cols)} numeric columns...")
            vif_scores = self._calculate_vif_for_features(X[vif_cols])
        else:
            vif_scores = {}

        # 10. Identify TOP-2 globally most predictive columns (for default interactions)
        composite_scores = self._compute_column_predictive_scores(X, y, base_feature_importance)
        all_cols_ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top columns: prefer numeric for interactions, but include any type
        top_global_cols = [col for col, _ in all_cols_ranked[:2]]
        self.log(f"Top global columns for default interactions: {top_global_cols}")

        # 11. NULL INTERVENTION
        self.log("Testing null intervention...")
        null_score, null_std, null_folds = self.evaluate_with_intervention(X_full, y, method='null_intervention')
        if null_score is not None:
            self.n_tests_performed += 1
            delta = self._get_delta(base_score, null_score)
            self._validate_and_append(
                ds_meta=ds_meta,
                column_name='NULL_INTERVENTION', method='null_intervention',
                delta=delta, delta_normalized=self._normalize_delta(delta, base_score),
                absolute_score=null_score, is_interaction=False,
                t_statistic=0.0, p_value=1.0, p_value_bonferroni=1.0,
                is_significant=False, is_significant_bonferroni=False,
            )

        # =====================================================================
        # 12. UNIVARIATE INTERVENTIONS + DEFAULT INTERACTIONS
        # =====================================================================
        self.log(f"Testing interventions on {len(selected_cols)} selected columns...")
        
        for col_idx, col in enumerate(selected_cols):
            self.log(f"[{col_idx+1}/{len(selected_cols)}] Column: {col}")
            
            col_meta = self.get_column_meta(X[col], y, baseline_importance=base_feature_importance.get(col, 0.0))
            col_meta['vif'] = vif_scores.get(col, -1.0)
            col_meta['composite_predictive_score'] = composite_scores.get(col, 0.0)
            
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
                # Skip log/sqrt/poly/binning — meaningless for cyclical data.
                # Keep impute_median/missing_indicator for null handling.
                methods = ['cyclical_encode', 'impute_median', 'missing_indicator']
                self.log(f"    🕐 Detected temporal component: {temporal_type} (period={temporal_period})")
                # Coerce string-encoded temporal columns to numeric so downstream
                # methods (median, sin/cos) work correctly
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if col in X_full.columns:
                        X_full[col] = pd.to_numeric(X_full[col], errors='coerce')
                    is_num = True
            elif is_num:
                methods = ['log_transform', 'impute_median', 'missing_indicator',
                          'quantile_binning', 'polynomial_square', 'sqrt_transform']
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
                    baseline_fold_scores=json.dumps(baseline_fold_scores) if self.store_fold_scores else None,
                    intervention_fold_scores=json.dumps(full_folds) if self.store_fold_scores else None,
                )
                
                sig_str = "✓" if sig_full else "✗"
                if individual_skip_reason:
                    indiv_str = f"indiv=SKIP({individual_skip_reason})"
                elif not np.isnan(indiv_delta_norm):
                    isig_str = "✓" if sig_indiv else "✗"
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

                    # Reverse: top_col is cat, col is num → group_mean with top as grouper
                    interaction_methods = []  # Will handle below
                    # Test: group_mean of col by top_col
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
            )

        # =====================================================================
        # 14. CROSS-COLUMN INTERACTIONS (top features × top features)
        # =====================================================================
        self.log("Selecting features for cross-column interactions...")
        
        num_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(X[c]) and not self._is_likely_date(X[c], c)]
        
        # v4: expanded interaction scope — more columns tested
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

        # Dynamic Budgets: Use up to 4 groupers and 3 value columns
        top_groupers = grouper_pool[:4] 
        top_values = value_pool[:3]


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
        top_arithmetic = arithmetic_candidates[:4]
        top_temporal = temporal_candidates[:3]
        top_cats = sorted(cat_cols, key=lambda c: composite_scores.get(c, 0), reverse=True)[:3]

        # A. Product interactions (top nums × top nums)
        if len(top_arithmetic) >= 2:
            self.log("Cross-column: Product Interactions...")
            for i, col_a in enumerate(top_arithmetic):
                for col_b in top_arithmetic[i+1:]:
                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method='product_interaction')
                    if full_s is None: continue
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method='product_interaction')
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{col_a}_x_{col_b}", 'product_interaction',
                        full_s, full_fs, indiv_base, indiv_new, col_a, col_b)

        
        # Define hard limits for arithmetic operations to save time
        MAX_ADD_TESTS = 4
        MAX_SUB_TESTS = 4
        MAX_ABS_TESTS = 3

        # A2. Addition interactions (top nums × top nums, scale-gated)
        if len(top_arithmetic) >= 2:
            self.log(f"Cross-column: Addition Interactions (Limit: {MAX_ADD_TESTS})...")
            add_count = 0
            # Iterate through all available, break when limit reached
            for i, col_a in enumerate(top_arithmetic):
                if add_count >= MAX_ADD_TESTS: break
                for col_b in top_arithmetic[i+1:]:
                    if add_count >= MAX_ADD_TESTS: break
                    
                    if not self._is_scale_compatible(col_a, col_b):
                        continue
                        
                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method='addition_interaction')
                    
                    if full_s is None: continue
                    
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method='addition_interaction')
                    
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{col_a}_plus_{col_b}", 'addition_interaction',
                        full_s, full_fs, indiv_base, indiv_new, col_a, col_b)
                    
                    add_count += 1

        # A3. Subtraction interactions (Directional: A-B and B-A)
        if len(top_arithmetic) >= 2:
            self.log(f"Cross-column: Subtraction Interactions (Limit: {MAX_SUB_TESTS})...")
            sub_count = 0
            for col_a in top_arithmetic:
                if sub_count >= MAX_SUB_TESTS: break
                for col_b in top_arithmetic:
                    if sub_count >= MAX_SUB_TESTS: break
                    if col_a == col_b: continue
                    
                    if not self._is_scale_compatible(col_a, col_b):
                        continue

                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method='subtraction_interaction')
                    
                    if full_s is None: continue
                    
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method='subtraction_interaction')
                    
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{col_a}_minus_{col_b}", 'subtraction_interaction',
                        full_s, full_fs, indiv_base, indiv_new, col_a, col_b)
                    
                    sub_count += 1

        # A4. Absolute difference interactions (Symmetric)
        if len(top_arithmetic) >= 2:
            self.log(f"Cross-column: Absolute Difference Interactions (Limit: {MAX_ABS_TESTS})...")
            abs_count = 0
            for i, col_a in enumerate(top_arithmetic):
                if abs_count >= MAX_ABS_TESTS: break
                for col_b in top_arithmetic[i+1:]:
                    if abs_count >= MAX_ABS_TESTS: break
                    
                    if not self._is_scale_compatible(col_a, col_b):
                        continue
                        
                    cache_key = f"pair_{col_a}_{col_b}"
                    indiv_base = self._get_individual_baseline(X_full, y, [col_a, col_b], cache_key=cache_key)
                    full_s, _, full_fs = self.evaluate_with_intervention(
                        X_full, y, col_to_transform=col_a, col_second=col_b, method='abs_diff_interaction')
                    
                    if full_s is None: continue
                    
                    indiv_new = self.evaluate_individual(
                        X_full[[col_a, col_b]], y, col_to_transform=col_a, col_second=col_b, method='abs_diff_interaction')
                    
                    self._log_interaction_result(
                        ds_meta, baseline_fold_scores, base_score,
                        f"{col_a}_absdiff_{col_b}", 'abs_diff_interaction',
                        full_s, full_fs, indiv_base, indiv_new, col_a, col_b)
                    
                    abs_count += 1

        

        # B. 3-Way interactions (multiplication, addition, ratio)
        if len(top_arithmetic) >= 3:
            self.log("Cross-column: 3-Way Interactions...")
            # v4: expanded to top 3 columns, multiple 3-way methods
            three_way_methods = ['three_way_interaction', 'three_way_addition', 'three_way_ratio']
            n_3way = min(3, len(top_arithmetic))
            for i in range(n_3way):
                for j in range(i+1, n_3way):
                    for k in range(j+1, n_3way):
                        a, b, c = top_arithmetic[i], top_arithmetic[j], top_arithmetic[k]
                        cache_key = f"triple_{a}_{b}_{c}"
                        indiv_base = self._get_individual_baseline(X_full, y, [a, b, c], cache_key=cache_key)
                        
                        for tw_method in three_way_methods:
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

        # C. Division interactions (expanded scope)
        if len(top_arithmetic) >= 2:
            self.log("Cross-column: Division Interactions...")
            for col_a in top_arithmetic[:5]:
                for col_b in top_arithmetic[:5]:
                    if col_a == col_b: continue
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
        if top_groupers and top_values:
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
        if len(top_cats) >= 2:
            self.log("Cross-column: Cat Concat...")
            for i, cat_a in enumerate(top_cats[:3]):
                for cat_b in top_cats[i+1:4]:
                    if X[cat_a].nunique() > len(X)*0.5 or X[cat_b].nunique() > len(X)*0.5:
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
        self.log(f"Analysis complete. Total tests: {self.n_tests_performed}")
        if self.n_tests_performed > 0:
            self.log(f"Bonferroni threshold: {0.05/self.n_tests_performed:.6f}")
        
        # Method tracker summary
        if self._method_tracker:
            self.log("Method effectiveness summary:")
            for method, tracker in sorted(self._method_tracker.items()):
                status = "DISABLED" if tracker.get('disabled') else "active"
                self.log(f"  {method}: {tracker['improved']}/{tracker['tested']} improved [{status}]")
        
        return pd.DataFrame(self.meta_data_log)

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

    def _log_interaction_result(self, ds_meta, baseline_fold_scores, base_score,
                                col_name, method, full_score, full_folds,
                                indiv_base, indiv_new, col_a, col_b, col_c=None,
                                interaction_source='cross_column'):
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
        )
        
        sig_str = "✓" if sig_full else "✗"
        isig_str = "✓" if sig_indiv else "✗"
        indiv_str = f"indiv={indiv_delta_norm:+.2f}% p={p_indiv:.4f} {isig_str}" if not np.isnan(indiv_delta_norm) else "indiv=N/A"
        self.log(f"  {col_name}: full={delta_full_norm:+.2f}% p={p_full:.4f} {sig_str} | {indiv_str}")


# =============================================================================
# OPENML PIPELINE
# =============================================================================

# ─── Curated temporal/date-heavy datasets for testing temporal detection ───
# Each entry: (dataset_id, description, expected_temporal_features)
# These are processed AFTER CC18 to enrich the meta-learning DB with temporal signal.
TEMPORAL_EXTRA_DATASETS = [
    # ─── HIGH QUALITY: Verified Tasks & Temporal Columns ───
    
    # Bike Sharing Demand (Regression)
    # Features: year, month, hour, weekday
    (42713, 'Bike_Sharing_Demand_Verified', 
     'hour, month, weekday → cyclical; regression'),

    # Seoul Bike Sharing (Regression)
    # Features: Hour, Date (string), Seasons
    (42736, 'Seoul_Bike_Sharing_Verified', 
     'Hour → cyclical; Date → extract; regression'),

    # Beijing PM2.5 Data (Regression)
    # Features: year, month, day, hour
    (42721, 'Beijing_PM25', 
     'month, day, hour → cyclical; regression'),

    # Electricity (Classification)
    # Features: date (numeric/string), day (1-7), period (0-48)
    # Note: Using ID 151 usually works, but if it fails, try 1471
    (151, 'Electricity_Elec2', 
     'day, period → cyclical; classification'),

    # Traffic Volume (Regression)
    # Features: date_time (string) -> needs parsing
    (42714, 'Metro_Interstate_Traffic_Volume', 
     'date_time → date_extract; regression'),

    # Online Shoppers Intention (Classification)
    # Features: Month (string "Feb", "Mar"), Weekend (bool), SpecialDay
    (42730, 'Online_Shoppers_Intention', 
     'Month → cyclical/cat; Weekend; classification'),

    # Avocados (Regression)
    # Features: Date (string)
    (42722, 'Avocado_Sales', 
     'Date → date_extract; regression'),

    # Appliances Energy Prediction (Regression)
    # Features: date (string)
    (42727, 'Appliances_Energy', 
     'date → date_extract; regression'),
     
    # Ozone Level Detection (Classification)
    # Features: Date (string)
    (1487, 'Ozone_Level_8hr', 
     'Date → date_extract; classification'),

    # ─── EXTRA: DateTime String Heavy ───
    
    # Bitcoin Heist (Classification)
    # Features: only has 'year' and 'day' integers, but good for simple cyclical
    (42617, 'Bitcoin_Heist', 
     'day → cyclical; classification'),

    # Walmart Store Sales (Regression)
    # Features: Date (string)
    (42732, 'Walmart_Store_Sales', 
     'Date → date_extract; regression')
]

# Subset for quick temporal testing (smaller datasets only)
# TEMPORAL_QUICK_DATASETS = [42712, 151, 46297, 42041, 44226]


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
        print(f"  ⚠️  Error resolving task for dataset {dataset_id}: {e}")
        # Return fallback to try loading dataset directly
        return -1, 'dataset_mode'

def run_openml_pipeline(max_tasks=50, output_dir='./meta_learning_output', 
                       n_folds=10, n_repeats=1, store_fold_scores=False,
                       include_temporal=True, temporal_only=False,
                       temporal_datasets=None):
    """Run the meta-learning data collection pipeline.
    
    Args:
        max_tasks: Maximum CC18 tasks to process.
        include_temporal: If True, also process TEMPORAL_EXTRA_DATASETS after CC18.
        temporal_only: If True, SKIP CC18 and only process temporal datasets.
        temporal_datasets: Optional list of dataset IDs to use instead of 
                          TEMPORAL_EXTRA_DATASETS. Pass TEMPORAL_QUICK_DATASETS
                          for a faster run.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'lightgbm_version': lgb.__version__,
        'n_folds': n_folds, 'n_repeats': n_repeats,
        'max_tasks': max_tasks,
        'store_fold_scores': store_fold_scores,
        'version': 'v4'
    }
    with open(os.path.join(output_dir, 'environment.json'), 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print("=" * 80)
    print("OPENML META-LEARNING PIPELINE (v4)")
    print("=" * 80)
    print(f"Output: {output_dir} | Max CC18 tasks: {max_tasks}")
    print(f"CV: {n_folds}-fold" + (f" × {n_repeats}" if n_repeats > 1 else ""))
    if temporal_only:
        print("Mode: TEMPORAL ONLY (skipping CC18)")
    elif include_temporal:
        print("Mode: CC18 + temporal extra datasets")
    print("=" * 80)
    
    # ── Build task list ──
    all_task_entries = []  # list of (task_id, source_label)
    
    if not temporal_only:
        try:
            suite = openml.study.get_suite('OpenML-CC18')
            print(f"✓ Loaded OpenML-CC18 ({len(suite.tasks)} tasks)")
            for tid in suite.tasks:
                all_task_entries.append((tid, 'CC18', None))
        except Exception as e:
            print(f"✗ Failed to load CC18: {e}")
            if not include_temporal:
                return
    
    # ── Resolve temporal extra datasets → task IDs ──
    if include_temporal or temporal_only:
        extra_ids = temporal_datasets if temporal_datasets is not None else \
                    [d[0] for d in TEMPORAL_EXTRA_DATASETS]
        
        # Build lookup for descriptions
        desc_map = {d[0]: d[2] for d in TEMPORAL_EXTRA_DATASETS}
        
        print(f"\n{'─'*40}")
        print(f"Resolving {len(extra_ids)} temporal extra datasets → tasks...")
        temporal_resolved = 0
        for did in extra_ids:
            desc = desc_map.get(did, '')
            print(f"  Dataset {did}: {desc[:60]}...", end=' ')
            tid, ttype = _resolve_task_for_dataset(did)
            if tid is not None:
                all_task_entries.append((tid, f'TEMPORAL(d={did})', ttype))
                print(f"→ task {tid} ({ttype})")
                temporal_resolved += 1
            else:
                print("→ NO TASK FOUND (skipped)")
        print(f"✓ Resolved {temporal_resolved}/{len(extra_ids)} temporal datasets")
        print(f"{'─'*40}\n")

    tasks_processed = 0
    tasks_skipped = 0
    tasks_failed = 0
    
    csv_file = os.path.join(output_dir, 'meta_learning_db.csv')
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    
    processed_tasks = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_tasks = set(json.load(f).get('processed_tasks', []))
            print(f"✓ Checkpoint: {len(processed_tasks)} done")
    
    cc18_limit_reached = False
    
    for task_id, source_label, forced_type in all_task_entries:
        if task_id != -1 and task_id in processed_tasks: continue
        
        # [Existing CC18 limit check logic here...]

        try:
            print(f"\n{'='*80}")
            
            # --- LOGIC BRANCH: TASK VS DATASET MODE ---
            dataset = None
            target_col = None
            task_type = forced_type
            
            if task_id == -1:
                # DATASET MODE: No Task ID found, load dataset directly
                # (The ID is stored in the source_label for parsing in this hacky fix, 
                #  or better, change how you store the list. 
                #  Assuming dataset_id was passed into resolve and we are iterating list.)
                
                # Extract Dataset ID from the source label "TEMPORAL(d=XXXX)"
                try:
                    import re
                    did_match = re.search(r'd=(\d+)', source_label)
                    if not did_match: raise ValueError("Could not parse dataset ID")
                    real_did = int(did_match.group(1))
                    
                    print(f"Dataset Mode: Loading Dataset ID {real_did} directly...")
                    dataset = openml.datasets.get_dataset(real_did)
                    target_col = dataset.default_target_attribute
                    
                    if not target_col:
                        print(f"⚠️ Skipping: Dataset {real_did} has no default target")
                        continue
                        
                    # Infer task type if not forced
                    if not task_type:
                        # Temporary load to check target type
                        _y_tmp = dataset.get_data(target=target_col, dataset_format='dataframe')[1]
                        task_type = 'regression' if pd.api.types.is_numeric_dtype(_y_tmp) and _y_tmp.nunique() > 20 else 'classification'
                        
                except Exception as e:
                    print(f"Failed to load dataset in fallback mode: {e}")
                    continue
            else:
                # TASK MODE: Standard OpenML flow
                print(f"Task {task_id} [{source_label}] [{tasks_processed+1}]")
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()
                target_col = dataset.default_target_attribute
                
                # Determine Task Type
                if task_type is None:
                    if hasattr(task, 'task_type_id'):
                        task_type = 'classification' if task.task_type_id == 1 else 'regression'
                    else:
                        task_type = 'regression' # Fallback guess

            # --- DATA LOADING (Common path) ---
            # Use the determined dataset object and target column
            print(f"Loading data... (Target: {target_col})")
            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            
            # [Size check...]
            if X.shape[0] * X.shape[1] > 15_000_000:
                 print(f"⚠️  Skipping: too large ({X.shape})")
                 if task_id != -1: processed_tasks.add(task_id)
                 continue

            # Double check task type based on loaded data
            if task_type is None:
                 task_type = 'regression' if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20 else 'classification'

            print(f"Type: {task_type} | Dataset: {dataset.name} | Shape: {X.shape} | Source: {source_label}")
            
            # [Data Prep for Classification...]
            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

            # [RUN COLLECTOR...]
            collector = MetaDataCollector(
                task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
                store_fold_scores=store_fold_scores, output_dir=output_dir)
            
            full_df = pd.concat([X, y], axis=1)
            df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
            
            # [SAVE RESULTS...]
            if df_result.empty:
                print(f"⚠️  Skipped")
                tasks_skipped += 1
            else:
                # Fill metadata
                df_result['openml_task_id'] = task_id if task_id != -1 else f"ds_{dataset.dataset_id}"
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                df_result['task_type'] = task_type
                
                MetaDataCollector.write_csv(df_result, csv_file)
                print(f"✅ Done ({len(df_result)} rows)")
                tasks_processed += 1
            
            if task_id != -1:
                processed_tasks.add(task_id)
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed_tasks': list(processed_tasks)}, f)

        except Exception as e:
            # [ERROR HANDLING...]
            print(f"❌ FAILED: {str(e)}")
            traceback.print_exc()
            tasks_failed += 1
            if task_id != -1: processed_tasks.add(task_id)
            continue
    
    print(f"\n{'='*80}\nDONE: {tasks_processed} ok, {tasks_skipped} skipped, {tasks_failed} failed")
    print(f"Total tasks attempted: {len([e for e in all_task_entries if e[0] in processed_tasks])}")
    if os.path.exists(csv_file):
        try:
            db = pd.read_csv(csv_file)
            print(f"Total rows: {len(db)} | Datasets: {db['dataset_name'].nunique()}")
            print(f"Significant full-model (p<0.05): {db['is_significant'].sum()}")
            print(f"Significant individual (p<0.05): {db['individual_is_significant'].sum()}")
            print(f"Bonferroni full: {db['is_significant_bonferroni'].sum()}")
            print(f"Bonferroni individual: {db['individual_is_significant_bonferroni'].sum()}")
        except:
            pass


if __name__ == "__main__":
    MAX_TASKS = 100
    OUTPUT_DIR = './meta_learning_output'
    N_FOLDS = 10
    N_REPEATS = 1
    STORE_FOLD_SCORES = False
    
    # ── Temporal dataset options ──
    # Set INCLUDE_TEMPORAL=True  to process temporal datasets AFTER CC18
    # Set TEMPORAL_ONLY=True     to SKIP CC18 and only process temporal datasets
    # Set TEMPORAL_IDS to TEMPORAL_QUICK_DATASETS for a fast temporal-only run
    INCLUDE_TEMPORAL = True
    TEMPORAL_ONLY = False
    TEMPORAL_IDS = None  # None = use full TEMPORAL_EXTRA_DATASETS list
    # TEMPORAL_IDS = TEMPORAL_QUICK_DATASETS  # Uncomment for quick test
    
    print("="*80)
    print("META-LEARNING PIPELINE v4")
    print("="*80)
    print(f"  Max CC18 tasks: {MAX_TASKS}")
    print(f"  CV: {N_FOLDS}-fold" + (f" × {N_REPEATS}" if N_REPEATS > 1 else ""))
    print(f"  Temporal: {'ONLY' if TEMPORAL_ONLY else 'included' if INCLUDE_TEMPORAL else 'disabled'}")
    print("="*80)
    
    run_openml_pipeline(
        max_tasks=MAX_TASKS,
        output_dir=OUTPUT_DIR,
        n_folds=N_FOLDS,
        n_repeats=N_REPEATS,
        store_fold_scores=STORE_FOLD_SCORES,
        include_temporal=INCLUDE_TEMPORAL,
        temporal_only=TEMPORAL_ONLY,
        temporal_datasets=TEMPORAL_IDS
    )