"""
Hyperparameter Tuning Meta-Data Collector
==========================================

Collects LightGBM hyperparameter performance data across hundreds of OpenML
datasets for training a hyperparameter recommendation meta-model.

DESIGN DECISIONS
================

1. STORE MULTIPLE HP CONFIGS PER DATASET (not just the best):
   - The meta-model needs to learn HOW different parameters affect performance
     on different dataset shapes, not just what the optimum is.
   - Storing ~20-30 configs per dataset lets the model see the performance
     LANDSCAPE: "increasing num_leaves from 15â†’63 helped this wide dataset
     but hurt this small one."
   - A model trained only on best configs can't learn this â€” it only sees
     one point per dataset.
   - We include a "default" config as an anchor so every dataset has a
     consistent reference point.

2. MULTIPLE PERFORMANCE METRICS per config:
   - Classification: accuracy, balanced_accuracy, f1_weighted, AUC, log_loss
   - Regression: RMSE, MAE, RÂ², MAPE (mean absolute percentage error)
   - Why? Different use cases optimize different metrics. Storing all lets
     the meta-model recommend HPs for ANY metric the user cares about.
   - We also store training wall-time per config â€” useful for "fast vs.
     accurate" trade-off recommendations.

3. LATIN HYPERCUBE SAMPLING for HP configurations:
   - Better space coverage than pure random sampling
   - 25 sampled configs + 1 default = 26 evaluations per dataset
   - Covers the important HP dimensions systematically

4. META FEATURES = dataset-level only (no column-level):
   - HP tuning is a dataset-level decision, not column-level
   - Reuses the rich statistical fingerprint from DataCollector_3
   - ~30 meta features per dataset

5. DERIVED TARGET VARIABLES per HP config:
   - primary_score: the main metric (AUC for classification, neg-RMSE for regression)
   - rank: ordinal rank within this dataset (1 = best config)
   - delta_vs_default: improvement over the default config
   - normalized_score: min-max normalized within dataset [0, 1]
   - These give the meta-model multiple ways to learn the HPâ†’performance mapping.

Usage:
    # Local sequential (small runs):
    python HPCollector.py --output_dir ./hp_tuning_output --task_list ./task_list.json

    # SLURM single-worker mode:
    python HPCollector.py --slurm_id $SLURM_ARRAY_TASK_ID --output_dir ./hp_tuning_output

    # Prerequisites: generate task_list.json using generate_task_list.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn import metrics
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from scipy.stats import loguniform
import warnings
import csv
import os
import json
import traceback
import math
import time
import gc
import psutil
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, r2_score
)
import networkx as nx

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Falling back to LHS sampling.")
    print("  Install with: pip install optuna")

import openml 
if "OPENML_CACHE_DIR" in os.environ:
    openml.config.cache_directory = os.environ["OPENML_CACHE_DIR"]
    
warnings.filterwarnings('ignore')

# =============================================================================
# [CRITICAL] GLOBAL CACHE CONFIGURATION
# =============================================================================
# Force OpenML to use /work immediately. This runs on import.
# 1. Check Env Var (set by SLURM script)
# 2. Fallback to hardcoded WORK path (if running locally without script)
# 3. Last resort: Home dir (only if others fail)

WORK_CACHE_DIR = os.environ.get('OPENML_CACHE_DIR', '/work/inestp05/lightgbm_project/openml_cache')

try:
    os.makedirs(WORK_CACHE_DIR, exist_ok=True)
    openml.config.cache_directory = WORK_CACHE_DIR
    print(f"GLOBAL: OpenML cache directory set to: {openml.config.cache_directory}")
except Exception as e:
    print(f"WARNING: Could not set OpenML cache to {WORK_CACHE_DIR}. Error: {e}")
# =============================================================================

warnings.filterwarnings('ignore')


# =============================================================================
# HP SEARCH SPACE DEFINITION
# =============================================================================

HP_SPACE = {
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

# Default config â€” always evaluated as reference point
DEFAULT_HP = {
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 1500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'max_bin': 255,
}


def _latin_hypercube_sample(n_samples, seed=42):
    """
    Generate HP configurations via Latin Hypercube Sampling.
    
    LHS divides each dimension into n_samples equal intervals and places
    exactly one sample in each interval. This guarantees better space
    coverage than pure random sampling.
    
    Returns: list of dicts, each a complete HP configuration.
    """
    rng = np.random.RandomState(seed)
    n_dims = len(HP_SPACE)
    hp_names = list(HP_SPACE.keys())
    
    # Generate LHS grid: for each dimension, create a permutation of intervals
    # Each row is a sample, each column is a dimension
    intervals = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        intervals[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    
    # Map [0, 1] intervals to actual HP values
    configs = []
    for i in range(n_samples):
        config = {}
        for d, name in enumerate(hp_names):
            spec = HP_SPACE[name]
            u = intervals[i, d]  # uniform in [0, 1]
            
            if spec['log']:
                # Log-uniform mapping
                log_low = np.log(spec['low'])
                log_high = np.log(spec['high'])
                val = np.exp(log_low + u * (log_high - log_low))
            else:
                # Linear mapping
                val = spec['low'] + u * (spec['high'] - spec['low'])
            
            if spec['type'] == 'int':
                val = int(round(val))
            else:
                val = round(val, 6)
            
            # Clamp to bounds
            val = max(spec['low'], min(spec['high'], val))
            config[name] = val
        
        configs.append(config)
    
    return configs


def _generate_hp_configs(n_samples=25, seed=42):
    """
    Generate the full set of HP configurations to evaluate per dataset.
    
    Returns: list of (config_dict, config_name) tuples.
    The first entry is always the default config.
    """
    configs = [
        (DEFAULT_HP.copy(), 'default'),
    ]
    
    lhs_configs = _latin_hypercube_sample(n_samples, seed=seed)
    for i, cfg in enumerate(lhs_configs):
        configs.append((cfg, f'lhs_{i:03d}'))
    
    return configs




# =============================================================================
# OPTUNA-BASED HP SAMPLING
# =============================================================================

def _create_optuna_study(seed=42, n_startup_trials=10):
    """
    Create an Optuna study with TPE sampler for HP optimization.

    TPE (Tree-structured Parzen Estimator) adapts sampling based on
    previous results. The first n_startup_trials use random sampling
    for exploration, then TPE kicks in for informed sampling.

    The default config is enqueued as the first trial so it always
    gets evaluated (anchor point for delta_vs_default).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for TPE sampling. pip install optuna")

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,   # model parameter correlations
        group=True,           # group correlated parameters
    )

    study = optuna.create_study(
        direction="maximize",  # primary_score: higher = better
        sampler=sampler,
    )

    # Enqueue the default config as trial #0
    study.enqueue_trial(DEFAULT_HP.copy())

    return study


def _optuna_suggest_hp(trial):
    """
    Suggest HP values from the search space using an Optuna trial.

    Uses the same HP_SPACE bounds as LHS for consistency.
    Returns a dict compatible with LightGBM parameter format.
    """
    config = {}
    for name, spec in HP_SPACE.items():
        if spec["log"] and spec["type"] == "float":
            config[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["log"] and spec["type"] == "int":
            config[name] = trial.suggest_int(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "int":
            config[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            config[name] = trial.suggest_float(name, spec["low"], spec["high"])
    return config

# =============================================================================
# CSV SCHEMA â€” strict column ordering for reliable merging
# =============================================================================

CSV_SCHEMA = [
    # --- Dataset-level metadata (30 fields) ---
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'class_imbalance_ratio', 'n_classes',
    'target_std', 'target_skew', 'target_kurtosis',
    'target_nunique_ratio',
    'landmarking_score', 'landmarking_score_norm',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'avg_numeric_sparsity', 'linearity_gap',
    'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
    'matrix_rank_ratio',
    'std_feature_importance', 'max_minus_min_importance',
    'pct_features_above_median_importance',
    'avg_skewness', 'avg_kurtosis',
    # --- HP configuration (10 fields) ---
    'hp_num_leaves', 'hp_max_depth', 'hp_learning_rate', 'hp_n_estimators',
    'hp_min_child_samples', 'hp_subsample', 'hp_colsample_bytree',
    'hp_reg_alpha', 'hp_reg_lambda', 'hp_max_bin',
    # --- HP derived features (6 fields) ---
    'hp_config_name', 'hp_is_default',
    'hp_leaves_depth_ratio',      # num_leaves / 2^max_depth â€” capacity utilization
    'hp_regularization_strength', # log(reg_alpha + reg_lambda) â€” total regularization
    'hp_sample_ratio',            # subsample * colsample_bytree â€” effective data usage
    'hp_lr_estimators_product',   # learning_rate * n_estimators â€” total learning budget
    # --- Performance metrics: Classification (6 fields) ---
    'metric_accuracy', 'metric_balanced_accuracy',
    'metric_f1_weighted', 'metric_auc', 'metric_log_loss',
    'metric_auc_std',
    # --- Performance metrics: Regression (5 fields) ---
    'metric_rmse', 'metric_mae', 'metric_r2', 'metric_mape',
    'metric_rmse_std',
    # --- Primary score and rankings (6 fields) ---
    'primary_score',           # AUC (clf) or neg_RMSE (reg) â€” the optimization target
    'primary_score_std',       # std across folds
    'rank_in_dataset',         # ordinal rank (1=best), filled in post-processing
    'delta_vs_default',        # primary_score - default_primary_score
    'normalized_score',        # min-max normalized [0,1] within dataset
    'pct_of_best',             # primary_score / best_primary_score â€” how close to best
    # --- Timing (2 fields) ---
    'train_time_seconds', 'actual_n_estimators',
    # --- Identifiers (4 fields) ---
    'openml_task_id', 'dataset_name', 'dataset_id', 'task_type',
]


# =============================================================================
# HP TUNING DATA COLLECTOR CLASS
# =============================================================================

class HPTuningCollector:
    """
    Collects hyperparameter tuning data across OpenML datasets.
    
    For each dataset:
    1. Computes rich dataset-level meta-features
    2. Evaluates ~26 HP configurations via repeated cross-validation
    3. Records multiple performance metrics per configuration
    4. Computes derived rankings and deltas
    """
    
    def __init__(self, task_type='classification', n_folds=5, n_repeats=3,
                 n_hp_configs=25, output_dir='./hp_tuning_output',
                 time_budget_seconds=3600, n_jobs=-1, seed=42,
                 sampler='optuna', n_startup_trials=None):
        self.task_type = task_type
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.n_hp_configs = n_hp_configs
        self.output_dir = output_dir
        self.time_budget_seconds = time_budget_seconds
        self.n_jobs = n_jobs
        self.seed = seed
        self.results = []
        self._start_time = None

        # Sampler selection: "optuna" (TPE) or "lhs" (Latin Hypercube)
        if sampler == 'optuna' and not OPTUNA_AVAILABLE:
            print("WARNING: optuna not available, falling back to LHS")
            sampler = 'lhs'
        self.sampler = sampler

        # n_startup_trials: how many random trials before TPE kicks in
        # Default: ~40% of total trials (good exploration/exploitation balance)
        if n_startup_trials is None:
            self.n_startup_trials = max(5, int(n_hp_configs * 0.4))
        else:
            self.n_startup_trials = n_startup_trials

        os.makedirs(output_dir, exist_ok=True)

        self.log_file = os.path.join(output_dir, 'hp_pipeline_log.txt')
        self.error_log_file = os.path.join(output_dir, 'hp_error_log.txt')
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
        log_msg = f"[{timestamp}] [{level}] [MEM: {mem_usage:.1f}MB] {message}"
        print(log_msg)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
        except:
            pass
    
    def log_error(self, message, exception=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_msg = f"[{timestamp}] ERROR: {message}\n"
        if exception:
            error_msg += f"Exception: {str(exception)}\n"
            error_msg += traceback.format_exc() + '\n'
        error_msg += '-' * 80 + '\n'
        print(f"âœ— {message}")
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
        except:
            pass
    
    def _is_over_budget(self):
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.time_budget_seconds
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    
    def _ensure_numeric_target(self, y):
        if pd.api.types.is_numeric_dtype(y):
            return y
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
    
    def _prepare_features(self, X):
        """Convert all columns to model-ready dtypes."""
        X = X.copy()
        for c in X.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            X[c] = X[c].astype('int64') / 10**9
            X[c] = X[c].fillna(-1)
        for c in X.select_dtypes(include=['object', 'category', 'bool']).columns:
            X[c] = X[c].astype('category').cat.codes
        # Fill remaining NaN for numeric
        for c in X.columns:
            if X[c].isnull().any():
                if pd.api.types.is_numeric_dtype(X[c]):
                    X[c] = X[c].fillna(X[c].median())
                else:
                    X[c] = X[c].fillna(-1)
        return X
    
    def _drop_useless_columns(self, X, y):
        """Drop ID-like, constant, and near-constant columns."""
        drop_cols = []
        for col in X.columns:
            nunique = X[col].nunique(dropna=True)
            # Constant
            if nunique <= 1:
                drop_cols.append(col)
                continue
            # Likely ID (unique ratio > 0.95 and numeric with monotonic pattern)
            if nunique > len(X) * 0.95:
                if pd.api.types.is_numeric_dtype(X[col]):
                    diffs = X[col].dropna().diff().dropna()
                    if len(diffs) > 0 and (diffs > 0).mean() > 0.95:
                        drop_cols.append(col)
                        continue
            # Near-constant (>99% same value)
            top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > 0.99:
                drop_cols.append(col)
                continue
        
        if drop_cols:
            self.log(f"  Dropping {len(drop_cols)} useless columns: {drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
            X = X.drop(columns=drop_cols)
        return X

    # =========================================================================
    # DATASET-LEVEL META FEATURES
    # =========================================================================
    
    def compute_dataset_meta(self, X, y):
        """
        Compute rich dataset-level meta-features.
        
        These describe the statistical "shape" of the dataset â€” dimensions,
        types, correlations, class balance, etc. The HP meta-model uses
        these to learn which HP settings work for which types of data.
        """
        ds = {}
        
        # --- Basic shape ---
        ds['n_rows'] = X.shape[0]
        ds['n_cols'] = X.shape[1]
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        ds['n_numeric_cols'] = len(numeric_cols)
        ds['n_cat_cols'] = len(cat_cols)
        ds['cat_ratio'] = len(cat_cols) / max(X.shape[1], 1)
        ds['missing_ratio'] = X.isnull().sum().sum() / max(X.shape[0] * X.shape[1], 1)
        ds['row_col_ratio'] = X.shape[0] / max(X.shape[1], 1)
        
        # --- Target properties ---
        if self.task_type == 'classification':
            vc = y.value_counts()
            ds['class_imbalance_ratio'] = vc.min() / vc.max() if len(vc) > 1 else 1.0
            ds['n_classes'] = len(vc)
            ds['target_std'] = np.nan
            ds['target_skew'] = np.nan
            ds['target_kurtosis'] = np.nan
            ds['target_nunique_ratio'] = len(vc) / max(len(y), 1)
        else:
            ds['class_imbalance_ratio'] = -1.0
            ds['n_classes'] = -1
            ds['target_std'] = float(y.std())
            ds['target_skew'] = float(skew(y.dropna()))
            ds['target_kurtosis'] = float(kurtosis(y.dropna()))
            ds['target_nunique_ratio'] = y.nunique() / max(len(y), 1)
        
        # --- Landmarking (simple model baseline) ---
        ds['landmarking_score'] = self._run_landmarking(X, y)
        if self.task_type == 'classification':
            ds['landmarking_score_norm'] = (ds['landmarking_score'] - 0.5) * 2
        else:
            ds['landmarking_score_norm'] = ds['landmarking_score']
        
        # --- Feature correlations ---
        if len(numeric_cols) > 1:
            sample_cols = numeric_cols[:500] if len(numeric_cols) > 500 else numeric_cols
            try:
                corr_matrix = X[sample_cols].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                ds['avg_feature_corr'] = float(upper_tri.stack().mean())
                ds['max_feature_corr'] = float(upper_tri.stack().max())
            except:
                ds['avg_feature_corr'] = 0.0
                ds['max_feature_corr'] = 0.0
        else:
            ds['avg_feature_corr'] = 0.0
            ds['max_feature_corr'] = 0.0
        
        # --- Target correlations ---
        if len(numeric_cols) > 0:
            y_num = self._ensure_numeric_target(y)
            target_corrs = []
            for col in numeric_cols:
                try:
                    if not X[col].isnull().all():
                        c = abs(X[col].corr(y_num))
                        if not np.isnan(c):
                            target_corrs.append(c)
                except:
                    pass
            ds['avg_target_corr'] = float(np.mean(target_corrs)) if target_corrs else 0.0
            ds['max_target_corr'] = float(np.max(target_corrs)) if target_corrs else 0.0
        else:
            ds['avg_target_corr'] = 0.0
            ds['max_target_corr'] = 0.0
        
        # --- Sparsity ---
        if len(numeric_cols) > 0:
            zero_ratios = [(X[col] == 0).mean() for col in numeric_cols]
            ds['avg_numeric_sparsity'] = float(np.mean(zero_ratios))
        else:
            ds['avg_numeric_sparsity'] = 0.0
        
        # --- Linearity gap ---
        ds['linearity_gap'] = self._compute_linearity_gap(X, y)
        
        # --- Correlation graph metrics ---
        if len(numeric_cols) > 1:
            ds.update(self._compute_corr_graph_metrics(
                X[numeric_cols[:500] if len(numeric_cols) > 500 else numeric_cols]))
        else:
            ds['corr_graph_components'] = 0
            ds['corr_graph_clustering'] = 0.0
            ds['corr_graph_density'] = 0.0
        
        # --- Matrix rank ratio ---
        if len(numeric_cols) > 1:
            ds['matrix_rank_ratio'] = self._compute_matrix_rank_ratio(X[numeric_cols])
        else:
            ds['matrix_rank_ratio'] = 0.0
        
        # --- Feature importance distribution ---
        imp_vals = self._get_baseline_importances(X, y)
        if imp_vals is not None and len(imp_vals) > 0 and imp_vals.max() > 0:
            imp_norm = imp_vals / imp_vals.max()
            ds['std_feature_importance'] = float(np.std(imp_norm))
            ds['max_minus_min_importance'] = float(imp_norm.max() - imp_norm.min())
            median_imp = np.median(imp_norm)
            ds['pct_features_above_median_importance'] = float((imp_norm > median_imp).mean())
        else:
            ds['std_feature_importance'] = 0.0
            ds['max_minus_min_importance'] = 0.0
            ds['pct_features_above_median_importance'] = 0.5
        
        # --- Aggregate column statistics ---
        if len(numeric_cols) > 0:
            skewness_vals = [float(skew(X[c].dropna())) for c in numeric_cols if len(X[c].dropna()) > 2]
            kurtosis_vals = [float(kurtosis(X[c].dropna())) for c in numeric_cols if len(X[c].dropna()) > 2]
            ds['avg_skewness'] = float(np.mean(skewness_vals)) if skewness_vals else 0.0
            ds['avg_kurtosis'] = float(np.mean(kurtosis_vals)) if kurtosis_vals else 0.0
        else:
            ds['avg_skewness'] = 0.0
            ds['avg_kurtosis'] = 0.0
        
        return ds
    
    # =========================================================================
    # META FEATURE HELPERS
    # =========================================================================
    
    def _run_landmarking(self, X, y):
        """Quick landmarking score using simple models."""
        try:
            X_proc = self._prepare_features(X.copy())
            if len(X_proc) > 50000:
                idx = np.random.RandomState(42).choice(len(X_proc), 50000, replace=False)
                X_proc = X_proc.iloc[idx]
                y_sub = y.iloc[idx]
            else:
                y_sub = y
            
            y_num = self._ensure_numeric_target(y_sub)
            
            from sklearn.model_selection import cross_val_score
            if self.task_type == 'classification':
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                scores = cross_val_score(model, X_proc, y_num, cv=3, scoring='accuracy')
            else:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
                scores = cross_val_score(model, X_proc, y_num, cv=3, scoring='neg_mean_squared_error')
                scores = -scores  # Make positive
            return float(np.mean(scores))
        except:
            return 0.5 if self.task_type == 'classification' else 0.0
    
    def _compute_linearity_gap(self, X, y):
        """Gap between linear and tree-based model performance."""
        try:
            X_proc = self._prepare_features(X.copy())
            y_num = self._ensure_numeric_target(y)
            
            if len(X_proc) > 50000:
                idx = np.random.RandomState(42).choice(len(X_proc), 50000, replace=False)
                X_proc, y_num = X_proc.iloc[idx], y_num.iloc[idx]
            
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            lin_scores, tree_scores = [], []
            
            for tr, vl in kf.split(X_proc):
                Xtr, Xvl = X_proc.iloc[tr], X_proc.iloc[vl]
                ytr, yvl = y_num.iloc[tr], y_num.iloc[vl]
                
                if self.task_type == 'classification' and len(np.unique(ytr)) == 2:
                    lin = LogisticRegression(max_iter=200, random_state=42)
                    lin.fit(Xtr, ytr)
                    lin_scores.append(roc_auc_score(yvl, lin.predict_proba(Xvl)[:, 1]))
                    tree = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    tree.fit(Xtr, ytr)
                    tree_scores.append(roc_auc_score(yvl, tree.predict_proba(Xvl)[:, 1]))
                elif self.task_type == 'regression':
                    lin = LinearRegression()
                    lin.fit(Xtr, ytr)
                    lin_scores.append(mean_squared_error(yvl, lin.predict(Xvl)))
                    tree = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    tree.fit(Xtr, ytr)
                    tree_scores.append(mean_squared_error(yvl, tree.predict(Xvl)))
                else:
                    return 0.0
            
            if self.task_type == 'classification':
                return float(np.mean(tree_scores) - np.mean(lin_scores))
            else:
                return float(np.mean(lin_scores) - np.mean(tree_scores)) if np.mean(tree_scores) > 0 else 0.0
        except:
            return 0.0
    
    def _compute_corr_graph_metrics(self, X_numeric):
        """Correlation graph metrics: components, clustering, density."""
        try:
            corr = X_numeric.corr().abs()
            # Build adjacency (edges where correlation > 0.5)
            adj = (corr > 0.5).astype(int).values
            np.fill_diagonal(adj, 0)
            G = nx.from_numpy_array(adj)
            return {
                'corr_graph_components': nx.number_connected_components(G),
                'corr_graph_clustering': float(nx.average_clustering(G)),
                'corr_graph_density': float(nx.density(G)),
            }
        except:
            return {'corr_graph_components': 0, 'corr_graph_clustering': 0.0, 'corr_graph_density': 0.0}
    
    def _compute_matrix_rank_ratio(self, X_numeric):
        """Ratio of matrix rank to number of columns (data redundancy)."""
        try:
            X_clean = X_numeric.dropna()
            if len(X_clean) == 0 or X_clean.shape[1] == 0:
                return 0.0
            if len(X_clean) > 10000:
                X_clean = X_clean.sample(10000, random_state=42)
            rank = np.linalg.matrix_rank(X_clean.values.astype(float))
            return float(rank / X_clean.shape[1])
        except:
            return 0.0
    
    def _get_baseline_importances(self, X, y):
        """Get feature importances from a quick default LightGBM model."""
        try:
            X_proc = self._prepare_features(X.copy())
            y_num = self._ensure_numeric_target(y)
            
            if self.task_type == 'classification':
                n_classes = y_num.nunique()
                params = {
                    'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31,
                    'max_depth': 5, 'random_state': 42, 'verbosity': -1, 'n_jobs': self.n_jobs,
                }
                if n_classes > 2:
                    params.update({'objective': 'multiclass', 'num_class': n_classes})
                model = lgb.LGBMClassifier(**params)
            else:
                params = {
                    'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31,
                    'max_depth': 5, 'random_state': 42, 'verbosity': -1, 'n_jobs': self.n_jobs,
                }
                model = lgb.LGBMRegressor(**params)
            
            model.fit(X_proc, y_num)
            return model.feature_importances_
        except:
            return None

    # =========================================================================
    # HP EVALUATION (CORE CV LOOP)
    # =========================================================================
    
    def evaluate_hp_config(self, X, y, hp_config):
        """
        Evaluate a single HP configuration via repeated cross-validation.
        
        Returns dict with all performance metrics, or None if failed.
        """
        y = self._ensure_numeric_target(y)
        n_classes = y.nunique()
        
        # Build LightGBM params
        params = hp_config.copy()
        params.update({
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': self.n_jobs,
        })
        
        # Use stratified CV for classification
        if self.task_type == 'classification':
            cv = RepeatedStratifiedKFold(
                n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=42)
        else:
            cv = RepeatedKFold(
                n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=42)
        
        # Per-fold metrics
        fold_metrics = defaultdict(list)
        actual_n_estimators_list = []
        
        t_start = time.time()
        
        for train_idx, val_idx in cv.split(X, y if self.task_type == 'classification' else None):
            X_train = self._prepare_features(X.iloc[train_idx].copy())
            X_val = self._prepare_features(X.iloc[val_idx].copy())
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                if self.task_type == 'classification':
                    p = params.copy()
                    if n_classes > 2:
                        p.update({'objective': 'multiclass', 'num_class': n_classes,
                                  'metric': 'multi_logloss'})
                    else:
                        p.update({'objective': 'binary', 'metric': 'binary_logloss'})
                    
                    model = lgb.LGBMClassifier(**p)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
                    
                    actual_n_estimators_list.append(model.best_iteration_)
                    
                    preds = model.predict(X_val, num_iteration=model.best_iteration_)
                    probs = model.predict_proba(X_val, num_iteration=model.best_iteration_)
                    
                    # Metrics
                    fold_metrics['accuracy'].append(accuracy_score(y_val, preds))
                    fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_val, preds))
                    fold_metrics['f1_weighted'].append(f1_score(y_val, preds, average='weighted', zero_division=0))
                    
                    try:
                        if n_classes > 2:
                            fold_metrics['auc'].append(roc_auc_score(
                                y_val, probs, multi_class='ovr',
                                labels=list(range(n_classes))))
                        else:
                            fold_metrics['auc'].append(roc_auc_score(y_val, probs[:, 1]))
                    except ValueError:
                        fold_metrics['auc'].append(np.nan)
                    
                    try:
                        fold_metrics['log_loss'].append(log_loss(y_val, probs, labels=list(range(n_classes))))
                    except:
                        fold_metrics['log_loss'].append(np.nan)
                    
                else:  # regression
                    p = params.copy()
                    p.update({'objective': 'regression', 'metric': 'l2'})
                    
                    model = lgb.LGBMRegressor(**p)
                    model.fit(X_train, y_train,
                              eval_set=[(X_val, y_val)],
                              callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
                    
                    actual_n_estimators_list.append(model.best_iteration_)
                    
                    preds = model.predict(X_val, num_iteration=model.best_iteration_)
                    
                    # Metrics
                    fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
                    fold_metrics['mae'].append(mean_absolute_error(y_val, preds))
                    fold_metrics['r2'].append(r2_score(y_val, preds))
                    
                    # MAPE (avoid division by zero)
                    mask = y_val != 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100
                        fold_metrics['mape'].append(mape)
                    else:
                        fold_metrics['mape'].append(np.nan)
                    
            except Exception as e:
                self.log_error(f"Fold failed for config: {e}", e)
                return None
        
        train_time = time.time() - t_start
        
        # Aggregate metrics
        result = {}
        if self.task_type == 'classification':
            result['metric_accuracy'] = float(np.nanmean(fold_metrics['accuracy']))
            result['metric_balanced_accuracy'] = float(np.nanmean(fold_metrics['balanced_accuracy']))
            result['metric_f1_weighted'] = float(np.nanmean(fold_metrics['f1_weighted']))
            result['metric_auc'] = float(np.nanmean(fold_metrics['auc']))
            result['metric_auc_std'] = float(np.nanstd(fold_metrics['auc']))
            result['metric_log_loss'] = float(np.nanmean(fold_metrics['log_loss']))
            # Regression fields â†’ NaN
            result['metric_rmse'] = np.nan
            result['metric_mae'] = np.nan
            result['metric_r2'] = np.nan
            result['metric_mape'] = np.nan
            result['metric_rmse_std'] = np.nan
            # Primary score for optimization
            result['primary_score'] = result['metric_auc']
            result['primary_score_std'] = result['metric_auc_std']
        else:
            result['metric_accuracy'] = np.nan
            result['metric_balanced_accuracy'] = np.nan
            result['metric_f1_weighted'] = np.nan
            result['metric_auc'] = np.nan
            result['metric_auc_std'] = np.nan
            result['metric_log_loss'] = np.nan
            # Regression metrics
            result['metric_rmse'] = float(np.nanmean(fold_metrics['rmse']))
            result['metric_mae'] = float(np.nanmean(fold_metrics['mae']))
            result['metric_r2'] = float(np.nanmean(fold_metrics['r2']))
            result['metric_mape'] = float(np.nanmean(fold_metrics['mape']))
            result['metric_rmse_std'] = float(np.nanstd(fold_metrics['rmse']))
            # Primary score: negative RMSE (higher = better, consistent with classification)
            result['primary_score'] = -result['metric_rmse']
            result['primary_score_std'] = result['metric_rmse_std']
        
        result['train_time_seconds'] = round(train_time, 2)
        result['actual_n_estimators'] = int(np.mean(actual_n_estimators_list))
        
        return result

    # =========================================================================
    # HP CONFIG â†’ ROW FEATURES
    # =========================================================================
    
    @staticmethod
    def _hp_config_to_features(hp_config, config_name):
        """Convert an HP config dict into row features, including derived features."""
        row = {}
        
        # Raw HP values
        for key in HP_SPACE.keys():
            row[f'hp_{key}'] = hp_config.get(key, np.nan)
        
        row['hp_config_name'] = config_name
        row['hp_is_default'] = int(config_name == 'default')
        
        # Derived HP features â€” these capture meaningful HP relationships
        # that are easier for the meta-model to learn from than raw values
        num_leaves = hp_config.get('num_leaves', 31)
        max_depth = hp_config.get('max_depth', 6)
        lr = hp_config.get('learning_rate', 0.05)
        n_est = hp_config.get('n_estimators', 1000)
        subsample = hp_config.get('subsample', 0.8)
        colsample = hp_config.get('colsample_bytree', 0.8)
        reg_alpha = hp_config.get('reg_alpha', 0.05)
        reg_lambda = hp_config.get('reg_lambda', 0.5)
        
        # Capacity utilization: how much of the tree's theoretical capacity is used
        # num_leaves / 2^max_depth. If 1.0 â†’ fully grown trees. If 0.01 â†’ very sparse.
        max_possible_leaves = 2 ** max_depth
        row['hp_leaves_depth_ratio'] = round(num_leaves / max(max_possible_leaves, 1), 6)
        
        # Total regularization strength (log scale since values span orders of magnitude)
        row['hp_regularization_strength'] = round(np.log1p(reg_alpha + reg_lambda), 6)
        
        # Effective data usage per tree
        row['hp_sample_ratio'] = round(subsample * colsample, 6)
        
        # Total learning budget: how much total "learning" happens
        # Low LR Ã— many trees â‰ˆ High LR Ã— few trees, but not exactly
        row['hp_lr_estimators_product'] = round(lr * n_est, 6)
        
        return row

    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================
    
    def collect(self, df, target_col, dataset_name="unknown"):
        """
        Main entry point: evaluate multiple HP configs on one dataset.

        Dispatches to Optuna TPE or LHS depending on self.sampler.
        Both paths produce identical output format.

        Steps:
        1. Clean and prepare data
        2. Compute dataset meta-features
        3. Generate/optimize HP configurations
        4. Evaluate each config via repeated CV
        5. Compute rankings and deltas
        6. Return DataFrame of results
        """
        self.log(f"{'='*80}")
        self.log(f"HP Tuning Collection: {dataset_name}")
        self.log(f"{'='*80}")

        self._start_time = time.time()
        self.results = []

        # --- Separate target ---
        y = df[target_col]
        X = df.drop(columns=[target_col])
        y = self._ensure_numeric_target(y)

        self.log(f"Raw shape: {X.shape}")

        # --- Drop useless columns ---
        X = self._drop_useless_columns(X, y)

        if X.shape[1] == 0:
            self.log("No features left after cleanup!", "ERROR")
            return pd.DataFrame()

        self.log(f"Shape after cleanup: {X.shape}")

        # --- Compute meta-features ---
        self.log("Computing dataset meta-features...")
        ds_meta = self.compute_dataset_meta(X, y)
        self.log(f"  Meta-features computed ({len(ds_meta)} features)")

        # --- Dispatch to sampler ---
        if self.sampler == 'optuna':
            config_results = self._collect_optuna(X, y, dataset_name)
        else:
            config_results = self._collect_lhs(X, y)

        if len(config_results) == 0:
            self.log("No configs succeeded!", "ERROR")
            return pd.DataFrame()

        # --- Compute rankings and deltas ---
        return self._build_result_dataframe(config_results, ds_meta)

    def _collect_lhs(self, X, y):
        """Original LHS-based collection (fallback)."""
        configs = _generate_hp_configs(n_samples=self.n_hp_configs, seed=self.seed)
        self.log(f"[LHS] Evaluating {len(configs)} HP configurations...")

        config_results = []
        for i, (hp_config, config_name) in enumerate(configs):
            if self._is_over_budget():
                self.log(f"  Time budget reached after {i}/{len(configs)} configs.", "WARNING")
                break

            self.log(f"  [{i+1}/{len(configs)}] Config: {config_name}")
            eval_result = self.evaluate_hp_config(X, y, hp_config)

            if eval_result is None:
                self.log(f"    FAILED -- skipping", "WARNING")
                continue

            self.log(f"    primary_score={eval_result['primary_score']:.5f} "
                     f"(+/-{eval_result['primary_score_std']:.5f}) "
                     f"time={eval_result['train_time_seconds']:.1f}s")
            config_results.append((hp_config, config_name, eval_result))

        return config_results

    def _collect_optuna(self, X, y, dataset_name="unknown"):
        """
        Optuna TPE-based collection.

        Creates an Optuna study, runs n_hp_configs + 1 trials
        (including the enqueued default), and returns ALL trial
        results in the same (config, name, metrics) format as LHS.
        """
        n_trials = self.n_hp_configs + 1  # +1 for default
        self.log(f"[Optuna TPE] Running {n_trials} trials "
                 f"({self.n_startup_trials} random startup + TPE)...")

        study = _create_optuna_study(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
        )

        # Shared state for the objective
        trial_results = []  # stores (config, name, eval_result) per trial
        collector_ref = self
        X_ref, y_ref = X, y

        def objective(trial):
            """Optuna objective: evaluate one HP config."""
            # Check time budget
            if collector_ref._is_over_budget():
                raise optuna.TrialPruned("Time budget exceeded")

            # Get HP config from Optuna
            hp_config = _optuna_suggest_hp(trial)

            # Determine config name
            if trial.number == 0:
                config_name = "default"
            else:
                config_name = f"tpe_{trial.number:03d}"

            collector_ref.log(f"  [Trial {trial.number+1}/{n_trials}] Config: {config_name}")

            # Evaluate via repeated CV
            eval_result = collector_ref.evaluate_hp_config(X_ref, y_ref, hp_config)

            if eval_result is None:
                collector_ref.log(f"    FAILED -- skipping", "WARNING")
                raise optuna.TrialPruned("Evaluation failed")

            collector_ref.log(f"    primary_score={eval_result['primary_score']:.5f} "
                             f"(+/-{eval_result['primary_score_std']:.5f}) "
                             f"time={eval_result['train_time_seconds']:.1f}s")

            # Store full result for later (Optuna only keeps the scalar)
            trial.set_user_attr("eval_result", eval_result)
            trial.set_user_attr("hp_config", hp_config)
            trial.set_user_attr("config_name", config_name)
            trial_results.append((hp_config, config_name, eval_result))

            return eval_result["primary_score"]

        # Run the study
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                catch=(Exception,),  # catch per-trial failures gracefully
                gc_after_trial=True,
            )
        except KeyboardInterrupt:
            self.log("Interrupted by user. Returning partial results.", "WARNING")

        # Log study summary
        n_complete = len([t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials 
                        if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len(study.trials) - n_complete - n_pruned
        self.log(f"  Optuna study complete: {n_complete} succeeded, "
                 f"{n_pruned} pruned, {n_failed} failed")

        if n_complete > 0:
            self.log(f"  Best trial: #{study.best_trial.number}, "
                     f"score={study.best_value:.5f}")

        return trial_results

    def _build_result_dataframe(self, config_results, ds_meta):
        """
        Build the output DataFrame from config results.

        Shared by both LHS and Optuna collection paths.
        Computes rankings, deltas, and normalized scores.
        """
        # Find default score for delta computation
        default_score = None
        for hp_config, config_name, eval_result in config_results:
            if config_name == 'default':
                default_score = eval_result['primary_score']
                break

        all_primary_scores = [r['primary_score'] for _, _, r in config_results]
        best_score = max(all_primary_scores)
        worst_score = min(all_primary_scores)
        score_range = best_score - worst_score if best_score != worst_score else 1e-8

        # Sort by primary_score descending for ranking
        sorted_results = sorted(config_results, key=lambda x: x[2]['primary_score'], reverse=True)

        # Build output rows
        for rank, (hp_config, config_name, eval_result) in enumerate(sorted_results, 1):
            row = {}
            row.update(ds_meta)
            row.update(self._hp_config_to_features(hp_config, config_name))
            row.update(eval_result)
            row['rank_in_dataset'] = rank
            row['delta_vs_default'] = (
                eval_result['primary_score'] - default_score if default_score is not None else np.nan)
            row['normalized_score'] = (
                (eval_result['primary_score'] - worst_score) / score_range)
            row['pct_of_best'] = (
                eval_result['primary_score'] / best_score if best_score != 0 else 1.0)
            self.results.append(row)

        df_result = pd.DataFrame(self.results)

        elapsed = time.time() - self._start_time
        self.log(f"Completed: {len(df_result)} rows in {elapsed:.1f}s")
        self.log(f"  Best: rank=1, score={best_score:.5f}")
        self.log(f"  Worst: rank={len(df_result)}, score={worst_score:.5f}")
        if default_score is not None:
            self.log(f"  Default: score={default_score:.5f}")

        return df_result
    
    # =========================================================================
    # CSV WRITING
    # =========================================================================
    
    @staticmethod
    def write_csv(df_result, csv_file):
        """Write results to CSV with strict schema enforcement."""
        # Ensure all schema columns exist
        for c in CSV_SCHEMA:
            if c not in df_result.columns:
                df_result[c] = np.nan
        
        df_out = df_result[CSV_SCHEMA]
        
        write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        
        STRING_FIELDS = {'hp_config_name', 'dataset_name', 'task_type'}
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_SCHEMA, extrasaction='ignore',
                                    quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writeheader()
            
            for _, row in df_out.iterrows():
                row_dict = {}
                for col in CSV_SCHEMA:
                    val = row[col]
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


# =============================================================================
# OPENML PIPELINE RUNNER
# =============================================================================

def _detect_openml_task_type(task):
    """Detect task type from OpenML task object."""
    try:
        tt = task.task_type_id
        if tt in (1, 3):
            return 'classification'
        elif tt in (2, 4):
            return 'regression'
    except:
        pass
    return None


def _infer_task_type(y):
    """Infer task type from the target column."""
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    if y.nunique() <= 20 and y.nunique() / len(y) < 0.05:
        return 'classification'
    return 'regression'


def _smart_size_reduction(X, y, task_type, cell_limit=100_000_000):
    """Reduce dataset size while preserving signal."""
    n_cells = X.shape[0] * X.shape[1]
    msg_parts = []
    
    # 1. Drop columns with >50% missing
    high_na = [c for c in X.columns if X[c].isnull().mean() > 0.5]
    if high_na:
        X = X.drop(columns=high_na)
        msg_parts.append(f"dropped {len(high_na)} high-NA cols")
    
    # 2. If still too large, sample rows
    n_cells = X.shape[0] * X.shape[1]
    if n_cells > cell_limit and X.shape[1] > 0:
        max_rows = cell_limit // X.shape[1]
        max_rows = max(max_rows, 10000)
        if X.shape[0] > max_rows:
            if task_type == 'classification':
                # Stratified sample
                from sklearn.model_selection import train_test_split
                _, X, _, y = train_test_split(X, y, test_size=max_rows/len(X),
                                               stratify=y, random_state=42)
            else:
                idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
                X, y = X.iloc[idx], y.iloc[idx]
            msg_parts.append(f"sampled to {len(X)} rows")
    
    return X, y, "; ".join(msg_parts) if msg_parts else "no reduction needed"


# =============================================================================
# SLURM SINGLE-WORKER MODE
# =============================================================================

def run_single_task(task_entry, output_dir, n_folds=5, n_repeats=3,
                    n_hp_configs=25, time_budget=3600, worker_id=None, seed=42,
                    sampler='optuna', n_startup_trials=None):
    """Process a single OpenML task (for SLURM array jobs)."""
    
    
    task_id = task_entry['task_id']
    dataset_id = task_entry.get('dataset_id')
    forced_type = task_entry.get('task_type')
    
    # Per-worker output files
    worker_tag = f"worker_{worker_id:05d}" if worker_id is not None else f"task_{task_id}"
    csv_file = os.path.join(output_dir, 'worker_csvs', f'hp_tuning_db_{worker_tag}.csv')
    checkpoint_file = os.path.join(output_dir, 'worker_checkpoints', f'hp_checkpoint_{worker_tag}.json')
    worker_log_dir = os.path.join(output_dir, 'worker_logs')
    
    os.makedirs(os.path.join(output_dir, 'worker_csvs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'worker_checkpoints'), exist_ok=True)
    os.makedirs(worker_log_dir, exist_ok=True)
    
    # Check if already done
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        if ckpt.get('status') == 'done':
            print(f"Worker {worker_id}: Task {task_id} already completed. Skipping.")
            return True
    
    print(f"{'='*80}")
    print(f"Worker {worker_id}: HP Tuning for task_id={task_id}, dataset_id={dataset_id}")
    print(f"{'='*80}")
    
    # Write in-progress checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({'task_id': task_id, 'dataset_id': dataset_id,
                   'status': 'in_progress', 'worker_id': worker_id}, f)
    
    try:
        # --- Load data ---
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
            return False
        
        if not target_col:
            print(f"  No default target attribute â€” skipping")
            with open(checkpoint_file, 'w') as f:
                json.dump({'task_id': task_id, 'dataset_id': dataset_id,
                           'status': 'skipped', 'reason': 'no_target'}, f)
            return False
        
        # Load data
        X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
        if task_type is None:
            task_type = _infer_task_type(y)
        
        # Size reduction
        raw_cells = X.shape[0] * X.shape[1]
        if raw_cells > 100_000_000:
            X, y, reduction_msg = _smart_size_reduction(X, y, task_type)
            print(f"  Size reduction: {reduction_msg}")
        
        print(f"  Type: {task_type} | Shape: {X.shape}")
        
        # Encode classification targets
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
        
        # --- Run collector ---
        collector = HPTuningCollector(
            task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
            n_hp_configs=n_hp_configs, output_dir=output_dir,
            time_budget_seconds=time_budget, seed=seed,
            sampler=sampler, n_startup_trials=n_startup_trials)
        
        full_df = pd.concat([X, y], axis=1)
        df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
        
        if df_result.empty:
            print(f"  Empty result â€” skipped")
            with open(checkpoint_file, 'w') as f:
                json.dump({'task_id': task_id, 'dataset_id': dataset_id,
                           'status': 'skipped', 'reason': 'empty_result'}, f)
            return False
        
        # Add identifiers
        df_result['openml_task_id'] = task_id if (task_id and task_id > 0) else f"ds_{dataset_id}"
        df_result['dataset_name'] = dataset.name
        df_result['dataset_id'] = dataset.dataset_id
        df_result['task_type'] = task_type
        
        # Write results
        HPTuningCollector.write_csv(df_result, csv_file)
        print(f"  Done: {len(df_result)} rows written to {csv_file}")
        
        # Mark as done
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'task_id': task_id, 'dataset_id': dataset_id,
                'status': 'done', 'n_rows': len(df_result),
                'dataset_name': dataset.name, 'task_type': task_type,
            }, f)
        
        del collector, full_df, df_result
        gc.collect()
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        with open(checkpoint_file, 'w') as f:
            json.dump({'task_id': task_id, 'dataset_id': dataset_id,
                       'status': 'failed', 'error': str(e)}, f)
        return False


# =============================================================================
# LOCAL SEQUENTIAL PIPELINE
# =============================================================================

def run_pipeline(output_dir='./hp_tuning_output', task_list_file=None,
                 n_folds=5, n_repeats=3, n_hp_configs=25,
                 time_budget=7200, seed=42, sampler='optuna',
                 n_startup_trials=None):
    """Run HP tuning data collection sequentially on all tasks."""
    import openml
    
    if task_list_file is None:
        task_list_file = os.path.join(output_dir, 'task_list.json')
    
    if not os.path.exists(task_list_file):
        print(f"ERROR: Task list not found: {task_list_file}")
        print("Run generate_task_list.py first.")
        return
    
    with open(task_list_file, 'r') as f:
        task_data = json.load(f)
    task_list = task_data['tasks']
    print(f"Loaded {len(task_list)} tasks from {task_list_file}")
    
    # Environment metadata
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'lightgbm_version': lgb.__version__,
        'n_folds': n_folds, 'n_repeats': n_repeats,
        'n_hp_configs': n_hp_configs,
        'n_tasks': len(task_list),
        'version': 'hp_v2_optuna',
        'sampler': sampler,
    }
    with open(os.path.join(output_dir, 'hp_environment.json'), 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print("=" * 80)
    print("HP TUNING META-LEARNING PIPELINE v1")
    print("=" * 80)
    print(f"  Output:      {output_dir}")
    print(f"  Tasks:       {len(task_list)}")
    print(f"  CV:          {n_folds}-fold Ã— {n_repeats} repeats")
    sampler_label = "Optuna TPE" if sampler == "optuna" else "LHS"
    print(f"  HP configs:  {n_hp_configs} {sampler_label} + 1 default = {n_hp_configs + 1}")
    print(f"  Time budget: {time_budget}s per dataset")
    print("=" * 80)
    
    csv_file = os.path.join(output_dir, 'hp_tuning_db.csv')
    checkpoint_file = os.path.join(output_dir, 'hp_checkpoint.json')
    
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
            print(f"\n{'='*80}")
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
                print("  No valid task_id or dataset_id â€” skipping")
                tasks_skipped += 1
                continue
            
            if not target_col:
                print(f"  No default target â€” skipping")
                tasks_skipped += 1
                processed_tasks.add(ckpt_key)
                continue
            
            # Infer type
            if task_type is None:
                _y_tmp = dataset.get_data(target=target_col, dataset_format='dataframe')[1]
                task_type = _infer_task_type(_y_tmp)
            
            # Load data
            X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
            if task_type is None:
                task_type = _infer_task_type(y)
            
            # Size reduction
            raw_cells = X.shape[0] * X.shape[1]
            if raw_cells > 100_000_000:
                X, y, reduction_msg = _smart_size_reduction(X, y, task_type)
                print(f"  Size reduction: {reduction_msg}")
            
            print(f"  Type: {task_type} | Shape: {X.shape}")
            
            # Encode targets
            if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
            
            # Run collector
            dataset_start = time.time()
            collector = HPTuningCollector(
                task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
                n_hp_configs=n_hp_configs, output_dir=output_dir,
                time_budget_seconds=time_budget, seed=seed,
                sampler=sampler, n_startup_trials=n_startup_trials)
            
            full_df = pd.concat([X, y], axis=1)
            df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
            dataset_elapsed = time.time() - dataset_start
            
            if df_result.empty:
                print(f"  Empty result â€” skipped ({dataset_elapsed:.1f}s)")
                tasks_skipped += 1
            else:
                df_result['openml_task_id'] = task_id if (task_id and task_id > 0) else f"ds_{dataset_id}"
                df_result['dataset_name'] = dataset.name
                df_result['dataset_id'] = dataset.dataset_id
                df_result['task_type'] = task_type
                
                HPTuningCollector.write_csv(df_result, csv_file)
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
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DONE: {tasks_processed} ok, {tasks_skipped} skipped, {tasks_failed} failed")
    if os.path.exists(csv_file):
        try:
            db = pd.read_csv(csv_file)
            n_datasets = db['dataset_name'].nunique()
            n_configs = len(db)
            print(f"Total rows: {n_configs} | Datasets: {n_datasets}")
            print(f"Avg configs/dataset: {n_configs / max(n_datasets, 1):.1f}")
            print(f"Primary score range: {db['primary_score'].min():.5f} â€“ {db['primary_score'].max():.5f}")
        except Exception:
            pass


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HP Tuning Meta-Learning Data Collector v1")
    parser.add_argument('--output_dir', default='./hp_tuning_output')
    parser.add_argument('--task_list', default=None,
                        help='Path to task_list.json (default: <output_dir>/task_list.json)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--n_hp_configs', type=int, default=25,
                        help='Number of HP configs to evaluate (+ 1 default). Default: 25')
    parser.add_argument('--time_budget', type=int, default=7200,
                        help='Max seconds per dataset')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--sampler', default='optuna', choices=['optuna', 'lhs'],
                        help='HP sampling strategy: optuna (TPE, default) or lhs (Latin Hypercube)')
    parser.add_argument('--n_startup_trials', type=int, default=None,
                        help='Optuna: number of random startup trials before TPE. Default: 40%% of n_hp_configs')
    
    # SLURM mode
    parser.add_argument('--slurm_id', type=int, default=None,
                        help='SLURM array task ID (single-worker mode)')
    
    args = parser.parse_args()
    
    if args.slurm_id is not None:
        # --- SLURM single-worker mode ---
        task_list_file = args.task_list or os.path.join(args.output_dir, 'task_list.json')
        
        if not os.path.exists(task_list_file):
            print(f"ERROR: Task list not found: {task_list_file}")
            sys.exit(1)
        
        import sys
        with open(task_list_file, 'r') as f:
            task_data = json.load(f)
        task_list = task_data['tasks']
        
        if args.slurm_id >= len(task_list):
            print(f"SLURM ID {args.slurm_id} >= task count {len(task_list)}. Nothing to do.")
            sys.exit(0)
        
        task_entry = task_list[args.slurm_id]
        print(f"SLURM worker {args.slurm_id}: task={task_entry}")
        
        success = run_single_task(
            task_entry=task_entry,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            n_hp_configs=args.n_hp_configs,
            time_budget=args.time_budget,
            worker_id=args.slurm_id,
            seed=args.seed,
            sampler=args.sampler,
            n_startup_trials=args.n_startup_trials,
        )
        sys.exit(0 if success else 1)
    
    else:
        # --- Local sequential mode ---
        run_pipeline(
            output_dir=args.output_dir,
            task_list_file=args.task_list,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            n_hp_configs=args.n_hp_configs,
            time_budget=args.time_budget,
            seed=args.seed,
            sampler=args.sampler,
            n_startup_trials=args.n_startup_trials,
        )