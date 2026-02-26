"""
train_meta_models.py — Train Meta-Models for Feature Engineering Recommendation
=================================================================================

Trains three independent LightGBM models (numerical, categorical, interaction)
on the collected meta-learning data. Each model learns:

    (dataset meta-features + column/pair meta-features + method) → calibrated_delta

Additionally trains a lightweight binary classifier per type for:
    → is_significant_bonferroni  (should we bother?)

Evaluation uses GroupKFold by dataset_name so no data from the same dataset
leaks between train and validation — this measures true generalization.

Usage:
    python train_meta_models.py \
        --numerical_csv   ./output_numerical/numerical_transforms_merged.csv \
        --categorical_csv ./output_categorical/categorical_transforms_merged.csv \
        --interaction_csv ./output_interactions/interaction_features_merged.csv \
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
    'n_numerical_cols',
    'pearson_corr', 'spearman_corr',
    'mutual_info_pair', 'mic_score', 'scale_ratio',
    'sum_importance', 'max_importance', 'min_importance',
    'sum_null_pct', 'max_null_pct',
    'sum_unique_ratio', 'abs_diff_unique_ratio',
    'sum_entropy', 'abs_diff_entropy',
    'sum_target_corr', 'abs_diff_target_corr',
    'sum_mi_target', 'abs_diff_mi_target',
    'both_binary',
]

# Regression target
TARGET_REGRESSION = 'calibrated_delta'

# Classification target
TARGET_CLASSIFICATION = 'is_significant_bonferroni'

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

def load_and_prepare(csv_path, collector_type):
    """
    Load a merged CSV and prepare features + targets.
    
    Returns:
        X (pd.DataFrame): Feature matrix (dataset meta + column/pair meta + method one-hot)
        y_reg (pd.Series): Regression target (calibrated_delta)
        y_cls (pd.Series): Classification target (is_significant_bonferroni)
        groups (pd.Series): Dataset name for grouped CV
        method_vocab (list): Ordered method names used for one-hot encoding
        feature_names (list): Final feature column names
    """
    print(f"\n{'='*60}")
    print(f"Loading {collector_type}: {csv_path}")
    
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
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")
    
    # Verify all expected columns exist
    missing = [c for c in meta_features if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns: {missing}")
        for c in missing:
            df[c] = 0.0
    
    # --- Drop rows with missing targets ---
    n_before = len(df)
    df = df.dropna(subset=[TARGET_REGRESSION])
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with missing {TARGET_REGRESSION}")
    
    # --- Prepare classification target ---
    # Handle string 'True'/'False' from CSV
    if TARGET_CLASSIFICATION in df.columns:
        y_cls = df[TARGET_CLASSIFICATION].map(
            {True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}
        ).fillna(0).astype(int)
    else:
        y_cls = (df[TARGET_REGRESSION].abs() > 1.0).astype(int)
    
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
    
    y_reg = df[TARGET_REGRESSION].astype(float).reset_index(drop=True)
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

def evaluate_grouped_cv(X, y_reg, y_cls, groups, n_splits=5):
    """
    Evaluate models with GroupKFold — no dataset leaks between folds.
    
    Returns dict with regression and classification metrics per fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    reg_metrics = {'mae': [], 'rmse': [], 'r2': [], 'median_ae': []}
    cls_metrics = {'auc': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
    
    feature_importances = np.zeros(X.shape[1])
    
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y_reg, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        # --- Regression ---
        y_train_reg, y_val_reg = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
        
        reg_model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
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
        
        cls_model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
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
# TRAIN FINAL MODELS (on all data)
# =============================================================================

def train_final_models(X, y_reg, y_cls):
    """Train final regressor and classifier on all available data."""
    
    # --- Regressor ---
    reg_model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    reg_model.fit(X, y_reg)
    
    # --- Classifier ---
    cls_model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    cls_model.fit(X, y_cls)
    
    return reg_model, cls_model


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_collector_type(csv_path, collector_type, output_dir, n_cv_splits=5):
    """
    Full pipeline for one collector type:
    1. Load data
    2. Grouped CV evaluation
    3. Train final models on all data
    4. Save models + config
    """
    if csv_path is None or not os.path.exists(csv_path):
        print(f"\n  Skipping {collector_type}: CSV not found at {csv_path}")
        return None
    
    # --- Load ---
    X, y_reg, y_cls, groups, method_vocab, feature_names = load_and_prepare(
        csv_path, collector_type
    )
    
    if len(X) < 50:
        print(f"  Skipping {collector_type}: only {len(X)} rows (need ≥50)")
        return None
    
    n_datasets = groups.nunique()
    actual_splits = min(n_cv_splits, n_datasets)
    if actual_splits < 2:
        print(f"  Skipping CV: only {n_datasets} unique dataset(s)")
        cv_results = None
        feat_importances = np.zeros(X.shape[1])
    else:
        # --- Evaluate ---
        print(f"\n  Evaluating with {actual_splits}-fold GroupKFold ({n_datasets} datasets)...")
        cv_results, feat_importances = evaluate_grouped_cv(
            X, y_reg, y_cls, groups, n_splits=actual_splits
        )
        
        print(f"\n  REGRESSION (calibrated_delta):")
        for metric in ['mae', 'rmse', 'r2', 'median_ae']:
            vals = cv_results['regression'][metric]
            print(f"    {metric:>10}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        
        print(f"\n  CLASSIFICATION (is_significant_bonferroni):")
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
    reg_model, cls_model = train_final_models(X, y_reg, y_cls)
    
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
    
    config = {
        'collector_type': collector_type,
        'dataset_features': DATASET_FEATURES,
        'column_features': column_features,
        'method_vocab': method_vocab,
        'feature_names': feature_names,
        'n_training_rows': len(X),
        'n_training_datasets': int(groups.nunique()),
        'target_regression': TARGET_REGRESSION,
        'target_classification': TARGET_CLASSIFICATION,
        'regressor_params': REGRESSOR_PARAMS,
        'classifier_params': CLASSIFIER_PARAMS,
        'cv_results': cv_results,
        'feature_importances': {
            fname: float(fval) 
            for fname, fval in imp_series.head(30).items()
        },
        'target_stats': {
            'calibrated_delta_mean': float(y_reg.mean()),
            'calibrated_delta_std': float(y_reg.std()),
            'calibrated_delta_median': float(y_reg.median()),
            'calibrated_delta_q25': float(y_reg.quantile(0.25)),
            'calibrated_delta_q75': float(y_reg.quantile(0.75)),
            'pct_significant': float(y_cls.mean()),
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
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train meta-models for feature engineering recommendation'
    )
    parser.add_argument(
        '--numerical_csv', type=str, default=None,
        help='Path to numerical_transforms_merged.csv'
    )
    parser.add_argument(
        '--categorical_csv', type=str, default=None,
        help='Path to categorical_transforms_merged.csv'
    )
    parser.add_argument(
        '--interaction_csv', type=str, default=None,
        help='Path to interaction_features_merged.csv'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./meta_models',
        help='Directory to save trained models and configs'
    )
    parser.add_argument(
        '--n_cv_splits', type=int, default=5,
        help='Number of GroupKFold splits for evaluation'
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("META-MODEL TRAINING")
    print("=" * 70)
    
    # --- Train each collector type ---
    all_reports = {}
    
    collector_csvs = {
        'numerical': args.numerical_csv,
        'categorical': args.categorical_csv,
        'interaction': args.interaction_csv,
    }
    
    for ctype, csv_path in collector_csvs.items():
        report = train_collector_type(
            csv_path=csv_path,
            collector_type=ctype,
            output_dir=args.output_dir,
            n_cv_splits=args.n_cv_splits,
        )
        if report:
            all_reports[ctype] = report
    
    # --- Save combined report ---
    report_path = os.path.join(args.output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Models saved to: {args.output_dir}")
    print(f"Training report: {report_path}")
    
    for ctype, report in all_reports.items():
        r2_str = "N/A"
        auc_str = "N/A"
        if report['cv_results']:
            r2_vals = report['cv_results']['regression'].get('r2', {})
            r2_str = f"{r2_vals.get('mean', 0):.3f}" if r2_vals else "N/A"
            auc_vals = report['cv_results']['classification'].get('auc', {})
            auc_str = f"{auc_vals.get('mean', 0):.3f}" if auc_vals else "N/A"
        print(f"  {ctype:>15}: {report['n_rows']:>6} rows, "
              f"{report['n_datasets']:>4} datasets, "
              f"R²={r2_str}, AUC={auc_str}")


if __name__ == '__main__':
    main()