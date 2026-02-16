"""
Pipeline Meta-Model Trainer
=============================

Trains meta-models from pipeline_meta_learning_db.csv produced by
PipelineDataCollector / PipelineCollector_slurm.

Five complementary models are trained:

1. **Pipeline Scorer** (LightGBM regressor)
   - Input:  dataset meta-features + pipeline config features + interaction features
   - Output: predicted pipeline_delta (improvement over baseline)
   - Use:    score candidate pipelines and pick the best
   - Trained on ALL rows (the full pipeline landscape)

2. **Pipeline Gate** (LightGBM binary classifier)
   - Input:  dataset meta-features + pipeline config features + interaction features
   - Output: P(pipeline significantly improves over baseline)
   - Use:    filter out pipelines predicted to hurt; high precision is critical
   - Trained on ALL rows, target = pipeline_improved (significant positive delta)

3. **Pipeline Ranker** (LightGBM lambdarank)
   - Input:  dataset meta-features + pipeline config features + interaction features
   - Output: relative ranking score within a dataset
   - Use:    rank candidate pipelines against each other
   - Trained on ALL rows, grouped by dataset, within-dataset relevance labels

4. **FE Sensitivity Predictor** (LightGBM regressor)
   - Input:  dataset meta-features only (no pipeline features)
   - Output: predicted max achievable delta across all pipeline archetypes
   - Use:    quick check -- "is FE worth trying at all for this dataset?"
   - Trained on best-pipeline-per-dataset (1 row per dataset)

5. **Archetype Recommender** (LightGBM multiclass classifier)
   - Input:  dataset meta-features only
   - Output: predicted best pipeline archetype family
   - Use:    fast path -- recommend an archetype without scoring all candidates
   - Trained on best-pipeline-per-dataset

Additionally:
- **Pipeline Archetype Profiles**: Summary statistics per archetype (when they
  help, average delta, typical dataset shape). Used for explainability and
  fallback recommendations.

Key design decisions:
- Target is pipeline_delta (not absolute score): the model learns *improvement*
  over baseline, which transfers across datasets with different baseline scores.
- Task-type normalization on delta: z-score pipeline_delta within classification
  and regression so the scorer sees a unified improvement landscape.
- Dataset x pipeline interaction features: captures how pipeline structure
  interacts with dataset characteristics (e.g., interactions_per_feature,
  encoding_coverage, transforms_per_log_rows).
- Within-dataset relevance for ranker: rank-based relevance labels within each
  dataset, not global quantiles.
- Gate model optimized for precision: better to skip a good pipeline than to
  recommend a harmful one.

Usage:
    python train_pipeline_meta_model.py \\
        --pipeline-db ./pipeline_meta_output/pipeline_meta_learning_db.csv \\
        --output-dir ./pipeline_meta_model

The trained models are saved as .pkl files and loaded by auto_fe_app_3.py.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
import argparse
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             f1_score, precision_score, recall_score,
                             balanced_accuracy_score, roc_auc_score,
                             ndcg_score)
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Dataset-level meta-features (from PipelineDataCollector, matches HPCollector)
DATASET_FEATURES = [
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'class_imbalance_ratio', 'n_classes',
    'target_std', 'target_skew',
    'landmarking_score', 'landmarking_score_norm',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'avg_numeric_sparsity', 'linearity_gap',
    'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
    'matrix_rank_ratio',
    'std_feature_importance', 'max_minus_min_importance',
    'pct_features_above_median_importance',
    # Pipeline-specific dataset features
    'relative_headroom', 'baseline_score', 'baseline_std',
]

# Pipeline configuration features (from _extract_pipeline_features)
PIPELINE_CONFIG_FEATURES = [
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
]

# Dataset x Pipeline interaction features (computed during preparation)
# These capture how pipeline structure relates to dataset characteristics.
INTERACTION_FEATURES = [
    'ix_interactions_per_feature',     # n_interactions_2way / n_cols
    'ix_transforms_per_log_rows',      # n_transforms_total / log1p(n_rows)
    'ix_encoding_coverage',            # n_encoding_transforms / max(n_cat_cols, 1)
    'ix_feature_expansion_vs_rows',    # estimated_features_added / log1p(n_rows)
    'ix_importance_x_transforms',      # avg_touched_col_importance * n_transforms_total
    'ix_headroom_x_coverage',          # relative_headroom * importance_coverage_top10
    'ix_interactions_x_corr',          # n_interactions_2way * avg_feature_corr
    'ix_complexity_x_diversity',       # n_transforms_total * method_diversity_ratio
]

# Full feature set for scorer, gate, and ranker (dataset + pipeline + interactions)
SCORER_FEATURES = DATASET_FEATURES + PIPELINE_CONFIG_FEATURES + INTERACTION_FEATURES + [
    'task_type_encoded', 'archetype_encoded',
]

# Feature set for FE sensitivity predictor and archetype recommender (dataset only)
DATASET_ONLY_FEATURES = DATASET_FEATURES + ['task_type_encoded']

# Pipeline archetype families (from PipelineDataCollector)
ARCHETYPE_NAMES = [
    'minimal_surgical', 'encoding_focused', 'interaction_heavy',
    'single_col_heavy', 'balanced_mix', 'kitchen_sink', 'row_stats_plus',
]


# =============================================================================
# DATA PREPARATION
# =============================================================================

class PipelineDataPreparator:
    """
    Prepares pipeline_meta_learning_db.csv for meta-model training.

    Handles:
    - Task type and archetype encoding
    - Task-type delta normalization (z-score within clf/reg)
    - Dataset x pipeline interaction feature computation
    - Missing value imputation
    - Best-pipeline-per-dataset extraction for sensitivity predictor
    """

    def __init__(self):
        self.task_type_encoder = LabelEncoder()
        self.archetype_encoder = LabelEncoder()
        self.fill_values = None
        self.feature_columns_scorer = None
        self.feature_columns_dataset = None
        self.task_type_delta_stats = {}

    def _compute_interaction_features(self, df):
        """
        Compute dataset x pipeline interaction features in-place.

        These give the scorer/ranker explicit signal about how pipeline
        structure interacts with dataset characteristics.
        """
        n_rows = df.get('n_rows', pd.Series(1, index=df.index))
        n_cols = df.get('n_cols', pd.Series(1, index=df.index))
        n_cat_cols = df.get('n_cat_cols', pd.Series(0, index=df.index))

        n_int_2way = df.get('n_interactions_2way', pd.Series(0, index=df.index))
        n_transforms = df.get('n_transforms_total', pd.Series(1, index=df.index))
        n_encoding = df.get('n_encoding_transforms', pd.Series(0, index=df.index))
        est_feat = df.get('estimated_features_added', pd.Series(0, index=df.index))
        avg_imp = df.get('avg_touched_col_importance', pd.Series(0, index=df.index))
        headroom = df.get('relative_headroom', pd.Series(0.5, index=df.index))
        imp_cov10 = df.get('importance_coverage_top10', pd.Series(0, index=df.index))
        avg_corr = df.get('avg_feature_corr', pd.Series(0, index=df.index))
        diversity = df.get('method_diversity_ratio', pd.Series(0.5, index=df.index))

        df['ix_interactions_per_feature'] = n_int_2way / n_cols.clip(lower=1)
        df['ix_transforms_per_log_rows'] = n_transforms / np.log1p(n_rows).clip(lower=1)
        df['ix_encoding_coverage'] = n_encoding / n_cat_cols.clip(lower=1)
        df['ix_feature_expansion_vs_rows'] = est_feat / np.log1p(n_rows).clip(lower=1)
        df['ix_importance_x_transforms'] = avg_imp * n_transforms
        df['ix_headroom_x_coverage'] = headroom * imp_cov10
        df['ix_interactions_x_corr'] = n_int_2way * avg_corr
        df['ix_complexity_x_diversity'] = n_transforms * diversity

    def prepare(self, df: pd.DataFrame):
        """
        Prepare the full pipeline database for training.

        Returns: dict with training-ready data.
        """
        df = df.copy()
        n_raw = len(df)

        # --- Basic cleanup ---
        df = df.dropna(subset=['pipeline_score']).reset_index(drop=True)
        print(f"  After dropping NaN pipeline_score: {len(df)} rows (was {n_raw})")

        # Drop rows where baseline failed (NaN baseline_score)
        if 'baseline_score' in df.columns:
            before = len(df)
            df = df.dropna(subset=['baseline_score']).reset_index(drop=True)
            if len(df) < before:
                print(f"  Dropped {before - len(df)} rows with NaN baseline_score")

        n_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else '?'
        print(f"  Clean data: {len(df)} rows, {n_datasets} datasets")

        # --- Encode task_type ---
        if 'task_type' in df.columns:
            df['task_type_encoded'] = self.task_type_encoder.fit_transform(
                df['task_type'].astype(str))
        else:
            df['task_type_encoded'] = 0

        # --- Encode archetype ---
        if 'pipeline_archetype' in df.columns:
            df['archetype_encoded'] = self.archetype_encoder.fit_transform(
                df['pipeline_archetype'].astype(str))
        else:
            df['archetype_encoded'] = 0

        # --- Compute pipeline_delta if missing ---
        if 'pipeline_delta' not in df.columns:
            df['pipeline_delta'] = df['pipeline_score'] - df['baseline_score']

        # --- Normalize pipeline_delta within task type ---
        # Classification deltas (AUC, small range) and regression deltas
        # (R², variable range) have incompatible scales. Z-score normalization
        # lets the scorer learn a unified improvement landscape.
        y_delta_raw = df['pipeline_delta'].copy()

        self.task_type_delta_stats = {}
        if 'task_type' in df.columns:
            y_delta_norm = df['pipeline_delta'].copy()
            for tt in df['task_type'].unique():
                mask = df['task_type'] == tt
                tt_deltas = df.loc[mask, 'pipeline_delta']
                tt_mean = float(tt_deltas.mean())
                tt_std = float(tt_deltas.std())
                if tt_std < 1e-10:
                    tt_std = 1.0
                self.task_type_delta_stats[tt] = {
                    'mean': tt_mean, 'std': tt_std,
                }
                y_delta_norm.loc[mask] = (tt_deltas - tt_mean) / tt_std
                print(f"  Task '{tt}': {mask.sum()} rows, "
                      f"delta mean={tt_mean:.6f}, std={tt_std:.6f}")
            y_delta = y_delta_norm
        else:
            y_delta = y_delta_raw

        # --- Compute pipeline_improved if missing ---
        if 'pipeline_improved' not in df.columns:
            # Fallback: positive delta as improvement (no significance test)
            df['pipeline_improved'] = (df['pipeline_delta'] > 0).astype(int)
        else:
            df['pipeline_improved'] = df['pipeline_improved'].astype(int)

        y_improved = df['pipeline_improved'].copy()

        # --- Ensure numeric ---
        all_feature_cols = list(set(
            DATASET_FEATURES + PIPELINE_CONFIG_FEATURES + [
                'task_type_encoded', 'archetype_encoded']
        ))
        for col in all_feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)

        # --- Compute interaction features ---
        self._compute_interaction_features(df)

        # --- Build scorer features (dataset + pipeline + interactions) ---
        scorer_cols = [c for c in SCORER_FEATURES if c in df.columns]
        self.feature_columns_scorer = scorer_cols
        X_scorer = df[scorer_cols].copy()

        # --- Build dataset-only features ---
        ds_cols = [c for c in DATASET_ONLY_FEATURES if c in df.columns]
        self.feature_columns_dataset = ds_cols

        # --- Impute missing values ---
        self.fill_values = X_scorer.median()
        X_scorer = X_scorer.fillna(self.fill_values)

        # --- Dataset grouping ---
        dataset_ids = None
        if 'dataset_name' in df.columns:
            ds_enc = LabelEncoder()
            dataset_ids = pd.Series(
                ds_enc.fit_transform(df['dataset_name'].astype(str)),
                index=df.index)

        # --- Best pipeline per dataset for sensitivity predictor ---
        X_dataset_best = None
        y_best_delta = None
        y_best_archetype = None
        best_archetype_labels = None

        if 'dataset_name' in df.columns:
            # For each dataset, find the pipeline with the best delta
            best_idx = df.groupby('dataset_name')['pipeline_delta'].idxmax().dropna()
            best_rows = df.loc[best_idx].copy()

            X_dataset_best = best_rows[ds_cols].copy()
            ds_fill = {c: self.fill_values[c] for c in ds_cols
                       if c in self.fill_values.index}
            X_dataset_best = X_dataset_best.fillna(ds_fill).reset_index(drop=True)

            # Target for sensitivity predictor: best achievable delta
            y_best_delta = best_rows['pipeline_delta'].reset_index(drop=True)

            # Target for archetype recommender: which archetype won
            if 'pipeline_archetype' in best_rows.columns:
                best_archetype_labels = best_rows['pipeline_archetype'].reset_index(drop=True)
                y_best_archetype = self.archetype_encoder.transform(
                    best_archetype_labels.astype(str))
                y_best_archetype = pd.Series(y_best_archetype)

            print(f"  Best-pipeline-per-dataset: {len(best_rows)} datasets")

        # --- Metadata ---
        n_datasets_val = df['dataset_name'].nunique() if 'dataset_name' in df.columns else -1
        metadata = {
            'n_samples': len(df),
            'n_datasets': n_datasets_val,
            'n_scorer_features': len(scorer_cols),
            'n_dataset_features': len(ds_cols),
            'scorer_features': scorer_cols,
            'dataset_features': ds_cols,
            'interaction_features': [c for c in INTERACTION_FEATURES if c in scorer_cols],
            'task_type_mapping': dict(zip(
                self.task_type_encoder.classes_.tolist(),
                self.task_type_encoder.transform(self.task_type_encoder.classes_).tolist()
            )),
            'archetype_mapping': dict(zip(
                self.archetype_encoder.classes_.tolist(),
                self.archetype_encoder.transform(self.archetype_encoder.classes_).tolist()
            )),
            'task_type_delta_stats': self.task_type_delta_stats,
            'delta_stats': {
                'mean': float(y_delta_raw.mean()),
                'std': float(y_delta_raw.std()),
                'min': float(y_delta_raw.min()),
                'max': float(y_delta_raw.max()),
                'pct_positive': float((y_delta_raw > 0).mean() * 100),
            },
            'improvement_rate': float(y_improved.mean() * 100),
            'training_timestamp': datetime.now().isoformat(),
        }

        return {
            'df_all': df,
            'X_scorer': X_scorer,
            'y_delta': y_delta,
            'y_delta_raw': y_delta_raw,
            'y_improved': y_improved,
            'X_dataset_best': X_dataset_best,
            'y_best_delta': y_best_delta,
            'y_best_archetype': y_best_archetype,
            'best_archetype_labels': best_archetype_labels,
            'dataset_ids': dataset_ids,
            'metadata': metadata,
        }

    def transform_scorer(self, X_new):
        """Transform new data for scorer/gate/ranker at inference time."""
        self._compute_interaction_features(X_new)
        X = X_new[[c for c in self.feature_columns_scorer if c in X_new.columns]].copy()
        for col in self.feature_columns_scorer:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns_scorer]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.fill_values[self.feature_columns_scorer])
        return X

    def transform_dataset(self, X_new):
        """Transform new data for sensitivity predictor / archetype recommender."""
        X = X_new[[c for c in self.feature_columns_dataset if c in X_new.columns]].copy()
        for col in self.feature_columns_dataset:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns_dataset]
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        ds_fill = {c: self.fill_values[c] for c in self.feature_columns_dataset
                   if c in self.fill_values.index}
        X = X.fillna(ds_fill)
        return X

    def normalize_delta(self, delta, task_type):
        """Normalize a raw pipeline_delta using stored task-type stats."""
        if task_type in self.task_type_delta_stats:
            stats = self.task_type_delta_stats[task_type]
            return (delta - stats['mean']) / stats['std']
        return delta

    def denormalize_delta(self, z_delta, task_type):
        """Convert a normalized z-score back to raw delta scale."""
        if task_type in self.task_type_delta_stats:
            stats = self.task_type_delta_stats[task_type]
            return z_delta * stats['std'] + stats['mean']
        return z_delta

    def save(self, path):
        state = {
            'task_type_encoder_classes': self.task_type_encoder.classes_.tolist(),
            'archetype_encoder_classes': self.archetype_encoder.classes_.tolist(),
            'fill_values': self.fill_values.to_dict(),
            'feature_columns_scorer': self.feature_columns_scorer,
            'feature_columns_dataset': self.feature_columns_dataset,
            'task_type_delta_stats': self.task_type_delta_stats,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.task_type_encoder.classes_ = np.array(state['task_type_encoder_classes'])
        self.archetype_encoder.classes_ = np.array(state['archetype_encoder_classes'])
        self.fill_values = pd.Series(state['fill_values'])
        self.feature_columns_scorer = state['feature_columns_scorer']
        self.feature_columns_dataset = state['feature_columns_dataset']
        self.task_type_delta_stats = state.get('task_type_delta_stats', {})


# =============================================================================
# MODEL 1: PIPELINE SCORER (regressor)
# =============================================================================

class PipelineScorerTrainer:
    """
    Predicts pipeline_delta given (dataset_meta + pipeline config + interactions).

    This is the workhorse model: at inference, generate N candidate pipelines,
    score each one, pick the highest-predicted-delta pipeline.

    Uses GroupKFold on dataset_name to prevent leakage.
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None

    def train(self, X, y, dataset_ids=None, n_folds=5):
        print(f"\n{'='*60}")
        print("Training Pipeline Scorer (regressor)...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = kf.split(X, y, groups=dataset_ids)
            print(f"  Using GroupKFold ({actual_folds} folds, {n_groups} datasets)")
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = kf.split(X)

        cv_mae, cv_rmse, cv_r2 = [], [], []
        best_iterations = []

        lgb_params = dict(
            n_estimators=1000, max_depth=7, learning_rate=0.03,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1, n_jobs=-1,
        )

        for fold_i, (train_idx, val_idx) in enumerate(split_iter):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**lgb_params)
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)

            best_iterations.append(model.best_iteration_)
            preds = model.predict(X_vl)
            cv_mae.append(mean_absolute_error(y_vl, preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y_vl, preds)))
            cv_r2.append(r2_score(y_vl, preds))

        self.cv_scores = {
            'mae_mean': float(np.mean(cv_mae)), 'mae_std': float(np.std(cv_mae)),
            'rmse_mean': float(np.mean(cv_rmse)), 'rmse_std': float(np.std(cv_rmse)),
            'r2_mean': float(np.mean(cv_r2)), 'r2_std': float(np.std(cv_r2)),
        }
        print(f"  CV MAE:  {self.cv_scores['mae_mean']:.5f} +/- {self.cv_scores['mae_std']:.5f}")
        print(f"  CV RMSE: {self.cv_scores['rmse_mean']:.5f} +/- {self.cv_scores['rmse_std']:.5f}")
        print(f"  CV R2:   {self.cv_scores['r2_mean']:.4f} +/- {self.cv_scores['r2_std']:.4f}")

        # Final model on all data using avg best_iteration
        avg_best_iter = int(np.mean(best_iterations))
        final_params = lgb_params.copy()
        final_params['n_estimators'] = max(avg_best_iter, 50)
        print(f"  Final model: n_estimators={final_params['n_estimators']} "
              f"(avg CV best_iteration, range {min(best_iterations)}-{max(best_iterations)})")

        self.model = lgb.LGBMRegressor(**final_params)
        self.model.fit(X, y)

        # Feature importance
        imp = pd.Series(self.model.feature_importances_, index=X.columns)
        imp_pct = (imp / imp.sum() * 100).sort_values(ascending=False)
        print(f"\n  Top 15 features:")
        for feat, val in imp_pct.head(15).items():
            bar = '#' * int(val / 2)
            print(f"    {feat:40s} {val:5.1f}% {bar}")

        return self.cv_scores

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'cv_scores': self.cv_scores}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')


# =============================================================================
# MODEL 2: PIPELINE GATE (binary classifier)
# =============================================================================

class PipelineGateTrainer:
    """
    Binary classifier: P(pipeline significantly improves baseline).

    Optimized for precision -- it's worse to recommend a harmful pipeline
    than to miss a good one. Users can always fall back to baseline.

    Uses GroupKFold to prevent leakage.
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None
        self.threshold = 0.5

    def train(self, X, y, dataset_ids=None, n_folds=5):
        print(f"\n{'='*60}")
        print("Training Pipeline Gate (binary classifier)...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

        pos_rate = y.mean()
        print(f"  Positive rate (pipeline_improved): {pos_rate:.1%}")

        if pos_rate < 0.01 or pos_rate > 0.99:
            print("  WARNING: Extremely imbalanced target. Gate may not be useful.")

        if dataset_ids is not None:
            n_groups = dataset_ids.nunique()
            actual_folds = min(n_folds, n_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = kf.split(X, y, groups=dataset_ids)
            print(f"  Using GroupKFold ({actual_folds} folds, {n_groups} datasets)")
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            split_iter = kf.split(X)

        # Compute scale_pos_weight for class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        lgb_params = dict(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            num_leaves=40, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=min(scale_pos_weight, 10.0),  # cap to avoid instability
            random_state=42, verbosity=-1, n_jobs=-1,
        )

        cv_auc, cv_f1, cv_prec, cv_recall, cv_bacc = [], [], [], [], []
        best_iterations = []
        all_oof_probs = np.zeros(len(X))
        all_oof_labels = np.zeros(len(X))

        for fold_i, (train_idx, val_idx) in enumerate(split_iter):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)

            best_iterations.append(model.best_iteration_)
            probs = model.predict_proba(X_vl)[:, 1]
            preds = (probs >= 0.5).astype(int)

            all_oof_probs[val_idx] = probs
            all_oof_labels[val_idx] = y_vl.values

            try:
                cv_auc.append(roc_auc_score(y_vl, probs))
            except ValueError:
                cv_auc.append(0.5)
            cv_f1.append(f1_score(y_vl, preds, zero_division=0))
            cv_prec.append(precision_score(y_vl, preds, zero_division=0))
            cv_recall.append(recall_score(y_vl, preds, zero_division=0))
            cv_bacc.append(balanced_accuracy_score(y_vl, preds))

        # Optimize threshold on OOF predictions for best F1
        best_f1_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.3, 0.8, 0.02):
            preds_t = (all_oof_probs >= thresh).astype(int)
            f1_t = f1_score(all_oof_labels, preds_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_f1_thresh = thresh
        self.threshold = best_f1_thresh

        # Also find a high-precision threshold (precision >= 0.7)
        prec_thresh = 0.5
        for thresh in np.arange(0.5, 0.95, 0.02):
            preds_t = (all_oof_probs >= thresh).astype(int)
            p = precision_score(all_oof_labels, preds_t, zero_division=0)
            r = recall_score(all_oof_labels, preds_t, zero_division=0)
            if p >= 0.7 and r > 0.1:
                prec_thresh = thresh
                break

        self.cv_scores = {
            'auc_mean': float(np.mean(cv_auc)), 'auc_std': float(np.std(cv_auc)),
            'f1_mean': float(np.mean(cv_f1)), 'f1_std': float(np.std(cv_f1)),
            'precision_mean': float(np.mean(cv_prec)), 'precision_std': float(np.std(cv_prec)),
            'recall_mean': float(np.mean(cv_recall)), 'recall_std': float(np.std(cv_recall)),
            'balanced_acc_mean': float(np.mean(cv_bacc)),
            'best_f1_threshold': float(best_f1_thresh),
            'best_f1_at_threshold': float(best_f1),
            'high_precision_threshold': float(prec_thresh),
        }

        print(f"  CV AUC:       {self.cv_scores['auc_mean']:.4f} +/- {self.cv_scores['auc_std']:.4f}")
        print(f"  CV F1:        {self.cv_scores['f1_mean']:.4f} +/- {self.cv_scores['f1_std']:.4f}")
        print(f"  CV Precision: {self.cv_scores['precision_mean']:.4f} +/- {self.cv_scores['precision_std']:.4f}")
        print(f"  CV Recall:    {self.cv_scores['recall_mean']:.4f} +/- {self.cv_scores['recall_std']:.4f}")
        print(f"  Optimal threshold (F1):         {best_f1_thresh:.2f} (F1={best_f1:.4f})")
        print(f"  High-precision threshold:       {prec_thresh:.2f}")

        # Final model
        avg_best_iter = int(np.mean(best_iterations))
        final_params = lgb_params.copy()
        final_params['n_estimators'] = max(avg_best_iter, 50)
        print(f"  Final model: n_estimators={final_params['n_estimators']}")

        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X, y)

        return self.cv_scores

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=None):
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'cv_scores': self.cv_scores,
                'threshold': self.threshold,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')
        self.threshold = state.get('threshold', 0.5)


# =============================================================================
# MODEL 3: PIPELINE RANKER (LambdaRank)
# =============================================================================

class PipelineRankerTrainer:
    """
    LightGBM ranker with lambdarank objective.

    Groups by dataset -- within each dataset, learns which pipelines
    are better relative to others.

    Uses within-dataset rank-based relevance labels (not global quantiles).
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None

    def _compute_within_dataset_relevance(self, y_deltas, dataset_ids, n_levels=5):
        """
        Compute relevance labels (0 to n_levels-1) within each dataset.

        Relevance is based on each pipeline's rank within its own dataset.
        """
        relevance = pd.Series(0, index=y_deltas.index, dtype=int)

        for ds_id in dataset_ids.unique():
            mask = dataset_ids == ds_id
            ds_deltas = y_deltas[mask]

            if len(ds_deltas) <= 1:
                relevance[mask] = n_levels // 2
                continue

            ranks = ds_deltas.rank(ascending=False, method='dense')
            n_configs = ranks.max()

            if n_configs <= 1:
                relevance[mask] = n_levels // 2
                continue

            normalized_rank = (ranks - 1) / (n_configs - 1)
            rel = ((1 - normalized_rank) * (n_levels - 1)).round().astype(int)
            rel = rel.clip(0, n_levels - 1)
            relevance[mask] = rel

        return relevance

    def train(self, X, y_deltas, dataset_ids, n_folds=5):
        print(f"\n{'='*60}")
        print("Training Pipeline Ranker (lambdarank)...")
        print(f"  Samples: {len(X)}, Datasets: {dataset_ids.nunique()}")

        if len(X) < 20 or dataset_ids.nunique() < 5:
            print("  WARNING: Too few samples/datasets for ranker. Skipping.")
            return None

        relevance = self._compute_within_dataset_relevance(y_deltas, dataset_ids, n_levels=5)
        print(f"  Relevance distribution: {dict(relevance.value_counts().sort_index())}")

        # Sort by dataset_id for grouping
        group_df = pd.DataFrame({
            'dataset_id': dataset_ids.values,
            'idx': range(len(X))
        })
        group_df = group_df.sort_values('dataset_id')
        sorted_idx = group_df['idx'].values

        X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
        rel_sorted = relevance.iloc[sorted_idx].reset_index(drop=True)
        group_sizes = group_df.groupby('dataset_id').size().values

        # Filter single-sample groups
        valid_groups = group_sizes > 1
        if valid_groups.sum() < 3:
            print("  WARNING: Too few multi-sample groups. Skipping.")
            return None

        cumsum = np.cumsum(np.concatenate([[0], group_sizes]))
        valid_indices = []
        valid_group_sizes = []
        for i, v in enumerate(valid_groups):
            if v:
                valid_indices.extend(range(cumsum[i], cumsum[i+1]))
                valid_group_sizes.append(group_sizes[i])

        X_rank = X_sorted.iloc[valid_indices].reset_index(drop=True)
        rel_rank = rel_sorted.iloc[valid_indices].reset_index(drop=True)
        group_sizes_valid = np.array(valid_group_sizes)

        print(f"  After filtering: {len(X_rank)} samples in {len(group_sizes_valid)} groups")

        # 80/20 split on groups
        n_groups = len(group_sizes_valid)
        n_train_groups = max(int(n_groups * 0.8), 1)

        train_end = sum(group_sizes_valid[:n_train_groups])
        X_tr = X_rank.iloc[:train_end]
        rel_tr = rel_rank.iloc[:train_end]
        groups_tr = group_sizes_valid[:n_train_groups]

        X_vl = X_rank.iloc[train_end:]
        rel_vl = rel_rank.iloc[train_end:]
        groups_vl = group_sizes_valid[n_train_groups:]

        train_set = lgb.Dataset(X_tr, label=rel_tr, group=groups_tr)
        val_set = lgb.Dataset(X_vl, label=rel_vl, group=groups_vl, reference=train_set)

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5, 10],
            'num_leaves': 40, 'learning_rate': 0.03,
            'max_depth': 6, 'min_child_samples': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
        }

        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
        self.model = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[val_set], callbacks=callbacks,
        )

        if self.model.best_score and 'valid_0' in self.model.best_score:
            ndcg = self.model.best_score['valid_0']
            self.cv_scores = {k: float(v) for k, v in ndcg.items()}
            print(f"  Validation NDCG: {self.cv_scores}")
        else:
            self.cv_scores = {}

        # Retrain on all data
        full_set = lgb.Dataset(X_rank, label=rel_rank, group=group_sizes_valid)
        best_iter = self.model.best_iteration if self.model.best_iteration else 100
        print(f"  Retraining on full data with {best_iter} iterations")
        self.model = lgb.train(params, full_set, num_boost_round=best_iter)

        return self.cv_scores

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'cv_scores': self.cv_scores}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')


# =============================================================================
# MODEL 4: FE SENSITIVITY PREDICTOR
# =============================================================================

class FESensitivityPredictor:
    """
    Predicts how much FE can potentially improve a dataset.

    Input:  dataset meta-features only (no pipeline features)
    Output: predicted max achievable delta across all pipeline archetypes

    This answers: "Is feature engineering worth trying for this dataset?"

    Trained on one row per dataset (the best pipeline's delta).
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None

    def train(self, X, y, n_folds=5):
        print(f"\n{'='*60}")
        print("Training FE Sensitivity Predictor...")
        print(f"  Samples (datasets): {len(X)}, Features: {X.shape[1]}")
        print(f"  Best delta stats: mean={y.mean():.5f}, "
              f"median={y.median():.5f}, std={y.std():.5f}")
        print(f"  Datasets with positive best-delta: {(y > 0).sum()}/{len(y)} "
              f"({(y > 0).mean():.1%})")

        if len(X) < 20:
            print("  WARNING: Too few datasets. Skipping sensitivity predictor.")
            return None

        actual_folds = min(n_folds, len(X))
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

        cv_mae, cv_r2 = [], []
        cv_spearman = []
        best_iterations = []

        lgb_params = dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=max(3, len(X) // 20),
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1, n_jobs=-1,
        )

        for train_idx, val_idx in kf.split(X):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**lgb_params)
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)

            best_iterations.append(model.best_iteration_)
            preds = model.predict(X_vl)
            cv_mae.append(mean_absolute_error(y_vl, preds))
            try:
                cv_r2.append(r2_score(y_vl, preds))
            except:
                cv_r2.append(0.0)
            try:
                rho, _ = spearmanr(y_vl, preds)
                cv_spearman.append(rho if not np.isnan(rho) else 0.0)
            except:
                cv_spearman.append(0.0)

        self.cv_scores = {
            'mae_mean': float(np.mean(cv_mae)), 'mae_std': float(np.std(cv_mae)),
            'r2_mean': float(np.mean(cv_r2)), 'r2_std': float(np.std(cv_r2)),
            'spearman_mean': float(np.mean(cv_spearman)),
        }
        print(f"  CV MAE:      {self.cv_scores['mae_mean']:.5f} +/- {self.cv_scores['mae_std']:.5f}")
        print(f"  CV R2:       {self.cv_scores['r2_mean']:.4f} +/- {self.cv_scores['r2_std']:.4f}")
        print(f"  CV Spearman: {self.cv_scores['spearman_mean']:.4f}")

        # Final model
        avg_best_iter = int(np.mean(best_iterations))
        final_params = lgb_params.copy()
        final_params['n_estimators'] = max(avg_best_iter, 30)
        print(f"  Final model: n_estimators={final_params['n_estimators']}")

        self.model = lgb.LGBMRegressor(**final_params)
        self.model.fit(X, y)

        # Feature importance
        imp = pd.Series(self.model.feature_importances_, index=X.columns)
        imp_pct = (imp / imp.sum() * 100).sort_values(ascending=False)
        print(f"\n  Top 10 features (what makes FE worthwhile?):")
        for feat, val in imp_pct.head(10).items():
            bar = '#' * int(val / 2)
            print(f"    {feat:40s} {val:5.1f}% {bar}")

        return self.cv_scores

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'cv_scores': self.cv_scores}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')


# =============================================================================
# MODEL 5: ARCHETYPE RECOMMENDER
# =============================================================================

class ArchetypeRecommender:
    """
    Predicts which pipeline archetype family works best for a dataset.

    Input:  dataset meta-features only
    Output: predicted best archetype (multiclass)

    This is the fast path: given a new dataset, immediately recommend an
    archetype without needing to score individual pipelines.
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None
        self.label_encoder = None  # shared with preparator's archetype_encoder

    def train(self, X, y, class_names=None, n_folds=5):
        print(f"\n{'='*60}")
        print("Training Archetype Recommender (multiclass)...")
        print(f"  Samples (datasets): {len(X)}, Features: {X.shape[1]}")

        n_classes = len(np.unique(y))
        print(f"  Classes: {n_classes}")

        if class_names is not None:
            dist = pd.Series(y).map(
                dict(enumerate(class_names)) if isinstance(class_names[0], str)
                else dict(zip(range(len(class_names)), class_names))
            )
            print(f"  Distribution:")
            for name, count in dist.value_counts().items():
                print(f"    {name:25s}: {count:4d} ({count/len(y):.1%})")

        if len(X) < 20 or n_classes < 2:
            print("  WARNING: Too few samples or classes. Skipping.")
            return None

        actual_folds = min(n_folds, len(X))
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

        cv_bacc, cv_f1 = [], []
        best_iterations = []

        lgb_params = dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=max(3, len(X) // 20),
            reg_alpha=0.1, reg_lambda=1.0,
            objective='multiclass', num_class=n_classes,
            random_state=42, verbosity=-1, n_jobs=-1,
        )

        for train_idx, val_idx in kf.split(X):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=callbacks)

            best_iterations.append(model.best_iteration_)
            preds = model.predict(X_vl)
            cv_bacc.append(balanced_accuracy_score(y_vl, preds))
            cv_f1.append(f1_score(y_vl, preds, average='weighted', zero_division=0))

        self.cv_scores = {
            'balanced_acc_mean': float(np.mean(cv_bacc)),
            'balanced_acc_std': float(np.std(cv_bacc)),
            'f1_weighted_mean': float(np.mean(cv_f1)),
            'f1_weighted_std': float(np.std(cv_f1)),
        }
        print(f"  CV Balanced Accuracy: {self.cv_scores['balanced_acc_mean']:.4f} "
              f"+/- {self.cv_scores['balanced_acc_std']:.4f}")
        print(f"  CV F1 (weighted):     {self.cv_scores['f1_weighted_mean']:.4f} "
              f"+/- {self.cv_scores['f1_weighted_std']:.4f}")

        # Final model
        avg_best_iter = int(np.mean(best_iterations))
        final_params = lgb_params.copy()
        final_params['n_estimators'] = max(avg_best_iter, 30)
        print(f"  Final model: n_estimators={final_params['n_estimators']}")

        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X, y)

        return self.cv_scores

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'cv_scores': self.cv_scores}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.cv_scores = state.get('cv_scores')


# =============================================================================
# ARCHETYPE PROFILING
# =============================================================================

def compute_archetype_profiles(df):
    """
    Build interpretable archetype profiles from the pipeline data.

    For each archetype, compute:
    - Average delta, improvement rate, significance rate
    - Typical dataset characteristics where this archetype wins
    - Average pipeline composition
    """
    print(f"\n{'='*60}")
    print("Computing Archetype Profiles...")

    if 'pipeline_archetype' not in df.columns:
        print("  WARNING: No pipeline_archetype column. Skipping.")
        return None

    profiles = {}

    for archetype in df['pipeline_archetype'].unique():
        arch_rows = df[df['pipeline_archetype'] == archetype]
        if len(arch_rows) == 0:
            continue

        delta = pd.to_numeric(arch_rows['pipeline_delta'], errors='coerce')
        improved = arch_rows['pipeline_improved'].astype(int)

        profile = {
            'n_pipelines': int(len(arch_rows)),
            'n_datasets': int(arch_rows['dataset_name'].nunique())
                if 'dataset_name' in arch_rows.columns else 0,
            'avg_delta': float(delta.mean()),
            'median_delta': float(delta.median()),
            'std_delta': float(delta.std()),
            'improvement_rate': float(improved.mean()),
            'avg_transforms': float(arch_rows['n_transforms_total'].mean())
                if 'n_transforms_total' in arch_rows.columns else 0,
        }

        # Dataset characteristics where this archetype appears
        for feat in ['n_rows', 'n_cols', 'cat_ratio', 'baseline_score',
                     'relative_headroom', 'linearity_gap']:
            if feat in arch_rows.columns:
                vals = pd.to_numeric(arch_rows[feat], errors='coerce').dropna()
                if len(vals) > 0:
                    profile[f'avg_{feat}'] = float(vals.mean())

        # Where this archetype is the BEST for a dataset
        if 'dataset_name' in df.columns:
            best_idx = df.groupby('dataset_name')['pipeline_delta'].idxmax().dropna()
            best_rows = df.loc[best_idx]
            best_for_this = best_rows[best_rows['pipeline_archetype'] == archetype]
            profile['n_datasets_where_best'] = int(len(best_for_this))
            profile['pct_datasets_where_best'] = float(
                len(best_for_this) / max(len(best_rows), 1) * 100)

        profiles[archetype] = profile

        print(f"  {archetype:25s}: {profile['n_pipelines']:4d} pipelines, "
              f"delta={profile['avg_delta']:+.5f}, "
              f"improved={profile['improvement_rate']:.1%}, "
              f"best-in={profile.get('n_datasets_where_best', '?')} datasets")

    return profiles


# =============================================================================
# PIPELINE ANALYSIS REPORT
# =============================================================================

def print_pipeline_analysis(df):
    """Print analysis of pipeline performance patterns."""
    print(f"\n{'='*60}")
    print("PIPELINE ANALYSIS")
    print(f"{'='*60}")

    if 'dataset_name' not in df.columns:
        return

    n_datasets = df['dataset_name'].nunique()
    n_pipelines = len(df)
    print(f"\n  {n_pipelines} pipelines across {n_datasets} datasets "
          f"(avg {n_pipelines/max(n_datasets,1):.1f}/dataset)")

    # 1. Overall improvement stats
    delta = pd.to_numeric(df['pipeline_delta'], errors='coerce')
    improved = df['pipeline_improved'].astype(int) if 'pipeline_improved' in df.columns else (delta > 0).astype(int)

    print(f"\n  Overall Delta Statistics:")
    print(f"    Mean:   {delta.mean():+.6f}")
    print(f"    Median: {delta.median():+.6f}")
    print(f"    Std:    {delta.std():.6f}")
    print(f"    Min:    {delta.min():+.6f}")
    print(f"    Max:    {delta.max():+.6f}")
    print(f"    Significantly improved: {improved.sum()}/{len(improved)} "
          f"({improved.mean():.1%})")

    # 2. Per-dataset: how many datasets benefit from ANY pipeline?
    if 'dataset_name' in df.columns:
        ds_best_delta = df.groupby('dataset_name')['pipeline_delta'].max()
        ds_any_improved = df.groupby('dataset_name')['pipeline_improved'].max()
        print(f"\n  Dataset-Level Summary:")
        print(f"    Datasets with any positive delta: "
              f"{(ds_best_delta > 0).sum()}/{n_datasets} ({(ds_best_delta > 0).mean():.1%})")
        print(f"    Datasets with significant improvement: "
              f"{ds_any_improved.sum()}/{n_datasets} ({ds_any_improved.mean():.1%})")
        print(f"    Best delta per dataset: mean={ds_best_delta.mean():+.5f}, "
              f"median={ds_best_delta.median():+.5f}")

    # 3. Archetype comparison
    if 'pipeline_archetype' in df.columns:
        print(f"\n  Archetype Comparison:")
        arch_stats = df.groupby('pipeline_archetype').agg(
            count=('pipeline_delta', 'size'),
            avg_delta=('pipeline_delta', 'mean'),
            pct_improved=('pipeline_improved', 'mean'),
        ).sort_values('avg_delta', ascending=False)

        for arch, row in arch_stats.iterrows():
            bar = '+' * max(0, int(row['avg_delta'] * 1000))
            bar = '-' * max(0, int(-row['avg_delta'] * 1000)) if not bar else bar
            print(f"    {arch:25s}: n={int(row['count']):4d}, "
                  f"delta={row['avg_delta']:+.5f}, "
                  f"improved={row['pct_improved']:.1%} {bar}")

    # 4. Feature correlations with delta
    print(f"\n  Feature Correlations with Delta (Spearman):")
    candidate_features = PIPELINE_CONFIG_FEATURES + [
        'relative_headroom', 'baseline_score', 'linearity_gap',
        'n_rows', 'n_cols', 'cat_ratio',
    ]
    corrs = {}
    for col in candidate_features:
        if col in df.columns:
            try:
                vals = pd.to_numeric(df[col], errors='coerce')
                r, p = spearmanr(vals.fillna(0), delta)
                if not np.isnan(r):
                    corrs[col] = (r, p)
            except:
                pass

    for col, (r, p) in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)[:15]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {col:40s}  r={r:+.4f}  {sig}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_pipeline_meta_model(pipeline_db_path: str,
                               output_dir: str = './pipeline_meta_model'):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PIPELINE META-MODEL TRAINING")
    print("=" * 70)

    # --- Load ---
    print(f"\nLoading: {pipeline_db_path}")
    df = pd.read_csv(pipeline_db_path)
    n_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else '?'
    print(f"  {len(df)} rows, {n_datasets} datasets")

    if len(df) == 0:
        print("ERROR: Empty database!")
        return

    # --- Analysis ---
    print_pipeline_analysis(df)

    # --- Prepare data ---
    print(f"\n{'='*70}")
    print("DATA PREPARATION")
    print(f"{'='*70}")

    preparator = PipelineDataPreparator()
    data = preparator.prepare(df)

    X_scorer = data['X_scorer']
    y_delta = data['y_delta']
    y_improved = data['y_improved']
    dataset_ids = data['dataset_ids']
    metadata = data['metadata']

    print(f"\n  Scorer features:  {X_scorer.shape[1]}")
    print(f"  Dataset features: {len(preparator.feature_columns_dataset)}")
    print(f"  Delta (normalized): mean={y_delta.mean():.5f}, std={y_delta.std():.5f}")
    raw = data['y_delta_raw']
    print(f"  Delta (raw):        mean={raw.mean():.6f}, std={raw.std():.6f}")
    print(f"  Improvement rate:   {y_improved.mean():.1%}")

    # --- Model 1: Pipeline Scorer ---
    scorer = PipelineScorerTrainer()
    scorer_cv = scorer.train(X_scorer, y_delta, dataset_ids=dataset_ids)

    # --- Model 2: Pipeline Gate ---
    gate = PipelineGateTrainer()
    gate_cv = gate.train(X_scorer, y_improved, dataset_ids=dataset_ids)

    # --- Model 3: Pipeline Ranker ---
    ranker = PipelineRankerTrainer()
    ranker_cv = None
    if dataset_ids is not None:
        ranker_cv = ranker.train(X_scorer, y_delta, dataset_ids)
    else:
        print("\n  WARNING: No dataset_ids -- skipping ranker.")

    # --- Model 4: FE Sensitivity Predictor ---
    sensitivity = FESensitivityPredictor()
    sensitivity_cv = None
    if data['X_dataset_best'] is not None and data['y_best_delta'] is not None:
        sensitivity_cv = sensitivity.train(
            data['X_dataset_best'], data['y_best_delta'])
    else:
        print("\n  WARNING: Cannot train sensitivity predictor (no best-per-dataset data).")

    # --- Model 5: Archetype Recommender ---
    archetype_rec = ArchetypeRecommender()
    archetype_cv = None
    if (data['X_dataset_best'] is not None and
            data['y_best_archetype'] is not None and
            len(np.unique(data['y_best_archetype'])) >= 2):
        archetype_cv = archetype_rec.train(
            data['X_dataset_best'],
            data['y_best_archetype'],
            class_names=preparator.archetype_encoder.classes_.tolist(),
        )
    else:
        print("\n  WARNING: Cannot train archetype recommender.")

    # --- Archetype Profiles ---
    archetype_profiles = compute_archetype_profiles(df)

    # --- Save everything ---
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")

    preparator.save(os.path.join(output_dir, 'pipeline_preparator.pkl'))
    scorer.save(os.path.join(output_dir, 'pipeline_scorer.pkl'))
    gate.save(os.path.join(output_dir, 'pipeline_gate.pkl'))
    ranker.save(os.path.join(output_dir, 'pipeline_ranker.pkl'))
    sensitivity.save(os.path.join(output_dir, 'pipeline_sensitivity.pkl'))
    archetype_rec.save(os.path.join(output_dir, 'pipeline_archetype_rec.pkl'))

    # Save archetype profiles
    if archetype_profiles is not None:
        with open(os.path.join(output_dir, 'pipeline_archetype_profiles.json'), 'w') as f:
            json.dump(archetype_profiles, f, indent=2, default=str)

    # Aggregate CV scores
    all_cv = {
        'scorer': scorer_cv,
        'gate': gate_cv,
        'ranker': ranker_cv,
        'sensitivity': sensitivity_cv,
        'archetype_recommender': archetype_cv,
    }
    with open(os.path.join(output_dir, 'pipeline_cv_scores.json'), 'w') as f:
        json.dump(all_cv, f, indent=2, default=str)

    # Metadata
    metadata['cv_scores'] = all_cv
    metadata['n_archetype_profiles'] = len(archetype_profiles) if archetype_profiles else 0
    metadata['gate_threshold'] = gate.threshold
    with open(os.path.join(output_dir, 'pipeline_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # --- Summary ---
    print(f"\n  Saved to {output_dir}/")
    print(f"  +-- pipeline_preparator.pkl            (data preparation state)")
    print(f"  +-- pipeline_scorer.pkl                (delta regressor, "
          f"R2={scorer_cv.get('r2_mean', 0):.4f})")
    print(f"  +-- pipeline_gate.pkl                  (improvement classifier, "
          f"AUC={gate_cv.get('auc_mean', 0):.4f})")
    print(f"  +-- pipeline_ranker.pkl                (lambdarank ranker)")
    print(f"  +-- pipeline_sensitivity.pkl           (FE sensitivity predictor, "
          f"R2={sensitivity_cv.get('r2_mean', 0) if sensitivity_cv else 0:.4f})")
    print(f"  +-- pipeline_archetype_rec.pkl         (archetype recommender, "
          f"BAcc={archetype_cv.get('balanced_acc_mean', 0) if archetype_cv else 0:.4f})")
    print(f"  +-- pipeline_archetype_profiles.json   (archetype statistics)")
    print(f"  +-- pipeline_cv_scores.json            (all CV metrics)")
    print(f"  +-- pipeline_metadata.json             (training metadata)")

    # Print inference example
    print(f"\n{'='*60}")
    print("INFERENCE EXAMPLE")
    print(f"{'='*60}")
    print("  # Quick check: is FE worth trying?")
    print("  from train_pipeline_meta_model import PipelineDataPreparator, FESensitivityPredictor")
    print("  preparator = PipelineDataPreparator()")
    print("  preparator.load('pipeline_meta_model/pipeline_preparator.pkl')")
    print("  sensitivity = FESensitivityPredictor()")
    print("  sensitivity.load('pipeline_meta_model/pipeline_sensitivity.pkl')")
    print("  # X_ds = preparator.transform_dataset(pd.DataFrame([ds_meta]))")
    print("  # predicted_max_delta = sensitivity.predict(X_ds)")
    print("")
    print("  # Full pipeline scoring:")
    print("  from train_pipeline_meta_model import PipelineScorerTrainer, PipelineGateTrainer")
    print("  scorer = PipelineScorerTrainer()")
    print("  scorer.load('pipeline_meta_model/pipeline_scorer.pkl')")
    print("  gate = PipelineGateTrainer()")
    print("  gate.load('pipeline_meta_model/pipeline_gate.pkl')")
    print("  # X_cand = preparator.transform_scorer(candidate_pipeline_features)")
    print("  # deltas = scorer.predict(X_cand)")
    print("  # will_help = gate.predict(X_cand)")

    print(f"\n>> Pipeline meta-model training complete!")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the pipeline meta-model')
    parser.add_argument('--pipeline-db', required=True,
                        help='Path to pipeline_meta_learning_db.csv')
    parser.add_argument('--output-dir', default='./pipeline_meta_model',
                        help='Where to save the models')
    args = parser.parse_args()

    train_pipeline_meta_model(args.pipeline_db, args.output_dir)
