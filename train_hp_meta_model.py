"""
HP Meta-Model Trainer (v2)
===========================

Trains meta-models from the hp_tuning_db.csv produced by HPCollector.py.

Three complementary models are trained:

1. **HP Config Scorer** (LightGBM regressor)
   - Input:  dataset meta-features + HP config features + interaction features
   - Output: predicted primary_score (task-type normalized z-score)
   - Use:    score candidate configs and pick the best
   - Trained on ALL rows (the full HP landscape)

2. **Direct HP Predictor** (one LightGBM regressor per HP parameter)
   - Input:  dataset meta-features only
   - Output: predicted optimal value for each HP parameter
   - Use:    single-pass HP prediction -- fast and simple
   - Trained on the TOP-K configs per dataset (weighted by normalized_score)
   - Uses GroupKFold on dataset_name to prevent cross-dataset leakage

3. **HP Config Ranker** (LightGBM lambdarank)
   - Input:  dataset meta-features + HP config features + interaction features
   - Output: relative ranking score
   - Use:    rank a set of candidate configs
   - Trained on ALL rows, grouped by dataset, with within-dataset relevance
   - GroupKFold CV for reliable NDCG estimates

4. **Ensemble Predictor** (combines all three models)
   - Predictor generates initial config, perturbations are scored and ranked
   - Weighted combination of scorer (absolute quality) + ranker (relative ordering)
   - Produces the most robust HP predictions

Additionally:
- **HP Consistency Constraints**: prevents contradictory predictions
  (e.g., num_leaves > 2^max_depth, learning budget violations)
- **Dataset Archetype Clustering**: K-Means on optimal HP profiles,
  producing interpretable archetypes for fallback and explainability.

Key improvements (v2):
- GroupKFold in DirectHPPredictor: prevents top-K rows from the same
  dataset leaking across train/val folds (was using plain KFold)
- GroupKFold CV in ranker: proper cross-validation instead of single
  80/20 split, giving more reliable NDCG estimates
- Ensemble inference: orchestrates scorer + predictor + ranker together
  for robust predictions (perturbation + scoring + ranking pipeline)
- HP consistency constraints: enforces num_leaves <= 2^max_depth,
  learning budget bounds, and proper integer rounding

Usage:
    python train_hp_meta_model.py --hp-db ./hp_tuning_output/hp_tuning_db.csv \\
                                  --output-dir ./hp_meta_model

The trained models are saved as .pkl files and can be loaded by the
Streamlit app to replace the rule-based archetype system.
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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans, SpectralClustering
import lightgbm as lgb

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Dataset-level meta-features (computed by HPCollector)
DATASET_FEATURES = [
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
]

# Raw HP parameter names as stored in the CSV
HP_PARAM_NAMES = [
    'hp_num_leaves', 'hp_max_depth', 'hp_learning_rate', 'hp_n_estimators',
    'hp_min_child_samples', 'hp_subsample', 'hp_colsample_bytree',
    'hp_reg_alpha', 'hp_reg_lambda', 'hp_max_bin',
]

# Derived HP features (computed by HPCollector, stored in CSV)
HP_DERIVED_FEATURES = [
    'hp_leaves_depth_ratio', 'hp_regularization_strength',
    'hp_sample_ratio', 'hp_lr_estimators_product',
]

# Dataset x HP interaction features (computed during preparation).
# These capture the key relationships the meta-model needs to learn:
# how HP choices interact with dataset characteristics.
INTERACTION_FEATURES = [
    'ix_data_per_leaf',           # n_rows / hp_min_child_samples
    'ix_effective_features',      # n_cols * hp_colsample_bytree
    'ix_effective_train_size',    # n_rows * hp_subsample
    'ix_complexity_ratio',        # row_col_ratio * hp_num_leaves
    'ix_leaf_density',            # hp_num_leaves / max(n_cols, 1)
    'ix_regularization_vs_size',  # hp_regularization_strength / log1p(n_rows)
]

# All HP features (raw + derived + interactions)
ALL_HP_FEATURES = HP_PARAM_NAMES + HP_DERIVED_FEATURES + INTERACTION_FEATURES

# Refinement dimensions (optional): encoded from classification_subtype, imbalance_level, high_dim_level
REFINEMENT_ENCODED = [
    'classification_subtype_encoded',
    'imbalance_level_encoded',
    'high_dim_level_encoded',
]

# Full feature set for scorer and ranker (dataset + HP + interactions)
SCORER_FEATURES = DATASET_FEATURES + ALL_HP_FEATURES + ['task_type_encoded'] + REFINEMENT_ENCODED

# Feature set for direct predictor (dataset only)
PREDICTOR_FEATURES = DATASET_FEATURES + ['task_type_encoded'] + REFINEMENT_ENCODED

# HP params to predict (without the "hp_" prefix in the target, with prefix in feature name)
HP_TARGETS = {
    'hp_num_leaves':        {'log_transform': False, 'description': 'Tree complexity (leaves per tree)'},
    'hp_max_depth':         {'log_transform': False, 'description': 'Maximum tree depth'},
    'hp_learning_rate':     {'log_transform': True,  'description': 'Boosting learning rate'},
    'hp_n_estimators':      {'log_transform': True,  'description': 'Number of boosting rounds'},
    'hp_min_child_samples': {'log_transform': False, 'description': 'Minimum samples per leaf'},
    'hp_subsample':         {'log_transform': False, 'description': 'Row subsampling ratio'},
    'hp_colsample_bytree':  {'log_transform': False, 'description': 'Column subsampling ratio'},
    'hp_reg_alpha':         {'log_transform': True,  'description': 'L1 regularization'},
    'hp_reg_lambda':        {'log_transform': True,  'description': 'L2 regularization'},
    'hp_max_bin':           {'log_transform': False, 'description': 'Max histogram bins'},
}


# Refinement dimension thresholds (same as add_refinement_columns / HPCollector)
IMBALANCE_RATIO_BALANCED = 0.2
IMBALANCE_RATIO_MODERATE = 0.05
HIGH_DIM_LOW = 20
HIGH_DIM_HIGH = 100


def _compute_refinement_dimensions(task_type: str, ds_meta: dict) -> dict:
    """Compute classification_subtype, imbalance_level, high_dim_level from task_type and ds_meta."""
    n_classes = ds_meta.get("n_classes", 2)
    class_imbalance_ratio = ds_meta.get("class_imbalance_ratio", 1.0)
    n_cols = ds_meta.get("n_cols", 0)
    if task_type == "regression":
        classification_subtype = "na"
        imbalance_level = "na"
    else:
        classification_subtype = "multiclass" if (n_classes is not None and n_classes > 2) else "binary"
        if class_imbalance_ratio is None or class_imbalance_ratio < 0:
            imbalance_level = "na"
        elif class_imbalance_ratio >= IMBALANCE_RATIO_BALANCED:
            imbalance_level = "balanced"
        elif class_imbalance_ratio >= IMBALANCE_RATIO_MODERATE:
            imbalance_level = "moderate"
        else:
            imbalance_level = "imbalanced"
    if n_cols is None or n_cols < HIGH_DIM_LOW:
        high_dim_level = "low"
    elif n_cols < HIGH_DIM_HIGH:
        high_dim_level = "medium"
    else:
        high_dim_level = "high"
    return {
        "classification_subtype": classification_subtype,
        "imbalance_level": imbalance_level,
        "high_dim_level": high_dim_level,
    }


# =============================================================================
# DATA PREPARATION
# =============================================================================

class HPDataPreparator:
    """
    Prepares hp_tuning_db.csv for meta-model training.

    Handles:
    - Task type encoding
    - Task-type score normalization (z-score within clf/reg)
    - Dataset x HP interaction feature computation
    - Missing value imputation
    - Extracting top-K configs per dataset for direct predictor
    """

    def __init__(self):
        self.task_type_encoder = LabelEncoder()
        self.fill_values = None
        self.feature_columns_scorer = None
        self.feature_columns_predictor = None
        self.task_type_score_stats = {}
        # When CSV has refinement columns, we fit LabelEncoders and store here; otherwise None
        self._refinement_encoders = None

    def _compute_interaction_features(self, df):
        """
        Compute dataset x HP interaction features in-place.

        These give the scorer/ranker explicit signal about how HP choices
        relate to dataset characteristics, rather than relying on LightGBM
        to discover these interactions implicitly.
        """
        n_rows = df.get('n_rows', pd.Series(1, index=df.index))
        n_cols = df.get('n_cols', pd.Series(1, index=df.index))
        row_col_ratio = df.get('row_col_ratio', n_rows / n_cols.clip(lower=1))

        hp_mcs = df.get('hp_min_child_samples', pd.Series(20, index=df.index))
        hp_col = df.get('hp_colsample_bytree', pd.Series(0.8, index=df.index))
        hp_sub = df.get('hp_subsample', pd.Series(0.8, index=df.index))
        hp_leaves = df.get('hp_num_leaves', pd.Series(31, index=df.index))
        hp_reg_str = df.get('hp_regularization_strength',
                            pd.Series(0.5, index=df.index))

        df['ix_data_per_leaf'] = n_rows / hp_mcs.clip(lower=1)
        df['ix_effective_features'] = n_cols * hp_col
        df['ix_effective_train_size'] = n_rows * hp_sub
        df['ix_complexity_ratio'] = row_col_ratio * hp_leaves
        df['ix_leaf_density'] = hp_leaves / n_cols.clip(lower=1)
        df['ix_regularization_vs_size'] = hp_reg_str / np.log1p(n_rows).clip(lower=1)

    def prepare(self, df: pd.DataFrame, top_k=3):
        """
        Prepare the full database for training.

        Returns: dict with keys:
            'df_all':       cleaned DataFrame (all rows)
            'X_scorer':     features for scorer/ranker (dataset + HP features)
            'y_primary':    primary_score target (normalized within task type)
            'y_primary_raw': raw primary_score (for analysis)
            'X_predictor':  features for direct predictor (dataset features only)
            'y_hp_targets': dict of {hp_name: Series} -- optimal HP values (top-K weighted)
            'sample_weights_predictor': weights for direct predictor (by normalized_score)
            'dataset_ids':  integer-encoded dataset groupings
            'task_types':   task_type per row (for task-aware training)
            'metadata':     training metadata dict
        """
        df = df.copy()
        n_raw = len(df)

        # --- Basic cleanup ---
        df = df.dropna(subset=['primary_score']).reset_index(drop=True)
        print(f"  After dropping NaN primary_score: {len(df)} rows (was {n_raw})")

        # --- Encode task_type ---
        if 'task_type' in df.columns:
            df['task_type_encoded'] = self.task_type_encoder.fit_transform(
                df['task_type'].astype(str))
        else:
            df['task_type_encoded'] = 0

        # --- Encode refinement dimensions when present ---
        refinement_cols = ['classification_subtype', 'imbalance_level', 'high_dim_level']
        if all(c in df.columns for c in refinement_cols):
            self._refinement_encoders = {}
            for name in refinement_cols:
                le = LabelEncoder()
                df[f'{name}_encoded'] = le.fit_transform(df[name].astype(str))
                self._refinement_encoders[name] = le
        else:
            self._refinement_encoders = None

        # --- Normalize primary_score within task type ---
        # Classification (AUC, 0-1) and regression (neg-RMSE, variable range) have
        # incompatible scales. Z-score normalization within task type lets the scorer
        # learn a unified performance landscape.
        y_primary_raw = df['primary_score'].copy()

        self.task_type_score_stats = {}
        if 'task_type' in df.columns:
            y_primary_norm = df['primary_score'].copy()
            for tt in df['task_type'].unique():
                mask = df['task_type'] == tt
                tt_scores = df.loc[mask, 'primary_score']
                tt_mean = float(tt_scores.mean())
                tt_std = float(tt_scores.std())
                if tt_std < 1e-10:
                    tt_std = 1.0
                self.task_type_score_stats[tt] = {
                    'mean': tt_mean, 'std': tt_std
                }
                y_primary_norm.loc[mask] = (tt_scores - tt_mean) / tt_std
                print(f"  Task '{tt}': {mask.sum()} rows, "
                      f"raw score mean={tt_mean:.5f}, std={tt_std:.5f}")
            y_primary = y_primary_norm
        else:
            y_primary = y_primary_raw

        # --- Ensure numeric ---
        all_feature_cols = list(set(
            DATASET_FEATURES + HP_PARAM_NAMES + HP_DERIVED_FEATURES
            + ['task_type_encoded'] + REFINEMENT_ENCODED
        ))
        for col in all_feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)

        # --- Compute interaction features ---
        self._compute_interaction_features(df)

        # --- Validate derived HP features ---
        # hp_leaves_depth_ratio should be num_leaves / 2^max_depth (range 0-1).
        # The app (auto_fe_app_3.py) historically computed num_leaves / max_depth
        # instead -- this check catches that mismatch.
        if 'hp_leaves_depth_ratio' in df.columns:
            ratio_max = df['hp_leaves_depth_ratio'].max()
            if ratio_max > 2.0:
                print(f"  WARNING: hp_leaves_depth_ratio max={ratio_max:.2f} -- "
                      f"expected <=1.0 (num_leaves / 2^max_depth). "
                      f"Check computation consistency with HPCollector.")

        # --- Build scorer features (dataset + HP + interactions) ---
        scorer_cols = [c for c in SCORER_FEATURES if c in df.columns]
        self.feature_columns_scorer = scorer_cols
        X_scorer = df[scorer_cols].copy()

        # --- Build predictor features (dataset only) ---
        pred_cols = [c for c in PREDICTOR_FEATURES if c in df.columns]
        self.feature_columns_predictor = pred_cols

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

        # --- Extract top-K configs per dataset for direct predictor ---
        # Using only the single best config is noisy (it's the best *sampled*
        # config from only ~25 LHS points, not the true optimum). Using the
        # top-K configs weighted by normalized_score gives smoother targets
        # and more training data.
        y_hp_targets = {}
        X_predictor_direct = None
        sample_weights_predictor = None
        dataset_ids_predictor = None

        if 'dataset_name' in df.columns:
            # Compute within-dataset rank if not present
            if 'rank_in_dataset' not in df.columns:
                df['rank_in_dataset'] = df.groupby('dataset_name')['primary_score'].rank(
                    ascending=False, method='dense').astype(int)

            # Compute normalized_score within dataset if not present
            if 'normalized_score' not in df.columns:
                df['normalized_score'] = 0.0
                for ds_name, group in df.groupby('dataset_name'):
                    idx = group.index
                    best = group['primary_score'].max()
                    worst = group['primary_score'].min()
                    rng = best - worst if best != worst else 1e-8
                    df.loc[idx, 'normalized_score'] = (
                        (group['primary_score'] - worst) / rng)

            # Take top-K configs per dataset
            top_k_mask = df['rank_in_dataset'] <= top_k
            top_rows = df[top_k_mask].copy()

            # Weight by normalized_score (best config gets weight ~1.0)
            weights = top_rows['normalized_score'].clip(lower=0.1)

            n_datasets = df['dataset_name'].nunique()
            print(f"  Top-{top_k} configs per dataset: {len(top_rows)} rows "
                  f"from {n_datasets} datasets "
                  f"(avg {len(top_rows)/max(n_datasets,1):.1f}/dataset)")

            X_predictor_direct = top_rows[pred_cols].copy()
            pred_fill = {c: self.fill_values[c] for c in pred_cols
                         if c in self.fill_values.index}
            X_predictor_direct = X_predictor_direct.fillna(pred_fill)

            sample_weights_predictor = weights.reset_index(drop=True)

            # Dataset grouping for predictor (needed for GroupKFold to prevent
            # top-K rows from the same dataset leaking across train/val)
            ds_enc_pred = LabelEncoder()
            dataset_ids_predictor = pd.Series(
                ds_enc_pred.fit_transform(top_rows['dataset_name'].astype(str)),
                index=top_rows.index).reset_index(drop=True)

            for hp_name, hp_spec in HP_TARGETS.items():
                if hp_name in top_rows.columns:
                    vals = top_rows[hp_name].copy()
                    if hp_spec['log_transform']:
                        vals = np.log1p(vals.clip(lower=1e-10))
                    y_hp_targets[hp_name] = vals.reset_index(drop=True)

            X_predictor_direct = X_predictor_direct.reset_index(drop=True)

        # --- Task types per row (for task-aware training) ---
        task_types = df['task_type'].copy() if 'task_type' in df.columns else None

        # --- Metadata ---
        n_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else -1
        metadata = {
            'n_samples': len(df),
            'n_datasets': n_datasets,
            'n_scorer_features': len(scorer_cols),
            'n_predictor_features': len(pred_cols),
            'scorer_features': scorer_cols,
            'predictor_features': pred_cols,
            'interaction_features': [c for c in INTERACTION_FEATURES if c in scorer_cols],
            'hp_targets': list(HP_TARGETS.keys()),
            'hp_log_transformed': {k: v['log_transform'] for k, v in HP_TARGETS.items()},
            'task_type_mapping': dict(zip(
                self.task_type_encoder.classes_.tolist(),
                self.task_type_encoder.transform(self.task_type_encoder.classes_).tolist()
            )),
            'task_type_score_stats': self.task_type_score_stats,
            'primary_score_stats': {
                'mean': float(y_primary_raw.mean()),
                'std': float(y_primary_raw.std()),
                'min': float(y_primary_raw.min()),
                'max': float(y_primary_raw.max()),
            },
            'primary_score_normalized': True,
            'predictor_top_k': top_k,
            'training_timestamp': datetime.now().isoformat(),
        }

        return {
            'df_all': df,
            'X_scorer': X_scorer,
            'y_primary': y_primary,
            'y_primary_raw': y_primary_raw,
            'X_predictor': X_predictor_direct,
            'y_hp_targets': y_hp_targets,
            'sample_weights_predictor': sample_weights_predictor,
            'dataset_ids': dataset_ids,
            'dataset_ids_predictor': dataset_ids_predictor,
            'task_types': task_types,
            'metadata': metadata,
        }

    def transform_scorer(self, X_new):
        """Transform new data for scorer model at inference time."""
        # Compute interaction features if not present
        self._compute_interaction_features(X_new)

        X = X_new[[c for c in self.feature_columns_scorer if c in X_new.columns]].copy()
        # Add any missing columns as NaN (will be filled)
        for col in self.feature_columns_scorer:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns_scorer]

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.fill_values[self.feature_columns_scorer])
        return X

    def transform_predictor(self, X_new):
        """Transform new data for predictor model at inference time."""
        X = X_new[self.feature_columns_predictor].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        pred_fill = {c: self.fill_values[c] for c in self.feature_columns_predictor
                     if c in self.fill_values.index}
        X = X.fillna(pred_fill)
        return X

    def normalize_score(self, score, task_type):
        """Normalize a raw primary_score using stored task-type stats."""
        if task_type in self.task_type_score_stats:
            stats = self.task_type_score_stats[task_type]
            return (score - stats['mean']) / stats['std']
        return score

    def denormalize_score(self, z_score, task_type):
        """Convert a normalized z-score back to raw primary_score scale."""
        if task_type in self.task_type_score_stats:
            stats = self.task_type_score_stats[task_type]
            return z_score * stats['std'] + stats['mean']
        return z_score

    def encode_refinement_dimensions(self, task_type: str, ds_meta: dict) -> dict:
        """
        Compute refinement dimensions (classification_subtype, imbalance_level, high_dim_level)
        from task_type and ds_meta, then encode with the encoders fitted during prepare().
        Returns dict with keys classification_subtype_encoded, imbalance_level_encoded, high_dim_level_encoded.
        When the preparator was not trained with refinement columns, returns {}.
        """
        if self._refinement_encoders is None:
            return {}
        dims = _compute_refinement_dimensions(task_type, ds_meta)
        out = {}
        for name, le in self._refinement_encoders.items():
            val = dims[name]
            if val in le.classes_:
                out[f'{name}_encoded'] = int(le.transform([val])[0])
            else:
                out[f'{name}_encoded'] = 0
        return out

    def save(self, path):
        state = {
            'task_type_encoder_classes': self.task_type_encoder.classes_.tolist(),
            'fill_values': self.fill_values.to_dict(),
            'feature_columns_scorer': self.feature_columns_scorer,
            'feature_columns_predictor': self.feature_columns_predictor,
            'task_type_score_stats': self.task_type_score_stats,
            'refinement_encoder_classes': {
                k: enc.classes_.tolist()
                for k, enc in self._refinement_encoders.items()
            } if self._refinement_encoders else None,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.task_type_encoder.classes_ = np.array(state['task_type_encoder_classes'])
        self.fill_values = pd.Series(state['fill_values'])
        self.feature_columns_scorer = state['feature_columns_scorer']
        self.feature_columns_predictor = state['feature_columns_predictor']
        self.task_type_score_stats = state.get('task_type_score_stats', {})
        enc_classes = state.get('refinement_encoder_classes')
        if enc_classes:
            self._refinement_encoders = {}
            for k, classes in enc_classes.items():
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                self._refinement_encoders[k] = le
        else:
            self._refinement_encoders = None


# =============================================================================
# MODEL 1: HP CONFIG SCORER
# =============================================================================

class HPScorerTrainer:
    """
    Predicts primary_score given (dataset_meta + HP config + interactions).

    This is the most flexible model: at inference, generate N candidate
    configs, score each one, pick the highest-scoring.

    Uses GroupKFold on dataset_name to prevent leakage (configs from the
    same dataset should not appear in both train and val).

    The final model uses the average best_iteration from CV to avoid
    overfitting (rather than training with a fixed n_estimators and no
    early stopping).
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None

    def train(self, X, y, dataset_ids=None, n_folds=5):
        print(f"\n{'='*60}")
        print(f"Training HP Config Scorer (regressor)...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

        # GroupKFold by dataset
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

        # Final model on all data, using avg best_iteration from CV
        # (prevents overfitting vs. the old approach of fixed 500 iters)
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
# MODEL 2: DIRECT HP PREDICTOR
# =============================================================================

class DirectHPPredictor:
    """
    Predicts optimal HP values directly from dataset meta-features.

    One regressor per HP parameter, trained on the top-K performing configs
    per dataset, weighted by normalized_score. At inference: one forward
    pass -> complete HP config.

    For log-scale parameters (learning_rate, n_estimators, reg_alpha,
    reg_lambda), we predict log(value) and exp() at inference for
    better regression stability.
    """

    def __init__(self):
        self.models = {}        # {hp_name: model}
        self.cv_scores = {}     # {hp_name: {mae, rmse, r2}}
        self.hp_specs = HP_TARGETS.copy()

    def train(self, X, y_targets, sample_weights=None, dataset_ids=None, n_folds=5):
        """
        Train one regressor per HP parameter.

        Args:
            X: dataset meta-features (top-K rows per dataset)
            y_targets: dict of {hp_name: Series of target values}
            sample_weights: optional weights per sample (e.g., normalized_score)
            dataset_ids: dataset grouping for GroupKFold (prevents top-K rows
                         from the same dataset leaking across train/val folds)
        """
        print(f"\n{'='*60}")
        print(f"Training Direct HP Predictor ({len(y_targets)} parameters)...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        if sample_weights is not None:
            print(f"  Using sample weights (mean={sample_weights.mean():.3f})")

        # Determine CV strategy
        use_group_kfold = dataset_ids is not None
        if use_group_kfold:
            n_groups = dataset_ids.nunique()
            print(f"  Using GroupKFold ({n_groups} datasets) -- no cross-dataset leakage")
        else:
            print(f"  WARNING: No dataset_ids -- using KFold (potential leakage)")

        if len(X) < 10:
            print("  WARNING: Too few datasets for meaningful training. Skipping.")
            return {}

        for hp_name, y_hp in y_targets.items():
            spec = self.hp_specs.get(hp_name, {})
            is_log = spec.get('log_transform', False)
            desc = spec.get('description', hp_name)

            # Align indices (drop NaN targets)
            valid_mask = y_hp.notna()
            X_clean = X[valid_mask.values].reset_index(drop=True)
            y_clean = y_hp[valid_mask].reset_index(drop=True)
            w_clean = None
            if sample_weights is not None:
                w_clean = sample_weights[valid_mask.values].reset_index(drop=True)
            ds_ids_clean = None
            if dataset_ids is not None:
                ds_ids_clean = dataset_ids[valid_mask.values].reset_index(drop=True)

            if len(X_clean) < 10:
                print(f"  {hp_name}: skipped (only {len(X_clean)} valid samples)")
                continue

            # Select CV strategy
            if ds_ids_clean is not None:
                n_groups_clean = ds_ids_clean.nunique()
                n_actual_folds = min(n_folds, n_groups_clean)
                if n_actual_folds < 2:
                    print(f"  {hp_name}: only {n_groups_clean} groups, using KFold fallback")
                    kf = KFold(n_splits=min(n_folds, len(X_clean)), shuffle=True, random_state=42)
                    split_iter = kf.split(X_clean)
                else:
                    kf = GroupKFold(n_splits=n_actual_folds)
                    split_iter = kf.split(X_clean, y_clean, groups=ds_ids_clean)
            else:
                n_actual_folds = min(n_folds, len(X_clean))
                kf = KFold(n_splits=n_actual_folds, shuffle=True, random_state=42)
                split_iter = kf.split(X_clean)

            cv_mae, cv_r2 = [], []
            best_iterations = []

            lgb_params = dict(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=max(3, len(X_clean) // 20),
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=-1, n_jobs=-1,
            )

            for train_idx, val_idx in split_iter:
                X_tr = X_clean.iloc[train_idx]
                X_vl = X_clean.iloc[val_idx]
                y_tr = y_clean.iloc[train_idx]
                y_vl = y_clean.iloc[val_idx]
                w_tr = w_clean.iloc[train_idx] if w_clean is not None else None

                model = lgb.LGBMRegressor(**lgb_params)
                callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_vl, y_vl)], callbacks=callbacks)

                best_iterations.append(model.best_iteration_)

                preds = model.predict(X_vl)
                cv_mae.append(mean_absolute_error(y_vl, preds))
                try:
                    cv_r2.append(r2_score(y_vl, preds))
                except:
                    cv_r2.append(0.0)

            self.cv_scores[hp_name] = {
                'mae_mean': float(np.mean(cv_mae)),
                'r2_mean': float(np.mean(cv_r2)),
                'n_samples': len(X_clean),
                'log_transformed': is_log,
            }

            # Final model on all data, using avg best_iteration from CV
            avg_best_iter = int(np.mean(best_iterations))
            final_params = lgb_params.copy()
            final_params['n_estimators'] = max(avg_best_iter, 30)

            model = lgb.LGBMRegressor(**final_params)
            model.fit(X_clean, y_clean,
                      sample_weight=w_clean if w_clean is not None else None)
            self.models[hp_name] = model

            r2_str = f"{self.cv_scores[hp_name]['r2_mean']:.3f}"
            quality = "+" if self.cv_scores[hp_name]['r2_mean'] > 0.1 else "~" if self.cv_scores[hp_name]['r2_mean'] > 0 else "-"
            log_str = " (log)" if is_log else ""
            print(f"  {quality} {hp_name:30s} R2={r2_str:>7s}  "
                  f"MAE={self.cv_scores[hp_name]['mae_mean']:.4f}{log_str}  "
                  f"[{desc}]  (n_est={final_params['n_estimators']})")

        return self.cv_scores

    def predict(self, X):
        """
        Predict optimal HP config for each row in X.

        Returns: dict of {hp_name: predicted_values}
        """
        predictions = {}
        for hp_name, model in self.models.items():
            spec = self.hp_specs.get(hp_name, {})
            raw_pred = model.predict(X)

            if spec.get('log_transform', False):
                # Inverse log transform
                raw_pred = np.expm1(raw_pred)

            predictions[hp_name] = raw_pred

        return predictions

    def predict_single(self, X_row):
        """
        Predict a single HP config dict for one dataset.

        Applies consistency constraints to prevent contradictory predictions
        (e.g., num_leaves=127 with max_depth=3, which is impossible since
        a tree of depth 3 can have at most 2^3=8 leaves).

        Args:
            X_row: DataFrame with 1 row of dataset meta-features

        Returns: dict of {param_name: value} (without "hp_" prefix)
        """
        preds = self.predict(X_row)
        config = {}
        for hp_name, values in preds.items():
            val = float(values[0])
            param_name = hp_name.replace('hp_', '')
            config[param_name] = val
        return self._enforce_consistency(config)

    @staticmethod
    def _enforce_consistency(config):
        """
        Apply inter-parameter consistency constraints to a predicted HP config.

        Constraints enforced:
        1. Bound clamping: all values within HP_SPACE bounds
        2. num_leaves <= 2^max_depth (tree capacity constraint)
        3. learning_rate * n_estimators in reasonable range (learning budget)
        4. min_child_samples scaled relative to implied data per leaf
        5. Integer rounding for int-typed parameters
        """
        try:
            from HPCollector import HP_SPACE
        except ImportError:
            # Fallback bounds if HPCollector not importable
            HP_SPACE = {
                'num_leaves':        {'low': 4,     'high': 256,   'type': 'int'},
                'max_depth':         {'low': 3,     'high': 15,    'type': 'int'},
                'learning_rate':     {'low': 0.005, 'high': 0.3,   'type': 'float'},
                'n_estimators':      {'low': 50,    'high': 3000,  'type': 'int'},
                'min_child_samples': {'low': 5,     'high': 100,   'type': 'int'},
                'subsample':         {'low': 0.5,   'high': 1.0,   'type': 'float'},
                'colsample_bytree':  {'low': 0.3,   'high': 1.0,   'type': 'float'},
                'reg_alpha':         {'low': 1e-8,  'high': 10.0,  'type': 'float'},
                'reg_lambda':        {'low': 1e-8,  'high': 10.0,  'type': 'float'},
                'max_bin':           {'low': 63,    'high': 511,   'type': 'int'},
            }

        # 1. Clamp to bounds and round integers
        for param_name, spec in HP_SPACE.items():
            if param_name in config:
                val = config[param_name]
                val = max(spec['low'], min(spec['high'], val))
                if spec['type'] == 'int':
                    val = int(round(val))
                config[param_name] = val

        # 2. Tree capacity constraint: num_leaves <= 2^max_depth
        if 'num_leaves' in config and 'max_depth' in config:
            max_possible = 2 ** config['max_depth']
            if config['num_leaves'] > max_possible:
                # Prefer adjusting num_leaves down (less disruptive than
                # increasing depth, which changes regularization behavior)
                config['num_leaves'] = max_possible

        # 3. Learning budget constraint: lr * n_estimators in [5, 300]
        # Too low = underfitting, too high = overfitting/wasted compute.
        if 'learning_rate' in config and 'n_estimators' in config:
            budget = config['learning_rate'] * config['n_estimators']
            if budget < 5:
                # Scale up n_estimators to reach minimum budget
                config['n_estimators'] = max(
                    int(round(5.0 / config['learning_rate'])),
                    HP_SPACE.get('n_estimators', {}).get('low', 50))
            elif budget > 300:
                # Scale down n_estimators to cap budget
                config['n_estimators'] = min(
                    int(round(300.0 / config['learning_rate'])),
                    HP_SPACE.get('n_estimators', {}).get('high', 3000))

        # Re-clamp after adjustments
        for param_name in ['num_leaves', 'n_estimators']:
            if param_name in config and param_name in HP_SPACE:
                spec = HP_SPACE[param_name]
                config[param_name] = max(spec['low'], min(spec['high'],
                                         int(round(config[param_name]))))

        return config

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'cv_scores': self.cv_scores,
                'hp_specs': self.hp_specs,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.models = state['models']
        self.cv_scores = state.get('cv_scores', {})
        self.hp_specs = state.get('hp_specs', HP_TARGETS)


# =============================================================================
# MODEL 3: HP CONFIG RANKER (LambdaRank)
# =============================================================================

class HPRankerTrainer:
    """
    LightGBM ranker with lambdarank objective.

    Groups by dataset -- within each dataset, learns which HP configs
    perform better relative to others.

    Uses within-dataset rank-based relevance labels (not global quantiles)
    so that relevance is meaningful within each group.
    """

    def __init__(self):
        self.model = None
        self.cv_scores = None

    def _compute_within_dataset_relevance(self, y_scores, dataset_ids, n_levels=5):
        """
        Compute relevance labels (0 to n_levels-1) within each dataset.

        Unlike global quantiles (which make cross-dataset comparisons
        that are meaningless for ranking), this assigns relevance based on
        each config's rank within its own dataset.
        """
        relevance = pd.Series(0, index=y_scores.index, dtype=int)

        for ds_id in dataset_ids.unique():
            mask = dataset_ids == ds_id
            ds_scores = y_scores[mask]

            if len(ds_scores) <= 1:
                relevance[mask] = n_levels // 2
                continue

            # Rank within this dataset (1 = best)
            ranks = ds_scores.rank(ascending=False, method='dense')
            n_configs = ranks.max()

            if n_configs <= 1:
                relevance[mask] = n_levels // 2
                continue

            # Map rank to relevance: rank 1 -> highest relevance, worst -> 0
            # normalized_rank in [0, 1] where 0 = best
            normalized_rank = (ranks - 1) / (n_configs - 1)
            # Invert and scale to [0, n_levels-1]
            rel = ((1 - normalized_rank) * (n_levels - 1)).round().astype(int)
            rel = rel.clip(0, n_levels - 1)
            relevance[mask] = rel

        return relevance

    def train(self, X, y_scores, dataset_ids, n_folds=5):
        print(f"\n{'='*60}")
        print(f"Training HP Config Ranker (lambdarank)...")
        print(f"  Samples: {len(X)}, Datasets: {dataset_ids.nunique()}")

        if len(X) < 20 or dataset_ids.nunique() < 5:
            print("  WARNING: Too few samples/datasets for ranker. Skipping.")
            return None

        # Convert scores to within-dataset relevance labels (0-4 scale)
        relevance = self._compute_within_dataset_relevance(y_scores, dataset_ids, n_levels=5)

        # Verify relevance distribution
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
        ds_sorted = dataset_ids.iloc[sorted_idx].reset_index(drop=True)
        group_sizes = group_df.groupby('dataset_id').size().values

        # Filter single-sample groups (can't rank a single item)
        valid_groups = group_sizes > 1
        if valid_groups.sum() < 3:
            print("  WARNING: Too few multi-sample groups. Skipping.")
            return None

        cumsum = np.cumsum(np.concatenate([[0], group_sizes]))
        valid_indices = []
        valid_group_sizes = []
        valid_group_ids = []
        for i, v in enumerate(valid_groups):
            if v:
                valid_indices.extend(range(cumsum[i], cumsum[i+1]))
                valid_group_sizes.append(group_sizes[i])
                valid_group_ids.extend([i] * group_sizes[i])

        X_rank = X_sorted.iloc[valid_indices].reset_index(drop=True)
        rel_rank = rel_sorted.iloc[valid_indices].reset_index(drop=True)
        group_sizes_valid = np.array(valid_group_sizes)
        group_ids_per_row = np.array(valid_group_ids)

        print(f"  After filtering: {len(X_rank)} samples in {len(group_sizes_valid)} groups")

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

        # --- GroupKFold CV on groups for reliable NDCG estimates ---
        n_valid_groups = len(group_sizes_valid)
        actual_folds = min(n_folds, n_valid_groups)

        if actual_folds < 2:
            print("  WARNING: Not enough groups for CV. Training without validation.")
            full_set = lgb.Dataset(X_rank, label=rel_rank, group=group_sizes_valid)
            self.model = lgb.train(params, full_set, num_boost_round=200)
            self.cv_scores = {}
            return self.cv_scores

        unique_group_ids = np.unique(group_ids_per_row)
        gkf = GroupKFold(n_splits=actual_folds)
        cv_ndcg_scores = defaultdict(list)
        best_iterations = []

        print(f"  Running {actual_folds}-fold GroupKFold CV...")

        for fold_i, (train_group_idx, val_group_idx) in enumerate(
                gkf.split(unique_group_ids, groups=unique_group_ids)):

            train_groups = set(unique_group_ids[train_group_idx])
            val_groups = set(unique_group_ids[val_group_idx])

            train_mask = np.isin(group_ids_per_row, list(train_groups))
            val_mask = np.isin(group_ids_per_row, list(val_groups))

            X_tr = X_rank[train_mask].reset_index(drop=True)
            rel_tr = rel_rank[train_mask].reset_index(drop=True)
            X_vl = X_rank[val_mask].reset_index(drop=True)
            rel_vl = rel_rank[val_mask].reset_index(drop=True)

            # Compute group sizes for train and val splits
            groups_tr = []
            for gid in sorted(train_groups):
                groups_tr.append(int((group_ids_per_row[train_mask] == gid).sum()))
            groups_vl = []
            for gid in sorted(val_groups):
                groups_vl.append(int((group_ids_per_row[val_mask] == gid).sum()))

            train_set = lgb.Dataset(X_tr, label=rel_tr, group=groups_tr)
            val_set = lgb.Dataset(X_vl, label=rel_vl, group=groups_vl, reference=train_set)

            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
            fold_model = lgb.train(
                params, train_set, num_boost_round=500,
                valid_sets=[val_set], callbacks=callbacks,
            )

            best_iterations.append(
                fold_model.best_iteration if fold_model.best_iteration else 100)

            if fold_model.best_score and 'valid_0' in fold_model.best_score:
                for metric_name, val in fold_model.best_score['valid_0'].items():
                    cv_ndcg_scores[metric_name].append(float(val))

        # Aggregate CV scores
        self.cv_scores = {}
        for metric_name, vals in cv_ndcg_scores.items():
            self.cv_scores[f'{metric_name}_mean'] = float(np.mean(vals))
            self.cv_scores[f'{metric_name}_std'] = float(np.std(vals))
        print(f"  CV NDCG: {self.cv_scores}")

        # --- Retrain on ALL data using avg best_iteration from CV ---
        avg_best_iter = int(np.mean(best_iterations))
        avg_best_iter = max(avg_best_iter, 50)
        print(f"  Retraining on full data with {avg_best_iter} iterations "
              f"(CV range: {min(best_iterations)}-{max(best_iterations)})")

        full_set = lgb.Dataset(X_rank, label=rel_rank, group=group_sizes_valid)
        self.model = lgb.train(params, full_set, num_boost_round=avg_best_iter)

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
# ENSEMBLE INFERENCE: COMBINING ALL THREE MODELS
# =============================================================================

class HPEnsemblePredictor:
    """
    Combines the scorer, predictor, and ranker for robust HP prediction.

    Strategy:
    1. Direct predictor produces an initial HP config from dataset meta-features.
    2. Generate N perturbations around that config (exploring the neighborhood).
    3. Score all candidates (initial + perturbations) with the scorer model.
    4. Rank all candidates with the ranker model.
    5. Select the best candidate by combining scorer and ranker signals.

    This captures the strengths of all three models:
    - Predictor: good starting point tuned to the dataset
    - Scorer: absolute quality estimate
    - Ranker: relative ordering accuracy
    """

    def __init__(self, preparator, scorer, predictor, ranker=None):
        """
        Args:
            preparator: HPDataPreparator (for feature transforms)
            scorer: HPScorerTrainer (scores configs)
            predictor: DirectHPPredictor (predicts initial config)
            ranker: HPRankerTrainer (ranks configs), optional
        """
        self.preparator = preparator
        self.scorer = scorer
        self.predictor = predictor
        self.ranker = ranker

    def predict(self, ds_meta_row, task_type='classification',
                n_perturbations=50, perturbation_scale=0.3, seed=42):
        """
        Predict optimal HP config for a single dataset.

        Args:
            ds_meta_row: DataFrame with 1 row of dataset meta-features
            task_type: 'classification' or 'regression'
            n_perturbations: number of perturbed configs to generate
            perturbation_scale: how far to perturb (0.0 = no change, 1.0 = full range)
            seed: random seed for perturbation generation

        Returns: dict with:
            'best_config': the selected HP config dict
            'initial_config': the predictor's raw output (before ensemble)
            'best_scorer_score': scorer's score for the best config
            'candidates_evaluated': number of candidates scored
            'method': 'ensemble' or 'predictor_only'
        """
        rng = np.random.RandomState(seed)

        # --- Step 1: Get initial prediction from direct predictor ---
        X_pred = self.preparator.transform_predictor(ds_meta_row)
        initial_config = self.predictor.predict_single(X_pred)

        # If scorer isn't available, fall back to predictor-only
        if self.scorer.model is None:
            return {
                'best_config': initial_config,
                'initial_config': initial_config,
                'best_scorer_score': None,
                'candidates_evaluated': 1,
                'method': 'predictor_only',
            }

        # --- Step 2: Generate perturbations around the initial config ---
        candidates = [initial_config]  # always include the predictor's output
        candidates.extend(
            self._generate_perturbations(initial_config, n_perturbations,
                                         perturbation_scale, rng))

        # --- Step 3: Score all candidates ---
        scored = self._score_candidates(candidates, ds_meta_row, task_type)

        # --- Step 4: Rank all candidates (if ranker available) ---
        if self.ranker is not None and self.ranker.model is not None:
            ranked = self._rank_candidates(scored, ds_meta_row, task_type)
        else:
            ranked = scored

        # --- Step 5: Select best by combined signal ---
        best = self._select_best(ranked)

        return {
            'best_config': best['config'],
            'initial_config': initial_config,
            'best_scorer_score': best.get('scorer_score'),
            'best_ranker_score': best.get('ranker_score'),
            'candidates_evaluated': len(candidates),
            'method': 'ensemble',
        }

    def _generate_perturbations(self, base_config, n_perturbations,
                                 scale, rng):
        """
        Generate perturbed configs around the base.

        For each perturbation, each parameter is independently nudged
        by a random fraction of its range (in log space for log params).
        """
        try:
            from HPCollector import HP_SPACE
        except ImportError:
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

        perturbations = []
        for _ in range(n_perturbations):
            config = {}
            for param_name, spec in HP_SPACE.items():
                base_val = base_config.get(param_name)
                if base_val is None:
                    continue

                # Perturbation in normalized space
                noise = rng.normal(0, scale)

                if spec.get('log', False):
                    # Perturb in log space
                    log_low = np.log(spec['low'])
                    log_high = np.log(spec['high'])
                    log_val = np.log(max(base_val, spec['low']))
                    log_range = log_high - log_low
                    new_log_val = log_val + noise * log_range
                    new_val = np.exp(np.clip(new_log_val, log_low, log_high))
                else:
                    # Perturb in linear space
                    val_range = spec['high'] - spec['low']
                    new_val = base_val + noise * val_range
                    new_val = np.clip(new_val, spec['low'], spec['high'])

                if spec['type'] == 'int':
                    new_val = int(round(new_val))
                config[param_name] = new_val

            # Apply consistency constraints
            config = DirectHPPredictor._enforce_consistency(config)
            perturbations.append(config)

        return perturbations

    def _score_candidates(self, candidates, ds_meta_row, task_type):
        """Score each candidate config using the scorer model."""
        scored = []

        for config in candidates:
            # Build a full feature row: dataset meta + HP config + interactions
            row = ds_meta_row.iloc[0].to_dict() if hasattr(ds_meta_row, 'iloc') else ds_meta_row

            # Add HP features
            for param_name, val in config.items():
                row[f'hp_{param_name}'] = val

            # Add derived HP features
            num_leaves = config.get('num_leaves', 31)
            max_depth = config.get('max_depth', 6)
            lr = config.get('learning_rate', 0.05)
            n_est = config.get('n_estimators', 1000)
            subsample = config.get('subsample', 0.8)
            colsample = config.get('colsample_bytree', 0.8)
            reg_alpha = config.get('reg_alpha', 0.05)
            reg_lambda = config.get('reg_lambda', 0.5)

            max_possible_leaves = 2 ** max_depth
            row['hp_leaves_depth_ratio'] = num_leaves / max(max_possible_leaves, 1)
            row['hp_regularization_strength'] = np.log1p(reg_alpha + reg_lambda)
            row['hp_sample_ratio'] = subsample * colsample
            row['hp_lr_estimators_product'] = lr * n_est

            row_df = pd.DataFrame([row])
            X_scorer = self.preparator.transform_scorer(row_df)

            scorer_score = float(self.scorer.predict(X_scorer)[0])

            scored.append({
                'config': config,
                'scorer_score': scorer_score,
            })

        return scored

    def _rank_candidates(self, scored_candidates, ds_meta_row, task_type):
        """Add ranker scores to the candidates."""
        for item in scored_candidates:
            config = item['config']
            row = ds_meta_row.iloc[0].to_dict() if hasattr(ds_meta_row, 'iloc') else ds_meta_row

            for param_name, val in config.items():
                row[f'hp_{param_name}'] = val

            num_leaves = config.get('num_leaves', 31)
            max_depth = config.get('max_depth', 6)
            lr = config.get('learning_rate', 0.05)
            n_est = config.get('n_estimators', 1000)
            subsample = config.get('subsample', 0.8)
            colsample = config.get('colsample_bytree', 0.8)
            reg_alpha = config.get('reg_alpha', 0.05)
            reg_lambda = config.get('reg_lambda', 0.5)

            max_possible_leaves = 2 ** max_depth
            row['hp_leaves_depth_ratio'] = num_leaves / max(max_possible_leaves, 1)
            row['hp_regularization_strength'] = np.log1p(reg_alpha + reg_lambda)
            row['hp_sample_ratio'] = subsample * colsample
            row['hp_lr_estimators_product'] = lr * n_est

            row_df = pd.DataFrame([row])
            X_scorer = self.preparator.transform_scorer(row_df)

            ranker_score = float(self.ranker.predict(X_scorer)[0])
            item['ranker_score'] = ranker_score

        return scored_candidates

    def _select_best(self, candidates):
        """
        Select the best candidate from scored + ranked candidates.

        Uses a weighted combination of scorer rank and ranker rank.
        Scorer gets 60% weight (absolute quality), ranker gets 40%
        (relative ordering).
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        # Rank by scorer score (higher = better)
        scorer_sorted = sorted(range(len(candidates)),
                               key=lambda i: candidates[i]['scorer_score'],
                               reverse=True)
        scorer_ranks = {idx: rank for rank, idx in enumerate(scorer_sorted)}

        # Rank by ranker score if available
        has_ranker = 'ranker_score' in candidates[0]
        if has_ranker:
            ranker_sorted = sorted(range(len(candidates)),
                                   key=lambda i: candidates[i]['ranker_score'],
                                   reverse=True)
            ranker_ranks = {idx: rank for rank, idx in enumerate(ranker_sorted)}
        else:
            ranker_ranks = scorer_ranks  # fall back to scorer only

        # Combined rank: weighted sum (lower = better)
        SCORER_WEIGHT = 0.6
        RANKER_WEIGHT = 0.4 if has_ranker else 0.0

        best_idx = min(range(len(candidates)),
                       key=lambda i: (SCORER_WEIGHT * scorer_ranks[i] +
                                      RANKER_WEIGHT * ranker_ranks.get(i, 0)))

        return candidates[best_idx]

    def save(self, output_dir):
        """Save the ensemble (just a reference file -- individual models saved separately)."""
        meta = {
            'has_scorer': self.scorer.model is not None,
            'has_predictor': len(self.predictor.models) > 0,
            'has_ranker': self.ranker is not None and self.ranker.model is not None,
        }
        with open(os.path.join(output_dir, 'hp_ensemble_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, output_dir, task_type=None):
        """Load ensemble from saved component models.
        If task_type is given (e.g. 'classification'), load from output_dir/task_type/.
        """
        if task_type is not None:
            output_dir = os.path.join(output_dir, task_type)
        preparator = HPDataPreparator()
        preparator.load(os.path.join(output_dir, 'hp_preparator.pkl'))

        scorer = HPScorerTrainer()
        scorer.load(os.path.join(output_dir, 'hp_scorer.pkl'))

        predictor = DirectHPPredictor()
        predictor.load(os.path.join(output_dir, 'hp_predictor.pkl'))

        ranker = HPRankerTrainer()
        ranker_path = os.path.join(output_dir, 'hp_ranker.pkl')
        if os.path.exists(ranker_path):
            ranker.load(ranker_path)
        else:
            ranker = None

        return cls(preparator, scorer, predictor, ranker)


# =============================================================================
# DATASET ARCHETYPE CLUSTERING
# =============================================================================

def compute_archetypes(df, n_clusters=6, cluster_method='kmeans'):
    """
    Cluster datasets by their optimal HP profiles.

    For each dataset, extract the best-performing HP config, then cluster
    the HP value vectors (K-Means or Spectral). Each cluster becomes a
    named archetype with an average HP config.

    cluster_method: 'kmeans' or 'spectral'. Spectral has no .predict() for
    new points; inference uses nearest training point's label (saved in pkl).

    Returns: dict with cluster info, or None if insufficient data.
    """
    print(f"\n{'='*60}")
    print(f"Computing Dataset Archetypes ({n_clusters} clusters, {cluster_method})...")

    if 'dataset_name' not in df.columns:
        print("  WARNING: No dataset_name column. Skipping archetypes.")
        return None

    # Get best config per dataset
    best_mask = df.groupby('dataset_name')['primary_score'].transform('max') == df['primary_score']
    best_rows = df[best_mask].drop_duplicates(subset='dataset_name', keep='first')

    if len(best_rows) < n_clusters * 2:
        print(f"  WARNING: Only {len(best_rows)} datasets -- too few for {n_clusters} clusters.")
        n_clusters = max(2, len(best_rows) // 3)
        print(f"  Reducing to {n_clusters} clusters.")

    # Extract HP values for clustering
    hp_cols = [c for c in HP_PARAM_NAMES if c in best_rows.columns]
    X_hp = best_rows[hp_cols].copy()

    # Log-transform skewed HP params before clustering
    for col in hp_cols:
        param = col.replace('hp_', '')
        if param in ['learning_rate', 'n_estimators', 'reg_alpha', 'reg_lambda']:
            X_hp[col] = np.log1p(X_hp[col].clip(lower=1e-10))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hp.fillna(X_hp.median()))
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Cluster
    if cluster_method == 'spectral':
        n_neighbors = min(10, max(2, len(X_scaled) // max(1, n_clusters)))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=42,
        )
        labels = model.fit_predict(X_scaled)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    best_rows = best_rows.copy()
    best_rows['archetype_cluster'] = labels

    # Build archetype profiles
    archetypes = {}

    for cluster_id in range(n_clusters):
        cluster_mask = best_rows['archetype_cluster'] == cluster_id
        cluster_rows = best_rows[cluster_mask]

        if len(cluster_rows) == 0:
            continue

        # Average HP config for this cluster
        avg_hp = {}
        for col in HP_PARAM_NAMES:
            if col in cluster_rows.columns:
                val = cluster_rows[col].mean()
                param = col.replace('hp_', '')
                if param in ['num_leaves', 'max_depth', 'n_estimators',
                             'min_child_samples', 'max_bin']:
                    avg_hp[param] = int(round(val))
                else:
                    avg_hp[param] = round(val, 4)

        # Cluster characterization from dataset meta-features
        avg_n_rows = cluster_rows['n_rows'].mean() if 'n_rows' in cluster_rows else 0
        avg_n_cols = cluster_rows['n_cols'].mean() if 'n_cols' in cluster_rows else 0
        avg_cat_ratio = cluster_rows['cat_ratio'].mean() if 'cat_ratio' in cluster_rows else 0

        # Auto-generate a descriptive name
        name_parts = []
        if avg_n_rows < 2000:
            name_parts.append("Small")
        elif avg_n_rows < 20000:
            name_parts.append("Medium")
        else:
            name_parts.append("Large")

        if avg_n_cols > 100:
            name_parts.append("Wide")

        if avg_cat_ratio > 0.5:
            name_parts.append("High-Cat")

        lr = avg_hp.get('learning_rate', 0.05)
        if lr < 0.02:
            name_parts.append("Low-LR")
        elif lr > 0.1:
            name_parts.append("High-LR")

        leaves = avg_hp.get('num_leaves', 31)
        if leaves > 60:
            name_parts.append("High-Capacity")
        elif leaves < 20:
            name_parts.append("Regularized")

        archetype_name = " / ".join(name_parts) if name_parts else f"Cluster-{cluster_id}"

        archetypes[archetype_name] = {
            'cluster_id': int(cluster_id),
            'n_datasets': int(len(cluster_rows)),
            'params': avg_hp,
            'dataset_profile': {
                'avg_n_rows': float(avg_n_rows),
                'avg_n_cols': float(avg_n_cols),
                'avg_cat_ratio': float(avg_cat_ratio),
            },
            'avg_primary_score': float(cluster_rows['primary_score'].mean()),
        }

        print(f"  {archetype_name} ({len(cluster_rows)} datasets):")
        print(f"    Data: ~{avg_n_rows:.0f} rows, ~{avg_n_cols:.0f} cols, "
              f"{avg_cat_ratio:.0%} categorical")
        print(f"    HPs:  leaves={avg_hp.get('num_leaves')}, "
              f"lr={avg_hp.get('learning_rate')}, "
              f"depth={avg_hp.get('max_depth')}, "
              f"n_est={avg_hp.get('n_estimators')}")

    out = {
        'archetypes': archetypes,
        'n_clusters': n_clusters,
        'scaler': scaler,
        'hp_cols_used': hp_cols,
        'cluster_method': cluster_method,
        'cluster_model': model,
    }
    if cluster_method == 'kmeans':
        out['kmeans_model'] = model  # backward compat for assign_config_to_archetype
    else:
        # Spectral has no .predict(); inference will use nearest training point
        out['X_scaled_train'] = X_scaled
        out['train_labels'] = labels
    return out


# =============================================================================
# HP ANALYSIS REPORT
# =============================================================================

def print_hp_analysis(df):
    """Print analysis of which HPs matter most across datasets."""
    print(f"\n{'='*60}")
    print("HP SENSITIVITY ANALYSIS")
    print(f"{'='*60}")

    if 'dataset_name' not in df.columns:
        return

    n_datasets = df['dataset_name'].nunique()

    # 1. How much does tuning help?
    if 'delta_vs_default' in df.columns and 'rank_in_dataset' in df.columns:
        best_only = df[df['rank_in_dataset'] == 1]
        if len(best_only) > 0:
            avg_improvement = best_only['delta_vs_default'].mean()
            pct_improved = (best_only['delta_vs_default'] > 0).mean() * 100
            max_improvement = best_only['delta_vs_default'].max()

            print(f"\n  Tuning vs Default ({n_datasets} datasets):")
            print(f"    Average improvement:  {avg_improvement:+.5f}")
            print(f"    Best-case gain:       {max_improvement:+.5f}")
            print(f"    % datasets improved:  {pct_improved:.1f}%")

    # 2. Score spread within datasets (how much do HPs matter?)
    if 'normalized_score' in df.columns:
        score_ranges = df.groupby('dataset_name')['primary_score'].agg(['min', 'max'])
        score_ranges['range'] = score_ranges['max'] - score_ranges['min']
        avg_range = score_ranges['range'].mean()
        print(f"\n  Score Range within Datasets:")
        print(f"    Average range:  {avg_range:.5f}")
        print(f"    Median range:   {score_ranges['range'].median():.5f}")
        print(f"    Max range:      {score_ranges['range'].max():.5f}")

    # 3. Which HP parameters have the highest correlation with performance?
    hp_cols = [c for c in HP_PARAM_NAMES if c in df.columns]
    if hp_cols and 'primary_score' in df.columns:
        print(f"\n  HP-Performance Correlations (Spearman):")
        corrs = {}
        for col in hp_cols:
            try:
                from scipy.stats import spearmanr
                r, p = spearmanr(df[col].fillna(0), df['primary_score'])
                corrs[col] = (r, p)
            except:
                pass

        for col, (r, p) in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {col:30s}  r={r:+.4f}  {sig}")

    # 4. Best HP value distributions
    if 'rank_in_dataset' in df.columns:
        best_only = df[df['rank_in_dataset'] == 1]
        if len(best_only) > 5:
            print(f"\n  Optimal HP Distributions (best config per dataset):")
            for col in hp_cols:
                if col in best_only.columns:
                    vals = best_only[col].dropna()
                    if len(vals) > 0:
                        print(f"    {col:30s}  "
                              f"median={vals.median():>10.4f}  "
                              f"IQR=[{vals.quantile(0.25):.4f}, {vals.quantile(0.75):.4f}]  "
                              f"range=[{vals.min():.4f}, {vals.max():.4f}]")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

MIN_SAMPLES_PER_TASK = 100


def _train_single_task(df_tt: pd.DataFrame, task_type: str, output_subdir: str,
                       top_k: int = 3, n_clusters: int = 6, cluster_method: str = 'kmeans'):
    """
    Train the full HP meta-model pipeline for one task type; save to output_subdir.
    Returns a dict with task_type, n_samples, n_datasets, cv_scores for manifest.
    """
    os.makedirs(output_subdir, exist_ok=True)
    n_samples = len(df_tt)
    n_datasets = df_tt['dataset_name'].nunique() if 'dataset_name' in df_tt.columns else 0

    print(f"\n{'='*70}")
    print(f"TASK TYPE: {task_type}  ({n_samples} rows, {n_datasets} datasets)")
    print(f"{'='*70}")

    preparator = HPDataPreparator()
    data = preparator.prepare(df_tt, top_k=top_k)

    X_scorer = data['X_scorer']
    y_primary = data['y_primary']
    dataset_ids = data['dataset_ids']

    print(f"  Scorer features: {X_scorer.shape[1]}, Predictor features: {len(preparator.feature_columns_predictor)}")

    scorer = HPScorerTrainer()
    scorer_cv = scorer.train(X_scorer, y_primary, dataset_ids=dataset_ids)

    predictor = DirectHPPredictor()
    predictor_cv = {}
    if data['X_predictor'] is not None and data['y_hp_targets']:
        predictor_cv = predictor.train(
            data['X_predictor'],
            data['y_hp_targets'],
            sample_weights=data.get('sample_weights_predictor'),
            dataset_ids=data.get('dataset_ids_predictor'),
        )
    else:
        print("  WARNING: Cannot train direct predictor (no best-config data).")

    ranker = HPRankerTrainer()
    ranker_cv = None
    if dataset_ids is not None:
        ranker_cv = ranker.train(X_scorer, y_primary, dataset_ids)
    else:
        print("  WARNING: No dataset_ids -- skipping ranker.")

    archetype_data = compute_archetypes(df_tt, n_clusters=n_clusters, cluster_method=cluster_method)

    preparator.save(os.path.join(output_subdir, 'hp_preparator.pkl'))
    scorer.save(os.path.join(output_subdir, 'hp_scorer.pkl'))
    predictor.save(os.path.join(output_subdir, 'hp_predictor.pkl'))
    ranker.save(os.path.join(output_subdir, 'hp_ranker.pkl'))

    ensemble = HPEnsemblePredictor(preparator, scorer, predictor, ranker)
    ensemble.save(output_subdir)

    if archetype_data is not None:
        with open(os.path.join(output_subdir, 'hp_archetypes.pkl'), 'wb') as f:
            pickle.dump(archetype_data, f)
        archetype_json = {
            name: {
                'params': info['params'],
                'n_datasets': info['n_datasets'],
                'dataset_profile': info['dataset_profile'],
                'avg_primary_score': info['avg_primary_score'],
            }
            for name, info in archetype_data['archetypes'].items()
        }
        with open(os.path.join(output_subdir, 'hp_archetypes.json'), 'w') as f:
            json.dump(archetype_json, f, indent=2)

    all_cv = {'scorer': scorer_cv, 'predictor': predictor_cv, 'ranker': ranker_cv}
    with open(os.path.join(output_subdir, 'hp_cv_scores.json'), 'w') as f:
        json.dump(all_cv, f, indent=2, default=str)

    metadata = data['metadata']
    metadata['cv_scores'] = all_cv
    metadata['n_archetypes'] = len(archetype_data['archetypes']) if archetype_data else 0
    with open(os.path.join(output_subdir, 'hp_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Saved to {output_subdir}/")
    return {
        'task_type': task_type,
        'n_samples': n_samples,
        'n_datasets': n_datasets,
        'cv_scores': all_cv,
    }


def train_hp_meta_model(hp_db_path: str, output_dir: str = './hp_meta_model',
                        top_k: int = 3, n_clusters: int = 6, per_task_type: bool = False,
                        holdout_ratio: float = 0.0, holdout_seed: int = 42,
                        cluster_method: str = 'kmeans'):
    """
    holdout_ratio: fraction of datasets to hold out for evaluation (0 = no hold-out).
    When > 0, those datasets are excluded from training and their names are saved to
    output_dir/holdout_datasets.json so the evaluation notebook can evaluate only on them.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HP META-MODEL TRAINING")
    print("=" * 70)

    # --- Load ---
    print(f"\nLoading: {hp_db_path}")
    df = pd.read_csv(hp_db_path)
    n_datasets = df['dataset_name'].nunique() if 'dataset_name' in df.columns else '?'
    print(f"  {len(df)} rows, {n_datasets} datasets")

    if len(df) == 0:
        print("ERROR: Empty database!")
        return

    # --- Hold-out split (by dataset) to avoid evaluation leakage ---
    holdout_datasets = []
    if holdout_ratio > 0 and 'dataset_name' in df.columns:
        all_ds = df['dataset_name'].unique().tolist()
        n = len(all_ds)
        if n >= 2:
            n_holdout = max(1, min(n - 1, int(round(n * holdout_ratio))))
            rng = np.random.RandomState(holdout_seed)
            perm = rng.permutation(n)
            holdout_datasets = [all_ds[i] for i in perm[-n_holdout:]]
            holdout_set = set(holdout_datasets)
            df = df[~df['dataset_name'].isin(holdout_set)].copy()
            print(f"\n  Hold-out: {n_holdout} datasets excluded for evaluation (ratio={holdout_ratio}, seed={holdout_seed})")
            print(f"  Training on {n - n_holdout} datasets, {len(df)} rows")
        else:
            print("\n  Hold-out skipped: fewer than 2 datasets.")
    elif holdout_ratio > 0:
        print("\n  Hold-out skipped: no 'dataset_name' column.")

    # --- Analysis (on training set only) ---
    print_hp_analysis(df)

    # --- Per-task-type branch ---
    if per_task_type and 'task_type' in df.columns:
        task_types = df['task_type'].dropna().unique().tolist()
        results = []
        for task_type in sorted(task_types):
            df_tt = df[df['task_type'] == task_type]
            if len(df_tt) < MIN_SAMPLES_PER_TASK:
                print(f"\n  Skipping task_type '{task_type}': {len(df_tt)} samples < {MIN_SAMPLES_PER_TASK}")
                continue
            out_subdir = os.path.join(output_dir, task_type)
            r = _train_single_task(df_tt, task_type, out_subdir, top_k=top_k, n_clusters=n_clusters, cluster_method=cluster_method)
            results.append(r)
        cv_summary = {}
        for r in results:
            tt = r['task_type']
            c = r['cv_scores']
            cv_summary[tt] = {
                'n_samples': r['n_samples'],
                'n_datasets': r['n_datasets'],
                'scorer_r2': (c.get('scorer') or {}).get('r2_mean'),
                'ranker_ndcg': c.get('ranker'),
            }
        manifest = {
            'per_task': True,
            'task_types': [r['task_type'] for r in results],
            'cv_summary': cv_summary,
        }
        with open(os.path.join(output_dir, 'hp_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        if holdout_datasets:
            holdout_meta = {
                'holdout_ratio': holdout_ratio,
                'seed': holdout_seed,
                'holdout_datasets': sorted(holdout_datasets),
                'n_holdout': len(holdout_datasets),
                'n_train': df['dataset_name'].nunique() if 'dataset_name' in df.columns else 0,
            }
            with open(os.path.join(output_dir, 'holdout_datasets.json'), 'w') as f:
                json.dump(holdout_meta, f, indent=2)
            print(f"  Hold-out list saved: {output_dir}/holdout_datasets.json ({len(holdout_datasets)} datasets)")
        print(f"\n>> Per-task training complete. Manifest: {output_dir}/hp_manifest.json")
        return

    # --- Single-model path ---
    print(f"\n{'='*70}")
    print("DATA PREPARATION")
    print(f"{'='*70}")

    preparator = HPDataPreparator()
    data = preparator.prepare(df, top_k=top_k)

    X_scorer = data['X_scorer']
    y_primary = data['y_primary']
    dataset_ids = data['dataset_ids']
    metadata = data['metadata']

    print(f"\n  Scorer features:    {X_scorer.shape[1]}")
    print(f"  Predictor features: {len(preparator.feature_columns_predictor)}")
    print(f"  Primary score (normalized): mean={y_primary.mean():.5f}, std={y_primary.std():.5f}")
    raw = data['y_primary_raw']
    print(f"  Primary score (raw):        mean={raw.mean():.5f}, std={raw.std():.5f}")

    # --- Model 1: HP Config Scorer ---
    scorer = HPScorerTrainer()
    scorer_cv = scorer.train(X_scorer, y_primary, dataset_ids=dataset_ids)

    # --- Model 2: Direct HP Predictor ---
    predictor = DirectHPPredictor()
    predictor_cv = {}
    if data['X_predictor'] is not None and data['y_hp_targets']:
        predictor_cv = predictor.train(
            data['X_predictor'],
            data['y_hp_targets'],
            sample_weights=data.get('sample_weights_predictor'),
            dataset_ids=data.get('dataset_ids_predictor'),
        )
    else:
        print("\n  WARNING: Cannot train direct predictor (no best-config data).")

    # --- Model 3: HP Config Ranker ---
    ranker = HPRankerTrainer()
    ranker_cv = None
    if dataset_ids is not None:
        ranker_cv = ranker.train(X_scorer, y_primary, dataset_ids)
    else:
        print("\n  WARNING: No dataset_ids -- skipping ranker.")

    # --- Archetypes ---
    archetype_data = compute_archetypes(df, n_clusters=n_clusters, cluster_method=cluster_method)

    # --- Save everything ---
    print(f"\n{'='*70}")
    print("SAVING MODELS")
    print(f"{'='*70}")

    preparator.save(os.path.join(output_dir, 'hp_preparator.pkl'))
    scorer.save(os.path.join(output_dir, 'hp_scorer.pkl'))
    predictor.save(os.path.join(output_dir, 'hp_predictor.pkl'))
    ranker.save(os.path.join(output_dir, 'hp_ranker.pkl'))

    # Build and save the ensemble predictor
    ensemble = HPEnsemblePredictor(preparator, scorer, predictor, ranker)
    ensemble.save(output_dir)

    # Save archetype data
    if archetype_data is not None:
        with open(os.path.join(output_dir, 'hp_archetypes.pkl'), 'wb') as f:
            pickle.dump(archetype_data, f)

        archetype_json = {
            name: {
                'params': info['params'],
                'n_datasets': info['n_datasets'],
                'dataset_profile': info['dataset_profile'],
                'avg_primary_score': info['avg_primary_score'],
            }
            for name, info in archetype_data['archetypes'].items()
        }
        with open(os.path.join(output_dir, 'hp_archetypes.json'), 'w') as f:
            json.dump(archetype_json, f, indent=2)

    # Aggregate CV scores
    all_cv = {
        'scorer': scorer_cv,
        'predictor': predictor_cv,
        'ranker': ranker_cv,
    }
    with open(os.path.join(output_dir, 'hp_cv_scores.json'), 'w') as f:
        json.dump(all_cv, f, indent=2, default=str)

    # Metadata
    metadata['cv_scores'] = all_cv
    metadata['n_archetypes'] = len(archetype_data['archetypes']) if archetype_data else 0
    with open(os.path.join(output_dir, 'hp_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    if holdout_datasets:
        holdout_meta = {
            'holdout_ratio': holdout_ratio,
            'seed': holdout_seed,
            'holdout_datasets': sorted(holdout_datasets),
            'n_holdout': len(holdout_datasets),
            'n_train': df['dataset_name'].nunique() if 'dataset_name' in df.columns else 0,
        }
        with open(os.path.join(output_dir, 'holdout_datasets.json'), 'w') as f:
            json.dump(holdout_meta, f, indent=2)
        print(f"  +-- holdout_datasets.json  ({len(holdout_datasets)} datasets for evaluation)")

    # --- Summary ---
    print(f"\n  Saved to {output_dir}/")
    print(f"  +-- hp_preparator.pkl       (data preparation state)")
    print(f"  +-- hp_scorer.pkl           (config scorer, R2={scorer_cv.get('r2_mean', 0):.4f})")
    print(f"  +-- hp_predictor.pkl        (direct HP predictor, {len(predictor.models)} params)")
    print(f"  +-- hp_ranker.pkl           (lambdarank ranker)")
    print(f"  +-- hp_ensemble_meta.json   (ensemble predictor metadata)")
    print(f"  +-- hp_archetypes.pkl/.json (learned archetypes)")
    print(f"  +-- hp_cv_scores.json       (all CV metrics)")
    print(f"  +-- hp_metadata.json        (training metadata)")

    # Print inference examples
    if predictor.models:
        print(f"\n{'='*60}")
        print("INFERENCE EXAMPLES")
        print(f"{'='*60}")
        print("\n  Option 1: Direct predictor (fast, single-pass):")
        print("    from train_hp_meta_model import HPDataPreparator, DirectHPPredictor")
        print("    preparator = HPDataPreparator()")
        print("    preparator.load('hp_meta_model/hp_preparator.pkl')")
        print("    predictor = DirectHPPredictor()")
        print("    predictor.load('hp_meta_model/hp_predictor.pkl')")
        print("    # ds_meta = compute_dataset_meta(X, y)  # from HPCollector")
        print("    # X_pred = preparator.transform_predictor(pd.DataFrame([ds_meta]))")
        print("    # optimal_hp = predictor.predict_single(X_pred)")
        print()
        print("  Option 2: Ensemble predictor (best quality, uses all 3 models):")
        print("    from train_hp_meta_model import HPEnsemblePredictor")
        print("    ensemble = HPEnsemblePredictor.load('hp_meta_model')")
        print("    # ds_meta = compute_dataset_meta(X, y)")
        print("    # result = ensemble.predict(pd.DataFrame([ds_meta]),")
        print("    #                           task_type='classification')")
        print("    # best_hp = result['best_config']")

    print(f"\n>> HP meta-model training complete!")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the HP meta-model')
    parser.add_argument('--hp-db', required=True,
                        help='Path to hp_tuning_db.csv')
    parser.add_argument('--output-dir', default='./hp_meta_model',
                        help='Where to save the models')
    parser.add_argument('--n-clusters', type=int, default=6,
                        help='Number of archetype clusters')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Top-K configs per dataset for predictor training')
    parser.add_argument('--per-task-type', action='store_true',
                        help='Train one model per task_type (if column present)')
    parser.add_argument('--holdout-ratio', type=float, default=0.0,
                        help='Fraction of datasets to hold out for evaluation (e.g. 0.2 for fair eval). 0=no hold-out')
    parser.add_argument('--holdout-seed', type=int, default=42,
                        help='Random seed for hold-out split')
    parser.add_argument('--cluster-method', choices=['kmeans', 'spectral'], default='kmeans',
                        help='Archetype clustering: kmeans or spectral (e.g. spectral with n_clusters=2 from 2_compare)')
    args = parser.parse_args()

    train_hp_meta_model(args.hp_db, args.output_dir, top_k=args.top_k,
                        n_clusters=args.n_clusters, per_task_type=args.per_task_type,
                        holdout_ratio=args.holdout_ratio, holdout_seed=args.holdout_seed,
                        cluster_method=args.cluster_method)