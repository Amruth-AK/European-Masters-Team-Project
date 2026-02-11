"""
Meta-Model Trainer (train_meta_model.py)

Trains an XGBoost/LightGBM meta-model from the meta_learning_db.csv
produced by the MetaDataCollector pipeline.

Usage:
    python train_meta_model.py --meta-db ./meta_learning_output/meta_learning_db.csv --output-dir ./meta_model

The trained model is saved to output-dir and can be loaded by the
Streamlit application (auto_fe_app.py) for inference.
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')


# =============================================================================
# RECOMMENDATION SCORE COMPUTATION
# =============================================================================

# Methods where individual evaluation is STRUCTURALLY UNINFORMATIVE.
# For these, the individual baseline (cat.codes on a single column) already
# captures what the encoding does. The individual delta is ~0 by construction.
# → Use 100% full-model signal instead of 40/60 split.
INDIVIDUAL_UNINFORMATIVE_METHODS = {
    'target_encoding',      # Single-col tree already learns category→target mapping
    'frequency_encoding',   # cat.codes gives similar ordering as frequency
    'group_mean',           # Creates a new column; individual baseline doesn't have it
    'group_std',            # Same reason
    'cat_concat',           # Same reason
    'hashing_encoding',     # Replaces encoding; tree learns same splits
    'row_stats',            # Global row-level features; no single column to evaluate
}


def confidence_weight(p_value):
    """
    Map p-value to confidence weight [0, 1] via -log10 scaling.
    
        p=0.001 → 1.00  (highly confident)
        p=0.01  → 0.67
        p=0.05  → 0.43
        p=0.10  → 0.33  (marginal — still contributes)
        p=0.50  → 0.10
        p=1.00  → 0.00  (no information)
    """
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return 0.0
    p_value = max(float(p_value), 1e-10)
    return min(-np.log10(p_value) / 3.0, 1.0)


def bonferroni_multiplier(p_bonferroni):
    """
    Bonus multiplier [1.0, 1.5] for surviving Bonferroni correction.
    Signals robustness against multiple testing.
    """
    if p_bonferroni is None or (isinstance(p_bonferroni, float) and np.isnan(p_bonferroni)):
        return 1.0
    return 1.0 + 0.5 * confidence_weight(p_bonferroni)


def compute_recommendation_score(row):
    """
    Compute 0-100 recommendation score from the 6 performance indicators.
    
    Scale:
        50 = neutral (no detectable effect)
        >50 = positive (higher = more confident improvement)
        <50 = negative (lower = more confident harm)
        ~70+ = strong recommendation
        ~30- = strong anti-recommendation
    
    Method-aware weighting:
        - For most methods: 40% full-model + 60% individual
        - For methods where individual eval is structurally uninformative
          (target_encoding, frequency_encoding, group_mean, etc.): 100% full-model
    """
    def safe_float(val, default=0.0):
        if val is None:
            return default
        try:
            f = float(val)
            return default if np.isnan(f) else f
        except (TypeError, ValueError):
            return default
    
    method = str(row.get('method', ''))
    
    # Full-model signal
    delta_full = safe_float(row.get('delta_normalized'))
    p_full = row.get('p_value')
    p_full_bonf = row.get('p_value_bonferroni')
    full_signal = delta_full * confidence_weight(p_full) * bonferroni_multiplier(p_full_bonf)
    
    # Individual signal
    delta_indiv = safe_float(row.get('individual_delta_normalized'))
    p_indiv = row.get('individual_p_value')
    p_indiv_bonf = row.get('individual_p_value_bonferroni')
    indiv_signal = delta_indiv * confidence_weight(p_indiv) * bonferroni_multiplier(p_indiv_bonf)
    
    # Method-aware weighting
    if method in INDIVIDUAL_UNINFORMATIVE_METHODS:
        # Individual eval is meaningless for this method → 100% full-model
        combined = full_signal
    else:
        # Standard weighting: individual isolates the effect better
        combined = 0.4 * full_signal + 0.6 * indiv_signal
    
    # Map to [0, 100] via tanh (smooth, bounded)
    # typical combined range: [-8, 8] based on observed deltas
    normalized = np.tanh(combined / 5.0)  # [-1, 1]
    score = 50 + normalized * 50  # [0, 100]
    
    return np.clip(score, 0, 100)


# =============================================================================
# META-DB PREPARATION
# =============================================================================

class MetaDBPreparator:
    """
    Prepares the raw meta_learning_db.csv for meta-model training.
    
    Steps:
    1. Filter out non-column-level entries
    2. Compute recommendation_score for each row
    3. Select and clean feature columns
    4. Encode categorical identifiers (method name, task type)
    5. Handle missing values
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
        Returns: (X, y, metadata_dict)
        """
        df = df.copy()
        
        # Filter non-column-level entries (keep row_stats — dataset-level features
        # give the meta-model enough signal to learn when row stats help)
        df = df[~df['method'].isin(['null_intervention'])].reset_index(drop=True)
        
        # Recommendation score
        df['recommendation_score'] = df.apply(compute_recommendation_score, axis=1)
        
        # Encode method
        df['method_encoded'] = self.method_encoder.fit_transform(df['method'].astype(str))
        
        # Encode task type
        if 'task_type' in df.columns:
            df['task_type_encoded'] = self.task_type_encoder.fit_transform(df['task_type'].astype(str))
        else:
            df['task_type_encoded'] = 0
        
        # Assemble feature columns
        feature_cols = []
        for col in self.DATASET_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        for col in self.COLUMN_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        feature_cols.extend(['method_encoded', 'task_type_encoded'])
        
        # Boolean fields: CSV may store as string 'True'/'False' — convert to int
        bool_fields = ['is_interaction', 'near_ceiling_flag']
        for bf in bool_fields:
            if bf in df.columns:
                df[bf] = df[bf].map(
                    lambda v: 1 if str(v).strip().lower() in ('true', '1', '1.0') else 0
                )
                feature_cols.append(bf)
        
        for extra in ['vif', 'composite_predictive_score',
                     'pairwise_corr_ab', 'pairwise_spearman_ab',
                     'pairwise_mi_ab', 'interaction_scale_ratio',
                     'is_temporal_component', 'temporal_period']:
            if extra in df.columns:
                feature_cols.append(extra)
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df['recommendation_score']
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        self.fill_values = X.median()
        X = X.fillna(self.fill_values)
        
        metadata = {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'score_distribution': {
                'mean': float(y.mean()), 'std': float(y.std()),
                'min': float(y.min()), 'max': float(y.max()),
                'pct_above_55': float((y > 55).mean()),
                'pct_above_60': float((y > 60).mean()),
                'pct_below_45': float((y < 45).mean()),
            },
            'method_mapping': dict(zip(
                self.method_encoder.classes_.tolist(),
                self.method_encoder.transform(self.method_encoder.classes_).tolist()
            )),
            'methods_in_db': df['method'].value_counts().to_dict(),
            'datasets_in_db': int(df['dataset_name'].nunique()) if 'dataset_name' in df.columns else -1,
        }
        
        return X, y, metadata
    
    def save(self, path: str):
        state = {
            'method_encoder_classes': self.method_encoder.classes_.tolist(),
            'task_type_encoder_classes': self.task_type_encoder.classes_.tolist(),
            'feature_columns': self.feature_columns,
            'fill_values': self.fill_values.to_dict(),
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


# =============================================================================
# META-MODEL TRAINER
# =============================================================================

class MetaModelTrainer:
    """Trains XGBoost (or LightGBM fallback) to predict recommendation scores."""
    
    def __init__(self, model_type='auto'):
        if model_type == 'auto':
            self.model_type = 'xgboost' if HAS_XGB else 'lightgbm'
        else:
            self.model_type = model_type
        self.model = None
        self.cv_scores = None
    
    def train(self, X, y, n_folds=5):
        print(f"\nTraining meta-model ({self.model_type})...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_mae, cv_rmse = [], []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._create_model()
            if self.model_type == 'lightgbm':
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                         callbacks=[lgb.early_stopping(30, verbose=False)], verbose=False)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            
            preds = model.predict(X_vl)
            cv_mae.append(mean_absolute_error(y_vl, preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y_vl, preds)))
        
        self.cv_scores = {
            'mae_mean': float(np.mean(cv_mae)), 'mae_std': float(np.std(cv_mae)),
            'rmse_mean': float(np.mean(cv_rmse)), 'rmse_std': float(np.std(cv_rmse)),
        }
        print(f"  CV MAE:  {self.cv_scores['mae_mean']:.3f} ± {self.cv_scores['mae_std']:.3f}")
        print(f"  CV RMSE: {self.cv_scores['rmse_mean']:.3f} ± {self.cv_scores['rmse_std']:.3f}")
        
        # Final model on all data
        self.model = self._create_model()
        self.model.fit(X, y, verbose=False)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            imp = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print(f"\n  Top 10 features:")
            for feat, val in imp.head(10).items():
                print(f"    {feat}: {val:.0f}")
        
        return self.cv_scores
    
    def _create_model(self):
        if self.model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0, n_jobs=-1,
                
            )
        else:
            return lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=-1, n_jobs=-1,
            )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'model_type': self.model_type,
                        'cv_scores': self.cv_scores}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.model_type = state['model_type']
        self.cv_scores = state.get('cv_scores')


# =============================================================================
# MAIN
# =============================================================================

def train_meta_model(meta_db_path: str, output_dir: str = './meta_model'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("META-MODEL TRAINING")
    print("=" * 70)
    
    # Load
    print(f"\nLoading: {meta_db_path}")
    df = pd.read_csv(meta_db_path)
    print(f"  {len(df)} rows, {df['dataset_name'].nunique() if 'dataset_name' in df.columns else '?'} datasets")
    
    # Prepare
    preparator = MetaDBPreparator()
    X, y, metadata = preparator.prepare(df)
    
    print(f"\nScore distribution:")
    print(f"  Mean: {y.mean():.1f}  Std: {y.std():.1f}")
    print(f"  >55 (positive): {(y > 55).mean()*100:.1f}%")
    print(f"  >60 (strong):   {(y > 60).mean()*100:.1f}%")
    print(f"  <45 (negative): {(y < 45).mean()*100:.1f}%")
    
    # Train
    trainer = MetaModelTrainer(model_type='auto')
    cv_scores = trainer.train(X, y)
    
    # Save everything
    preparator.save(os.path.join(output_dir, 'preparator.pkl'))
    trainer.save(os.path.join(output_dir, 'meta_model.pkl'))
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'cv_scores.json'), 'w') as f:
        json.dump(cv_scores, f, indent=2)
    
    # Save the score distribution as a separate analysis file
    score_analysis = df.copy()
    score_analysis['recommendation_score'] = score_analysis.apply(compute_recommendation_score, axis=1)
    score_summary = score_analysis.groupby('method')['recommendation_score'].agg(['mean', 'std', 'count']).round(2)
    score_summary = score_summary.sort_values('mean', ascending=False)
    score_summary.to_csv(os.path.join(output_dir, 'method_score_summary.csv'))
    print(f"\n  Method score summary:")
    print(score_summary.to_string())
    
    print(f"\n✓ Saved to {output_dir}/")
    print(f"  - preparator.pkl")
    print(f"  - meta_model.pkl")
    print(f"  - metadata.json")
    print(f"  - cv_scores.json")
    print(f"  - method_score_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the meta-model for Auto-FE')
    parser.add_argument('--meta-db', required=True, help='Path to meta_learning_db.csv')
    parser.add_argument('--output-dir', default='./meta_model', help='Where to save the model')
    args = parser.parse_args()
    
    train_meta_model(args.meta_db, args.output_dir)