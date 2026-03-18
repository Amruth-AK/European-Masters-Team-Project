"""
mlcompass.evaluation.training — LightGBM model training & data preparation
==========================================================================
"""

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from mlcompass.constants import BASE_PARAMS


def prepare_data_for_model(X_train, X_val):
    """Encode categoricals for LightGBM.

    Returns:
        X_tr       – encoded training frame (copy)
        X_vl       – encoded validation frame (copy)
        col_encoders – dict mapping col name → {'encoder': LabelEncoder | None,
                                                  'median': float}
    """
    X_tr, X_vl = X_train.copy(), X_val.copy()
    for _c in X_tr.columns:
        if _c not in X_vl.columns:
            X_vl[_c] = 0.0
    extra_val = [c for c in X_vl.columns if c not in X_tr.columns]
    if extra_val:
        X_vl = X_vl.drop(columns=extra_val)
    X_vl = X_vl[X_tr.columns]
    col_encoders = {}
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]):
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_vl[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_vl[col] = le.transform(X_vl[col].astype(str))
            col_encoders[col] = {'encoder': le, 'median': None}
        else:
            col_encoders[col] = {'encoder': None, 'median': None}
        med = X_tr[col].median() if pd.api.types.is_numeric_dtype(X_tr[col]) else -999
        col_encoders[col]['median'] = float(med)
        X_tr[col] = X_tr[col].fillna(med)
        X_vl[col] = X_vl[col].fillna(med)
    return X_tr, X_vl, col_encoders


def train_lgbm_model(X_train, y_train, X_val, y_val, n_classes,
                     apply_imbalance: bool = False,
                     imbalance_strategy: str = 'none',
                     base_params: dict = None):
    """Train a LightGBM classifier with early stopping on validation set."""
    X_tr, X_vl, col_encoders = prepare_data_for_model(X_train, X_val)

    params = (base_params or BASE_PARAMS).copy()
    if apply_imbalance:
        if imbalance_strategy in ('binary', 'low') and n_classes == 2:
            params['is_unbalance'] = True
        elif imbalance_strategy in ('multiclass_moderate', 'low'):
            params['class_weight'] = 'balanced'

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model, X_tr.columns.tolist(), col_encoders
