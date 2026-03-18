"""
mlcompass.evaluation.metrics — Model evaluation & threshold optimization
=======================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, log_loss,
)


def metrics_at_threshold(y_true, y_proba, threshold):
    """Compute classification metrics using a custom probability threshold.

    Parameters
    ----------
    y_true  : array-like — Ground-truth binary labels (0/1).
    y_proba : array-like — Predicted probabilities for the positive class.
    threshold : float — Decision boundary (0–1).

    Returns
    -------
    dict with keys: ``threshold``, ``accuracy``, ``f1``, ``precision``,
    ``recall``.
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        'threshold': float(threshold),
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# backward-compatible alias
_metrics_at_threshold = metrics_at_threshold


def find_optimal_thresholds(y_true, y_proba, n_steps=200):
    """Sweep thresholds and return the optimal one for each metric.

    Parameters
    ----------
    y_true  : array-like — Ground-truth binary labels (0/1).
    y_proba : array-like — Predicted probabilities for the positive class.
    n_steps : int — Number of threshold steps to evaluate (default 200).

    Returns
    -------
    dict — Keys are ``'f1'``, ``'precision'``, ``'recall'``, ``'accuracy'``.
    Each value is a dict with ``'threshold'`` (float) and ``'value'`` (float)
    representing the threshold that maximises that metric and the metric
    value achieved.
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.01, 0.99, n_steps)

    best = {
        'f1':        {'threshold': 0.5, 'value': 0.0},
        'precision': {'threshold': 0.5, 'value': 0.0},
        'recall':    {'threshold': 0.5, 'value': 0.0},
        'accuracy':  {'threshold': 0.5, 'value': 0.0},
    }

    for t in thresholds:
        m = metrics_at_threshold(y_true, y_proba, t)
        for key in best:
            if m[key] > best[key]['value']:
                best[key] = {'threshold': float(t), 'value': float(m[key])}

    return best


# backward-compatible alias
_find_optimal_thresholds = find_optimal_thresholds


def evaluate_on_set(model, X, y, train_columns, n_classes, col_encoders=None,
                    threshold=None):
    """Evaluate model on a dataset. Returns metrics dict.

    threshold : float or None
        Custom probability threshold for binary classification (0–1).
        When provided and n_classes == 2, predictions are derived from
        ``y_pred_proba[:, 1] >= threshold`` instead of ``model.predict()``.
        All downstream metrics (accuracy, F1, precision, recall, confusion
        matrix) will reflect the custom threshold. Ignored for multiclass.
    """
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            if col_encoders and col in col_encoders and col_encoders[col]['encoder'] is not None:
                le = col_encoders[col]['encoder']
                known = set(le.classes_)
                fallback = le.classes_[0]
                X_enc[col] = le.transform(
                    X_enc[col].astype(str).map(lambda v: v if v in known else fallback)
                )
            else:
                le = LabelEncoder()
                le.fit(X_enc[col].astype(str))
                X_enc[col] = le.transform(X_enc[col].astype(str))
        if col_encoders and col in col_encoders and col_encoders[col]['median'] is not None:
            fill_val = col_encoders[col]['median']
        elif pd.api.types.is_numeric_dtype(X_enc[col]):
            fill_val = X_enc[col].median()
        else:
            fill_val = -999
        X_enc[col] = X_enc[col].fillna(fill_val)

    for c in train_columns:
        if c not in X_enc.columns:
            X_enc[c] = 0
    X_enc = X_enc[train_columns]

    y_pred_proba = model.predict_proba(X_enc)

    if threshold is not None and n_classes == 2:
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    else:
        y_pred = model.predict(X_enc)

    metrics = {}
    metrics['_y_pred'] = y_pred
    metrics['_y_pred_proba'] = y_pred_proba
    metrics['accuracy'] = float(accuracy_score(y, y_pred))

    try:
        if n_classes == 2:
            metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba[:, 1]))
        else:
            metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted'))
    except Exception:
        metrics['roc_auc'] = None

    try:
        avg = 'binary' if n_classes == 2 else 'weighted'
        metrics['f1'] = float(f1_score(y, y_pred, average=avg, zero_division=0))
        metrics['precision'] = float(precision_score(y, y_pred, average=avg, zero_division=0))
        metrics['recall'] = float(recall_score(y, y_pred, average=avg, zero_division=0))
    except Exception:
        metrics['f1'] = None; metrics['precision'] = None; metrics['recall'] = None

    try:
        metrics['log_loss'] = float(log_loss(y, y_pred_proba))
    except Exception:
        metrics['log_loss'] = None

    try:
        from sklearn.metrics import confusion_matrix as _cm
        metrics['confusion_matrix'] = _cm(y, y_pred).tolist()
        metrics['y_classes'] = [str(c) for c in sorted(set(y))]
    except Exception:
        metrics['confusion_matrix'] = None
        metrics['y_classes'] = None

    try:
        if n_classes == 2:
            from sklearn.metrics import roc_curve as _roc_curve
            fpr, tpr, _ = _roc_curve(y, y_pred_proba[:, 1])
            metrics['roc_data'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': metrics.get('roc_auc') or 0.0,
            }
        else:
            metrics['roc_data'] = None
    except Exception:
        metrics['roc_data'] = None

    try:
        if n_classes == 2:
            from sklearn.metrics import precision_recall_curve as _pr_curve
            from sklearn.metrics import average_precision_score as _ap_score
            _pr_prec, _pr_rec, _pr_thresh = _pr_curve(y, y_pred_proba[:, 1])
            metrics['pr_data'] = {
                'precision': _pr_prec.tolist(),
                'recall':    _pr_rec.tolist(),
                'thresholds': _pr_thresh.tolist(),
                'avg_precision': float(_ap_score(y, y_pred_proba[:, 1])),
            }
            metrics['_y_true']  = np.array(y).tolist()
            metrics['_y_proba'] = y_pred_proba[:, 1].tolist()
        else:
            metrics['pr_data'] = None
    except Exception:
        metrics['pr_data'] = None

    # Optimal thresholds per metric (binary only)
    try:
        if n_classes == 2:
            metrics['optimal_thresholds'] = find_optimal_thresholds(
                y, y_pred_proba[:, 1]
            )
        else:
            metrics['optimal_thresholds'] = None
    except Exception:
        metrics['optimal_thresholds'] = None

    return metrics


def predict_on_set(model, X, train_columns, n_classes, col_encoders=None,
                   threshold=None):
    """Generate predictions without ground-truth labels (predict-only mode).

    threshold : float or None
        Custom probability threshold for binary classification (0–1).
        When provided and n_classes == 2, predictions are derived from
        ``y_pred_proba[:, 1] >= threshold`` instead of ``model.predict()``.
        Ignored for multiclass problems.
    """
    X_enc = X.copy()
    for col in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            if col_encoders and col in col_encoders and col_encoders[col]['encoder'] is not None:
                le = col_encoders[col]['encoder']
                known = set(le.classes_)
                fallback = le.classes_[0]
                X_enc[col] = le.transform(
                    X_enc[col].astype(str).map(lambda v: v if v in known else fallback)
                )
            else:
                le_local = LabelEncoder()
                le_local.fit(X_enc[col].astype(str))
                X_enc[col] = le_local.transform(X_enc[col].astype(str))
        if col_encoders and col in col_encoders and col_encoders[col]['median'] is not None:
            fill_val = col_encoders[col]['median']
        elif pd.api.types.is_numeric_dtype(X_enc[col]):
            fill_val = X_enc[col].median()
        else:
            fill_val = -999
        X_enc[col] = X_enc[col].fillna(fill_val)

    for c in train_columns:
        if c not in X_enc.columns:
            X_enc[c] = 0
    X_enc = X_enc[train_columns]

    y_pred_proba = model.predict_proba(X_enc)

    if threshold is not None and n_classes == 2:
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    else:
        y_pred = model.predict(X_enc)

    return {
        '_y_pred': y_pred,
        '_y_pred_proba': y_pred_proba,
    }
