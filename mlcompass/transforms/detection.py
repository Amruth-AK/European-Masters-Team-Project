"""
mlcompass.transforms.detection — Day-of-week / text / date column detection
==========================================================================
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Day-of-week constants
# ---------------------------------------------------------------------------

_DOW_FULL  = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
_DOW_ABBR  = {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'}
_DOW_ABBR2 = {'mo', 'tu', 'we', 'th', 'fr', 'sa', 'su'}
_DOW_ALL   = _DOW_FULL | _DOW_ABBR | _DOW_ABBR2

_DOW_TO_INT = {
    'monday': 0, 'mon': 0, 'mo': 0,
    'tuesday': 1, 'tue': 1, 'tu': 1,
    'wednesday': 2, 'wed': 2, 'we': 2,
    'thursday': 3, 'thu': 3, 'th': 3,
    'friday': 4, 'fri': 4, 'fr': 4,
    'saturday': 5, 'sat': 5, 'sa': 5,
    'sunday': 6, 'sun': 6, 'su': 6,
}


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_dow_columns(X, already_date_cols=None):
    """
    Return a set of column names that appear to contain day-of-week labels
    (Mon, Monday, etc.).
    """
    already_date_cols = set(already_date_cols or {})
    dow_cols = set()
    for col in X.columns:
        if col in already_date_cols:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        s = X[col].dropna().astype(str)
        if len(s) < 3:
            continue
        unique_lower = {v.strip().lower() for v in s.unique()}
        if not unique_lower.issubset(_DOW_ALL):
            continue
        recognised = s.str.strip().str.lower().isin(_DOW_ALL)
        if float(recognised.mean()) >= 0.80:
            dow_cols.add(col)
    return dow_cols


def detect_text_columns(X, date_cols=None, dow_cols=None):
    """
    Return a dict of col -> description for columns that look like free-form text.
    Heuristic: object dtype, avg character length > 30, unique_ratio > 10 %.
    """
    date_cols = set(date_cols or {})
    dow_cols  = set(dow_cols  or {})
    text_cols = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        if col in date_cols:
            continue
        if col in dow_cols:
            continue
        s = X[col].dropna().astype(str)
        if len(s) < 5:
            continue
        avg_len = float(s.str.len().mean())
        unique_ratio = X[col].nunique() / max(len(X[col]), 1)
        if avg_len > 30 and unique_ratio > 0.10:
            text_cols[col] = f"avg {avg_len:.0f} chars/value, {unique_ratio:.0%} unique"
    return text_cols
