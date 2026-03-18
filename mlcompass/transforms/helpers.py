"""
mlcompass.transforms.helpers — Feature name sanitization, date/text extraction helpers
=====================================================================================
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Target / feature-name utilities
# ---------------------------------------------------------------------------

def ensure_numeric_target(y):
    if pd.api.types.is_numeric_dtype(y):
        return y
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)


def sanitize_feature_names(df):
    df.columns = [re.sub(r'[\[\]\{\}":,]', '_', str(c)) for c in df.columns]
    if df.columns.duplicated().any():
        cols = list(df.columns)
        seen = {}
        for i, c in enumerate(cols):
            if c in seen:
                seen[c] += 1
                cols[i] = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
        df.columns = cols
    return df


# ---------------------------------------------------------------------------
# Date feature extraction helpers
# ---------------------------------------------------------------------------

def _to_datetime_safe(series):
    """
    Parse a string/object series to a proper tz-naive datetime64[ns] Series.

    Strategy:
      1. Try format='mixed' directly (fast, handles the common case).
      2. If that raises due to mixed timezones, strip trailing tz markers
         (Z, +HH:MM, -HH:MM) from each value and retry format='mixed'.
      3. Final fallback: utc=True + strip timezone.
    Always returns a tz-naive datetime64 Series so .dt accessors work everywhere.
    """
    try:
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
    except (ValueError, TypeError):
        try:
            stripped = series.astype(str).str.replace(
                r'Z$|[+-]\d{2}:\d{2}$|[+-]\d{4}$', '', regex=True
            ).str.strip()
            parsed = pd.to_datetime(stripped, errors='coerce', format='mixed')
        except Exception:
            try:
                parsed = pd.to_datetime(series, errors='coerce', utc=True)
            except Exception:
                parsed = pd.to_datetime(series, errors='coerce')

    if getattr(parsed.dtype, 'tz', None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _apply_date_features(series, min_date=None, col_type='datetime', selected_features=None):
    """
    Parse a date-like series and return a DataFrame of extracted base features.
    min_date: pd.Timestamp — reference point for days_since_min (fit from training).
    col_type: 'date' | 'time' | 'datetime' — controls which components are extracted.
    selected_features: list of feature names to include (None = all available).
    """
    parsed = _to_datetime_safe(series)
    feats = pd.DataFrame(index=series.index)

    def _include(name):
        return selected_features is None or name in selected_features

    if col_type != 'time':
        if _include('year'):
            feats['year']       = parsed.dt.year.fillna(0).astype(int)
        if _include('month'):
            feats['month']      = parsed.dt.month.fillna(1).astype(int)
        if _include('day'):
            feats['day']        = parsed.dt.day.fillna(1).astype(int)
        if _include('dayofweek'):
            feats['dayofweek']  = parsed.dt.dayofweek.fillna(0).astype(int)
        if _include('is_weekend'):
            feats['is_weekend'] = (parsed.dt.dayofweek >= 5).astype(int)
        if _include('quarter'):
            feats['quarter']    = parsed.dt.quarter.fillna(1).astype(int)
        if _include('weekofyear'):
            try:
                feats['weekofyear'] = parsed.dt.isocalendar().week.fillna(1).astype(int).values
            except Exception:
                feats['weekofyear'] = parsed.dt.week.fillna(1).astype(int)
        if _include('days_since_min'):
            if min_date is None:
                min_date = parsed.min()
            feats['days_since_min'] = (parsed - min_date).dt.days.fillna(0).astype(float)
        elif min_date is None:
            min_date = parsed.min()

    if col_type in ('time', 'datetime') and _include('hour'):
        feats['hour'] = parsed.dt.hour.fillna(0).astype(int)

    if min_date is None:
        min_date = parsed.min()
    return feats, min_date


# Cyclic encoding helpers — per component

_CYCLICAL_PERIODS = {
    'month': ('month',      12),
    'dow':   ('dayofweek',   7),
    'dom':   ('day',        31),
    'hour':  ('hour',       24),
}

def _apply_date_cyclical(df, col_prefix, component=None):
    """
    Add cyclic sin/cos encoding for one (or all legacy) date components.
    """
    feats = pd.DataFrame(index=df.index)

    if component is None:
        targets = ['month', 'dow']
    else:
        targets = [component]

    for comp in targets:
        src_suffix, period = _CYCLICAL_PERIODS[comp]
        src_col = f"{col_prefix}{src_suffix}"
        if src_col not in df.columns:
            continue
        vals = df[src_col]
        feats[f"{col_prefix}{comp}_sin"] = np.sin(2 * np.pi * vals / period)
        feats[f"{col_prefix}{comp}_cos"] = np.cos(2 * np.pi * vals / period)

    return feats


# ---------------------------------------------------------------------------
# Text feature extraction helpers
# ---------------------------------------------------------------------------

def _apply_text_stats(series, fields=None):
    """Surface-level text statistics — no fitting required."""
    _ALL_FIELDS = ['word_count', 'char_count', 'avg_word_len',
                   'uppercase_ratio', 'digit_ratio', 'punct_ratio']
    if fields is None:
        fields = _ALL_FIELDS
    fields_set = set(fields)

    s = series.fillna('').astype(str)
    feats = pd.DataFrame(index=series.index)
    words = s.str.split()
    if 'word_count' in fields_set:
        feats['word_count']      = words.apply(len)
    if 'char_count' in fields_set:
        feats['char_count']      = s.str.len()
    if 'avg_word_len' in fields_set:
        feats['avg_word_len']    = words.apply(
            lambda ws: float(np.mean([len(w) for w in ws])) if ws else 0.0
        )
    if 'uppercase_ratio' in fields_set:
        feats['uppercase_ratio'] = s.apply(
            lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
        )
    if 'digit_ratio' in fields_set:
        feats['digit_ratio']     = s.apply(
            lambda t: sum(1 for c in t if c.isdigit()) / max(len(t), 1)
        )
    if 'punct_ratio' in fields_set:
        feats['punct_ratio']     = s.apply(
            lambda t: sum(1 for c in t if not c.isalnum() and not c.isspace()) / max(len(t), 1)
        )
    return feats
