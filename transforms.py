"""
transforms.py — Feature Transform Helpers & Apply Logic
========================================================
Contains:
  - Day-of-week detection constants and helpers
  - Date / text feature extraction helpers
  - fit_and_apply_suggestions  (training data)
  - apply_fitted_to_test       (test / inference data)

Imported by recommend_app.py — no imports back to recommend_app.
"""

import re
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


# Needed for ensure_numeric_target, called inside fit_and_apply_suggestions
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
# Day-of-week categorical column detection
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


def detect_dow_columns(X, already_date_cols=None):
    """
    Return a set of column names that appear to contain day-of-week labels
    (Mon, Monday, etc.).  Numeric columns and already-detected date/time columns
    are excluded.  Requires that all unique non-null values are recognised
    weekday tokens (i.e. the column contains *only* day names).
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
    Already-identified date columns and day-of-week columns are excluded.
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


# --------------------------------------------------------------------------
# Date feature extraction helpers
# --------------------------------------------------------------------------

def _to_datetime_safe(series):
    """
    Parse a string/object series to a proper tz-naive datetime64[ns] Series.

    Problem: pd.to_datetime(..., format='mixed') raises ValueError when the
    input contains both timezone-aware strings (e.g. '2023-01-01T08:00:00Z')
    and timezone-naive strings (e.g. '01 Jan 2023 08:00').  This crashes
    _apply_date_features with "Mixed timezones detected."

    Strategy:
      1. Try format='mixed' directly (fast, handles the common case).
      2. If that raises due to mixed timezones, strip trailing tz markers
         (Z, +HH:MM, -HH:MM) from each value and retry format='mixed'.
         This preserves naive values that would otherwise be coerced to NaN
         by the utc=True fallback.
      3. Final fallback: utc=True + strip timezone.
    Always returns a tz-naive datetime64 Series so .dt accessors work everywhere.
    """
    try:
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
    except (ValueError, TypeError):
        # Mixed tz-aware / tz-naive — strip timezone markers and retry
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

    # Strip timezone if present so .dt.year etc. always work
    if getattr(parsed.dtype, 'tz', None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _apply_date_features(series, min_date=None, col_type='datetime', selected_features=None):
    """
    Parse a date-like series and return a DataFrame of extracted base features.
    min_date: pd.Timestamp — reference point for days_since_min (fit from training).
    col_type: 'date' | 'time' | 'datetime' — controls which components are extracted.
      - 'time'     → hour only (date components like year/month/dow are meaningless
                     because pd.to_datetime injects today's date for every row)
      - 'date'     → year, month, day, dayofweek, quarter, is_weekend, days_since_min
      - 'datetime' → all of the above plus hour
    selected_features: list of feature names to include (None = all available).
      Allowed values: 'year', 'month', 'day', 'dayofweek', 'is_weekend',
                      'quarter', 'weekofyear', 'hour', 'days_since_min'
    Cyclic sin/cos features are NOT included here — use _apply_date_cyclical for those.
    """
    parsed = _to_datetime_safe(series)
    feats = pd.DataFrame(index=series.index)

    # Helper: should we include a feature?
    def _include(name):
        return selected_features is None or name in selected_features

    if col_type != 'time':
        # Date components are only meaningful when the column carries real date info.
        # For time-only columns (e.g. "08:12"), pd.to_datetime injects today's date
        # into every row, making year/month/day/dayofweek identical across all rows.
        if _include('year'):
            feats['year']       = parsed.dt.year.fillna(0).astype(int)
        if _include('month'):
            feats['month']      = parsed.dt.month.fillna(1).astype(int)
        if _include('day'):
            feats['day']        = parsed.dt.day.fillna(1).astype(int)
        if _include('dayofweek'):
            feats['dayofweek']  = parsed.dt.dayofweek.fillna(0).astype(int)   # 0=Mon…6=Sun
        if _include('is_weekend'):
            feats['is_weekend'] = (parsed.dt.dayofweek >= 5).astype(int)
        if _include('quarter'):
            feats['quarter']    = parsed.dt.quarter.fillna(1).astype(int)
        if _include('weekofyear'):
            try:
                feats['weekofyear'] = parsed.dt.isocalendar().week.fillna(1).astype(int).values
            except Exception:
                feats['weekofyear'] = parsed.dt.week.fillna(1).astype(int)
        # Days since earliest training date (meaningless for time-only columns)
        if _include('days_since_min'):
            if min_date is None:
                min_date = parsed.min()
            feats['days_since_min'] = (parsed - min_date).dt.days.fillna(0).astype(float)
        elif min_date is None:
            min_date = parsed.min()

    # Hour — only for columns that carry a time component
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

    Parameters
    ----------
    df         : DataFrame containing already-extracted date columns.
    col_prefix : e.g. 'purchase_date_' — expects columns like {prefix}month, {prefix}dayofweek, …
    component  : one of 'month' | 'dow' | 'dom' | 'hour', or None (legacy: month + dow).

    Returns a DataFrame of new cyclic columns (empty if prerequisites are missing).
    """
    feats = pd.DataFrame(index=df.index)

    if component is None:
        # Legacy all-in-one: month + dow only
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


# --------------------------------------------------------------------------
# Text feature extraction helpers
# --------------------------------------------------------------------------

def _apply_text_stats(series, fields=None):
    """Surface-level text statistics — no fitting required.
    
    fields: list of field names to compute. If None, all fields are computed.
    """
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


# =============================================================================
# TRANSFORM FIT / APPLY PIPELINE
# =============================================================================

def fit_and_apply_suggestions(X_train, y_train, suggestions):
    """
    Apply selected suggestions to training data.
    Returns: (X_enhanced, fitted_params_list)
    fitted_params_list stores everything needed to replay on test data.
    """
    X_enh = X_train.copy()
    y_num = ensure_numeric_target(y_train)
    # Normalise index to 0..n-1 so that any pd.concat reset_index calls inside
    # transforms (e.g. onehot_encoding) never leave X_enh with a different index
    # than y_num, which would silently NaN-fill target-encoding OOF columns.
    X_enh = X_enh.reset_index(drop=True)
    y_num = y_num.reset_index(drop=True)
    fitted = []

    # Ensure missing_indicator always runs before impute_median for the same
    # column.  If impute_median runs first it fills all NaNs, leaving
    # missing_indicator with nothing to flag (all-zeros column).
    def _suggestion_order_key(s):
        if s['method'] == 'missing_indicator':
            return 0
        if s['method'] == 'impute_median':
            return 1
        return 2

    suggestions = sorted(suggestions, key=_suggestion_order_key)

    for sug in suggestions:
        method = sug['method']
        col = sug['column']
        col_b = sug.get('column_b')
        params = {'method': method, 'type': sug['type'], 'column': col, 'column_b': col_b}

        try:
            if sug['type'] == 'numerical':
                if method == 'log_transform':
                    fill = float(X_train[col].median())
                    temp = X_train[col].fillna(fill)
                    offset = float(abs(temp.min()) + 1) if temp.min() <= 0 else 0.0
                    params.update({'fill': fill, 'offset': offset})
                    # Warn when the shift swamps the column's natural variance.
                    # A huge offset compresses almost all variation into a tiny
                    # log range, making the transform effectively useless.
                    col_std = float(X_train[col].std())
                    if offset > 3 * col_std and col_std > 0:
                        st.warning(
                            f"⚠️ `log_transform` on **{col}**: the required shift "
                            f"({offset:.2g}) is >3× the column std ({col_std:.2g}). "
                            f"The transform will compress most variance — consider "
                            f"skipping it or using quantile binning instead."
                        )
                    X_enh[col] = np.log1p(X_enh[col].fillna(fill) + offset)

                elif method == 'sqrt_transform':
                    fill = float(X_train[col].median())
                    temp = X_train[col].fillna(fill)
                    offset = float(abs(temp.min()) + 1) if temp.min() < 0 else 0.0
                    params.update({'fill': fill, 'offset': offset})
                    X_enh[col] = np.sqrt(X_enh[col].fillna(fill) + offset)

                elif method == 'polynomial_square':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_sq'] = X_enh[col].fillna(med) ** 2
                    params['new_cols'] = [f'{col}_sq']

                elif method == 'polynomial_cube':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_cube'] = X_enh[col].fillna(med) ** 3
                    params['new_cols'] = [f'{col}_cube']

                elif method == 'reciprocal_transform':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[f'{col}_recip'] = 1.0 / (X_enh[col].fillna(med).abs() + 1e-5)
                    params['new_cols'] = [f'{col}_recip']

                elif method == 'quantile_binning':
                    _, bin_edges = pd.qcut(X_train[col].dropna(), q=5, retbins=True, duplicates='drop')
                    params['bin_edges'] = bin_edges.tolist()
                    clipped = X_enh[col].clip(bin_edges[0], bin_edges[-1])
                    X_enh[col] = pd.cut(clipped, bins=bin_edges, labels=False, include_lowest=True)

                elif method == 'impute_median':
                    med = float(X_train[col].median())
                    params['median'] = med
                    X_enh[col] = X_enh[col].fillna(med)

                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)
                    params['new_cols'] = [f'{col}_is_na']

            elif sug['type'] == 'categorical':
                if method == 'frequency_encoding':
                    freq = X_train[col].astype(str).value_counts(normalize=True).to_dict()
                    params['freq_map'] = freq
                    X_enh[col] = X_enh[col].astype(str).map(freq).fillna(0).astype(float)

                elif method == 'target_encoding':
                    str_col = X_enh[col].astype(str)   # use X_enh (reset index) not X_train
                    gm = float(y_num.mean())
                    # Full-data smooth map — stored for use at test time only.
                    agg = y_num.groupby(str_col).agg(['count', 'mean'])
                    smooth = ((agg['count'] * agg['mean'] + 10 * gm) / (agg['count'] + 10)).to_dict()
                    params.update({'smooth_map': smooth, 'global_mean': gm})
                    # 5-fold OOF encoding for training rows to prevent in-sample
                    # leakage: each row is encoded using statistics from the other
                    # folds only, then the full-data map is used on test data.
                    from sklearn.model_selection import KFold as _KFold
                    oof_encoded = pd.Series(gm, index=X_enh.index, dtype=float)
                    _kf = _KFold(n_splits=5, shuffle=True, random_state=42)
                    for _tr_idx, _val_idx in _kf.split(X_enh):
                        _fold_str = str_col.iloc[_tr_idx]
                        _fold_y   = y_num.iloc[_tr_idx]
                        _fold_agg = _fold_y.groupby(_fold_str).agg(['count', 'mean'])
                        _fold_map = (
                            (_fold_agg['count'] * _fold_agg['mean'] + 10 * gm)
                            / (_fold_agg['count'] + 10)
                        ).to_dict()
                        _val_str = str_col.iloc[_val_idx]
                        oof_encoded.iloc[_val_idx] = _val_str.map(_fold_map).fillna(gm).values
                    X_enh[col] = oof_encoded

                elif method == 'onehot_encoding':
                    train_dummies = pd.get_dummies(X_train[col].astype(str), prefix=col, drop_first=True)
                    params['dummy_columns'] = train_dummies.columns.tolist()
                    enh_dummies = pd.get_dummies(X_enh[col].astype(str), prefix=col, drop_first=True)
                    enh_dummies = enh_dummies.reindex(columns=train_dummies.columns, fill_value=0)
                    X_enh = X_enh.drop(columns=[col])
                    X_enh = pd.concat([X_enh.reset_index(drop=True), enh_dummies.reset_index(drop=True)], axis=1)

                elif method == 'hashing_encoding':
                    X_enh[col] = X_enh[col].astype(str).apply(
                        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 32
                    )

                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)
                    params['new_cols'] = [f'{col}_is_na']

            elif sug['type'] == 'interaction':
                a, b = col, col_b
                if method == 'product_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_x_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) * X_enh[b].fillna(mb)
                    params['new_cols'] = [nc]

                elif method == 'division_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_div_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) / (X_enh[b].fillna(mb).abs() + 1e-5)
                    params['new_cols'] = [nc]

                elif method == 'addition_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_plus_{b}'
                    X_enh[nc] = X_enh[a].fillna(ma) + X_enh[b].fillna(mb)
                    params['new_cols'] = [nc]

                elif method == 'abs_diff_interaction':
                    ma, mb = float(X_train[a].median()), float(X_train[b].median())
                    params.update({'med_a': ma, 'med_b': mb})
                    nc = f'{a}_absdiff_{b}'
                    X_enh[nc] = (X_enh[a].fillna(ma) - X_enh[b].fillna(mb)).abs()
                    params['new_cols'] = [nc]

                elif method == 'group_mean':
                    a_num = pd.api.types.is_numeric_dtype(X_train[a])
                    cat_c, num_c = (b, a) if a_num else (a, b)
                    grp = X_train[num_c].groupby(X_train[cat_c].astype(str)).mean().to_dict()
                    fv = float(X_train[num_c].mean())
                    params.update({'grp_map': grp, 'fill_val': fv, 'cat_col': cat_c, 'num_col': num_c})
                    nc = f'grpmean_{num_c}_by_{cat_c}'
                    X_enh[nc] = X_enh[cat_c].astype(str).map(grp).fillna(fv).astype(float)
                    params['new_cols'] = [nc]

                elif method == 'group_std':
                    a_num = pd.api.types.is_numeric_dtype(X_train[a])
                    cat_c, num_c = (b, a) if a_num else (a, b)
                    grp = X_train[num_c].groupby(X_train[cat_c].astype(str)).std().to_dict()
                    # Use mean of per-group stds as fallback (better than global std)
                    fv = float(np.nanmean(list(grp.values()))) if grp else 0.0
                    params.update({'grp_map': grp, 'fill_val': fv, 'cat_col': cat_c, 'num_col': num_c})
                    nc = f'grpstd_{num_c}_by_{cat_c}'
                    X_enh[nc] = X_enh[cat_c].astype(str).map(grp).fillna(fv).astype(float)
                    params['new_cols'] = [nc]

                elif method == 'cat_concat':
                    nc = f'{a}_concat_{b}'
                    combined = X_enh[a].astype(str) + '_' + X_enh[b].astype(str)
                    le = LabelEncoder(); le.fit(combined)
                    params['le_classes'] = le.classes_.tolist()
                    X_enh[nc] = le.transform(combined)
                    params['new_cols'] = [nc]

            elif sug['type'] == 'row':
                numeric_cols_row = [c for c in X_train.columns
                                    if pd.api.types.is_numeric_dtype(X_train[c])]
                X_num = X_enh[numeric_cols_row].copy()
                # Fill NaN with column medians (computed from training set)
                col_medians = X_train[numeric_cols_row].median().to_dict()
                params['col_medians'] = col_medians
                X_num_filled = X_num.apply(lambda s: s.fillna(col_medians.get(s.name, 0)))

                if method == 'row_numeric_stats':
                    # Respect per-stat selection from UI; default to all stats.
                    _ALL_ROW_STATS = ['row_mean', 'row_median', 'row_sum',
                                      'row_std', 'row_min', 'row_max', 'row_range']
                    _selected_stats = sug.get('selected_row_stats') or _ALL_ROW_STATS
                    # Guard: ensure at least one stat is included
                    if not _selected_stats:
                        _selected_stats = _ALL_ROW_STATS

                    _stat_fns = {
                        'row_mean':   lambda df: df.mean(axis=1),
                        'row_median': lambda df: df.median(axis=1),
                        'row_sum':    lambda df: df.sum(axis=1),
                        'row_std':    lambda df: df.std(axis=1).fillna(0),
                        'row_min':    lambda df: df.min(axis=1),
                        'row_max':    lambda df: df.max(axis=1),
                    }
                    _new_stat_cols = []
                    for _stat in _selected_stats:
                        if _stat == 'row_range':
                            # range depends on min and max
                            _min = (X_enh['row_min'] if 'row_min' in X_enh.columns
                                    else X_num_filled.min(axis=1))
                            _max = (X_enh['row_max'] if 'row_max' in X_enh.columns
                                    else X_num_filled.max(axis=1))
                            X_enh['row_range'] = _max - _min
                            _new_stat_cols.append('row_range')
                        elif _stat in _stat_fns:
                            X_enh[_stat] = _stat_fns[_stat](X_num_filled)
                            _new_stat_cols.append(_stat)
                    params['new_cols'] = _new_stat_cols
                    params['selected_row_stats'] = _selected_stats

                elif method == 'row_zero_stats':
                    zero_mask = (X_num_filled == 0)
                    X_enh['row_zero_count']      = zero_mask.sum(axis=1)
                    X_enh['row_zero_percentage'] = zero_mask.mean(axis=1)
                    params['new_cols'] = ['row_zero_count', 'row_zero_percentage']

                elif method == 'row_missing_stats':
                    _rm_cols = [c for c in X_train.columns if c in X_enh.columns]
                    miss_mask = X_enh[_rm_cols].isnull()
                    X_enh['row_missing_count']      = miss_mask.sum(axis=1)
                    X_enh['row_missing_percentage'] = miss_mask.mean(axis=1)
                    params['new_cols'] = ['row_missing_count', 'row_missing_percentage']
                    params['all_cols'] = X_train.columns.tolist()

            elif sug['type'] == 'date':
                if method == 'date_features':
                    _sel_feats = sug.get('selected_date_features', None)
                    date_feats, min_date = _apply_date_features(
                        X_enh[col], col_type=sug.get('col_type', 'datetime'),
                        selected_features=_sel_feats)
                    params['min_date'] = min_date
                    params['col_type'] = sug.get('col_type', 'datetime')
                    params['selected_date_features'] = _sel_feats
                    prefix = f'{col}_'
                    date_feats.columns = [prefix + c for c in date_feats.columns]
                    params['new_cols'] = date_feats.columns.tolist()
                    params['col_prefix'] = prefix
                    X_enh = pd.concat([X_enh.reset_index(drop=True),
                                       date_feats.reset_index(drop=True)], axis=1)
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'date_cyclical':
                    prefix = f'{col}_'
                    params['col_prefix'] = prefix
                    cyclic_feats = _apply_date_cyclical(X_enh, prefix, component=None)
                    if not cyclic_feats.empty:
                        params['new_cols'] = cyclic_feats.columns.tolist()
                        X_enh = pd.concat([X_enh.reset_index(drop=True),
                                           cyclic_feats.reset_index(drop=True)], axis=1)

                elif method in ('date_cyclical_month', 'date_cyclical_dow',
                                'date_cyclical_dom',  'date_cyclical_hour'):
                    prefix = f'{col}_'
                    component = method.replace('date_cyclical_', '')
                    params['col_prefix'] = prefix
                    params['component']  = component
                    cyclic_feats = _apply_date_cyclical(X_enh, prefix, component=component)
                    if not cyclic_feats.empty:
                        params['new_cols'] = cyclic_feats.columns.tolist()
                        X_enh = pd.concat([X_enh.reset_index(drop=True),
                                           cyclic_feats.reset_index(drop=True)], axis=1)

                elif method == 'dow_ordinal':
                    # Map weekday name → 0-6, drop the original string column
                    mapped = X_enh[col].astype(str).str.strip().str.lower().map(_DOW_TO_INT)
                    X_enh[f'{col}_dow'] = mapped.fillna(-1).astype(int)
                    params['new_cols'] = [f'{col}_dow']
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'dow_cyclical':
                    # sin/cos of the ordinal column produced by dow_ordinal
                    ordinal_col = f'{col}_dow'
                    if ordinal_col not in X_enh.columns:
                        # Compute on the fly if dow_ordinal wasn't applied first
                        mapped = X_enh[col].astype(str).str.strip().str.lower().map(_DOW_TO_INT)
                        vals = mapped.fillna(0).astype(float)
                        drop_orig = True
                    else:
                        vals = X_enh[ordinal_col].astype(float)
                        drop_orig = False
                    X_enh[f'{col}_dow_sin'] = np.sin(2 * np.pi * vals / 7)
                    X_enh[f'{col}_dow_cos'] = np.cos(2 * np.pi * vals / 7)
                    params['new_cols'] = [f'{col}_dow_sin', f'{col}_dow_cos']
                    if drop_orig and col in X_enh.columns:
                        X_enh = X_enh.drop(columns=[col])

            elif sug['type'] == 'text':
                if method == 'text_stats':
                    # Use only the sub-fields the user selected (stored in suggestion)
                    _text_fields = sug.get('text_stat_fields', None)
                    params['text_stat_fields'] = _text_fields
                    text_feats = _apply_text_stats(X_enh[col], fields=_text_fields)
                    prefix = f'{col}_'
                    text_feats.columns = [prefix + c for c in text_feats.columns]
                    params['new_cols'] = text_feats.columns.tolist()
                    X_enh = pd.concat([X_enh.reset_index(drop=True),
                                       text_feats.reset_index(drop=True)], axis=1)
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'text_tfidf':
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    tfidf = TfidfVectorizer(max_features=20, strip_accents='unicode',
                                           analyzer='word', stop_words='english')
                    # col may have been dropped by a prior text_stats transform; fall back to X_train
                    text_source = X_enh[col] if col in X_enh.columns else X_train[col]
                    corpus = text_source.fillna('').astype(str).tolist()
                    tfidf_mat = tfidf.fit_transform(corpus).toarray()
                    feat_names = [f'{col}_tfidf_{fn}' for fn in tfidf.get_feature_names_out()]
                    tfidf_df = pd.DataFrame(tfidf_mat, columns=feat_names,
                                            index=X_enh.index)
                    params['tfidf_vectorizer'] = tfidf
                    params['new_cols'] = feat_names
                    X_enh = pd.concat([X_enh.reset_index(drop=True),
                                       tfidf_df.reset_index(drop=True)], axis=1)
                    # col may already be gone if text_stats ran first — guard the drop
                    if col in X_enh.columns:
                        X_enh = X_enh.drop(columns=[col])

            fitted.append(params)

        except Exception as e:
            # For transforms that are supposed to DROP the original column and replace
            # it with new features (date_features, text_stats, text_tfidf, dow_ordinal),
            # a failure leaves the raw column in X_enh.  That column then goes through
            # prepare_data_for_model which LabelEncodes it — producing a useless
            # high-cardinality integer column (e.g. 120 unique timestamps → 120 labels).
            # Drop the column proactively so it can't silently corrupt the model.
            _DROPS_SOURCE = {
                'date_features', 'text_stats', 'text_tfidf',
                'dow_ordinal', 'dow_cyclical',
            }
            if method in _DROPS_SOURCE and col in X_enh.columns:
                X_enh = X_enh.drop(columns=[col])
                st.warning(
                    f"⚠️ `{method}` on **{col}** failed ({e}) — "
                    f"column dropped to prevent useless label-encoding of raw strings."
                )
            else:
                st.warning(f"Failed to apply {method} on {col}: {e}")
            continue

    # Defragment the DataFrame after column-by-column additions to avoid
    # PerformanceWarning and improve downstream operation speed.
    X_enh = X_enh.copy()
    return X_enh, fitted


def apply_fitted_to_test(X_test, fitted_params_list):
    """Apply pre-fitted transforms to test data."""
    X_enh = X_test.copy()

    for params in fitted_params_list:
        method = params['method']
        col = params['column']
        col_b = params.get('column_b')

        try:
            if params['type'] == 'numerical':
                if method == 'log_transform':
                    X_enh[col] = np.log1p(X_enh[col].fillna(params['fill']) + params['offset'])
                elif method == 'sqrt_transform':
                    X_enh[col] = np.sqrt(X_enh[col].fillna(params['fill']) + params['offset'])
                elif method == 'polynomial_square':
                    X_enh[f'{col}_sq'] = X_enh[col].fillna(params['median']) ** 2
                elif method == 'polynomial_cube':
                    X_enh[f'{col}_cube'] = X_enh[col].fillna(params['median']) ** 3
                elif method == 'reciprocal_transform':
                    X_enh[f'{col}_recip'] = 1.0 / (X_enh[col].fillna(params['median']).abs() + 1e-5)
                elif method == 'quantile_binning':
                    _edges = params['bin_edges']
                    clipped = X_enh[col].clip(_edges[0], _edges[-1])
                    X_enh[col] = pd.cut(clipped, bins=_edges,
                                         labels=False, include_lowest=True)
                elif method == 'impute_median':
                    X_enh[col] = X_enh[col].fillna(params['median'])
                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif params['type'] == 'categorical':
                if method == 'frequency_encoding':
                    X_enh[col] = X_enh[col].astype(str).map(params['freq_map']).fillna(0).astype(float)
                elif method == 'target_encoding':
                    X_enh[col] = X_enh[col].astype(str).map(params['smooth_map']).fillna(params['global_mean']).astype(float)
                elif method == 'onehot_encoding':
                    dummies = pd.get_dummies(X_enh[col].astype(str), prefix=col, drop_first=True)
                    dummies = dummies.reindex(columns=params['dummy_columns'], fill_value=0)
                    X_enh = X_enh.drop(columns=[col])
                    X_enh = pd.concat([X_enh.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
                elif method == 'hashing_encoding':
                    X_enh[col] = X_enh[col].astype(str).apply(
                        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 32
                    )
                elif method == 'missing_indicator':
                    X_enh[f'{col}_is_na'] = X_enh[col].isnull().astype(int)

            elif params['type'] == 'interaction':
                a, b = col, col_b
                if method == 'product_interaction':
                    X_enh[f'{a}_x_{b}'] = X_enh[a].fillna(params['med_a']) * X_enh[b].fillna(params['med_b'])
                elif method == 'division_interaction':
                    X_enh[f'{a}_div_{b}'] = X_enh[a].fillna(params['med_a']) / (X_enh[b].fillna(params['med_b']).abs() + 1e-5)
                elif method == 'addition_interaction':
                    X_enh[f'{a}_plus_{b}'] = X_enh[a].fillna(params['med_a']) + X_enh[b].fillna(params['med_b'])
                elif method == 'abs_diff_interaction':
                    X_enh[f'{a}_absdiff_{b}'] = (X_enh[a].fillna(params['med_a']) - X_enh[b].fillna(params['med_b'])).abs()
                elif method == 'group_mean':
                    cc, nc_ = params['cat_col'], params['num_col']
                    X_enh[f'grpmean_{nc_}_by_{cc}'] = X_enh[cc].astype(str).map(params['grp_map']).fillna(params['fill_val']).astype(float)
                elif method == 'group_std':
                    cc, nc_ = params['cat_col'], params['num_col']
                    X_enh[f'grpstd_{nc_}_by_{cc}'] = X_enh[cc].astype(str).map(params['grp_map']).fillna(params['fill_val']).astype(float)
                elif method == 'cat_concat':
                    nc = f'{a}_concat_{b}'
                    combined = X_enh[a].astype(str) + '_' + X_enh[b].astype(str)
                    inv = {v: i for i, v in enumerate(params['le_classes'])}
                    X_enh[nc] = combined.map(inv).fillna(-1).astype(int)

            elif params['type'] == 'row':
                col_medians = params.get('col_medians', {})
                numeric_cols_row = list(col_medians.keys())
                # Only use numeric cols that exist in test data
                numeric_cols_row = [c for c in numeric_cols_row if c in X_enh.columns]
                X_num = X_enh[numeric_cols_row].copy()
                X_num_filled = X_num.apply(lambda s: s.fillna(col_medians.get(s.name, 0)))

                if method == 'row_numeric_stats':
                    _sel_stats = params.get('selected_row_stats') or [
                        'row_mean', 'row_median', 'row_sum', 'row_std',
                        'row_min', 'row_max', 'row_range',
                    ]
                    _stat_fns_t = {
                        'row_mean':   lambda df: df.mean(axis=1),
                        'row_median': lambda df: df.median(axis=1),
                        'row_sum':    lambda df: df.sum(axis=1),
                        'row_std':    lambda df: df.std(axis=1).fillna(0),
                        'row_min':    lambda df: df.min(axis=1),
                        'row_max':    lambda df: df.max(axis=1),
                    }
                    for _st in _sel_stats:
                        if _st == 'row_range':
                            _mn = (X_enh['row_min'] if 'row_min' in X_enh.columns
                                   else X_num_filled.min(axis=1))
                            _mx = (X_enh['row_max'] if 'row_max' in X_enh.columns
                                   else X_num_filled.max(axis=1))
                            X_enh['row_range'] = _mx - _mn
                        elif _st in _stat_fns_t:
                            X_enh[_st] = _stat_fns_t[_st](X_num_filled)

                elif method == 'row_zero_stats':
                    zero_mask = (X_num_filled == 0)
                    X_enh['row_zero_count']      = zero_mask.sum(axis=1)
                    X_enh['row_zero_percentage'] = zero_mask.mean(axis=1)

                elif method == 'row_missing_stats':
                    all_cols = [c for c in params.get('all_cols', X_enh.columns.tolist())
                                if c in X_enh.columns]
                    miss_mask = X_enh[all_cols].isnull()
                    X_enh['row_missing_count']      = miss_mask.sum(axis=1)
                    X_enh['row_missing_percentage'] = miss_mask.mean(axis=1)

            elif params['type'] == 'date':
                if method == 'date_features' and col in X_enh.columns:
                    date_feats, _ = _apply_date_features(X_enh[col],
                                                         min_date=params.get('min_date'),
                                                         col_type=params.get('col_type', 'datetime'),
                                                         selected_features=params.get('selected_date_features'))
                    prefix = f'{col}_'
                    date_feats.columns = [prefix + c for c in date_feats.columns]
                    X_enh = pd.concat([X_enh.reset_index(drop=True),
                                       date_feats.reset_index(drop=True)], axis=1)
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'date_cyclical':
                    prefix = params.get('col_prefix', f'{col}_')
                    cyclic_feats = _apply_date_cyclical(X_enh, prefix, component=None)
                    if not cyclic_feats.empty:
                        X_enh = pd.concat([X_enh.reset_index(drop=True),
                                           cyclic_feats.reset_index(drop=True)], axis=1)

                elif method in ('date_cyclical_month', 'date_cyclical_dow',
                                'date_cyclical_dom',  'date_cyclical_hour'):
                    prefix    = params.get('col_prefix', f'{col}_')
                    component = params.get('component', method.replace('date_cyclical_', ''))
                    cyclic_feats = _apply_date_cyclical(X_enh, prefix, component=component)
                    if not cyclic_feats.empty:
                        X_enh = pd.concat([X_enh.reset_index(drop=True),
                                           cyclic_feats.reset_index(drop=True)], axis=1)

                elif method == 'dow_ordinal':
                    if col in X_enh.columns:
                        mapped = X_enh[col].astype(str).str.strip().str.lower().map(_DOW_TO_INT)
                        X_enh[f'{col}_dow'] = mapped.fillna(-1).astype(int)
                        X_enh = X_enh.drop(columns=[col])

                elif method == 'dow_cyclical':
                    ordinal_col = f'{col}_dow'
                    if ordinal_col in X_enh.columns:
                        vals = X_enh[ordinal_col].astype(float)
                    elif col in X_enh.columns:
                        vals = X_enh[col].astype(str).str.strip().str.lower().map(_DOW_TO_INT).fillna(0).astype(float)
                        X_enh = X_enh.drop(columns=[col])
                    else:
                        continue
                    X_enh[f'{col}_dow_sin'] = np.sin(2 * np.pi * vals / 7)
                    X_enh[f'{col}_dow_cos'] = np.cos(2 * np.pi * vals / 7)

            elif params['type'] == 'text':
                if method == 'text_stats' and col in X_enh.columns:
                    _fields = params.get('text_stat_fields', None)
                    text_feats = _apply_text_stats(X_enh[col], fields=_fields)
                    prefix = f'{col}_'
                    text_feats.columns = [prefix + c for c in text_feats.columns]
                    X_enh = pd.concat([X_enh.reset_index(drop=True),
                                       text_feats.reset_index(drop=True)], axis=1)
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'text_tfidf':
                    tfidf = params.get('tfidf_vectorizer')
                    if tfidf is not None:
                        # col may have been dropped by a prior text_stats transform; fall back to X_test
                        text_source = X_enh[col] if col in X_enh.columns else X_test[col]
                        corpus = text_source.fillna('').astype(str).tolist()
                        tfidf_mat = tfidf.transform(corpus).toarray()
                        feat_names = params.get('new_cols', [])
                        tfidf_df = pd.DataFrame(tfidf_mat, columns=feat_names,
                                                index=X_enh.index)
                        X_enh = pd.concat([X_enh.reset_index(drop=True),
                                           tfidf_df.reset_index(drop=True)], axis=1)
                        if col in X_enh.columns:
                            X_enh = X_enh.drop(columns=[col])

        except Exception as _e:
            _DROPS_SOURCE = {
                'date_features', 'text_stats', 'text_tfidf',
                'dow_ordinal', 'dow_cyclical',
            }
            _method = params.get('method')
            _col    = params.get('column')
            if _method in _DROPS_SOURCE and _col and _col in X_enh.columns:
                X_enh = X_enh.drop(columns=[_col])
                st.warning(
                    f"⚠️ Test-time `{_method}` on **{_col}** failed ({_e}) — "
                    f"column dropped to prevent useless label-encoding of raw strings."
                )
            else:
                st.warning(f"⚠️ Test-time transform failed ({_method} on `{_col}`): {_e}")
            continue

    # Defragment the DataFrame after column-by-column additions to avoid
    # PerformanceWarning and improve downstream operation speed.
    X_enh = X_enh.copy()
    return X_enh