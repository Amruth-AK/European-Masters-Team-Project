"""
mlcompass.analysis.profiling — Column type detection, ID/constant detection, overrides
=====================================================================================
"""

import numpy as np
import pandas as pd

from mlcompass.transforms.helpers import _to_datetime_safe
from mlcompass.transforms.detection import (
    detect_dow_columns, detect_text_columns, _DOW_ALL,
)


# ---------------------------------------------------------------------------
# Problematic column detection
# ---------------------------------------------------------------------------

def detect_problematic_columns(X, known_date_cols=None, known_text_cols=None):
    """
    Detect columns that should be excluded from (or handled carefully in) suggestions.

    Returns a dict:
        'id_columns':          {col: reason}
        'constant_columns':    {col: reason}
        'binary_num_columns':  {col: reason}
        'high_missing_columns':{col: reason}
    """
    known_date_cols = set(known_date_cols or {})
    known_text_cols = set(known_text_cols or {})
    id_cols = {}
    constant_cols = {}
    binary_num_cols = {}
    high_missing_cols = {}

    n = len(X)

    ID_EXACT = {'id', 'idx', 'index', 'key', 'no', 'num', 'number', 'row', '#'}
    ID_SUBSTRINGS = ['_id', 'id_', '_idx', 'idx_', '_index', 'index_',
                     '_key', 'key_', '_no', '_num', '_number', 'rownum', 'row_num',
                     'record_id', 'sample_id', 'obs_id', 'entry_id']

    def _name_looks_like_id(col):
        c = col.lower().strip()
        if c in ID_EXACT:
            return True
        return any(sub in c for sub in ID_SUBSTRINGS)

    for col in X.columns:
        s = X[col]
        n_unique = s.nunique(dropna=True)
        non_null = s.dropna()
        null_pct = s.isnull().mean()

        # Constant
        if n_unique <= 1:
            reason = "all values are null" if n_unique == 0 else "all values are identical"
            constant_cols[col] = reason
            continue

        # ID-like
        unique_ratio = n_unique / max(n, 1)

        if pd.api.types.is_numeric_dtype(s) and n_unique > 10:
            is_id = False
            id_reason = None

            try:
                float_vals = non_null.astype(float)
                if (n_unique == len(non_null) and
                        (float_vals.is_monotonic_increasing or
                         float_vals.is_monotonic_decreasing)):
                    is_id = True
                    id_reason = (
                        f"monotonic all-unique sequence (index-like), "
                        f"unique_ratio={unique_ratio:.2f}"
                    )
            except Exception:
                pass

            if not is_id and unique_ratio >= 0.95:
                try:
                    sorted_vals = np.sort(non_null.values.astype(float))
                    diffs = np.diff(sorted_vals)
                    if len(diffs) > 0 and diffs.mean() != 0:
                        step_cv = diffs.std() / abs(diffs.mean())
                        if step_cv < 0.05:
                            is_id = True
                            id_reason = (
                                f"sequential values — unique_ratio={unique_ratio:.2f}, "
                                f"step≈{diffs.mean():.2g} (step CV={step_cv:.3f})"
                            )
                except Exception:
                    pass

                if not is_id and _name_looks_like_id(col):
                    is_id = True
                    id_reason = (
                        f"ID-like column name with near-unique values "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )

            elif unique_ratio >= 0.80 and _name_looks_like_id(col):
                is_id = True
                id_reason = (
                    f"ID-like column name with high uniqueness "
                    f"(unique_ratio={unique_ratio:.2f})"
                )

            if is_id:
                id_cols[col] = id_reason
                continue

        elif not pd.api.types.is_numeric_dtype(s):
            if col in known_date_cols or col in known_text_cols:
                pass
            else:
                avg_len = float(s.dropna().astype(str).str.len().mean()) if len(s.dropna()) > 0 else 0
                looks_like_text = avg_len > 25
                if unique_ratio >= 0.95 and _name_looks_like_id(col) and not looks_like_text:
                    id_cols[col] = (
                        f"string column with near-unique values and ID-like name "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )
                    continue
                elif unique_ratio >= 0.80 and _name_looks_like_id(col) and not looks_like_text:
                    id_cols[col] = (
                        f"ID-like column name with high uniqueness "
                        f"(unique_ratio={unique_ratio:.2f})"
                    )
                    continue

        # High-missing
        if null_pct > 0.50:
            high_missing_cols[col] = (
                f"{null_pct*100:.0f}% missing — only impute/missing-indicator transforms considered"
            )

        # Binary numeric
        if pd.api.types.is_numeric_dtype(s) and n_unique == 2:
            binary_num_cols[col] = "binary numeric — non-trivial transforms skipped"

    return {
        'id_columns': id_cols,
        'constant_columns': constant_cols,
        'binary_num_columns': binary_num_cols,
        'high_missing_columns': high_missing_cols,
    }


# ---------------------------------------------------------------------------
# Column type info (unified)
# ---------------------------------------------------------------------------

_COL_TYPE_ICONS = {
    'Numerical': '🔢', 'Categorical': '🏷️', 'Binary': '⚡',
    'Date': '📅', 'DateTime': '🕐', 'Time only': '⏱️',
    'Day-of-Week': '📆', 'Free Text': '📝', 'ID': '🆔', 'Constant': '⚫',
}


def _override_options_for(info):
    """Return the valid override options for a column based on its dtype."""
    base = ['Auto', 'Numerical', 'Categorical', 'Binary']
    if not info.get('is_numeric', False):
        base += ['Date', 'DateTime', 'Time only', 'Day-of-Week', 'Free Text']
    return base


def get_column_type_info(X):
    """Run all detectors; return per-column summary dict."""
    date_col_map = detect_date_columns(X)
    dow_cols     = detect_dow_columns(X, already_date_cols=set(date_col_map))
    text_col_map = detect_text_columns(X, date_cols=date_col_map, dow_cols=dow_cols)
    skipped      = detect_problematic_columns(
        X,
        known_date_cols=set(date_col_map) | dow_cols,
        known_text_cols=set(text_col_map),
    )
    result = {}
    for col in X.columns:
        s        = X[col]
        nu       = int(s.nunique(dropna=True))
        mp       = float(s.isnull().mean())
        non_null = s.dropna()
        sample   = str(non_null.iloc[0])[:50] if len(non_null) > 0 else ''
        drop_suggested, drop_reason = False, ''

        if col in skipped['constant_columns']:
            detected = 'Constant'; drop_suggested = True
            drop_reason = skipped['constant_columns'][col]
        elif col in skipped['id_columns']:
            detected = 'ID'; drop_suggested = True
            drop_reason = skipped['id_columns'][col]
        elif col in date_col_map:
            ct = date_col_map[col]['col_type']
            detected = {'datetime': 'DateTime', 'date': 'Date', 'time': 'Time only'}.get(ct, 'Date')
        elif col in dow_cols:
            detected = 'Day-of-Week'
        elif col in text_col_map:
            detected = 'Free Text'
        elif pd.api.types.is_numeric_dtype(s):
            detected = 'Binary' if nu <= 2 else 'Numerical'
        else:
            detected = 'Binary' if nu <= 2 else 'Categorical'

        if not drop_suggested and mp > 0.80:
            drop_suggested = True; drop_reason = f"{mp*100:.0f}% values missing"

        result[col] = {
            'detected': detected, 'icon': _COL_TYPE_ICONS.get(detected, ''),
            'n_unique': nu, 'missing_pct': mp, 'sample': sample,
            'drop_suggested': drop_suggested, 'drop_reason': drop_reason,
            'is_numeric': bool(pd.api.types.is_numeric_dtype(s)),
        }
    return result


def _validate_col_override(col, override_type, info, X):
    """Return (severity, message) if override looks wrong, else None."""
    if override_type in ('Auto',):
        return None
    is_num = info.get('is_numeric', False)
    nu     = info.get('n_unique', 0)
    s      = X[col].dropna().astype(str)

    if override_type in ('Date', 'DateTime', 'Time only') and len(s):
        try:
            pr = float(pd.to_datetime(s, errors='coerce', format='mixed').notna().mean())
            if pr < 0.50:
                return ('warning', f"Only {pr*100:.0f}% of values parse as dates — are you sure?")
        except Exception:
            pass
    if override_type == 'Numerical' and not is_num:
        try:
            pr = float(pd.to_numeric(X[col], errors='coerce').notna().mean())
            if pr < 0.70:
                return ('error', f"Only {pr*100:.0f}% of values are numeric — use Categorical instead.")
            if pr < 0.95:
                return ('warning', f"{(1-pr)*100:.0f}% of values will become NaN.")
        except Exception:
            return ('error', "Values don't appear to be numeric.")
    if override_type == 'Free Text' and not is_num and len(s):
        avg_len = float(s.str.len().mean())
        if avg_len < 10:
            return ('warning', f"Average value length is {avg_len:.1f} chars — looks more like Categorical.")
    if override_type == 'Day-of-Week' and not is_num and len(s):
        recognised = {v.strip().lower() for v in s.unique()} & _DOW_ALL
        pct = len(recognised) / max(s.nunique(), 1)
        if pct < 0.50:
            return ('warning', f"Only {pct*100:.0f}% of unique values match weekday names.")
    if override_type == 'Binary' and nu > 2:
        return ('error', f"Column has {nu} unique values — Binary requires ≤ 2.")
    if override_type == 'Categorical' and nu > 500:
        return ('warning', f"{nu} unique values is very high — consider Free Text or Drop.")
    return None


def _apply_type_reassignments(X, type_reassignments, date_col_map, text_col_map,
                               dow_cols, skipped_info):
    """Apply non-Drop type overrides to detection dicts."""
    _DTYPE_MAP = {'Date': 'date', 'DateTime': 'datetime', 'Time only': 'time'}
    date_col_map = dict(date_col_map); text_col_map = dict(text_col_map)
    dow_cols = set(dow_cols)
    id_cols  = dict(skipped_info.get('id_columns', {}))
    const_cols = dict(skipped_info.get('constant_columns', {}))
    for col, t in type_reassignments.items():
        if col not in X.columns:
            continue
        date_col_map.pop(col, None); text_col_map.pop(col, None)
        dow_cols.discard(col); id_cols.pop(col, None); const_cols.pop(col, None)
        if t in _DTYPE_MAP:
            date_col_map[col] = {'parse_rate': 1.0, 'col_type': _DTYPE_MAP[t]}
        elif t == 'Free Text':
            text_col_map[col] = 'user-specified'
        elif t == 'Day-of-Week':
            dow_cols.add(col)
    skipped_info = dict(skipped_info)
    skipped_info['id_columns'] = id_cols; skipped_info['constant_columns'] = const_cols
    return X, date_col_map, text_col_map, dow_cols, skipped_info


# ---------------------------------------------------------------------------
# Date column detection
# ---------------------------------------------------------------------------

def detect_date_columns(X):
    """
    Return a dict of col -> {'parse_rate', 'col_type'} for temporal columns.
    col_type is one of: 'datetime', 'date', 'time'.
    """
    date_cols = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        s = X[col].dropna().astype(str)
        if len(s) < 5:
            continue
        try:
            parsed = _to_datetime_safe(s)
            parse_rate = float(parsed.notna().mean())
            if parse_rate < 0.70:
                continue

            valid = parsed.dropna()
            has_hour = float((valid.dt.hour != 0).mean()) >= 0.20
            n_unique_dates = valid.dt.normalize().nunique()
            has_date = n_unique_dates >= max(3, int(len(valid) * 0.02))

            if has_hour and not has_date:
                col_type = 'time'
            elif has_date and has_hour:
                col_type = 'datetime'
            else:
                col_type = 'date'

            date_cols[col] = {'parse_rate': parse_rate, 'col_type': col_type}
        except Exception:
            pass
    return date_cols


def detect_date_has_hour(X, col):
    """Return True if a detected date column carries a time component."""
    try:
        parsed = pd.to_datetime(X[col].dropna().astype(str), errors='coerce', format='mixed').dropna()
        if len(parsed) == 0:
            return False
        return float((parsed.dt.hour != 0).mean()) >= 0.20
    except Exception:
        return False
