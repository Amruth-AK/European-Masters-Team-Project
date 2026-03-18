"""
mlcompass.transforms.applicator — Fit & apply transform suggestions
==================================================================
"""

import logging
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mlcompass.transforms.helpers import (
    ensure_numeric_target,
    _apply_date_features, _apply_date_cyclical, _apply_text_stats,
)
from mlcompass.transforms.detection import _DOW_TO_INT

logger = logging.getLogger("mlcompass")


def fit_and_apply_suggestions(X_train, y_train, suggestions):
    """
    Apply selected suggestions to training data.
    Returns: (X_enhanced, fitted_params_list)
    fitted_params_list stores everything needed to replay on test data.
    """
    X_enh = X_train.copy()
    y_num = ensure_numeric_target(y_train)
    X_enh = X_enh.reset_index(drop=True)
    y_num = y_num.reset_index(drop=True)
    fitted = []

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
                    col_std = float(X_train[col].std())
                    if offset > 3 * col_std and col_std > 0:
                        logger.warning(
                            "log_transform on %s: the required shift (%.2g) is >3x "
                            "the column std (%.2g). The transform will compress most "
                            "variance — consider skipping it or using quantile binning "
                            "instead.", col, offset, col_std
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
                    str_col = X_enh[col].astype(str)
                    gm = float(y_num.mean())
                    agg = y_num.groupby(str_col).agg(['count', 'mean'])
                    smooth = ((agg['count'] * agg['mean'] + 10 * gm) / (agg['count'] + 10)).to_dict()
                    params.update({'smooth_map': smooth, 'global_mean': gm})
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
                col_medians = X_train[numeric_cols_row].median().to_dict()
                params['col_medians'] = col_medians
                X_num_filled = X_num.apply(lambda s: s.fillna(col_medians.get(s.name, 0)))

                if method == 'row_numeric_stats':
                    _ALL_ROW_STATS = ['row_mean', 'row_median', 'row_sum',
                                      'row_std', 'row_min', 'row_max', 'row_range']
                    _selected_stats = sug.get('selected_row_stats') or _ALL_ROW_STATS
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
                    mapped = X_enh[col].astype(str).str.strip().str.lower().map(_DOW_TO_INT)
                    X_enh[f'{col}_dow'] = mapped.fillna(-1).astype(int)
                    params['new_cols'] = [f'{col}_dow']
                    X_enh = X_enh.drop(columns=[col])

                elif method == 'dow_cyclical':
                    ordinal_col = f'{col}_dow'
                    if ordinal_col not in X_enh.columns:
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
                    if col in X_enh.columns:
                        X_enh = X_enh.drop(columns=[col])

            fitted.append(params)

        except Exception as e:
            _DROPS_SOURCE = {
                'date_features', 'text_stats', 'text_tfidf',
                'dow_ordinal', 'dow_cyclical',
            }
            if method in _DROPS_SOURCE and col in X_enh.columns:
                X_enh = X_enh.drop(columns=[col])
                logger.warning(
                    "%s on %s failed (%s) — column dropped to prevent "
                    "useless label-encoding of raw strings.", method, col, e
                )
            else:
                logger.warning("Failed to apply %s on %s: %s", method, col, e)
            continue

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
                logger.warning(
                    "Test-time %s on %s failed (%s) — column dropped to "
                    "prevent useless label-encoding of raw strings.",
                    _method, _col, _e
                )
            else:
                logger.warning("Test-time transform failed (%s on %s): %s",
                               _method, _col, _e)
            continue

    X_enh = X_enh.copy()
    return X_enh
