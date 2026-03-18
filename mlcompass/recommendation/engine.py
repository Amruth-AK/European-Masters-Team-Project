"""
mlcompass.recommendation.engine — Suggestion generation & deduplication
======================================================================
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from mlcompass.constants import (
    NUMERICAL_METHODS, CATEGORICAL_METHODS,
    INTERACTION_METHODS_NUM_NUM, INTERACTION_METHODS_CAT_NUM, INTERACTION_METHODS_CAT_CAT,
    ROW_FAMILIES, METHOD_DESCRIPTIONS,
    _IMBALANCE_MODERATE, _IMBALANCE_MULTICLASS_RATIO_CAP, _IMBALANCE_MULTICLASS_DOMINANT,
)
from mlcompass.transforms.helpers import ensure_numeric_target
from mlcompass.transforms.detection import detect_dow_columns, detect_text_columns
from mlcompass.analysis.meta_features import (
    get_dataset_meta, get_row_dataset_meta,
    get_numeric_column_meta, get_categorical_column_meta,
    get_pair_meta_features, get_baseline_importances,
    should_test_numerical, should_test_categorical,
)
from mlcompass.analysis.profiling import (
    detect_date_columns, detect_problematic_columns, _apply_type_reassignments,
)
from mlcompass.analysis.advisories import generate_dataset_advisories
from mlcompass.recommendation.meta_models import build_feature_vector, _is_near_bijection


def generate_suggestions(X, y, meta_models, baseline_score, baseline_std,
                         progress_cb=None, type_reassignments=None,
                         real_n_rows=None, include_imbalance=True):
    """
    Run meta-models on all applicable (column, method) combinations.

    real_n_rows: when X is a subsample (quick mode), pass the full dataset's
        row count so meta-models receive accurate n_rows / row_col_ratio.
    include_imbalance: if True (default), inject a class_weight_balance
        suggestion when the class imbalance ratio >= _IMBALANCE_MODERATE.
        Set to False to suppress imbalance suggestions entirely.
    """
    y_numeric = ensure_numeric_target(y)
    ds_meta = get_dataset_meta(X, y)
    ds_meta['baseline_score'] = baseline_score
    ds_meta['baseline_std'] = baseline_std
    ds_meta['relative_headroom'] = max(1.0 - baseline_score, 0.001)

    if real_n_rows is not None and real_n_rows > len(X):
        ds_meta['n_rows']        = real_n_rows
        ds_meta['row_col_ratio'] = real_n_rows / max(X.shape[1], 1)

    importances = get_baseline_importances(X, y)
    imp_ranks = importances.rank(ascending=False, pct=True)

    # Pre-detect date and text columns
    date_col_map     = detect_date_columns(X)
    dow_cols_pre     = detect_dow_columns(X, already_date_cols=set(date_col_map))
    text_col_map_pre = detect_text_columns(X, date_cols=date_col_map, dow_cols=dow_cols_pre)

    # Column-level safeguards
    skipped_info = detect_problematic_columns(
        X,
        known_date_cols=set(date_col_map) | dow_cols_pre,
        known_text_cols=set(text_col_map_pre),
    )

    # Apply user type reassignments
    if type_reassignments:
        X, date_col_map, text_col_map_pre, dow_cols_pre, skipped_info = (
            _apply_type_reassignments(
                X, type_reassignments,
                date_col_map, text_col_map_pre, dow_cols_pre, skipped_info,
            )
        )

    fully_skip = set(skipped_info['id_columns']) | set(skipped_info['constant_columns'])

    suggestions = []
    total_steps = 0
    done_steps = 0
    _leakage_col_metas = {}

    numeric_cols = [c for c in X.columns
                    if pd.api.types.is_numeric_dtype(X[c]) and c not in fully_skip]
    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c]) and c not in fully_skip]
    if 'numerical' in meta_models: total_steps += len(numeric_cols) * len(NUMERICAL_METHODS)
    if 'categorical' in meta_models: total_steps += len(cat_cols) * len(CATEGORICAL_METHODS)
    if 'interaction' in meta_models: total_steps += 50
    if 'row' in meta_models: total_steps += len(ROW_FAMILIES)
    total_steps = max(total_steps, 1)

    # --- Numerical suggestions ---
    if 'numerical' in meta_models and numeric_cols:
        mm = meta_models['numerical']
        for col in numeric_cols:
            col_meta = get_numeric_column_meta(
                X[col], y,
                importance=float(importances.get(col, 0)),
                importance_rank_pct=float(imp_ranks.get(col, 0.5))
            )
            _leakage_col_metas[col] = col_meta
            combined_meta = {**ds_meta, **col_meta}

            for method in NUMERICAL_METHODS:
                done_steps += 1
                if progress_cb: progress_cb(done_steps / total_steps)
                if not should_test_numerical(method, col_meta, X[col]):
                    continue
                if method not in mm['method_vocab']:
                    continue

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]
                suggestions.append({
                    'type': 'numerical',
                    'column': col,
                    'column_b': None,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                    'meta': {k: col_meta.get(k) for k in [
                        'null_pct', 'skewness', 'pct_negative', 'outlier_ratio',
                        'zeros_ratio', 'coeff_variation', 'unique_ratio', 'is_binary',
                    ]},
                })

    # --- Categorical suggestions ---
    if 'categorical' in meta_models and cat_cols:
        mm = meta_models['categorical']
        for col in cat_cols:
            col_meta = get_categorical_column_meta(
                X[col], y,
                importance=float(importances.get(col, 0)),
                importance_rank_pct=float(imp_ranks.get(col, 0.5))
            )
            _leakage_col_metas[col] = col_meta
            combined_meta = {**ds_meta, **col_meta}

            for method in CATEGORICAL_METHODS:
                done_steps += 1
                if progress_cb: progress_cb(done_steps / total_steps)
                if not should_test_categorical(method, col_meta):
                    continue
                if method not in mm['method_vocab']:
                    continue

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]
                suggestions.append({
                    'type': 'categorical',
                    'column': col,
                    'column_b': None,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                    'meta': {k: col_meta.get(k) for k in [
                        'null_pct', 'n_unique', 'unique_ratio', 'is_high_cardinality',
                        'is_low_cardinality', 'rare_category_pct', 'top_category_dominance',
                    ]},
                })

    # --- Interaction suggestions ---
    if 'interaction' in meta_models and X.shape[1] >= 2:
        mm = meta_models['interaction']
        imp_dict = importances.to_dict()

        sorted_num = sorted(numeric_cols, key=lambda c: imp_dict.get(c, 0), reverse=True)[:8]
        sorted_cat = sorted(cat_cols, key=lambda c: imp_dict.get(c, 0), reverse=True)[:6]
        tested = set()

        pairs = []
        # num+num
        _num_num_done = False
        for i, a in enumerate(sorted_num):
            if _num_num_done:
                break
            for b in sorted_num[i+1:]:
                if len(pairs) >= 30:
                    _num_num_done = True
                    break
                if _is_near_bijection(X[a], X[b]): continue
                pairs.append((a, b, INTERACTION_METHODS_NUM_NUM))
        # cat+num
        cnt = 0
        for cat_c in sorted_cat:
            for num_c in sorted_num:
                if cnt >= 20: break
                if _is_near_bijection(X[cat_c], X[num_c]): continue
                pairs.append((cat_c, num_c, INTERACTION_METHODS_CAT_NUM))
                cnt += 1
        # cat+cat
        cnt = 0
        for i, a in enumerate(sorted_cat):
            for b in sorted_cat[i+1:]:
                if cnt >= 10: break
                if X[a].nunique() > len(X)*0.5 or X[b].nunique() > len(X)*0.5:
                    continue
                if _is_near_bijection(X[a], X[b]): continue
                pairs.append((a, b, INTERACTION_METHODS_CAT_CAT))
                cnt += 1

        for col_a, col_b, methods in pairs:
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in tested: continue
            tested.add(pair_key)

            pair_meta = get_pair_meta_features(
                X[col_a], X[col_b], y,
                imp_a=float(imp_dict.get(col_a, 0)),
                imp_b=float(imp_dict.get(col_b, 0))
            )
            combined_meta = {**ds_meta, **pair_meta}

            for method in methods:
                done_steps += 1
                if progress_cb: progress_cb(min(done_steps / total_steps, 1.0))
                if method not in mm['method_vocab']:
                    continue

                a_num = pd.api.types.is_numeric_dtype(X[col_a])
                b_num = pd.api.types.is_numeric_dtype(X[col_b])

                if method == 'division_interaction':
                    if X[col_a].nunique() <= 2 or X[col_b].nunique() <= 2: continue
                    denom = X[col_b]
                    if (denom.fillna(0) == 0).any(): continue

                if method in ('group_mean', 'group_std'):
                    if a_num == b_num: continue
                    cat_col = col_a if not a_num else col_b
                    cat_unique_ratio = X[cat_col].nunique() / max(len(X), 1)
                    if cat_unique_ratio > 0.30: continue

                if method == 'cat_concat':
                    joint_card = X[col_a].nunique() * X[col_b].nunique()
                    if joint_card > min(len(X) * 0.5, 500): continue

                if method in ('product_interaction', 'addition_interaction', 'abs_diff_interaction'):
                    if a_num and b_num:
                        if X[col_a].nunique() <= 2 and X[col_b].nunique() <= 2: continue

                fv = build_feature_vector(combined_meta, method, mm['config'])
                pred = mm['booster'].predict(fv)[0]

                display_a, display_b = sorted([col_a, col_b])
                suggestions.append({
                    'type': 'interaction',
                    'column': col_a,
                    'column_b': col_b,
                    'method': method,
                    'predicted_delta': float(pred),
                    'description': METHOD_DESCRIPTIONS.get(method, method),
                    'meta': {k: pair_meta.get(k) for k in [
                        'pearson_corr', 'spearman_corr', 'mutual_info_pair',
                        'imp_a', 'imp_b',
                    ]},
                })

    # --- Row suggestions ---
    if 'row' in meta_models:
        mm = meta_models['row']
        row_meta = get_row_dataset_meta(X)
        combined_meta = {**ds_meta, **row_meta}
        numeric_cols_for_row = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        has_zeros = any(
            (X[c].fillna(0) == 0).any() for c in numeric_cols_for_row
        ) if numeric_cols_for_row else False
        has_missing = X.isnull().any().any()

        for family in ROW_FAMILIES:
            done_steps += 1
            if progress_cb: progress_cb(min(done_steps / total_steps, 1.0))
            if family not in mm['method_vocab']:
                continue

            if family == 'row_numeric_stats' and len(numeric_cols_for_row) < 2:
                continue
            if family == 'row_zero_stats' and (len(numeric_cols_for_row) < 2 or not has_zeros):
                continue
            if family == 'row_missing_stats' and not has_missing:
                continue

            fv = build_feature_vector(combined_meta, family, mm['config'])
            pred = mm['booster'].predict(fv)[0]
            suggestions.append({
                'type': 'row',
                'column': '(all numeric cols)',
                'column_b': None,
                'method': family,
                'predicted_delta': float(pred),
                'description': METHOD_DESCRIPTIONS.get(family, family),
                'meta': {k: row_meta.get(k) for k in [
                    'avg_missing_pct', 'max_missing_pct', 'pct_cells_zero',
                    'pct_rows_with_any_missing', 'pct_rows_with_any_zero',
                ]},
            })

    # Pin all applicable row families
    if 'row' in meta_models:
        _row_applicable = {
            'row_numeric_stats': len(numeric_cols_for_row) >= 2,
            'row_zero_stats':    len(numeric_cols_for_row) >= 2 and has_zeros,
            'row_missing_stats': has_missing,
        }
        for _fam, _applicable in _row_applicable.items():
            if not _applicable:
                continue
            already_in = any(s['method'] == _fam for s in suggestions)
            if not already_in:
                try:
                    _row_meta_pin = get_row_dataset_meta(X)
                    _combined_pin = {**ds_meta, **_row_meta_pin}
                    _mm_pin       = meta_models['row']
                    _fv_pin       = build_feature_vector(_combined_pin, _fam, _mm_pin['config'])
                    _raw_pin      = float(_mm_pin['booster'].predict(_fv_pin)[0])
                except Exception:
                    _raw_pin = 0.0
                suggestions.append({
                    'type': 'row', 'column': '(all numeric cols)', 'column_b': None,
                    'method': _fam,
                    'predicted_delta':     0.5,
                    'predicted_delta_raw': _raw_pin,
                    'description': METHOD_DESCRIPTIONS.get(_fam, _fam),
                    'pinned': True,
                })
            else:
                for s in suggestions:
                    if s['method'] == _fam:
                        s['pinned'] = True
                        if s['predicted_delta'] < 0:
                            s['predicted_delta'] = 0.0

    # ── Inject date-column suggestions ───────────────────────────────────────
    _CYCLICAL_COMPONENT_META = {
        'date_cyclical_month': {
            'label':  '📅 Cyclic month',
            'icon':   '🔵',
            'hint':   (
                "**When to enable:** your target follows a seasonal calendar pattern "
                "(retail sales, energy demand, weather-driven behaviour, tax deadlines). "
                "Wrapping month 1–12 into sin/cos means the model sees Dec and Jan as "
                "neighbours rather than opposites. "
                "**Skip if** you only need to identify *which* month something happened "
                "and there is no wrap-around seasonality."
            ),
        },
        'date_cyclical_dow': {
            'label':  '📅 Cyclic day-of-week',
            'icon':   '🟢',
            'hint':   (
                "**When to enable:** weekly rhythms matter — footfall, support-ticket "
                "volume, fraud rates, or any pattern where Sunday and Monday are "
                "behaviourally adjacent. "
                "**Skip if** the ordinal weekday identity is more important than "
                "its position in the weekly cycle."
            ),
        },
        'date_cyclical_dom': {
            'label':  '📅 Cyclic day-of-month',
            'icon':   '🟡',
            'hint':   (
                "**When to enable:** billing cycles, salary payment dates, or any "
                "month-end / month-start effect where the 31st and the 1st should be "
                "treated as adjacent. "
                "**Skip if** you only need the raw day number and there is no "
                "meaningful wrap-around between month boundaries."
            ),
        },
        'date_cyclical_hour': {
            'label':  '📅 Cyclic hour',
            'icon':   '🟠',
            'hint':   (
                "**When to enable:** intra-day patterns where 23:00 and 00:00 are "
                "adjacent — rush-hour traffic, overnight fraud, shift-based workloads, "
                "or any circadian rhythm. "
                "**Skip if** the absolute hour is sufficient or if the column contains "
                "only date (no time) information."
            ),
        },
    }

    for col, col_info in date_col_map.items():
        parse_rate = col_info['parse_rate']
        col_type   = col_info['col_type']

        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'date_features',
            'col_type':            col_type,
            'predicted_delta':     0.50,
            'predicted_delta_raw': 0.0,
            'description':         (
                (
                    "Time extraction — hour"
                    if col_type == 'time'
                    else METHOD_DESCRIPTIONS['date_features']
                ) + f" (parse rate {parse_rate:.0%})"
            ),
            'pinned':       True,
            'auto_checked': True,
        })

        if col_type == 'time':
            components_to_offer = ['date_cyclical_hour']
        elif col_type == 'date':
            components_to_offer = ['date_cyclical_month', 'date_cyclical_dow', 'date_cyclical_dom']
        else:
            components_to_offer = [
                'date_cyclical_month', 'date_cyclical_dow',
                'date_cyclical_dom',   'date_cyclical_hour',
            ]

        for rank, comp_method in enumerate(components_to_offer, start=1):
            cmeta = _CYCLICAL_COMPONENT_META[comp_method]
            suggestions.append({
                'type':                'date',
                'column':              col,
                'column_b':            None,
                'method':              comp_method,
                'predicted_delta':     0.49 - rank * 0.001,
                'predicted_delta_raw': 0.0,
                'description':         (
                    f"{METHOD_DESCRIPTIONS[comp_method]}  ·  {cmeta['hint']}"
                ),
                'pinned':              False,
                'auto_checked':        False,
            })

    # ── Inject day-of-week categorical column suggestions ─────────────────────
    dow_cols = dow_cols_pre
    fully_skip |= dow_cols
    for col in dow_cols:
        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'dow_ordinal',
            'predicted_delta':     0.50,
            'predicted_delta_raw': 0.0,
            'description':         (
                "Day-of-week ordinal encoding — maps Mon→0, Tue→1, … Sun→6. "
                "Gives the model a numeric representation that preserves weekday order."
            ),
            'pinned':       True,
            'auto_checked': True,
        })
        cmeta = _CYCLICAL_COMPONENT_META['date_cyclical_dow']
        suggestions.append({
            'type':                'date',
            'column':              col,
            'column_b':            None,
            'method':              'dow_cyclical',
            'predicted_delta':     0.49,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"Day-of-week cyclical encoding — sin/cos of weekday (0–6); "
                f"keeps Sunday and Monday adjacent.  ·  {cmeta['hint']}"
            ),
            'pinned':              False,
            'auto_checked':        False,
        })

    # ── Inject text-column suggestions ────────────────────────────────────────
    text_col_map = text_col_map_pre
    _TEXT_STAT_FIELDS = [
        ('word_count',      'Word count'),
        ('char_count',      'Char count'),
        ('avg_word_len',    'Avg word length'),
        ('uppercase_ratio', 'Uppercase %'),
        ('digit_ratio',     'Digit %'),
        ('punct_ratio',     'Punctuation %'),
    ]
    for col, reason in text_col_map.items():
        suggestions.append({
            'type':                'text',
            'column':              col,
            'column_b':            None,
            'method':              'text_stats',
            'predicted_delta':     0.5,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"{METHOD_DESCRIPTIONS['text_stats']} — {reason}"
            ),
            'pinned': True,
            'text_stat_fields': [f for f, _ in _TEXT_STAT_FIELDS],
        })
        suggestions.append({
            'type':                'text',
            'column':              col,
            'column_b':            None,
            'method':              'text_tfidf',
            'predicted_delta':     0.5,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"{METHOD_DESCRIPTIONS['text_tfidf']} — {reason}"
            ),
            'pinned': True,
        })

    # ── Inject imbalance suggestion ─────────────────────────────────────────
    y_counts_adv       = pd.Series(y_numeric).value_counts()
    imbalance_ratio_adv = float(y_counts_adv.max() / max(y_counts_adv.min(), 1))
    dominant_frac_adv   = float(y_counts_adv.max() / max(len(y_numeric), 1))
    is_binary_adv       = (y_counts_adv.shape[0] == 2)

    # Only inject when caller allows it AND ratio exceeds the moderate threshold
    if include_imbalance and imbalance_ratio_adv >= _IMBALANCE_MODERATE:
        if is_binary_adv:
            _adv_strategy   = 'binary'
            _adv_auto_check = False
        else:
            _severe = (
                imbalance_ratio_adv > _IMBALANCE_MULTICLASS_RATIO_CAP
                or dominant_frac_adv > _IMBALANCE_MULTICLASS_DOMINANT
            )
            _adv_strategy   = 'none' if _severe else 'multiclass_moderate'
            _adv_auto_check = not _severe

        method_hint = {'binary': 'is_unbalance=True',
                       'multiclass_moderate': 'class_weight=balanced',
                       'none': 'no reweighting (severe imbalance)',
                       }[_adv_strategy]
        suggestions.append({
            'type':                'imbalance',
            'column':              '(model parameter)',
            'column_b':            None,
            'method':              'class_weight_balance',
            'predicted_delta':     1.0,
            'predicted_delta_raw': 0.0,
            'description':         (
                f"Class reweighting — ratio {imbalance_ratio_adv:.1f}:1 "
                f"(dominant class {dominant_frac_adv:.0%}) — {method_hint}"
            ),
            'pinned':              True,
            'imbalance_ratio':     imbalance_ratio_adv,
            'dominant_frac':       dominant_frac_adv,
            'n_classes_imb':       int(y_counts_adv.shape[0]),
            'imbalance_strategy':  _adv_strategy,
            'auto_checked':        _adv_auto_check,
        })

    # ── Cross-type normalization ──────────────────────────────────────────────
    by_type = defaultdict(list)
    for s in suggestions:
        if s['type'] not in ('imbalance', 'date', 'text') and not s.get('pinned'):
            by_type[s['type']].append(s)

    for type_suggestions in by_type.values():
        vals = np.array([s['predicted_delta'] for s in type_suggestions], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        for s in type_suggestions:
            s['predicted_delta_raw'] = s['predicted_delta']
            s['predicted_delta'] = float((s['predicted_delta'] - mu) / sigma) if sigma > 1e-9 else 0.0

    for s in suggestions:
        if s.get('pinned') and 'predicted_delta_raw' not in s:
            s['predicted_delta_raw'] = s['predicted_delta']

    suggestions.sort(key=lambda s: s['predicted_delta'], reverse=True)

    advisories = generate_dataset_advisories(X, y)

    # ── Leakage detection ─────────────────────────────────────────────────────
    n_classes_lk = int(y_numeric.nunique())
    target_probs = y_numeric.value_counts(normalize=True).values
    target_entropy = float(-np.sum(target_probs * np.log2(np.clip(target_probs, 1e-12, None))))
    mi_leakage_thresh = max(0.95 * target_entropy, 0.95)

    leaky_cols = {}
    for col, cm in _leakage_col_metas.items():
        spear = cm.get('spearman_corr_target', 0.0)
        mi    = cm.get('mutual_info_score', 0.0)
        reasons = []
        if spear >= 0.95:
            reasons.append(f"Spearman |ρ| = {spear:.4f} ≥ 0.95")
        if mi >= mi_leakage_thresh:
            reasons.append(f"Mutual info = {mi:.4f} ≥ {mi_leakage_thresh:.3f} (≥ 95% of target entropy)")
        if reasons:
            leaky_cols[col] = " | ".join(reasons)

    if leaky_cols:
        col_list = "\n".join(f"- `{c}`: {r}" for c, r in leaky_cols.items())
        advisories.insert(0, {
            'category': 'leakage',
            'severity': 'high',
            'title': f'⚠️ Potential Data Leakage — {len(leaky_cols)} column(s) suspiciously correlated with target',
            'detail': (
                "The following features have near-perfect statistical association with the "
                "target. This is a strong signal of **data leakage** — the column may be "
                "derived from the target, recorded after the event, or otherwise unavailable "
                "at real inference time. Consider removing or carefully auditing these columns "
                "before drawing any conclusions from model performance.\n\n"
                + col_list
            ),
            'code_hint': "# Drop suspect columns before training\nX = X.drop(columns=" + str(list(leaky_cols.keys())) + ")",
        })

    return suggestions, skipped_info, advisories, ds_meta


def deduplicate_suggestions(suggestions):
    """
    For single-column transforms, keep only the best method per column among
    in-place transforms.  Additive transforms are kept unconditionally.
    For interactions, keep only the best method per pair.
    """
    ADDITIVE_METHODS = {
        'missing_indicator',
        'polynomial_square',
        'polynomial_cube',
        'reciprocal_transform',
    }

    best_single = {}
    additive_suggestions = []
    best_interaction = {}
    row_suggestions = []

    for s in suggestions:
        if s['type'] in ('numerical', 'categorical'):
            if s['method'] in ADDITIVE_METHODS:
                additive_suggestions.append(s)
            else:
                key = (s['type'], s['column'])
                if key not in best_single or s['predicted_delta'] > best_single[key]['predicted_delta']:
                    best_single[key] = s
        elif s['type'] in ('row', 'date', 'text', 'imbalance'):
            row_suggestions.append(s)
        else:
            col_a = s['column'] or ''
            col_b = s.get('column_b') or ''
            pair = tuple(sorted([col_a, col_b]))
            key = (pair, s['method'])
            if key not in best_interaction or s['predicted_delta'] > best_interaction[key]['predicted_delta']:
                best_interaction[key] = s

    deduped = (list(best_single.values()) + additive_suggestions
               + list(best_interaction.values()) + row_suggestions)
    deduped.sort(key=lambda s: s['predicted_delta'], reverse=True)
    return deduped


def recommended_top_k(X):
    """
    Compute a sensible default for the number of suggestions to apply,
    scaled by dataset rows and columns.

    Parameters
    ----------
    X : pd.DataFrame or None
        Training feature matrix.  When *None* a safe fallback of 10 is
        returned.

    Returns
    -------
    int
        Recommended number of top-ranked suggestions to pre-select /
        apply by default.
    """
    if X is None:
        return 10

    n_rows, n_cols = X.shape

    # Row-based tier
    if   n_rows < 500:    k = 3
    elif n_rows < 2_000:  k = 5
    elif n_rows < 10_000: k = 8
    elif n_rows < 50_000: k = 12
    else:                 k = 15

    # Wider datasets → more FE opportunities → bump up
    if   n_cols > 30: k = min(k + 3, 20)
    elif n_cols > 15: k = min(k + 1, 20)

    return k
