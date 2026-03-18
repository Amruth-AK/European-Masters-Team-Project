"""
mlcompass.recommendation.verdicts — Post-training suggestion diagnostics
=======================================================================
"""


def _compute_suggestion_verdicts(
    fitted_params,
    suggestions,
    selected_indices,
    enhanced_model,
    enhanced_train_cols,
    baseline_model,
    baseline_train_cols,
    baseline_val_metrics,
    enhanced_val_metrics,
    apply_imbalance=False,
):
    """
    Attribute each applied suggestion's impact using feature importances.

    Returns
    -------
    verdicts : list[dict]
        One entry per applied transform + one for imbalance if used.
    low_imp_orig : dict[str, float]
        Original columns (in baseline) with < 0.5 % importance.
    """
    enh_imps  = enhanced_model.feature_importances_
    enh_total = max(float(enh_imps.sum()), 1.0)
    enh_pct   = {c: float(v / enh_total * 100)
                 for c, v in zip(enhanced_train_cols, enh_imps)}

    base_imps  = baseline_model.feature_importances_
    base_total = max(float(base_imps.sum()), 1.0)
    base_pct   = {c: float(v / base_total * 100)
                  for c, v in zip(baseline_train_cols, base_imps)}

    original_cols = set(baseline_train_cols)

    verdicts = []

    # Dynamic importance thresholds
    n_enh_cols = max(len(enhanced_train_cols), 1)
    _thr_good     = max(0.5, 50.0 / n_enh_cols)
    _thr_marginal = _thr_good / 4.0

    # Per-transform verdicts
    for p in fitted_params:
        method   = p.get('method', '')
        col      = p.get('column', '')
        col_b    = p.get('column_b')
        sug_type = p.get('type', '')
        new_cols = p.get('new_cols') or []

        sug_idx = None
        search_order = list(selected_indices or []) + [
            j for j in range(len(suggestions)) if j not in set(selected_indices or [])
        ]
        for i in search_order:
            if i < len(suggestions):
                s = suggestions[i]
                if (s['method'] == method
                        and s['column'] == col
                        and s.get('column_b') == col_b):
                    sug_idx = i
                    break

        if new_cols:
            total_imp = sum(enh_pct.get(c, 0.0) for c in new_cols)
            col_names = ", ".join(f"`{c}`" for c in new_cols[:2]) + ("…" if len(new_cols) > 2 else "")
            if total_imp >= _thr_good:
                verdict = 'good'
                reason  = f"New column(s) {col_names} hold {total_imp:.2f}% of enhanced model importance"
            elif total_imp >= _thr_marginal:
                verdict = 'marginal'
                reason  = (f"New column(s) {col_names} hold only {total_imp:.2f}% importance "
                           f"— minimal contribution (threshold: {_thr_good:.2f}%)")
            else:
                verdict = 'bad'
                reason  = (f"New column(s) {col_names} hold {total_imp:.2f}% importance "
                           f"— effectively no contribution (threshold: {_thr_good:.2f}%)")
        else:
            base_i = base_pct.get(col, 0.0)
            enh_i  = enh_pct.get(col, 0.0)
            delta_i = enh_i - base_i
            delta_str = f"{delta_i:+.2f}%" if abs(delta_i) >= 0.01 else "unchanged"
            if enh_i >= _thr_good:
                verdict = 'good'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline)")
            elif enh_i >= _thr_marginal:
                verdict = 'marginal'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline) — limited gain (threshold: {_thr_good:.2f}%)")
            else:
                verdict = 'bad'
                reason  = (f"Column holds {enh_i:.2f}% importance in enhanced model "
                           f"({delta_str} vs baseline) — transform had no measurable effect (threshold: {_thr_good:.2f}%)")

        # Date sub-feature pruning
        bad_date_subfeatures = []
        col_prefix_for_date = ''
        if method == 'date_features' and new_cols and verdict == 'good':
            col_prefix_for_date = p.get('col_prefix', f'{col}_')
            for _nc in new_cols:
                if enh_pct.get(_nc, 0.0) < _thr_good:
                    _sub_name = (_nc[len(col_prefix_for_date):]
                                 if _nc.startswith(col_prefix_for_date) else _nc)
                    bad_date_subfeatures.append(_sub_name)

        # Row-stat sub-feature pruning
        bad_row_stats = []
        if method == 'row_numeric_stats' and new_cols and verdict == 'good':
            for _nc in new_cols:
                if enh_pct.get(_nc, 0.0) < _thr_good:
                    bad_row_stats.append(_nc)

        verdicts.append({
            'sug_idx':             sug_idx,
            'method':              method,
            'column':              col,
            'column_b':            col_b,
            'type':                sug_type,
            'new_cols':            new_cols,
            'verdict':             verdict,
            'reason':              reason,
            'bad_date_subfeatures': bad_date_subfeatures,
            'col_prefix':          col_prefix_for_date,
            'bad_row_stats':       bad_row_stats,
        })

    # Class-imbalance verdict
    if apply_imbalance:
        imb_idx = None
        for i in (selected_indices or []):
            if i < len(suggestions) and suggestions[i].get('type') == 'imbalance':
                imb_idx = i
                break

        b_f1   = float(baseline_val_metrics.get('f1')   or 0)
        e_f1   = float(enhanced_val_metrics.get('f1')    or 0)
        b_prec = float(baseline_val_metrics.get('precision') or 0)
        e_prec = float(enhanced_val_metrics.get('precision') or 0)
        b_rec  = float(baseline_val_metrics.get('recall') or 0)
        e_rec  = float(enhanced_val_metrics.get('recall') or 0)

        _STRONG_BASELINE_THRESHOLD = 0.90

        b_already_strong = (
            b_f1   >= _STRONG_BASELINE_THRESHOLD and
            b_prec >= _STRONG_BASELINE_THRESHOLD and
            b_rec  >= _STRONG_BASELINE_THRESHOLD
        )

        if b_already_strong and e_f1 <= b_f1 + 0.005:
            imb_verdict = 'marginal'
            imb_reason  = (
                f"Baseline was already strong (F1={b_f1:.3f}, P={b_prec:.3f}, R={b_rec:.3f}) — "
                f"class reweighting had no meaningful benefit (F1: {b_f1:.3f} → {e_f1:.3f}). "
                f"Consider disabling it."
            )
        elif b_already_strong and e_f1 < b_f1 - 0.005:
            imb_verdict = 'bad'
            imb_reason  = (
                f"Baseline was already strong (F1={b_f1:.3f}, P={b_prec:.3f}, R={b_rec:.3f}) — "
                f"class reweighting hurt performance (F1: {b_f1:.3f} → {e_f1:.3f}). "
                f"Disable it; the model handles the imbalance on its own."
            )
        elif e_f1 < b_f1 - 0.02:
            imb_verdict = 'bad'
            imb_reason  = (f"F1 degraded: {b_f1:.3f} → {e_f1:.3f}. "
                           f"Class reweighting may be hurting overall performance.")
        elif e_rec > b_rec + 0.15 and e_prec < b_prec - 0.15:
            imb_verdict = 'marginal'
            imb_reason  = (f"Recall improved ({b_rec:.3f} → {e_rec:.3f}) but precision "
                           f"dropped ({b_prec:.3f} → {e_prec:.3f}). "
                           f"Minority class is being over-predicted.")
        elif e_f1 >= b_f1 - 0.005:
            imb_verdict = 'good'
            imb_reason  = f"F1 maintained or improved: {b_f1:.3f} → {e_f1:.3f}"
        else:
            imb_verdict = 'marginal'
            imb_reason  = (f"Mixed results — F1: {b_f1:.3f} → {e_f1:.3f}, "
                           f"Recall: {b_rec:.3f} → {e_rec:.3f}")

        verdicts.append({
            'sug_idx':  imb_idx,
            'method':   'class_weight_balance',
            'column':   '(model param)',
            'column_b': None,
            'type':     'imbalance',
            'new_cols': [],
            'verdict':  imb_verdict,
            'reason':   imb_reason,
        })

    # Low-importance original columns
    low_imp_orig = {
        c: pct
        for c, pct in base_pct.items()
        if pct < 0.5 and c in original_cols
    }

    return verdicts, low_imp_orig
