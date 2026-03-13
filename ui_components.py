"""
ui_components.py — Streamlit UI Rendering Functions
====================================================
Contains all reusable rendering helpers for the Feature Engineering
Recommender. These functions are stateless with respect to business logic —
they only read from st.session_state and call st.* widget functions.

Imported by recommend_app.py — no imports back to recommend_app.
"""

import streamlit as st
import pandas as pd

from app_constants import (
    _GROUPS_BY_ID,
    _IMBALANCE_SEVERE,
    METHOD_DESCRIPTIONS,
    _CUSTOM_METHODS,
)

# =============================================================================
# GROUPED SUGGESTIONS UI HELPERS
# =============================================================================

def _suggestion_label(s):
    """One-line label for a suggestion dict."""
    col = s.get("column", "")
    col_b = s.get("column_b")
    if s["type"] == "row":
        col = "(all numeric cols)"
    elif col_b:
        col = f"{col} × {col_b}"
    delta = s.get("predicted_delta_raw", s.get("predicted_delta", 0))
    sign = "+" if delta >= 0 else ""
    return col, s["method"], f"{sign}{delta:.4f}", s.get("description", ""), delta


def _delta_color(delta):
    return "#3fb950" if delta >= 0.002 else ("#d29922" if delta >= 0 else "#f85149")


def _on_suggest_change(ck, persist_ck):
    """on_change callback: immediately persists a suggestion checkbox's value.

    Fired by Streamlit at interaction time, BEFORE the next script re-run,
    so the persistent key is always up-to-date even when the widget is
    subsequently hidden by "Show fewer" and Streamlit removes the widget key.

    Note: expander open/closed state is now managed natively via st.expander(key=),
    so no manual expander_key manipulation is needed here.
    """
    st.session_state[persist_ck] = st.session_state.get(ck, False)
    # Record that the user has explicitly touched this checkbox so the restore
    # logic in _render_group_card can safely trust the persist key going forward
    # (important for the imbalance suggestion, which has auto_checked=False and
    # must not be silently reset by an erroneous persist-key write).
    _idx = ck.replace("suggest_check_", "")
    st.session_state[f"_ck_user_touched_{_idx}"] = True


@st.fragment
def _render_group_card(group_id, group_suggs, all_suggestions):
    """Render one collapsible problem-group card with per-suggestion checkboxes.

    Decorated with @st.fragment so that interactions inside one card (checkbox
    toggles, show-more, info buttons) only rerun this card — not the whole page.
    Combined with st.expander(key=...), the open/closed state is fully persistent
    and independent of every other group card.
    """
    g = _GROUPS_BY_ID[group_id]

    cols_affected = list(dict.fromkeys(
        s["column"] for s in group_suggs
        if s.get("column") and s["column"] != "(all numeric cols)" and s["type"] != "row"
    ))

    # Summary stats for the expander label
    # Use individual _ck_persist_{idx} keys — never cleared by Streamlit (not widget keys)
    n_checked  = sum(1 for s in group_suggs
                     if st.session_state.get(
                         f"_ck_persist_{all_suggestions.index(s)}",
                         s.get('auto_checked', True)
                     ))
    best_delta = max(
        (s.get("predicted_delta_raw", s.get("predicted_delta", 0)) for s in group_suggs),
        default=0,
    )
    best_str   = f"{best_delta:+.4f}" if best_delta else ""
    pinned     = any(s.get("pinned") for s in group_suggs)

    # The label must be stable across reruns so Streamlit can track the
    # expanded/collapsed state by widget identity (label-based in older Streamlit).
    # n_checked is intentionally excluded here — it changes on every checkbox
    # toggle, which would make Streamlit treat the expander as a new widget and
    # reset it to collapsed every time. We show the count inside instead.
    label = (
        f"{g['icon']} **{g['title']}** "
        f"— {len(group_suggs)} suggestion{'s' if len(group_suggs) != 1 else ''}"
        + (f", best Δ {best_str}" if best_str else "")
        + (" 📌 recommended" if pinned else "")
    )

    _expander_open_key = f"_expander_open_{group_id}"
    # key= lets Streamlit natively persist open/closed state across reruns,
    # including plain manual clicks — no callback needed.
    # Seed False on first appearance so groups start collapsed.
    if _expander_open_key not in st.session_state:
        st.session_state[_expander_open_key] = False
    with st.expander(label, key=_expander_open_key):
        # Show the selection count at the top of the body (stable label means
        # this won't cause the collapse-on-click bug).
        _ck_color = "#3fb950" if n_checked else "#8b949e"
        _ck_label = f"✅ {n_checked} selected" if n_checked else "☐ none selected"
        st.markdown(
            f"<p style='font-size:0.78rem;color:{_ck_color};"
            f"margin:0 0 8px 0'>{_ck_label}</p>",
            unsafe_allow_html=True,
        )
        # ── Special imbalance explanation block ───────────────────────────────
        if group_id == 'imbalance' and group_suggs:
            imb_sug   = group_suggs[0]
            ratio     = imb_sug.get('imbalance_ratio', 0)
            dom_frac  = imb_sug.get('dominant_frac', 0)
            n_cls     = imb_sug.get('n_classes_imb', 2)
            strategy  = imb_sug.get('imbalance_strategy', 'none')
            is_binary = (n_cls == 2)

            # ── What was detected ─────────────────────────────────────────────
            if strategy == 'low':
                det_color = '#3fb950'   # green — low imbalance
            elif ratio >= _IMBALANCE_SEVERE:
                det_color = '#f0883e'
            else:
                det_color = '#d29922'
            _imb_label = f"Low {'binary' if is_binary else f'{n_cls}-class'}" if strategy == 'low' else ('Binary' if is_binary else f'{n_cls}-class')
            st.markdown(
                f"<div style='padding:10px 14px;border-left:3px solid {det_color};"
                f"background:#1a1009;border-radius:4px;margin-bottom:10px'>"
                f"<b>Detected:</b> {_imb_label} imbalance — "
                f"<b>{ratio:.1f}:1</b> max/min ratio &nbsp;·&nbsp; "
                f"dominant class makes up <b>{dom_frac:.0%}</b> of training data"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── What the enhanced model will do ──────────────────────────────
            if strategy == 'low':
                _low_param = '`is_unbalance=True`' if is_binary else '`class_weight=balanced`'
                st.markdown(
                    f"ℹ️ **Low class imbalance — unlikely to have a strong effect**\n\n"
                    f"Your class distribution is relatively balanced ({ratio:.1f}:1). "
                    f"Class reweighting ({_low_param}) is available but **not pre-enabled** "
                    f"because the model will likely handle this distribution well on its own.\n\n"
                    f"You can still enable it below if you notice that minority-class "
                    f"recall or F1 is lower than expected after training."
                )
            elif strategy == 'binary':
                st.markdown(
                    "⚠️ **Auto-decision: unchecked by default — train without it first**\n\n"
                    "Class reweighting (`is_unbalance=True`) is available but **not pre-enabled**. "
                    "Many binary classifiers already handle imbalance well without intervention — "
                    "applying reweighting when the model is already strong can *hurt* AUC and "
                    "calibration by distorting the learned decision boundary.\n\n"
                    "**Recommended workflow:**\n"
                    "1. Train the baseline first (checkbox unchecked)\n"
                    "2. If **recall, precision, or F1 are low** (e.g. < 0.80) on the validation "
                    "or test set, come back and enable this\n"
                    "3. If all metrics are already strong (≥ 0.90), there is no need — "
                    "the model is already handling the imbalance on its own"
                )
            elif strategy == 'multiclass_moderate':
                st.markdown(
                    "✅ **Auto-decision: apply `class_weight='balanced'`**\n\n"
                    "Moderate multiclass imbalance — sklearn will compute per-class "
                    "weights inversely proportional to frequency. At this ratio the "
                    "correction is gentle enough to help without distorting the "
                    "dominant class."
                )
            else:
                st.markdown(
                    f"⚠️ **Auto-decision: skip class reweighting**\n\n"
                    f"Your imbalance is severe for a multiclass problem "
                    f"({ratio:.1f}:1, dominant class {dom_frac:.0%}). "
                    f"`class_weight='balanced'` would assign ~{ratio:.0f}× more "
                    f"weight to minority classes — the model would essentially "
                    f"ignore the dominant class and collapse to noise-fitting on "
                    f"the rare classes (this is the exact scenario you observed).\n\n"
                    f"**What to try instead:**\n"
                    f"- Leave reweighting **unchecked** (default) — LightGBM with "
                    f"default settings handles this well.\n"
                    f"- Focus on **feature engineering** to help the model distinguish "
                    f"minority classes from the dominant one.\n"
                    f"- If minority class recall is critical, evaluate with macro-F1 "
                    f"rather than accuracy or ROC-AUC and accept lower accuracy."
                )
            st.markdown(
                "<span style='color:#8b949e;font-size:0.82rem'>"
                "You can override this decision by ticking/unticking the checkbox below — "
                "it only affects the **enhanced model**; the baseline is always trained "
                "without any reweighting."
                "</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")
        # ── Standard explanation for all other groups ─────────────────────────
        else:
            explain = g["explain"]
            if cols_affected:
                sample  = cols_affected[:4]
                col_str = ", ".join(f"`{c}`" for c in sample)
                if len(cols_affected) > 4:
                    col_str += f" (+{len(cols_affected) - 4} more)"
                explain += f"\n\n**Columns flagged:** {col_str}"
            st.markdown(explain)
        st.markdown("")

        # Column headers
        hc = st.columns([0.4, 2.2, 1.8, 1.1, 3.5])
        for col_widget, lbl in zip(hc, ["", "Column(s)", "Method", "Δ AUC", "Description"]):
            col_widget.markdown(
                f"<span style='font-size:0.72rem;color:#8b949e;"
                f"text-transform:uppercase;letter-spacing:0.07em'>{lbl}</span>",
                unsafe_allow_html=True,
            )

        # "Show more" toggle for large groups
        _show_all_key = f"_show_all_{group_id}"
        _INITIAL_SHOW = 5
        show_all = st.session_state.get(_show_all_key, False)
        visible_suggs = group_suggs if show_all else group_suggs[:_INITIAL_SHOW]

        for s in visible_suggs:
            idx = all_suggestions.index(s)
            col_disp, method, delta_str, desc, delta_val = _suggestion_label(s)
            is_custom = s.get("custom", False)
            is_pinned = s.get("pinned", False)
            is_cyclical_component = method in (
                'date_cyclical_month', 'date_cyclical_dow',
                'date_cyclical_dom',   'date_cyclical_hour',
            )

            check_col, col_col, meth_col, delta_col, desc_col = st.columns([0.4, 2.2, 1.8, 1.1, 3.5])

            ck          = f"suggest_check_{idx}"
            _persist_ck = f"_ck_persist_{idx}"

            # Always restore ck from the persistent key before rendering.
            # _ck_persist_ is the single source of truth: it is seeded at
            # analysis time and updated ONLY by _on_suggest_change when the
            # user explicitly toggles the checkbox.  We must NOT write back
            # from ck → _ck_persist_ at render time because Streamlit can
            # reinitialise a widget that was hidden (not rendered last run) to
            # its component default (False), which would silently corrupt the
            # persisted selection.
            #
            # Extra guard for imbalance suggestions: the persist key can be
            # seeded to False (auto_checked=False for binary) and an erroneous
            # external write (e.g. a pre-fix auto-deselect) could leave it False
            # even after the user checked it.  We only trust the persist key for
            # imbalance once the user has explicitly interacted (_ck_user_touched_).
            _is_imbalance = s.get('type') == 'imbalance'
            _user_touched = st.session_state.get(f"_ck_user_touched_{idx}", False)
            if _is_imbalance and not _user_touched:
                # Not yet touched by the user — use the seeded auto_checked value
                # directly so a stale persist key can't override it.
                st.session_state[ck] = s.get("auto_checked", False)
            else:
                st.session_state[ck] = st.session_state.get(
                    _persist_ck, s.get("auto_checked", True)
                )

            check_col.checkbox(
                " ", key=ck, label_visibility="collapsed",
                on_change=_on_suggest_change,
                args=(ck, _persist_ck),
            )

            badge = ""
            if is_pinned:
                badge += " <span style='font-size:0.65rem;color:#d29922'>📌</span>"
            if is_custom:
                badge += " <span style='font-size:0.65rem;color:#a371f7'>[custom]</span>"
            col_col.markdown(
                f"<span style='font-family:monospace;font-size:0.8rem;color:#e6edf3'>"
                f"{col_disp}</span>{badge}",
                unsafe_allow_html=True,
            )
            meth_col.markdown(
                f"<span style='font-size:0.8rem;color:#79c0ff'>{method}</span>",
                unsafe_allow_html=True,
            )
            delta_col.markdown(
                f"<span style='font-family:monospace;font-weight:700;"
                f"font-size:0.88rem;color:{_delta_color(delta_val)}'>{delta_str}</span>",
                unsafe_allow_html=True,
            )
            # For cyclical-component suggestions, show a compact (ℹ️) expand toggle
            # instead of the long inline guidance block.
            if is_cyclical_component:
                _cyclical_labels = {
                    'date_cyclical_month': ('🔵 Month cyclical', '#1a3a5c',
                        "Wraps month 1–12 so **Dec and Jan are adjacent**. "
                        "Enable for seasonal calendar patterns: retail, energy, weather, tax."),
                    'date_cyclical_dow': ('🟢 Day-of-week cyclical', '#1a3a2a',
                        "Wraps Mon–Sun so **Sunday and Monday are adjacent**. "
                        "Enable for weekly rhythms: footfall, fraud rates, support volume."),
                    'date_cyclical_dom': ('🟡 Day-of-month cyclical', '#2a2a10',
                        "Wraps day 1–31 so **the 31st and 1st are adjacent**. "
                        "Enable for billing cycles, salary dates, month-end reporting."),
                    'date_cyclical_hour': ('🟠 Hour cyclical', '#2a1a0a',
                        "Wraps 0–23 so **23:00 and 00:00 are adjacent**. "
                        "Enable for intra-day patterns: rush hour, overnight fraud, shift rhythms."),
                }
                lbl, bg, guidance = _cyclical_labels.get(method, (method, '#1a1a2a', ''))
                # Show a compact short label; guidance is behind an info expander
                desc_col.markdown(
                    f"<span style='font-size:0.78rem;color:#8b949e'>{lbl}</span>",
                    unsafe_allow_html=True,
                )
                _info_key = f"_cycl_info_{idx}"
                if st.session_state.get(_info_key, False):
                    st.markdown(
                        f"<div style='margin-left:28px;margin-bottom:4px;padding:6px 12px;"
                        f"border-left:3px solid #444;background:{bg};border-radius:4px;"
                        f"font-size:0.78rem;color:#c9d1d9'>"
                        f"<b>{lbl}</b> — {guidance}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("▲ Hide", key=f"_cycl_hide_{idx}", width="content"):
                        st.session_state[_info_key] = False
                        st.rerun()
                else:
                    if st.button("ℹ️ When to use", key=f"_cycl_show_{idx}", width="content"):
                        st.session_state[_info_key] = True
                        st.rerun()
            else:
                desc_col.markdown(
                    f"<span style='font-size:0.78rem;color:#8b949e'>{desc}</span>",
                    unsafe_allow_html=True,
                )
                # For text_stats, add per-field checkboxes below the row
                if method == 'text_stats' and s.get('text_stat_fields') is not None:
                    _TEXT_STAT_FIELDS = [
                        ('word_count',      'Word count'),
                        ('char_count',      'Char count'),
                        ('avg_word_len',    'Avg word length'),
                        ('uppercase_ratio', 'Uppercase %'),
                        ('digit_ratio',     'Digit %'),
                        ('punct_ratio',     'Punctuation %'),
                    ]
                    _parent_checked = st.session_state.get(ck, True)
                    # When parent is unchecked, auto-deselect all sub-fields
                    if not _parent_checked:
                        for _fk, _ in _TEXT_STAT_FIELDS:
                            _fkey = f"_text_stat_{idx}_{_fk}"
                            st.session_state[_fkey] = False

                    _stat_cols_row = st.columns([1] * len(_TEXT_STAT_FIELDS))
                    st.caption(
                        "Select which text statistics to compute. "
                        "word_count & char_count are fast and broadly useful; "
                        "uppercase_ratio, digit_ratio and punct_ratio help with "
                        "reviews, product descriptions and mixed-content fields."
                    )
                    _selected_fields = []
                    for _fi, (_fk, _fl) in enumerate(_TEXT_STAT_FIELDS):
                        _fkey = f"_text_stat_{idx}_{_fk}"
                        if _fkey not in st.session_state:
                            st.session_state[_fkey] = True
                        _fval = _stat_cols_row[_fi].checkbox(
                            _fl, key=_fkey, disabled=not _parent_checked,
                        )
                        if _fval and _parent_checked:
                            _selected_fields.append(_fk)
                    # Update the suggestion's field list so fit_and_apply uses it
                    s['text_stat_fields'] = _selected_fields if _selected_fields else [_TEXT_STAT_FIELDS[0][0]]

                # For date_features, add per-component granularity checkboxes
                if method == 'date_features':
                    _col_type_d = s.get('col_type', 'datetime')
                    # Build available sub-features based on col_type
                    _DATE_SUB_FEATS = []
                    if _col_type_d != 'time':
                        _DATE_SUB_FEATS += [
                            ('year',         'Year',          '📅'),
                            ('month',        'Month',         '🗓️'),
                            ('day',          'Day of month',  '🔢'),
                            ('dayofweek',    'Day of week',   '📆'),
                            ('is_weekend',   'Is weekend',    '🏖️'),
                            ('weekofyear',   'Week of year',  '📊'),
                            ('quarter',      'Quarter',       '🗃️'),
                            ('days_since_min','Days since min','⏳'),
                        ]
                    if _col_type_d in ('time', 'datetime'):
                        _DATE_SUB_FEATS += [('hour', 'Hour', '🕐')]

                    _parent_checked = st.session_state.get(ck, True)
                    _d_feat_keys = {_fk: f"_date_feat_{idx}_{_fk}" for _fk, _, _ in _DATE_SUB_FEATS}

                    # When parent unchecked → deselect all
                    if not _parent_checked:
                        for _fk in _d_feat_keys:
                            st.session_state[_d_feat_keys[_fk]] = False
                    else:
                        # When parent becomes checked again, restore defaults if all were cleared
                        _all_false = all(
                            not st.session_state.get(_d_feat_keys[_fk], True)
                            for _fk in _d_feat_keys
                        )
                        if _all_false:
                            for _fk in _d_feat_keys:
                                st.session_state[_d_feat_keys[_fk]] = True

                    st.markdown(
                        "<span style='font-size:0.75rem;color:#8b949e'>"
                        "📐 **Select sub-features to extract:**</span>",
                        unsafe_allow_html=True,
                    )
                    _n_date_cols = min(4, len(_DATE_SUB_FEATS))
                    _d_rows = [_DATE_SUB_FEATS[i:i+_n_date_cols]
                                for i in range(0, len(_DATE_SUB_FEATS), _n_date_cols)]
                    _selected_date_feats = []
                    for _drow in _d_rows:
                        _dcols = st.columns([1] * _n_date_cols + [1] * (_n_date_cols - len(_drow)))
                        for _dci, (_fk, _fl, _icon) in enumerate(_drow):
                            _dk = _d_feat_keys[_fk]
                            if _dk not in st.session_state:
                                st.session_state[_dk] = True
                            # Apply any pending auto-deselect (set before widget instantiation)
                            _pending_ds = st.session_state.get('_pending_date_deselect', {})
                            if idx in _pending_ds and _fk in _pending_ds[idx]:
                                st.session_state[_dk] = False
                            _dval = _dcols[_dci].checkbox(
                                f"{_icon} {_fl}", key=_dk, disabled=not _parent_checked,
                            )
                            if _dval and _parent_checked:
                                _selected_date_feats.append(_fk)
                    # Clear any consumed pending deselections for this suggestion
                    _pending_ds = st.session_state.get('_pending_date_deselect', {})
                    if idx in _pending_ds:
                        del _pending_ds[idx]
                        st.session_state['_pending_date_deselect'] = _pending_ds
                    s['selected_date_features'] = _selected_date_feats if _selected_date_feats else None

                # For row_numeric_stats, add per-stat checkboxes
                if method == 'row_numeric_stats':
                    _ROW_STAT_FIELDS = [
                        ('row_mean',   'Mean',    '📊'),
                        ('row_median', 'Median',  '📍'),
                        ('row_sum',    'Sum',     '➕'),
                        ('row_std',    'Std dev', '〰️'),
                        ('row_min',    'Min',     '⬇️'),
                        ('row_max',    'Max',     '⬆️'),
                        ('row_range',  'Range',   '↕️'),
                    ]
                    _parent_checked = st.session_state.get(ck, True)
                    if not _parent_checked:
                        for _fk, _, _ in _ROW_STAT_FIELDS:
                            st.session_state[f"_row_stat_{idx}_{_fk}"] = False
                    else:
                        # When parent becomes checked again after being unchecked,
                        # auto-restore all sub-fields to True
                        _all_false = all(
                            not st.session_state.get(f"_row_stat_{idx}_{_fk}", True)
                            for _fk, _, _ in _ROW_STAT_FIELDS
                        )
                        if _all_false:
                            for _fk, _, _ in _ROW_STAT_FIELDS:
                                st.session_state[f"_row_stat_{idx}_{_fk}"] = True

                    st.markdown(
                        "<div style='background:#0d1117;border:1px solid #21262d;"
                        "border-radius:6px;padding:8px 12px;margin:4px 0 6px 0'>"
                        "<span style='font-size:0.75rem;color:#8b949e'>"
                        "📐 <b>Select row statistics to compute</b> — each adds one column per row across all numeric features:"
                        "</span></div>",
                        unsafe_allow_html=True,
                    )
                    _row_stat_cols = st.columns(len(_ROW_STAT_FIELDS))
                    _selected_row_stats = []
                    for _fi, (_fk, _fl, _icon) in enumerate(_ROW_STAT_FIELDS):
                        _rfkey = f"_row_stat_{idx}_{_fk}"
                        if _rfkey not in st.session_state:
                            st.session_state[_rfkey] = True   # all on by default
                        # Apply any pending auto-deselect staged by the auto-deselect
                        # button handler (mirrors the _pending_date_deselect pattern).
                        _pending_rs = st.session_state.get('_pending_row_stat_deselect', {})
                        if idx in _pending_rs and _fk in _pending_rs[idx]:
                            st.session_state[_rfkey] = False
                        _rfval = _row_stat_cols[_fi].checkbox(
                            f"{_icon} {_fl}", key=_rfkey, disabled=not _parent_checked,
                        )
                        if _rfval and _parent_checked:
                            _selected_row_stats.append(_fk)
                    # Clear any consumed pending deselections for this suggestion
                    _pending_rs = st.session_state.get('_pending_row_stat_deselect', {})
                    if idx in _pending_rs:
                        del _pending_rs[idx]
                        st.session_state['_pending_row_stat_deselect'] = _pending_rs
                    # Fallback: always keep at least one stat
                    s['selected_row_stats'] = _selected_row_stats if _selected_row_stats else ['row_sum', 'row_max', 'row_mean']

        # Show more / show less button
        if len(group_suggs) > _INITIAL_SHOW:
            _n_hidden = len(group_suggs) - _INITIAL_SHOW
            if show_all:
                if st.button(f"▲ Show fewer", key=f"_showmore_{group_id}"):
                    st.session_state[_show_all_key] = False
                    st.rerun()
            else:
                if st.button(f"▼ Show {_n_hidden} more", key=f"_showmore_{group_id}"):
                    st.session_state[_show_all_key] = True
                    st.rerun()


def _render_custom_step_adder(X_train, existing_suggestions):
    """
    Expander that lets the user define a custom preprocessing step and
    append it to st.session_state.suggestions.
    """
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols     = [c for c in X_train.columns if c not in numeric_cols]
    all_cols     = X_train.columns.tolist()

    with st.expander("➕ Add a custom preprocessing step"):
        st.markdown("Define your own transform — it will be appended to the suggestions "
                    "and auto-selected.")

        c1, c2 = st.columns(2)
        step_type = c1.selectbox(
            "Transform type",
            ["numerical", "categorical", "interaction", "row"],
            key="_custom_type",
        )
        available_methods = _CUSTOM_METHODS.get(step_type, [])
        step_method = c2.selectbox(
            "Method",
            available_methods,
            format_func=lambda m: METHOD_DESCRIPTIONS.get(m, m),
            key="_custom_method",
        )

        col_a, col_b_widget = st.columns(2)

        if step_type == "numerical":
            col_options = numeric_cols if numeric_cols else all_cols
            step_col = col_a.selectbox("Column", col_options, key="_custom_col_a")
            step_col_b = None
        elif step_type == "categorical":
            col_options = cat_cols if cat_cols else all_cols
            step_col = col_a.selectbox("Column", col_options, key="_custom_col_a")
            step_col_b = None
        elif step_type == "interaction":
            step_col   = col_a.selectbox("Column A", all_cols, key="_custom_col_a")
            step_col_b = col_b_widget.selectbox(
                "Column B",
                [c for c in all_cols if c != step_col] or all_cols,
                key="_custom_col_b",
            )
        else:  # row
            step_col   = "(all numeric cols)"
            step_col_b = None
            col_a.info("Row transforms apply to all numeric columns — no column selection needed.")

        if st.button("Add custom step", type="secondary"):
            new_s = {
                "type":                step_type,
                "column":              step_col,
                "column_b":            step_col_b,
                "method":              step_method,
                "predicted_delta":     0.0,
                "predicted_delta_raw": 0.0,
                "description":         METHOD_DESCRIPTIONS.get(step_method, step_method),
                "custom":              True,
                "auto_checked":        True,   # ensure re-init loop always selects custom steps
            }
            # Avoid exact duplicates
            already = any(
                s["type"] == new_s["type"]
                and s["column"] == new_s["column"]
                and s.get("column_b") == new_s.get("column_b")
                and s["method"] == new_s["method"]
                for s in existing_suggestions
            )
            if already:
                st.warning("This exact step is already in the list.")
            else:
                new_idx = len(st.session_state.suggestions)
                st.session_state.suggestions.append(new_s)
                st.session_state[f"suggest_check_{new_idx}"]          = True
                st.session_state[f"_ck_persist_{new_idx}"]            = True   # seed persist key
                st.session_state[f"_initial_auto_checked_{new_idx}"]  = True   # treat custom as always protected
                st.success(f"Added: **{step_method}** on `{step_col}`")
                st.rerun()


def _sidebar_progress(ss):
    """Render a visual step progress tracker in the sidebar."""
    steps = [
        ("① Upload Data",       ss.get("X_train") is not None),
        ("② Analyze & Suggest", bool(ss.get("suggestions"))),
        ("③ Train Models",      ss.get("baseline_model") is not None),
        ("④ Test & Compare",    bool(ss.get("_test_baseline_metrics"))),
        ("⑤ Analysis & Recs",   bool(ss.get("_test_baseline_metrics"))),
    ]
    active_idx = next((i for i, (_, done) in enumerate(steps) if not done), len(steps))
    st.markdown("**Progress**")
    for i, (label, done) in enumerate(steps):
        if done:
            icon, color = "✅", "#3fb950"
        elif i == active_idx:
            icon, color = "▶️", "#58a6ff"
        else:
            icon, color = "🔒", "#8b949e"
        st.markdown(
            f"<span style='color:{color};font-size:0.85rem'>{icon} {label}</span>",
            unsafe_allow_html=True,
        )


def _render_locked_step(message: str) -> None:
    """Render a subtle 'step not yet unlocked' placeholder."""
    st.markdown(
        f"<div style='padding:14px 18px;background:#0d1117;border:1px dashed #30363d;"
        f"border-radius:6px;color:#484f58;font-size:0.87rem;margin:8px 0'>"
        f"🔒&ensp;{message}"
        f"</div>",
        unsafe_allow_html=True,
    )