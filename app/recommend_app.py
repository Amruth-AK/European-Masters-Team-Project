"""
recommend_app.py — Feature Engineering Recommendation Tool (UI only)
=====================================================================

Streamlit application that:
1. Accepts a training CSV + target column
2. Analyzes the dataset via trained meta-models
3. Suggests preprocessing transforms ranked by predicted impact
4. Trains two LightGBM models (baseline vs. enhanced)
5. Accepts a test CSV and compares both models

All core computation is in the `mlcompass` library; this file contains only
the Streamlit UI orchestration (the main() function).

Usage:
    streamlit run recommend_app.py
"""

import os
import re
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import importlib.resources
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- mlcompass library (all pure computation) ---
from mlcompass import (
    ensure_numeric_target,
    sanitize_feature_names,
    get_column_type_info,
    load_meta_models,
    generate_suggestions,
    deduplicate_suggestions,
    recommended_top_k,
    fit_and_apply_suggestions,
    apply_fitted_to_test,
    prepare_data_for_model,
    train_lgbm_model,
    evaluate_on_set,
    predict_on_set,
    _compute_suggestion_verdicts,
    _metrics_at_threshold,
    _find_optimal_thresholds,
    _override_options_for,
    _validate_col_override,
)
from mlcompass.constants import (
    BASE_PARAMS,
    METHOD_DESCRIPTIONS,
    _SUGGESTION_GROUPS,
    _METHOD_TO_GROUP,
    _IMBALANCE_MODERATE,
)
from mlcompass.analysis.profiling import _COL_TYPE_ICONS

# --- Streamlit-specific UI ---
from ui_components import (
    _suggestion_label,
    _delta_color,
    _on_suggest_change,
    _render_group_card,
    _render_custom_step_adder,
    _sidebar_progress,
    _render_locked_step,
)
from chat_component import render_chat_sidebar

# Wrap load_meta_models with Streamlit caching so it's only called once
load_meta_models = st.cache_resource(load_meta_models)


def main():
    st.set_page_config(
        page_title="ML Compass",
        page_icon="🧭",
        layout="wide",
    )

    # ── Inject subtle style improvements + scroll-position preserver ────────
    st.markdown("""
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        div[data-testid="stMetricDelta"] svg { display: none; }
        /* Prevent Streamlit from greying out the UI during script reruns */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main .block-container,
        .stApp { opacity: 1 !important; transition: none !important; }
        /* Hide the "running" top-bar decoration flash */
        [data-testid="stStatusWidget"] { visibility: hidden; }
        </style>
        <script>
        (function () {
          /* Preserve the main-panel scroll position across Streamlit reruns.
             Streamlit's own JS resets scroll on every rerun; we save it to
             sessionStorage before the reset and restore it 120 ms later. */
          const KEY = '__st_scroll_y';
          function getMain() {
            return (
              document.querySelector('[data-testid="stMain"]') ||
              document.querySelector('section.main')
            );
          }
          function save() {
            const el = getMain();
            if (el) sessionStorage.setItem(KEY, el.scrollTop);
          }
          function restore() {
            const el = getMain();
            const saved = parseInt(sessionStorage.getItem(KEY) || '0', 10);
            if (el && saved > 0) el.scrollTop = saved;
          }
          /* Attach scroll listener once main element is in DOM. */
          function attachListener() {
            const el = getMain();
            if (!el) { setTimeout(attachListener, 80); return; }
            el.addEventListener('scroll', save, { passive: true });
          }
          /* On every script execution (= every Streamlit rerun) restore scroll
             after a short delay so Streamlit's own reset fires first. */
          restore();
          setTimeout(restore, 120);
          attachListener();

          /* ── Sidebar scroll preservation ──────────────────────────────
             Approach: use a single, stable scrollable element and restore
             its scroll position after every Streamlit rerun.
             Key insight: on resize, Streamlit replaces sidebar DOM but the
             scrollable container selector stays predictable. We use a 
             MutationObserver only to detect when the container is replaced,
             then re-attach and immediately restore. */
          const SB_KEY = '__st_sidebar_scroll_y';
          let _sbLastSaved = 0;
          let _sbRestoreTimer = null;
          let _sbListener = null;

          function getSidebarScroller() {
            // Walk up from stSidebarContent to find the actual scrolling container
            const candidates = [
              document.querySelector('section[data-testid="stSidebar"] > div:first-child'),
              document.querySelector('[data-testid="stSidebarContent"]'),
              document.querySelector('[data-testid="stSidebar"] > div'),
            ];
            for (const el of candidates) {
              if (el && el.scrollHeight > el.clientHeight) return el;
            }
            return candidates.find(Boolean) || null;
          }

          function saveSidebarScroll() {
            const sb = getSidebarScroller();
            if (sb && sb.scrollTop > 0) {
              _sbLastSaved = sb.scrollTop;
              try { sessionStorage.setItem(SB_KEY, _sbLastSaved); } catch(e) {}
            }
          }

          function restoreSidebarScroll() {
            const saved = parseInt(sessionStorage.getItem(SB_KEY) || '0', 10);
            if (saved <= 0) return;
            const sb = getSidebarScroller();
            if (sb) {
              sb.scrollTop = saved;
              // Double-tap after layout settles
              setTimeout(() => { const s2 = getSidebarScroller(); if (s2) s2.scrollTop = saved; }, 80);
            }
          }

          function attachSidebarListener() {
            const sb = getSidebarScroller();
            if (!sb) return;
            if (_sbListener) sb.removeEventListener('scroll', _sbListener);
            _sbListener = () => saveSidebarScroll();
            sb.addEventListener('scroll', _sbListener, { passive: true });
            restoreSidebarScroll();
          }

          // Watch the sidebar root for DOM changes (resize / rerun replaces content)
          const _sidebarRoot = document.querySelector('[data-testid="stSidebar"]');
          if (_sidebarRoot) {
            new MutationObserver(() => {
              // Debounce: wait until DOM settles before re-attaching
              clearTimeout(_sbRestoreTimer);
              _sbRestoreTimer = setTimeout(() => {
                attachSidebarListener();
              }, 50);
            }).observe(_sidebarRoot, { childList: true, subtree: false });
          }

          restoreSidebarScroll();
          setTimeout(attachSidebarListener, 100);
          setTimeout(restoreSidebarScroll, 300);
        })();
        </script>
    """, unsafe_allow_html=True)

    st.title("🧭 ML Compass")
    st.caption("Upload a classification dataset → get data-driven transform suggestions → compare baseline vs. enhanced LightGBM model")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown("""
**What it does:** This tool uses trained *meta-models* — models that have learned which feature engineering transforms tend to help on datasets with similar characteristics — to recommend preprocessing steps tailored to your specific dataset.

**The workflow:**
1. **Upload** your training CSV and pick the target column
2. **Analyze** — meta-models score each potential transform (log-transform, frequency encoding, interaction features, etc.) against your dataset's properties
3. **Review** the ranked suggestions, check/uncheck what to apply
4. **Train** a baseline LightGBM and an enhanced one (with your selected transforms), then compare validation metrics
5. **Test** — upload a held-out test CSV to get a clean head-to-head comparison

**Meta-models directory:** Set this in the sidebar to point at your trained models (run `train_meta_models.py` to generate them).
        """)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        _default_model_dir = str(importlib.resources.files("mlcompass") / "data" / "meta_models")
        model_dir = st.text_input("Meta-models directory", value=_default_model_dir)

        # Dynamic default: scale with dataset rows AND columns
        _ds = st.session_state.get('X_train')
        _auto_k = recommended_top_k(_ds)

        # Reset slider to new auto value when a fresh dataset is loaded
        _top_k_sig = f"{getattr(_ds, 'shape', None)}"
        if st.session_state.get('_top_k_shape_sig') != _top_k_sig:
            st.session_state['_top_k_shape_sig'] = _top_k_sig
            st.session_state['_top_k_slider']    = _auto_k

        top_k = st.slider(
            "Pre-select top N suggestions", 1, 30,
            key='_top_k_slider',
            help=(
                f"Auto-set to **{_auto_k}** based on dataset size "
                + "("
                + (f"{_ds.shape[0]:,} rows × {_ds.shape[1]} cols" if hasattr(_ds, 'shape') else "? rows × ? cols")
                + "). Adjust as needed."
            ),
        )
        delta_threshold = st.number_input(
            "Confidence filter",
            value=0.0, step=0.05, format="%.2f",
            help="Hides suggestions below this z-score rank within their type. "
                 "0 = show all above-average, 1 = top ~16% only. "
                 "Raw Δ AUC values are shown in the results table.",
        )
        st.divider()
        st.markdown("**⚡ Quick Mode**")
        _quick_mode = st.toggle(
            "Sample rows for faster analysis",
            value=st.session_state.get('_quick_mode', False),
            key='_quick_mode',
            help=(
                "Sub-samples the dataset before running meta-feature analysis "
                "(correlations, MI scores, landmarking). Useful for wide/large "
                "datasets where the Analyze step is slow. The sample is used only "
                "for *analysis*; training always uses the full data unless you "
                "explicitly opt in below."
            ),
        )
        if _quick_mode:
            _xt = st.session_state.get('X_train')
            _n_full = _xt.shape[0] if _xt is not None else 0
            # Dynamic default: 10% of the dataset, clamped between 2 000 and 20 000.
            # This keeps the sample large enough for stable meta-feature estimates
            # while scaling sensibly with dataset size.
            #   20 k rows  → 2 000 (lower clamp)
            #   50 k rows  → 5 000
            #   100 k rows → 10 000
            #   300 k rows → 20 000 (upper clamp)
            _default_n = min(20_000, max(2_000, _n_full // 10))
            st.number_input(
                "Sample size (rows)",
                min_value=500, max_value=max(50_000, _n_full),
                value=st.session_state.get('_quick_n', _default_n), step=500,
                key='_quick_n',
                help=(
                    f"Default is 10% of your dataset ({_default_n:,} rows), "
                    f"clamped between 2 000 and 20 000. "
                    "Increase for more accurate meta-feature estimates; "
                    "decrease for faster analysis on very wide datasets."
                ),
            )
            st.toggle(
                "Also train on sample (not full data)",
                value=st.session_state.get('_quick_train_sample', False),
                key='_quick_train_sample',
                help="If on, both analysis AND model training use the sample. Faster but less representative.",
            )

        st.divider()
        _sidebar_progress(st.session_state)
        st.divider()
        if st.button("🔄 Reset session", help="Clear all state and start over", use_container_width=True):
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()
        render_chat_sidebar(st.session_state)

    # Load meta-models
    if os.path.isdir(model_dir):
        meta_models = load_meta_models(model_dir)
        if meta_models:
            st.sidebar.success(f"✅ Models loaded: {', '.join(meta_models.keys())}")
        else:
            st.sidebar.warning("No model files found in that directory.")
            meta_models = {}
    else:
        with st.sidebar.expander("⚠️ Setup required", expanded=True):
            st.warning(
                f"Directory **`{model_dir}`** not found.  \n"
                "Run `train_meta_models.py` to generate models, "
                "then point the path above to that directory."
            )
        meta_models = {}

    # --- Session state init ---
    for key in ['train_df', 'target_col', 'X_train', 'y_train', 'suggestions',
                'selected_indices', 'baseline_model', 'enhanced_model',
                'baseline_train_cols', 'enhanced_train_cols', 'fitted_params',
                'n_classes', 'label_encoder', 'baseline_val_metrics', 'enhanced_val_metrics',
                'baseline_col_encoders', 'enhanced_col_encoders',
                'skipped_info', 'advisories', 'X_train_enhanced',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_baseline_metrics', '_test_enhanced_metrics',
                '_test_file_name']:
        if key not in st.session_state:
            st.session_state[key] = None

    # =========================================================================
    # STEP 1: UPLOAD TRAINING DATA
    # =========================================================================
    st.header("① Upload Training Data")

    uploaded_train = st.file_uploader("Upload training CSV", type=['csv'], key='train_upload')

    if uploaded_train is not None:
        try:
            try:
                df = pd.read_csv(uploaded_train, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_train.seek(0)
                df = pd.read_csv(uploaded_train, encoding='latin-1')
        except Exception as _parse_err:
            st.error(
                f"**Could not read your file:** {_parse_err}  \n"
                "Make sure it's a valid CSV with a header row and comma-separated values."
            )
            st.stop()
        df = sanitize_feature_names(df)
        st.session_state.train_df = df

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            st.dataframe(df.head(5), use_container_width=True, height=200)

        # Target selection — auto-detect by common name, fallback to last column
        _col_list = df.columns.tolist()
        _TARGET_NAMES = {
            'target', 'label', 'labels', 'class', 'classes',
            'y', 'outcome', 'response', 'output', 'result',
            'dependent', 'dependent_variable', 'churn', 'fraud',
            'survived', 'survival', 'defaulted', 'default',
        }
        _auto_target_idx = len(_col_list) - 1  # fallback: last column
        for _ci, _cn in enumerate(_col_list):
            if _cn.lower().strip().replace('-', '_').replace(' ', '_') in _TARGET_NAMES:
                _auto_target_idx = _ci
                break
        _auto_detected = (_auto_target_idx < len(_col_list) - 1)
        target_col = st.selectbox(
            "Select target column",
            options=_col_list,
            index=_auto_target_idx,
            help=(
                f"🎯 Auto-detected from column name: **`{_col_list[_auto_target_idx]}`**"
                if _auto_detected
                else "Defaulted to the last column — adjust if needed."
            ),
        )
        if _auto_detected and target_col == _col_list[_auto_target_idx]:
            st.caption(f"🎯 Target auto-detected: `{target_col}`")
        st.session_state.target_col = target_col

        y_raw = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode target
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        st.session_state.label_encoder = le
        n_classes = y.nunique()
        st.session_state.n_classes = n_classes

        st.write(f"**Task**: {'Binary' if n_classes == 2 else 'Multiclass'} classification "
                 f"({n_classes} classes)")

        class_dist = y_raw.value_counts()
        # Colour-coded bar chart: largest class grey, minority classes amber/red
        try:
            import plotly.graph_objects as go
            _max_count = class_dist.max()
            _bar_colors = [
                "#3fb950" if v == _max_count else ("#f0883e" if v >= _max_count * 0.2 else "#f85149")
                for v in class_dist.values
            ]
            _fig_dist = go.Figure(go.Bar(
                x=[str(c) for c in class_dist.index],
                y=class_dist.values,
                marker_color=_bar_colors,
                text=[f"{v:,}" for v in class_dist.values],
                textposition="outside",
            ))
            _fig_dist.update_layout(
                margin=dict(t=10, b=10, l=0, r=0),
                height=180,
                xaxis_title="Class",
                yaxis_title="Count",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=11),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#30363d"),
            )
            st.plotly_chart(_fig_dist, use_container_width=True)
        except Exception:
            st.bar_chart(class_dist, height=150)

        # Only (re-)set X_train from the raw file when no column changes have been
        # applied yet.  Once Apply Changes has run, X_train holds the modified
        # dataset and must not be overwritten by the raw file on subsequent reruns.
        if not st.session_state.get('_col_type_applied', False):
            st.session_state.X_train = X
        st.session_state.y_train = y

        # Snapshot raw X for reset capability (once per unique dataset)
        _raw_sig = f"{X.shape}__{list(X.columns)}"
        if st.session_state.get('_raw_sig') != _raw_sig:
            st.session_state['X_train_raw']     = X
            st.session_state['_raw_sig']         = _raw_sig
            st.session_state['_applied_drops']   = []
            st.session_state['_applied_types']   = {}
            st.session_state['_col_type_applied'] = False
            # Clear all downstream state so the new dataset starts fresh
            for _stale_key in [
                'suggestions', 'skipped_info', 'advisories', 'selected_indices',
                'baseline_model', 'enhanced_model',
                'baseline_train_cols', 'enhanced_train_cols',
                'baseline_val_metrics', 'enhanced_val_metrics',
                'baseline_col_encoders', 'enhanced_col_encoders',
                'fitted_params', 'X_train_enhanced',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_baseline_metrics', '_test_enhanced_metrics',
                '_test_file_sig', '_test_file_name',
                # Presentation / report keys that would show stale data
                '_val_metrics_rows', '_report_class_dist',
                '_fi_b_pct', '_fi_e_original', '_fi_e_new',
                '_fi_orig_pct', '_fi_new_pct',
                '_suggestion_verdicts', '_low_imp_cols', '_verdicts_stale',
                '_analyze_baseline_score', '_analyze_baseline_std',
                'apply_imbalance',
                '_optuna_best_params', '_optuna_best_score',
            ]:
                st.session_state[_stale_key] = None

    # =========================================================================
    # COLUMN TYPE REVIEW — detect → review → Apply Changes → then Analyze
    # =========================================================================
    if st.session_state.X_train is not None:
        X_raw = st.session_state.get('X_train_raw', st.session_state.X_train)

        # Detect types on raw X, cache by signature
        _xt_sig_ct = f"{X_raw.shape}__{list(X_raw.columns[:5])}"
        if st.session_state.get('_col_type_sig') != _xt_sig_ct:
            with st.spinner("Detecting column types…"):
                st.session_state['_col_type_info'] = get_column_type_info(X_raw)
                st.session_state['_col_type_sig']  = _xt_sig_ct
                # Bump version so per-row widget keys reset on new dataset
                st.session_state['_col_type_ver'] = (
                    st.session_state.get('_col_type_ver', 0) + 1
                )

        col_type_info    = st.session_state.get('_col_type_info', {})
        _ct_ver          = st.session_state.get('_col_type_ver', 0)
        _already_applied = st.session_state.get('_col_type_applied', False)
        _applied_drops   = st.session_state.get('_applied_drops', [])
        _applied_types   = st.session_state.get('_applied_types', {})

        # ── Expander title ────────────────────────────────────────────────────
        _n_drop_sug  = sum(1 for v in col_type_info.values() if v['drop_suggested'])
        _type_counts = {}
        for v in col_type_info.values():
            _type_counts[v['detected']] = _type_counts.get(v['detected'], 0) + 1
        _summary_parts = []
        if _already_applied and (_applied_drops or _applied_types):
            _summary_parts.append("✅ Changes applied")
        elif _n_drop_sug:
            _summary_parts.append(f"⚠️ {_n_drop_sug} drop suggestion{'s' if _n_drop_sug!=1 else ''}")
        for _t in ('Free Text', 'Date', 'DateTime', 'Time only', 'Day-of-Week'):
            if _type_counts.get(_t):
                _summary_parts.append(f"{_COL_TYPE_ICONS.get(_t,'')} {_type_counts[_t]} {_t}")
        _ct_label = "🔬 Column Types & Drop Suggestions"
        if _summary_parts:
            _ct_label += "  —  " + "  ·  ".join(_summary_parts)
        _ct_label += "  *(optional)*"

        with st.expander(_ct_label, expanded=bool(_n_drop_sug)):
            # ── Applied state banner + preview ────────────────────────────────
            if _already_applied and (_applied_drops or _applied_types):
                _banner_parts = []
                if _applied_drops:
                    _banner_parts.append(
                        f"**{len(_applied_drops)}** column{'s' if len(_applied_drops)!=1 else ''} dropped"
                    )
                if _applied_types:
                    _banner_parts.append(
                        f"**{len(_applied_types)}** type override{'s' if len(_applied_types)!=1 else ''}"
                    )
                st.success("Changes applied: " + "  ·  ".join(_banner_parts) +
                           ". Click **Analyze Dataset** below to regenerate suggestions.")

                # ── Preview of the working dataset ───────────────────────────
                X_working_preview = st.session_state['X_train']
                _prev_cols = st.columns([1, 3])
                _prev_cols[0].metric("Remaining columns",
                                     f"{X_working_preview.shape[1]} / {X_raw.shape[1]}")
                _prev_cols[0].metric("Rows", X_working_preview.shape[0])
                with _prev_cols[1]:
                    st.caption("Working dataset preview (first 5 rows):")
                    st.dataframe(X_working_preview.head(5), use_container_width=True,
                                 height=195)

                if st.button("↩️ Reset to original columns", key="_ct_reset_btn"):
                    st.session_state['X_train']           = st.session_state['X_train_raw']
                    st.session_state['_applied_drops']    = []
                    st.session_state['_applied_types']    = {}
                    st.session_state['_col_type_applied'] = False
                    st.session_state['_col_type_ver']    += 1   # reset row widget keys
                    st.rerun()
                st.divider()

            st.caption(
                "Auto-detected column types below. Use **Override** to correct a misdetection, "
                "or tick **Drop** to exclude a column. Override options are filtered to "
                "what's valid for each column's data type. "
                "Click **Apply Changes** to update the working dataset before analyzing."
            )

            # ── Pre-initialise ALL widget keys so hidden columns keep their state ──
            for _col_init, _info_init in col_type_info.items():
                _k_ovr  = f"_ct_ovr_{_col_init}_v{_ct_ver}"
                _k_drop = f"_ct_drop_{_col_init}_v{_ct_ver}"
                if _k_ovr  not in st.session_state:
                    st.session_state[_k_ovr]  = 'Auto'
                if _k_drop not in st.session_state:
                    st.session_state[_k_drop] = _info_init['drop_suggested']

            # ── Compute pending state from ALL columns (reads previous-run values) ──
            _pending_drops = []
            _pending_types = {}
            _val_issues    = []
            for _col_p, _info_p in col_type_info.items():
                _dv = st.session_state.get(f"_ct_drop_{_col_p}_v{_ct_ver}",
                                           _info_p['drop_suggested'])
                _ov = st.session_state.get(f"_ct_ovr_{_col_p}_v{_ct_ver}", 'Auto')
                if _dv:
                    _pending_drops.append(_col_p)
                elif _ov != 'Auto':
                    _pending_types[_col_p] = _ov
                    _res = _validate_col_override(_col_p, _ov, _info_p, X_raw)
                    if _res:
                        _val_issues.append((_col_p, _ov, _res[0], _res[1]))

            _has_pending = bool(_pending_drops or _pending_types)
            _has_errors  = any(s == 'error' for *_, s, _ in _val_issues)
            _parts: list = []
            if _pending_drops:
                _parts.append(f"drop **{len(_pending_drops)}** column{'s' if len(_pending_drops)!=1 else ''}")
            if _pending_types:
                _parts.append(f"**{len(_pending_types)}** type override{'s' if len(_pending_types)!=1 else ''}")

            def _do_apply_changes():
                _X_work = st.session_state['X_train_raw'].copy()
                if _pending_drops:
                    _X_work = _X_work.drop(
                        columns=[c for c in _pending_drops if c in _X_work.columns]
                    )
                st.session_state['X_train']           = _X_work
                st.session_state['_applied_drops']    = _pending_drops
                st.session_state['_applied_types']    = _pending_types
                st.session_state['_col_type_applied'] = True
                st.session_state.suggestions      = None
                st.session_state.skipped_info     = None
                st.session_state.selected_indices = None
                st.rerun()

            # ── Apply Changes button — TOP ────────────────────────────────────
            _ap_top = st.columns([3, 1])
            _ap_top[0].caption(
                ("Pending: " + ", ".join(_parts) + ".") if _has_pending
                else "No changes pending — all columns use auto-detected types."
            )
            if _ap_top[1].button("✅ Apply Changes", type="primary",
                                  disabled=(not _has_pending or _has_errors),
                                  key="_ct_apply_btn_top"):
                _do_apply_changes()

            st.markdown("<hr style='margin:6px 0 10px 0;border-color:#30363d'>",
                        unsafe_allow_html=True)

            # ── Search + Tag filter ───────────────────────────────────────────
            _tag_type_counts: dict = {}
            for _ti in col_type_info.values():
                _td = _ti['detected']
                _tag_type_counts[_td] = _tag_type_counts.get(_td, 0) + 1
            _n_drop_total = sum(1 for _v in col_type_info.values() if _v['drop_suggested'])

            _TAG_DEFS = [
                ('⚠️ Drop suggested', '__drop__'),
                ('📝 Free Text',      'Free Text'),
                ('📅 Date',           'Date'),
                ('🕐 DateTime',       'DateTime'),
                ('⏱️ Time only',      'Time only'),
                ('📆 Day-of-Week',    'Day-of-Week'),
                ('🔢 Numerical',      'Numerical'),
                ('🏷️ Categorical',    'Categorical'),
                ('⚡ Binary',         'Binary'),
                ('🆔 ID',             'ID'),
                ('⚫ Constant',       'Constant'),
            ]
            _available_tags = []
            for _tlabel, _tkey in _TAG_DEFS:
                _cnt = _n_drop_total if _tkey == '__drop__' else _tag_type_counts.get(_tkey, 0)
                if _cnt > 0:
                    _available_tags.append(f"{_tlabel} ({_cnt})")

            _filt_row = st.columns([2, 4])
            with _filt_row[0]:
                _search_val = st.text_input(
                    "_ct_search_lbl", key="_ct_search",
                    placeholder="🔍  Search column name…",
                    label_visibility="collapsed",
                )
            with _filt_row[1]:
                _active_tags = st.multiselect(
                    "_ct_tags_lbl", options=_available_tags,
                    key="_ct_tags",
                    placeholder="🏷️  Filter by type tag — none = show all…",
                    label_visibility="collapsed",
                )

            # Map active tag labels back to type keys
            _active_tag_keys: set = set()
            _filter_drop_flag = False
            for _atag in _active_tags:
                for _tlabel, _tkey in _TAG_DEFS:
                    if _atag.startswith(_tlabel):
                        if _tkey == '__drop__':
                            _filter_drop_flag = True
                        else:
                            _active_tag_keys.add(_tkey)

            def _col_visible(col, info):
                if _search_val and _search_val.lower() not in col.lower():
                    return False
                if not _active_tags:
                    return True  # no filter → show all
                if _filter_drop_flag and info['drop_suggested']:
                    return True
                return info['detected'] in _active_tag_keys

            # ── Sort: problematic / special columns first ─────────────────────
            def _col_sort_key(item):
                _c, _i = item
                _d = _i['detected']
                if   _d == 'Constant':                                    _r = 0
                elif _d == 'ID':                                          _r = 1
                elif _i['drop_suggested']:                                _r = 2
                elif _d == 'Free Text':                                   _r = 3
                elif _d in ('Date', 'DateTime', 'Time only', 'Day-of-Week'): _r = 4
                elif _d == 'Binary':                                      _r = 5
                elif _d == 'Categorical':                                 _r = 6
                else:                                                     _r = 7
                return (_r, _c)

            _sorted_items = sorted(col_type_info.items(), key=_col_sort_key)
            _n_visible    = sum(1 for _c, _i in _sorted_items if _col_visible(_c, _i))
            _n_total      = len(_sorted_items)

            if _n_visible < _n_total:
                st.caption(
                    f"Showing **{_n_visible}** of {_n_total} columns "
                    f"— clear filters above to see all."
                )

            # ── Per-row table (manual layout so each row gets filtered options) ──
            _hc = st.columns([2.0, 1.8, 2.2, 0.7, 0.8, 0.9, 2.0, 2.5])
            for _hl, _hcol in zip(
                ["Column", "Detected", "Override", "Drop", "Unique", "Missing", "Sample", "Drop Reason"],
                _hc
            ):
                _hcol.markdown(
                    f"<span style='font-size:0.72rem;color:#8b949e;"
                    f"text-transform:uppercase;letter-spacing:0.06em'>{_hl}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<hr style='margin:4px 0 6px 0;border-color:#30363d'>",
                        unsafe_allow_html=True)

            for col, info in _sorted_items:
                if not _col_visible(col, info):
                    continue

                _row = st.columns([2.0, 1.8, 2.2, 0.7, 0.8, 0.9, 2.0, 2.5])
                _name_color = "color:#f0883e;" if info['drop_suggested'] else ""
                _row[0].markdown(
                    f"<span style='font-family:monospace;font-size:0.82rem;{_name_color}'>`{col}`</span>",
                    unsafe_allow_html=True,
                )
                _row[1].markdown(
                    f"<span style='font-size:0.82rem'>{info['icon']} {info['detected']}</span>",
                    unsafe_allow_html=True,
                )

                # Filtered override options — no temporal/text for numeric columns
                _opts    = _override_options_for(info)
                _ovr_key  = f"_ct_ovr_{col}_v{_ct_ver}"
                _drop_key = f"_ct_drop_{col}_v{_ct_ver}"

                _ovr_val  = _row[2].selectbox(
                    "##", options=_opts, key=_ovr_key,
                    label_visibility="collapsed",
                )
                _drop_val = _row[3].checkbox(
                    "##", key=_drop_key,
                    label_visibility="collapsed",
                )

                _row[4].markdown(
                    f"<span style='font-size:0.82rem'>{info['n_unique']}</span>",
                    unsafe_allow_html=True,
                )
                _row[5].markdown(
                    f"<span style='font-size:0.82rem'>{info['missing_pct']*100:.0f}%</span>",
                    unsafe_allow_html=True,
                )
                _row[6].markdown(
                    f"<span style='font-size:0.78rem;color:#8b949e'>{info['sample'][:30]}</span>",
                    unsafe_allow_html=True,
                )
                if info['drop_reason']:
                    _row[7].markdown(
                        f"<span style='font-size:0.75rem;color:#f0883e'>{info['drop_reason']}</span>",
                        unsafe_allow_html=True,
                    )

            # ── Validation messages ───────────────────────────────────────────
            if _val_issues:
                st.markdown("")
                for _vc, _vt, _vs, _vm in _val_issues:
                    if _vs == 'error':
                        st.error(f"**`{_vc}` → {_vt}:** {_vm}", icon="🚫")
                    else:
                        st.warning(f"**`{_vc}` → {_vt}:** {_vm}", icon="⚠️")

            # ── Apply Changes button — BOTTOM ─────────────────────────────────
            st.markdown("")
            _ap_cols = st.columns([3, 1])
            _ap_cols[0].caption(
                ("Pending: " + ", ".join(_parts) + ".") if _has_pending
                else "No changes pending — all columns use auto-detected types."
            )
            if _ap_cols[1].button("✅ Apply Changes", type="primary",
                                   disabled=(not _has_pending or _has_errors),
                                   key="_ct_apply_btn"):
                _do_apply_changes()

    # =========================================================================
    # STEP 2: ANALYZE & SUGGEST
    # =========================================================================
    st.divider()
    st.header("② Analyze & Get Suggestions")
    if st.session_state.X_train is not None:

        if not meta_models:
            st.warning("No meta-models loaded. Please check the model directory.")
        else:
            _analyze_cols = st.columns([2, 1])
            if _analyze_cols[0].button("🔍 Analyze Dataset", type="primary"):
                # Clear stale MODEL results only — deliberately keep suggestions/
                # selected_indices alive so Step ③ stays visible while the new
                # analysis runs (prevents the "Step 3 disappears" flash).
                for _stale_k in [
                    'baseline_model', 'enhanced_model',
                    'baseline_train_cols', 'enhanced_train_cols',
                    'baseline_val_metrics', 'enhanced_val_metrics',
                    'baseline_col_encoders', 'enhanced_col_encoders',
                    'fitted_params', 'X_train_enhanced',
                    'X_test_raw', 'X_test_enhanced', '_test_df_original',
                    '_test_baseline_metrics', '_test_enhanced_metrics',
                    '_test_file_sig', '_test_file_name',
                    '_suggestion_verdicts', '_low_imp_cols',
                    '_val_metrics_rows', '_fi_b_pct', '_fi_e_original',
                    '_fi_e_new', '_fi_orig_pct', '_fi_new_pct',
                    'apply_imbalance',
                ]:
                    st.session_state[_stale_k] = None
                # Note: suggestions / selected_indices are NOT cleared here.
                # They will be overwritten below once the new analysis finishes.

                X = st.session_state.X_train
                y = st.session_state.y_train
                n_classes = st.session_state.n_classes

                # ── Quick-mode sub-sampling ──────────────────────────────────
                _qm_on      = st.session_state.get('_quick_mode', False)
                _qm_n       = int(st.session_state.get('_quick_n', max(2_000, len(X) // 10)))
                _qm_sampled = False          # did we actually subsample?
                X_for_analysis, y_for_analysis = X, y
                if _qm_on and len(X) > _qm_n:
                    # Use stratified sampling to preserve class balance
                    try:
                        from sklearn.model_selection import train_test_split as _tts
                        _keep_frac = _qm_n / len(X)
                        _, _qm_X_s, _, _qm_y_s = _tts(
                            X, y, test_size=_keep_frac, random_state=42,
                            stratify=y if y.value_counts().min() >= 2 else None,
                        )
                        X_for_analysis = _qm_X_s.reset_index(drop=True)
                        y_for_analysis = _qm_y_s.reset_index(drop=True)
                    except Exception:
                        # Fallback to random sample if stratification fails
                        _qm_idx        = X.sample(n=_qm_n, random_state=42).index
                        X_for_analysis = X.loc[_qm_idx].reset_index(drop=True)
                        y_for_analysis = y.loc[_qm_idx].reset_index(drop=True)
                    _qm_sampled    = True
                    st.info(
                        f"⚡ Quick mode — analyzing **{_qm_n:,}** of **{len(X):,}** rows "
                        f"({_qm_n/len(X)*100:.0f}%) with stratified sampling (class balance preserved). "
                        "Dataset-level shape (n_rows, row_col_ratio) uses the real count "
                        "so meta-model predictions are not skewed."
                    )
                # ─────────────────────────────────────────────────────────────
                # Compute quick baseline on the analysis subset so meta-models
                # receive the real baseline_score / headroom.
                with st.spinner("Computing quick baseline..."):
                    try:
                        _stratify = y_for_analysis if y_for_analysis.value_counts().min() >= 2 else None
                        if _stratify is None:
                            st.warning("Some classes have only 1 sample — stratified split disabled for quick baseline.")
                        _X_tr, _X_vl, _y_tr, _y_vl = train_test_split(
                            X_for_analysis, y_for_analysis, test_size=0.2, random_state=42, stratify=_stratify
                        )
                        _X_tr_enc, _X_vl_enc, _ = prepare_data_for_model(_X_tr, _X_vl)
                        _quick = lgb.LGBMClassifier(**{**BASE_PARAMS, 'n_estimators': 100})
                        _quick.fit(_X_tr_enc, _y_tr,
                                   eval_set=[(_X_vl_enc, _y_vl)],
                                   callbacks=[lgb.early_stopping(10, verbose=False)])
                        if n_classes == 2:
                            _bs = roc_auc_score(_y_vl, _quick.predict_proba(_X_vl_enc)[:, 1])
                        else:
                            _bs = roc_auc_score(_y_vl, _quick.predict_proba(_X_vl_enc),
                                                multi_class='ovr', average='weighted')
                        _bs_std = float(abs(min(_bs - 0.5, 1.0 - _bs)) * 0.1 + 0.01)
                    except Exception as _e:
                        st.warning(f"Quick baseline failed ({_e}), defaulting to 0.5")
                        _bs, _bs_std = 0.5, 0.05
                st.session_state['_analyze_baseline_score'] = _bs
                st.session_state['_analyze_baseline_std']   = _bs_std
                st.info(f"Quick baseline ROC-AUC: **{_bs:.4f}**")

                progress_bar = st.progress(0, text="Generating suggestions...")

                def update_progress(pct):
                    progress_bar.progress(min(pct, 1.0), text=f"Analyzing... {pct*100:.0f}%")

                suggestions, skipped_info, advisories, ds_meta = generate_suggestions(
                    X_for_analysis, y_for_analysis, meta_models,
                    baseline_score=_bs,
                    baseline_std=_bs_std,
                    progress_cb=update_progress,
                    type_reassignments=st.session_state.get('_applied_types', {}),
                    real_n_rows=len(X) if _qm_sampled else None,
                )
                progress_bar.empty()

                # Filter and deduplicate
                suggestions = [s for s in suggestions if s['predicted_delta'] >= delta_threshold]
                suggestions = deduplicate_suggestions(suggestions)

                st.session_state.suggestions = suggestions
                st.session_state.skipped_info = skipped_info
                st.session_state.advisories = advisories
                st.session_state.ds_meta = ds_meta
                st.success(f"Generated {len(suggestions)} suggestions")
                # Inform the user that interaction search is capped so they
                # know the list may not be exhaustive on wide datasets.
                _n_interact = sum(1 for s in suggestions if s['type'] == 'interaction')
                if _n_interact > 0:
                    st.caption(
                        f"ℹ️ Interaction search is capped at the top-30 num×num, "
                        f"20 cat×num, and 10 cat×cat pairs (ranked by baseline importance). "
                        f"On wide datasets some column pairs may not have been evaluated."
                    )

        # Display suggestions
        if st.session_state.suggestions:
            suggestions = st.session_state.suggestions

            # ── Reset analysis button ─────────────────────────────────────────
            if _analyze_cols[1].button("🔄 Reset Analysis", help="Clear suggestions and start fresh"):
                for _k in ['suggestions', 'skipped_info', 'advisories', 'selected_indices',
                            'baseline_model', 'enhanced_model',
                            'baseline_train_cols', 'enhanced_train_cols',
                            'baseline_val_metrics', 'enhanced_val_metrics',
                            'baseline_col_encoders', 'enhanced_col_encoders',
                            'fitted_params', 'X_train_enhanced',
                            'X_test_raw', 'X_test_enhanced', '_test_df_original',
                            '_test_baseline_metrics', '_test_enhanced_metrics',
                            '_test_file_sig', '_test_file_name']:
                    st.session_state[_k] = None
                for _ek in list(st.session_state.keys()):
                    if _ek.startswith("_expander_open_"):
                        del st.session_state[_ek]
                st.rerun()
            # ─────────────────────────────────────────────────────────────────

            # ── Initialise checkbox states on new analysis ────────────────────
            # The signature captures the *analysis identity* (dataset + target +
            # the system-generated suggestions).  Custom steps added by the user
            # must NOT change it — otherwise every custom addition would reset
            # all user (de)selections.
            _n_system_suggestions = sum(
                1 for s in suggestions if not s.get('custom')
            )
            _sig = (
                f"{_n_system_suggestions}"
                f"__{suggestions[0]['method'] if suggestions else ''}"
                f"__{st.session_state.X_train.shape if st.session_state.X_train is not None else ''}"
                f"__{st.session_state.target_col or ''}"
            )
            if st.session_state.get("_suggestions_sig") != _sig:
                st.session_state._suggestions_sig = _sig
                default_k = min(top_k, len(suggestions))
                for i, s in enumerate(suggestions):
                    _raw = s.get("predicted_delta_raw", s.get("predicted_delta", 0))
                    val = s['auto_checked'] if 'auto_checked' in s else (i < default_k and _raw > 0)
                    st.session_state[f"suggest_check_{i}"]         = val
                    st.session_state[f"_ck_persist_{i}"]           = val   # seed persistent key
                    st.session_state[f"_initial_auto_checked_{i}"] = val   # immutable: was this system-ticked?
                # Reset group expansion so groups start collapsed for the new analysis
                for _ek in list(st.session_state.keys()):
                    if _ek.startswith("_expander_open_"):
                        del st.session_state[_ek]
            # ─────────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────

            # ── Dataset advisories ────────────────────────────────────────────
            advisories = st.session_state.get('advisories') or []
            if advisories:
                _SEVERITY_ICON  = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}
                _SEVERITY_COLOR = {'high': '#4a1010', 'medium': '#3d3010', 'low': '#0e2a3d'}
                for adv in advisories:
                    icon = _SEVERITY_ICON.get(adv['severity'], '💡')
                    _adv_cat = adv['category'].replace(' ', '_')
                    _adv_open_key = f"_adv_open_{_adv_cat}"
                    _adv_expanded = st.session_state.get(_adv_open_key, adv['severity'] == 'high')
                    with st.expander(f"{icon} **{adv['title']}**", expanded=_adv_expanded):
                        st.markdown(adv['detail'])
                        suggested = adv.get('suggested_params')
                        if suggested:
                            _hp_key = f"_custom_hp_{_adv_cat}"
                            if _hp_key not in st.session_state:
                                st.session_state[_hp_key] = BASE_PARAMS.copy()
                            st.markdown("**Model hyperparameters** *(used for the enhanced model)*")
                            _hp_btn_key = f"_hp_btn_{_adv_cat}"
                            if st.button(f"⚡ Apply suggested hyperparameters", key=_hp_btn_key):
                                merged = BASE_PARAMS.copy()
                                merged.update(suggested)
                                st.session_state[_hp_key] = merged
                                for _pk in ['num_leaves', 'min_child_samples', 'reg_alpha',
                                            'reg_lambda', 'subsample', 'colsample_bytree']:
                                    if _pk in merged:
                                        st.session_state[f"_hp_{_adv_cat}_{_pk}"] = merged[_pk]
                                st.session_state[_adv_open_key] = True
                                st.rerun()

                            _cur_hp = st.session_state[_hp_key]
                            _hp_cols = st.columns(3)
                            _editable_params = [
                                ('num_leaves',        'Num leaves',        int,   4,   256),
                                ('min_child_samples', 'Min child samples',  int,   1,   500),
                                ('reg_alpha',         'reg_alpha (L1)',     float, 0.0, 10.0),
                                ('reg_lambda',        'reg_lambda (L2)',    float, 0.0, 10.0),
                                ('subsample',         'Subsample',          float, 0.1, 1.0),
                                ('colsample_bytree',  'Colsample bytree',   float, 0.1, 1.0),
                            ]
                            for _pi, (_pk, _pl, _ptype, _pmin, _pmax) in enumerate(_editable_params):
                                _col = _hp_cols[_pi % 3]
                                _wkey = f"_hp_{_adv_cat}_{_pk}"
                                # Use session_state widget value if present (persists across reruns),
                                # otherwise fall back to the stored HP dict value
                                _default_val = st.session_state.get(_wkey, _cur_hp.get(_pk, BASE_PARAMS.get(_pk, _pmin)))
                                if _ptype == int:
                                    _new_val = _col.number_input(
                                        _pl, min_value=_pmin, max_value=_pmax,
                                        value=int(_default_val), step=1, key=_wkey,
                                    )
                                else:
                                    _new_val = _col.number_input(
                                        _pl, min_value=float(_pmin), max_value=float(_pmax),
                                        value=float(_default_val), step=0.05,
                                        format="%.2f", key=_wkey,
                                    )
                                st.session_state[_hp_key][_pk] = _new_val
            # ─────────────────────────────────────────────────────────────────

            # ── Quick stats row ───────────────────────────────────────────────
            n_pos = sum(
                1 for i, s in enumerate(suggestions)
                if s.get("predicted_delta_raw", s.get("predicted_delta", 0)) > 0
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total suggestions", len(suggestions))
            mc2.metric("Positive Δ AUC", n_pos)
            mc3.metric("Pre-selected", sum(
                st.session_state.get(f"_ck_persist_{i}", False)
                for i in range(len(suggestions))
            ))
            # ─────────────────────────────────────────────────────────────────

            # ── Bulk-action bar ───────────────────────────────────────────────
            _n_currently_selected = sum(
                st.session_state.get(f"_ck_persist_{i}", False)
                for i in range(len(suggestions))
            )
            _bulk_label_col, _bulk_btn_col = st.columns([5, 3])
            _bulk_label_col.markdown(
                f"<div style='padding:8px 0 2px 0'>"
                f"<span style='font-size:0.92rem;color:#e6edf3'>"
                f"<b>{_n_currently_selected}</b>"
                f"<span style='color:#8b949e'> of {len(suggestions)} transforms selected</span>"
                f"</span></div>",
                unsafe_allow_html=True,
            )

            # Buttons in their own styled row
            st.markdown(
                "<p style='font-size:0.75rem;color:#8b949e;"
                "text-transform:uppercase;letter-spacing:0.07em;"
                "margin:0 0 4px 0'>Quick selection</p>",
                unsafe_allow_html=True,
            )
            _bc1, _bc2, _bc3 = st.columns(3)
            if _bc1.button(
                "✅ Select All",
                help="Check every suggestion in the list",
                use_container_width=True,
            ):
                for i in range(len(suggestions)):
                    st.session_state[f"suggest_check_{i}"] = True
                    st.session_state[f"_ck_persist_{i}"]   = True
                st.rerun()
            if _bc2.button(
                "☐ Deselect All",
                help="Uncheck everything — start from a clean slate",
                use_container_width=True,
            ):
                for i in range(len(suggestions)):
                    st.session_state[f"suggest_check_{i}"] = False
                    st.session_state[f"_ck_persist_{i}"]   = False
                st.rerun()
            if _bc3.button(
                f"⭐ Top {top_k} Only",
                help=f"Select only the top {top_k} suggestions ranked by predicted Δ AUC (auto-sized from your dataset)",
                use_container_width=True,
                type="primary",
            ):
                _sorted_idxs = sorted(
                    range(len(suggestions)),
                    key=lambda i: suggestions[i].get("predicted_delta_raw", suggestions[i].get("predicted_delta", 0)),
                    reverse=True,
                )
                for i in range(len(suggestions)):
                    _val = i in _sorted_idxs[:top_k]
                    st.session_state[f"suggest_check_{i}"] = _val
                    st.session_state[f"_ck_persist_{i}"]   = _val
                st.rerun()
            # ─────────────────────────────────────────────────────────────────

            # ── Problem-grouped suggestion cards ─────────────────────────────
            st.markdown("### Suggestions by Problem Type")
            st.caption("Check or uncheck individual transforms. Custom steps can be added below.")

            # Bucket suggestions into groups (ungrouped methods fall into "other")
            groups_seen = {}
            ungrouped   = []
            for s in suggestions:
                gid = _METHOD_TO_GROUP.get(s["method"])
                if gid:
                    groups_seen.setdefault(gid, []).append(s)
                else:
                    ungrouped.append(s)

            # Render in defined order
            for g in _SUGGESTION_GROUPS:
                gid = g["id"]
                if gid not in groups_seen:
                    continue
                _render_group_card(gid, groups_seen[gid], suggestions)

            # Ungrouped fallback
            if ungrouped:
                st.markdown("**Other transforms**")
                # Custom steps first
                ungrouped.sort(key=lambda s: (not s.get('custom', False),))
                for s in ungrouped:
                    idx = suggestions.index(s)
                    col_disp, method, delta_str, desc, delta_val = _suggestion_label(s)
                    ck          = f"suggest_check_{idx}"
                    _persist_ck = f"_ck_persist_{idx}"
                    st.session_state[ck] = st.session_state.get(
                        _persist_ck, s.get('auto_checked', True)
                    )
                    cc, nc, mc, dc, dsc = st.columns([0.4, 2.2, 1.8, 1.1, 3.5])
                    cc.checkbox(
                        " ", key=ck, label_visibility="collapsed",
                        on_change=_on_suggest_change, args=(ck, _persist_ck),
                    )
                    nc.markdown(f"`{col_disp}`")
                    mc.markdown(f"<span style='color:#79c0ff;font-size:0.8rem'>{method}</span>",
                                unsafe_allow_html=True)
                    dc.markdown(
                        f"<span style='color:{_delta_color(delta_val)};font-weight:700;"
                        f"font-family:monospace'>{delta_str}</span>",
                        unsafe_allow_html=True,
                    )
                    dsc.markdown(f"<span style='color:#8b949e;font-size:0.78rem'>{desc}</span>",
                                 unsafe_allow_html=True)
                st.markdown("---")
            # ─────────────────────────────────────────────────────────────────

            # ── Custom step adder ─────────────────────────────────────────────
            _render_custom_step_adder(st.session_state.X_train, suggestions)
            # ─────────────────────────────────────────────────────────────────

            # ── Resolve selected_indices — read from persistent keys ──────────
            st.session_state.selected_indices = [
                i for i in range(len(suggestions))
                if st.session_state.get(f"_ck_persist_{i}", False)
            ]
            n_sel = len(st.session_state.selected_indices)
            if n_sel:
                st.info(f"**{n_sel} transform(s) selected** — scroll down to train models.")
            else:
                st.warning("No transforms selected. Check at least one suggestion above.")
            # ─────────────────────────────────────────────────────────────────
    else:
        _render_locked_step("Upload a training CSV in **①** to unlock this step.")

    # =========================================================================
    # STEP 3: TRAIN MODELS
    # =========================================================================
    st.divider()
    st.header("③ Train Baseline & Enhanced Models")
    if st.session_state.selected_indices is not None and st.session_state.suggestions:

        # ── Hyperparameters + Optuna (unified expander) ───────────────────────
        # Key design decision: number_inputs use NO `key=` argument.
        # Their values are read from / written to _user_hp_overrides (a plain
        # session_state dict).  This avoids Streamlit's "cannot modify widget
        # state after instantiation" error entirely — there are no widget-bound
        # keys to conflict with when Apply / Reset updates the dict and reruns.
        _HP_EDIT_KEY = "_user_hp_overrides"
        _ALL_EDITABLE_PARAMS = [
            # (param_key,          label,               type,  min,   max,   step,  help)
            ('n_estimators',      'N Estimators',       int,   50,    2000,  1,     'Number of boosting rounds'),
            ('learning_rate',     'Learning Rate',      float, 0.001, 0.5,   0.005, 'Step size shrinkage'),
            ('num_leaves',        'Num Leaves',         int,   4,     512,   1,     'Max leaves per tree'),
            ('max_depth',         'Max Depth',          int,   -1,    30,    1,     '-1 = unlimited'),
            ('subsample',         'Subsample',          float, 0.1,   1.0,   0.05,  'Row subsampling ratio'),
            ('colsample_bytree',  'Colsample Bytree',   float, 0.1,   1.0,   0.05,  'Feature subsampling ratio'),
            ('min_child_samples', 'Min Child Samples',  int,   1,     500,   1,     'Min samples per leaf'),
            ('reg_alpha',         'reg_alpha (L1)',      float, 0.0,   20.0,  0.1,   'L1 regularisation'),
            ('reg_lambda',        'reg_lambda (L2)',     float, 0.0,   20.0,  0.1,   'L2 regularisation'),
        ]
        if _HP_EDIT_KEY not in st.session_state:
            st.session_state[_HP_EDIT_KEY] = BASE_PARAMS.copy()

        with st.expander("⚙️ Hyperparameters & Optimization (Enhanced Model)", expanded=False):

            # ── Section 1: HP editor ─────────────────────────────────────────
            st.caption(
                "💡 **Recommended: keep the defaults** — they are well-tuned for most "
                "tabular classification tasks. The **baseline model always uses fixed "
                "defaults** so comparisons stay fair. Edit only if you have a specific "
                "reason, or use Optuna below to search automatically."
            )

            _hpc1, _hpc2 = st.columns([1, 5])
            _do_reset = _hpc1.button("↺ Reset to defaults", key="_hp_reset_btn")
            _hpc2.markdown(
                "<span style='color:#8b949e;font-size:0.82rem'>"
                "Defaults: n_estimators=300 · lr=0.05 · num_leaves=31 · "
                "max_depth=6 · subsample=0.8 · colsample_bytree=0.8"
                "</span>",
                unsafe_allow_html=True,
            )
            if _do_reset:
                st.session_state[_HP_EDIT_KEY] = BASE_PARAMS.copy()
                st.rerun()

            # Render number_inputs WITHOUT key= to avoid widget-state conflicts.
            # value= is always driven from our dict; return values are written back.
            _hp_grid = st.columns(3)
            for _pi, (_pk, _pl, _ptype, _pmin, _pmax, _pstep, _phelp) in enumerate(_ALL_EDITABLE_PARAMS):
                _cur_val = st.session_state[_HP_EDIT_KEY].get(_pk, BASE_PARAMS.get(_pk, _pmin))
                with _hp_grid[_pi % 3]:
                    if _ptype == int:
                        _ret = st.number_input(
                            _pl, min_value=_pmin, max_value=_pmax,
                            value=int(_cur_val), step=_pstep, help=_phelp,
                        )
                    else:
                        _ret = st.number_input(
                            _pl, min_value=float(_pmin), max_value=float(_pmax),
                            value=float(_cur_val), step=float(_pstep),
                            format="%.3f", help=_phelp,
                        )
                    st.session_state[_HP_EDIT_KEY][_pk] = _ret

            # Live diff banner
            _changed = {
                k: (BASE_PARAMS.get(k), st.session_state[_HP_EDIT_KEY].get(k))
                for k, *_ in _ALL_EDITABLE_PARAMS
                if st.session_state[_HP_EDIT_KEY].get(k) != BASE_PARAMS.get(k)
            }
            if _changed:
                _diff_parts = [f"`{k}`: {v[0]} → **{v[1]}**" for k, v in _changed.items()]
                st.info("📝 Modified from defaults: " + " · ".join(_diff_parts))
            else:
                st.success("✅ Using default parameters")

            # ── Section 2: Optuna ────────────────────────────────────────────
            st.divider()
            st.markdown("**🔍 Optuna Hyperparameter Search** *(optional)*")

            if not _OPTUNA_AVAILABLE:
                st.warning("Optuna is not installed. Run `pip install optuna` to enable.")
            else:
                st.caption(
                    "Runs a Bayesian (TPE) search on your **selected transforms** "
                    "(and class reweighting if enabled). "
                    "Hit **Apply** afterwards to load the result into the editor above."
                )

                _oc1, _oc2 = st.columns([1, 3])
                _n_optuna_trials = _oc1.number_input(
                    "Trials", min_value=5, max_value=500,
                    value=st.session_state.get("_optuna_n_trials_val", 30),
                    step=5,
                    help="More trials → better params, but slower. 20–50 is a good start.",
                )
                st.session_state["_optuna_n_trials_val"] = int(_n_optuna_trials)

                # Show previous result + Apply button (always visible once a run is done)
                if st.session_state.get('_optuna_best_params'):
                    _ob   = st.session_state['_optuna_best_params']
                    _oscr = st.session_state.get('_optuna_best_score', 0)
                    _oc2.success(
                        f"✅ Best ROC-AUC: **{_oscr:.4f}** · "
                        + ", ".join(f"`{k}={round(v, 4) if isinstance(v, float) else v}`"
                                    for k, v in _ob.items())
                    )
                    if st.button("📥 Apply Optuna params", key="_optuna_apply_btn", type="secondary"):
                        merged = BASE_PARAMS.copy()
                        merged.update(_ob)
                        # Simply overwrite our dict and rerun — no widget-key manipulation needed
                        st.session_state[_HP_EDIT_KEY] = merged
                        st.rerun()

                if st.button("⚡ Run Optuna Search", key="_optuna_run_btn", type="primary"):
                    _X_opt = st.session_state.get('X_train')
                    _y_opt = st.session_state.get('y_train')
                    if _X_opt is None or _y_opt is None:
                        st.warning("Upload a training CSV first.")
                    elif st.session_state.selected_indices is None or st.session_state.suggestions is None:
                        st.warning("Run **② Analyze Dataset** first so transforms are available.")
                    else:
                        _n_trials_run = int(st.session_state["_optuna_n_trials_val"])
                        _opt_prog  = st.progress(0.0, text="Preparing transformed data for Optuna…")
                        _opt_log   = st.empty()
                        _trial_log = []

                        # ── Resolve selected FE suggestions & imbalance ──────
                        _opt_selected = [st.session_state.suggestions[i]
                                         for i in st.session_state.selected_indices]
                        _opt_fe_suggestions = [s for s in _opt_selected
                                               if s.get('type') != 'imbalance']

                        # Resolve imbalance settings (mirrors training logic)
                        _opt_imb_suggestion = next(
                            (s for s in st.session_state.suggestions
                             if s.get('type') == 'imbalance'), None,
                        )
                        _opt_imb_idx = (
                            st.session_state.suggestions.index(_opt_imb_suggestion)
                            if _opt_imb_suggestion else None
                        )
                        _opt_user_wants_imb = (
                            st.session_state.get(
                                f"_ck_persist_{_opt_imb_idx}",
                                _opt_imb_suggestion.get('auto_checked', False))
                            if _opt_imb_suggestion else False
                        )
                        _opt_imb_strategy = (
                            _opt_imb_suggestion.get('imbalance_strategy', 'none')
                            if _opt_imb_suggestion else 'none'
                        )
                        _opt_apply_imbalance = (
                            _opt_user_wants_imb and _opt_imb_strategy != 'none'
                        )

                        # ── Pre-compute transformed data once (fixed split) ──
                        _yo = ensure_numeric_target(_y_opt.copy())
                        _strat = _yo if _yo.value_counts().min() >= 2 else None
                        _Xtr_o, _Xvl_o, _ytr_o, _yvl_o = train_test_split(
                            _X_opt.copy(), _yo, test_size=0.2,
                            random_state=42, stratify=_strat,
                        )

                        if _opt_fe_suggestions:
                            _Xtr_t, _opt_fitted = fit_and_apply_suggestions(
                                _Xtr_o, _ytr_o, _opt_fe_suggestions)
                            _Xvl_t = apply_fitted_to_test(_Xvl_o, _opt_fitted)
                        else:
                            _Xtr_t, _Xvl_t = _Xtr_o, _Xvl_o

                        _Xtr_enc, _Xvl_enc, _ = prepare_data_for_model(
                            _Xtr_t, _Xvl_t)
                        _opt_n_classes = int(_yo.nunique())

                        _opt_status_parts = []
                        if _opt_fe_suggestions:
                            _opt_status_parts.append(
                                f"{len(_opt_fe_suggestions)} transform(s)")
                        if _opt_apply_imbalance:
                            _opt_status_parts.append(
                                f"imbalance={_opt_imb_strategy}")
                        _opt_status = (
                            " + ".join(_opt_status_parts) if _opt_status_parts
                            else "raw features"
                        )
                        _opt_prog.progress(
                            0.0, text=f"Running Optuna on {_opt_status}…")

                        def _optuna_objective(trial):
                            _p = {
                                'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
                                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                                'num_leaves':        trial.suggest_int('num_leaves', 10, 200),
                                'max_depth':         trial.suggest_int('max_depth', 3, 12),
                                'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
                                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                                'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),
                                'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 5.0),
                                'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
                            }
                            # Apply imbalance handling if user enabled it
                            if _opt_apply_imbalance:
                                if _opt_imb_strategy in ('binary', 'low') and _opt_n_classes == 2:
                                    _p['is_unbalance'] = True
                                elif _opt_imb_strategy in ('multiclass_moderate', 'low'):
                                    _p['class_weight'] = 'balanced'
                            try:
                                _m = lgb.LGBMClassifier(**_p)
                                _m.fit(_Xtr_enc, _ytr_o,
                                       eval_set=[(_Xvl_enc, _yvl_o)],
                                       callbacks=[lgb.early_stopping(20, verbose=False)])
                                if _opt_n_classes == 2:
                                    return roc_auc_score(_yvl_o, _m.predict_proba(_Xvl_enc)[:, 1])
                                return roc_auc_score(_yvl_o, _m.predict_proba(_Xvl_enc),
                                                     multi_class='ovr', average='weighted')
                            except Exception:
                                return 0.0

                        def _optuna_cb(study, trial):
                            _frac = (trial.number + 1) / _n_trials_run
                            _opt_prog.progress(
                                min(_frac, 1.0),
                                text=f"Trial {trial.number + 1}/{_n_trials_run} "
                                     f"— best so far: {study.best_value:.4f} "
                                     f"({_opt_status})",
                            )
                            _trial_log.append(
                                f"Trial {trial.number + 1:03d}: {trial.value:.4f}"
                                + (" ★" if trial.value == study.best_value else "")
                            )
                            _opt_log.markdown(
                                "<details><summary style='font-size:0.8rem;color:#8b949e'>"
                                "Trial log</summary>"
                                "<pre style='font-size:0.75rem;max-height:160px;overflow:auto'>"
                                + "\n".join(_trial_log[-20:])
                                + "</pre></details>",
                                unsafe_allow_html=True,
                            )

                        _study = optuna.create_study(
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42),
                        )
                        _study.optimize(_optuna_objective, n_trials=_n_trials_run,
                                        callbacks=[_optuna_cb])

                        st.session_state['_optuna_best_params'] = _study.best_params
                        st.session_state['_optuna_best_score']  = _study.best_value
                        # Rerun immediately so the Apply button appears without extra clicks
                        st.rerun()
        # ─────────────────────────────────────────────────────────────────────

        if st.button("🚀 Train Both Models", type="primary"):
            # Clear any stale test results from a previous training run so that
            # step ④ starts fresh and doesn't show outdated comparisons.
            for _stale_k in [
                '_test_baseline_metrics', '_test_enhanced_metrics',
                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                '_test_file_sig', '_test_file_name',
                '_suggestion_verdicts',
            ]:
                st.session_state[_stale_k] = None

            X = st.session_state.X_train
            y = st.session_state.y_train
            n_classes = st.session_state.n_classes

            # ── Quick-mode: optionally train on sample too ────────────────────
            if (st.session_state.get('_quick_mode', False)
                    and st.session_state.get('_quick_train_sample', False)):
                _qm_n_tr = int(st.session_state.get('_quick_n', 5_000))
                if len(X) > _qm_n_tr:
                    _qm_tr_idx = X.sample(n=_qm_n_tr, random_state=42).index
                    X = X.loc[_qm_tr_idx].reset_index(drop=True)
                    y = y.loc[_qm_tr_idx].reset_index(drop=True)
                    st.info(f"⚡ Training on quick-mode sample ({_qm_n_tr:,} rows)")
            # ─────────────────────────────────────────────────────────────────

            selected_suggestions = [st.session_state.suggestions[i]
                                     for i in st.session_state.selected_indices]

            # Feature engineering suggestions only (imbalance is a model param, not a transform)
            fe_suggestions = [s for s in selected_suggestions if s.get('type') != 'imbalance']

            # Resolve imbalance strategy: start from the suggestion's auto-detection
            # (which is already encoded in the suggestion metadata), then respect
            # the user's checkbox override.
            _imb_suggestion = next(
                (s for s in st.session_state.suggestions if s.get('type') == 'imbalance'),
                None,
            )
            _imb_idx = (
                st.session_state.suggestions.index(_imb_suggestion)
                if _imb_suggestion else None
            )
            # User checked/unchecked the box?  Absent → fall back to auto_checked default.
            _user_wants_imbalance = (
                st.session_state.get(f"_ck_persist_{_imb_idx}",
                                     _imb_suggestion.get('auto_checked', False))
                if _imb_suggestion else False
            )
            _imbalance_strategy = (
                _imb_suggestion.get('imbalance_strategy', 'none')
                if _imb_suggestion else 'none'
            )
            apply_imbalance     = _user_wants_imbalance and _imbalance_strategy != 'none'
            st.session_state['apply_imbalance'] = apply_imbalance  # persisted for post-training analysis

            # Compute ratio/dominant_frac for the caption (may differ from suggestion
            # if the dataset was changed without re-analyzing).
            _y_counts_enh        = pd.Series(y).value_counts()
            _imbalance_ratio_enh = float(_y_counts_enh.max() / max(_y_counts_enh.min(), 1))
            _dominant_class_frac = float(_y_counts_enh.max() / len(y))

            # Resolve effective model params for the enhanced model.
            # Priority (lowest → highest):
            #   BASE_PARAMS → advisory panel overrides → always-visible HP editor
            _effective_params = BASE_PARAMS.copy()
            # Advisory panels (small dataset, high dimensionality)
            for _adv_key in ['_custom_hp_Small_Dataset', '_custom_hp_High_Dimensionality']:
                if _adv_key in st.session_state and st.session_state[_adv_key]:
                    _effective_params.update(st.session_state[_adv_key])
            # Always-visible HP editor is the final word
            if st.session_state.get('_user_hp_overrides'):
                _effective_params.update(st.session_state['_user_hp_overrides'])

            # Split for validation
            # Store class distribution snapshot for the report
            try:
                st.session_state['_report_class_dist'] = {
                    str(k): int(v)
                    for k, v in pd.Series(y).value_counts().sort_index().items()
                }
            except Exception:
                pass

            _stratify = y if y.value_counts().min() >= 2 else None
            if _stratify is None:
                st.warning("Some classes have only 1 sample — stratified split disabled.")
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=_stratify
            )

            # --- Compute baseline score ---
            with st.spinner("Computing baseline score...", show_time=True):
                try:
                    X_tr_enc, X_vl_enc, _ = prepare_data_for_model(X_tr, X_vl)
                    quick_model = lgb.LGBMClassifier(**{**BASE_PARAMS, 'n_estimators': 100})
                    quick_model.fit(X_tr_enc, y_tr, eval_set=[(X_vl_enc, y_vl)],
                                    callbacks=[lgb.early_stopping(10, verbose=False)])
                    if n_classes == 2:
                        baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc)[:, 1])
                    else:
                        baseline_score = roc_auc_score(y_vl, quick_model.predict_proba(X_vl_enc),
                                                        multi_class='ovr', average='weighted')
                except Exception as e:
                    st.error(f"Baseline failed: {e}")
                    baseline_score = 0.5
            st.info(f"Quick baseline ROC-AUC: **{baseline_score:.4f}**")

            # --- Apply transforms ---
            with st.spinner(f"Applying {len(fe_suggestions)} transforms...", show_time=True):
                X_tr_enh, fitted_params = fit_and_apply_suggestions(X_tr, y_tr, fe_suggestions)
                X_vl_enh = apply_fitted_to_test(X_vl, fitted_params)

                # The enhanced model is trained with transforms fitted on X_tr.
                # We use those same fitted_params at test time so the feature
                # statistics (medians, encoding maps, …) always match what the
                # model saw during training.
                st.session_state.fitted_params = fitted_params

                # Fit on the full dataset solely to produce the download artefact
                # (a fully-transformed version of the training data the user can
                # export).  This is NOT used for test evaluation.
                X_full_enh, _ = fit_and_apply_suggestions(X, y, fe_suggestions)
                st.session_state.X_train_enhanced = X_full_enh  # saved for download

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Baseline Model")
                st.caption("No class reweighting, no feature engineering")
                with st.spinner("Training baseline...", show_time=True):
                    baseline_model, baseline_cols, baseline_col_enc = train_lgbm_model(
                        X_tr, y_tr, X_vl, y_vl, n_classes, apply_imbalance=False,
                        base_params=BASE_PARAMS.copy(),
                    )
                    baseline_metrics = evaluate_on_set(baseline_model, X_vl, y_vl, baseline_cols, n_classes, baseline_col_enc)
                st.session_state.baseline_model = baseline_model
                st.session_state.baseline_train_cols = baseline_cols
                st.session_state.baseline_val_metrics = baseline_metrics

            with col2:
                enh_label_parts = [f"+{len(fe_suggestions)} transforms"]
                if apply_imbalance:
                    enh_label_parts.append("+ class reweighting (auto)")
                if _effective_params != BASE_PARAMS:
                    enh_label_parts.append("+ custom HP")
                st.subheader(f"Enhanced Model ({', '.join(enh_label_parts)})")
                if apply_imbalance:
                    _strat_label = '`is_unbalance=True`' if _imbalance_strategy in ('binary',) or (_imbalance_strategy == 'low' and n_classes == 2) else '`class_weight=balanced`'
                    st.caption(
                        f"Class imbalance **auto-detected** ({_imbalance_ratio_enh:.1f}:1, "
                        f"dominant class {_dominant_class_frac:.0%}) — "
                        f"{_strat_label} applied automatically"
                    )
                elif n_classes > 2 and _imbalance_ratio_enh >= _IMBALANCE_MODERATE:
                    st.caption(
                        f"⚠️ Severe multiclass imbalance detected ({_imbalance_ratio_enh:.1f}:1, "
                        f"dominant class {_dominant_class_frac:.0%}) — "
                        f"class reweighting **skipped** (would over-penalise minority classes "
                        f"~{_imbalance_ratio_enh:.0f}× and collapse model on dominant class)"
                    )
                with st.spinner("Training enhanced...", show_time=True):
                    enhanced_model, enhanced_cols, enhanced_col_enc = train_lgbm_model(
                        X_tr_enh, y_tr, X_vl_enh, y_vl, n_classes,
                        apply_imbalance=apply_imbalance,
                        imbalance_strategy=_imbalance_strategy,
                        base_params=_effective_params,
                    )
                    enhanced_metrics = evaluate_on_set(
                        enhanced_model, X_vl_enh, y_vl, enhanced_cols, n_classes, enhanced_col_enc
                    )
                st.session_state.enhanced_model = enhanced_model
                st.session_state.enhanced_train_cols = enhanced_cols
                st.session_state.enhanced_val_metrics = enhanced_metrics

            # ── Compact validation metrics table ──────────────────────────────
            _metrics_order_val = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss']
            _val_rows = []
            for _m in _metrics_order_val:
                _bv = baseline_metrics.get(_m)
                _ev = enhanced_metrics.get(_m)
                if _bv is not None and _ev is not None:
                    _diff = _ev - _bv
                    _better = _diff < 0 if _m == 'log_loss' else _diff > 0
                    _val_rows.append({
                        'Metric': _m.replace('_', ' ').upper(),
                        'Baseline': f"{_bv:.4f}",
                        'Enhanced': f"{_ev:.4f}",
                        'Δ': f"{_diff:+.4f}",
                        '': '✅' if _better else ('⚖️' if _diff == 0 else '↘️'),
                    })
            # Save to session state so the table persists across reruns (e.g. test upload)
            st.session_state['_val_metrics_rows'] = _val_rows

            # ── Feature Importance — stored for persistent display ─────────────
            try:
                # Normalise to % for both models
                _b_imps = baseline_model.feature_importances_
                _e_imps = enhanced_model.feature_importances_
                _b_pct = pd.Series(
                    (_b_imps / max(_b_imps.sum(), 1)) * 100, index=baseline_cols
                )
                _e_pct = pd.Series(
                    (_e_imps / max(_e_imps.sum(), 1)) * 100, index=enhanced_cols
                )

                # Split enhanced features into original vs new
                _orig_col_set = set(baseline_cols)
                _e_original = _e_pct[[c for c in enhanced_cols if c in _orig_col_set]]
                _e_new      = _e_pct[[c for c in enhanced_cols if c not in _orig_col_set]]

                _orig_pct_total = float(_e_original.sum())
                _new_pct_total  = float(_e_new.sum())

                # Save FI data to session state so it stays visible after test evaluation
                st.session_state['_fi_b_pct']       = _b_pct
                st.session_state['_fi_e_original']   = _e_original
                st.session_state['_fi_e_new']        = _e_new
                st.session_state['_fi_orig_pct']     = _orig_pct_total
                st.session_state['_fi_new_pct']      = _new_pct_total

            except Exception as _fi_err:
                st.info(f"Could not build importance display: {_fi_err}")
            # ─────────────────────────────────────────────────────────────────
            # Train final models on full training data
            with st.spinner("Training final models on full training set...", show_time=True):
                X_tr_full, _, baseline_full_enc = prepare_data_for_model(X, X.iloc[:1])
                # Baseline intentionally uses BASE_PARAMS (no user overrides) so it
                # measures the raw starting point unaffected by any tuning decisions.
                final_baseline = lgb.LGBMClassifier(**BASE_PARAMS)
                final_baseline.fit(X_tr_full, y)
                st.session_state.baseline_model = final_baseline
                st.session_state.baseline_train_cols = X_tr_full.columns.tolist()
                st.session_state.baseline_col_encoders = baseline_full_enc

                # Enhanced model uses _effective_params (which may include advisory HP
                # overrides chosen by the user) so the test comparison reflects the full
                # effect of every suggestion — transforms + hyperparameter tuning.
                enh_params = _effective_params.copy()
                if apply_imbalance:
                    if _imbalance_strategy in ('binary',) or (_imbalance_strategy == 'low' and n_classes == 2):
                        enh_params['is_unbalance'] = True
                    elif _imbalance_strategy in ('multiclass_moderate', 'low'):
                        enh_params['class_weight'] = 'balanced'
                    # 'none' → severe multiclass: no reweighting
                X_enh_enc, _, enhanced_full_enc = prepare_data_for_model(X_full_enh, X_full_enh.iloc[:1])
                final_enhanced = lgb.LGBMClassifier(**enh_params)
                final_enhanced.fit(X_enh_enc, y)
                st.session_state.enhanced_model = final_enhanced
                st.session_state.enhanced_train_cols = X_enh_enc.columns.tolist()
                st.session_state.enhanced_col_encoders = enhanced_full_enc

            st.success("Both models trained on full training data and ready for test evaluation.")
            # (download buttons are rendered below, outside this block, so they persist)

            # ── Compute post-training diagnosis (store for persistent display) ──
            try:
                _verdicts, _low_imp = _compute_suggestion_verdicts(
                    fitted_params        = st.session_state.fitted_params or [],
                    suggestions          = st.session_state.suggestions or [],
                    selected_indices     = st.session_state.selected_indices or [],
                    enhanced_model       = enhanced_model,
                    enhanced_train_cols  = enhanced_cols,
                    baseline_model       = baseline_model,
                    baseline_train_cols  = baseline_cols,
                    baseline_val_metrics = baseline_metrics,
                    enhanced_val_metrics = enhanced_metrics,
                    apply_imbalance      = apply_imbalance,
                )
                st.session_state['_suggestion_verdicts'] = _verdicts
                st.session_state['_low_imp_cols']        = _low_imp
                st.session_state['_verdicts_stale']      = False  # fresh results
            except Exception as _diag_err:
                st.session_state['_suggestion_verdicts'] = []
                st.session_state['_low_imp_cols']        = {}
                st.warning(f"Post-training diagnosis could not be computed: {_diag_err}")
            # ─────────────────────────────────────────────────────────────────

        # ── Persistent Validation Metrics (shown whenever models are trained) ──
        if st.session_state.baseline_model is not None and st.session_state.get('_val_metrics_rows'):
            st.divider()
            st.markdown("**📋 Validation Metrics**")
            st.dataframe(
                pd.DataFrame(st.session_state['_val_metrics_rows']),
                hide_index=True, use_container_width=True,
            )

        # ── Persistent Feature Importance (shown whenever models are trained) ──
        if st.session_state.baseline_model is not None and st.session_state.get('_fi_b_pct') is not None:
            st.divider()
            with st.expander("📊 Feature Importance", expanded=True):
                _fi_b_pct     = st.session_state['_fi_b_pct']
                _fi_e_orig    = st.session_state['_fi_e_original']
                _fi_e_new     = st.session_state['_fi_e_new']
                _fi_orig_pct  = st.session_state['_fi_orig_pct']
                _fi_new_pct   = st.session_state['_fi_new_pct']

                st.caption(
                    "Sorted by importance. Click any column header to re-sort.  "
                    f"Original features hold **{_fi_orig_pct:.1f}%** | "
                    f"Engineered features hold **{_fi_new_pct:.1f}%** "
                    f"({len(_fi_e_new)} new columns)"
                )

                def _make_fi_table(series, top_n=None):
                    s = series.sort_values(ascending=False)
                    if top_n is not None:
                        s = s.head(top_n)
                    s = s.round(3)
                    df_fi = s.reset_index()
                    df_fi.columns = ['Feature', 'Importance %']
                    return df_fi

                def _make_fi_table_styled(series, top_n=None, flagged_cols=None):
                    """Like _make_fi_table but returns a Styler with flagged rows greyed out."""
                    df_fi = _make_fi_table(series, top_n=top_n)
                    if not flagged_cols:
                        return df_fi
                    flagged = set(flagged_cols)

                    def _style_row(row):
                        if row['Feature'] in flagged:
                            return [
                                'background-color:#1e1e1e;color:#ffffff',
                                'background-color:#1e1e1e;color:#ffffff',
                            ]
                        return ['', '']

                    return df_fi.style.apply(_style_row, axis=1)

                # Build the set of flagged engineered column names from verdicts.
                # Includes: (a) all new_cols from bad/marginal transforms,
                #           (b) individual bad_row_stats,
                #           (c) individual bad_date_subfeatures.
                _verdicts_for_fi = st.session_state.get('_suggestion_verdicts') or []
                _flagged_eng_cols = set()
                for _vv in _verdicts_for_fi:
                    if _vv.get('verdict') in ('bad', 'marginal'):
                        _flagged_eng_cols.update(_vv.get('new_cols') or [])
                    # Partial bad row stats (parent is good, individual stats aren't)
                    _flagged_eng_cols.update(_vv.get('bad_row_stats') or [])
                    # Partial bad date sub-features
                    _prefix = _vv.get('col_prefix', '')
                    for _sf in (_vv.get('bad_date_subfeatures') or []):
                        _flagged_eng_cols.add(f"{_prefix}{_sf}")

                _max_imp = max(
                    float(_fi_b_pct.max()) if not _fi_b_pct.empty else 0,
                    float(_fi_e_orig.max()) if not _fi_e_orig.empty else 0,
                    float(_fi_e_new.max())  if not _fi_e_new.empty  else 0,
                    1.0,
                )

                _tab_b, _tab_orig, _tab_eng = st.tabs([
                    "Baseline model",
                    f"Enhanced — original features ({_fi_orig_pct:.1f}%)",
                    f"Enhanced — engineered features ({_fi_new_pct:.1f}%, {len(_fi_e_new)} cols)",
                ])

                _fi_col_cfg = {
                    "Importance %": st.column_config.ProgressColumn(
                        "Importance %",
                        format="%.3f%%",
                        min_value=0.0,
                        max_value=_max_imp,
                    )
                }

                with _tab_b:
                    st.dataframe(
                        _make_fi_table(_fi_b_pct),
                        hide_index=True,
                        use_container_width=True,
                        column_config=_fi_col_cfg,
                    )
                with _tab_orig:
                    if not _fi_e_orig.empty:
                        st.dataframe(
                            _make_fi_table(_fi_e_orig),
                            hide_index=True,
                            use_container_width=True,
                            column_config=_fi_col_cfg,
                        )
                    else:
                        st.info("No original features retained in the enhanced model.")
                with _tab_eng:
                    if not _fi_e_new.empty:
                        _top_new = _fi_e_new.sort_values(ascending=False)
                        _n_flagged_eng = sum(1 for c in _top_new.index if c in _flagged_eng_cols)
                        _flagged_caption = (
                            f"  ·  **{_n_flagged_eng} greyed out** — from underperforming transforms "
                            f"(grey = transform failed, not just low rank)"
                            if _n_flagged_eng else ""
                        )
                        st.caption(
                            f"Top engineered feature: **{_top_new.index[0]}** "
                            f"({_top_new.iloc[0]:.2f}%). "
                            "Drop in original-feature importance is expected — signal is shared across more columns."
                            + _flagged_caption
                        )
                        st.dataframe(
                            _make_fi_table_styled(_fi_e_new, flagged_cols=_flagged_eng_cols),
                            hide_index=True,
                            use_container_width=True,
                            column_config=_fi_col_cfg,
                        )
                    else:
                        st.info("No engineered features were added.")

            # ── Stale results banner ───────────────────────────────────────────
            if st.session_state.get('_verdicts_stale'):
                st.warning(
                    "⚠️ These results are from the **previous** training run. "
                    "Scroll up to **③ Train Models** and re-train to refresh them.",
                    icon=None,
                )

            # ── Auto-deselect shortcut below Feature Importance ────────────────
            _verdicts_fi = st.session_state.get('_suggestion_verdicts') or []
            _bad_marginal_fi = [
                v for v in _verdicts_fi
                if v.get('verdict') in ('bad', 'marginal')
                and v.get('type') != 'imbalance'   # imbalance is never auto-deselected
            ]
            _deselectable_fi = [
                v for v in _bad_marginal_fi
                if v.get('sug_idx') is not None and v.get('type') != 'imbalance'
            ]
            # Also collect date_features verdicts that are overall 'good' but have
            # individual zero-importance sub-features (e.g. is_weekend) to prune.
            _date_subfeature_fi = [
                v for v in _verdicts_fi
                if v.get('bad_date_subfeatures') and v.get('sug_idx') is not None
                and v.get('verdict') not in ('bad', 'marginal')
            ]
            # Also collect row_numeric_stats verdicts that are overall 'good' but have
            # individual low-importance stats (e.g. row_range) to prune.
            _row_stat_fi = [
                v for v in _verdicts_fi
                if v.get('bad_row_stats') and v.get('sug_idx') is not None
                and v.get('verdict') not in ('bad', 'marginal')
            ]
            _total_fi_actions = (
                len(_deselectable_fi)
                + len(_date_subfeature_fi)
                + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_fi)
            )
            # Show button whenever there are bad/marginal transforms or bad date/row sub-features
            if _bad_marginal_fi or _date_subfeature_fi or _row_stat_fi:
                st.markdown("")
                if st.button(
                    f"⚡ Auto-deselect {_total_fi_actions} bad/marginal transform(s) & re-train",
                    help="Unticks underperforming transforms in Step ② and resets Step ③ — scroll up to re-train.",
                    type="primary",
                    key="_auto_deselect_fi",
                ):
                    for _v in _deselectable_fi:
                        st.session_state[f"_ck_persist_{_v['sug_idx']}"] = False
                    # Stage date sub-feature deselections in a non-widget key.
                    # ui_components applies them before st.checkbox is called on the next run.
                    _pending = st.session_state.get('_pending_date_deselect', {})
                    for _v in _date_subfeature_fi:
                        _si = _v['sug_idx']
                        _pending[_si] = list(_v['bad_date_subfeatures'])
                    st.session_state['_pending_date_deselect'] = _pending
                    # Stage row-stat sub-feature deselections in a non-widget key.
                    _pending_rs = st.session_state.get('_pending_row_stat_deselect', {})
                    for _v in _row_stat_fi:
                        _si = _v['sug_idx']
                        _pending_rs[_si] = list(_v['bad_row_stats'])
                    st.session_state['_pending_row_stat_deselect'] = _pending_rs
                    for _rk in [
                        'baseline_model', 'enhanced_model',
                        'baseline_train_cols', 'enhanced_train_cols',
                        'baseline_val_metrics', 'enhanced_val_metrics',
                        'baseline_col_encoders', 'enhanced_col_encoders',
                        'fitted_params', 'X_train_enhanced',
                        'X_test_raw', 'X_test_enhanced', '_test_df_original',
                        '_test_baseline_metrics', '_test_enhanced_metrics',
                    ]:
                        st.session_state[_rk] = None
                    st.session_state['_verdicts_stale'] = True
                    st.rerun()

            # ── Bottom-2% feature importance warning ──────────────────────────
            _fi_b_pct_warn = st.session_state.get('_fi_b_pct')
            if _fi_b_pct_warn is not None and not st.session_state.get('_verdicts_stale'):
                try:
                    _n_feats = len(_fi_b_pct_warn)
                    # Bottom 2% means columns below the 2nd percentile of importance
                    _p2_thresh = float(np.percentile(_fi_b_pct_warn.values, 2))
                    _useless = _fi_b_pct_warn[
                        (_fi_b_pct_warn <= _p2_thresh) & (_fi_b_pct_warn < 0.05)
                    ]
                    if len(_useless) >= 2 and len(_useless) < _n_feats:
                        _useless_names = _useless.sort_values().index.tolist()
                        with st.expander(
                            f"🗑️ {len(_useless_names)} potentially useless feature(s) "
                            f"(bottom 2% of baseline importance)",
                            expanded=False,
                        ):
                            st.markdown(
                                "These columns contribute less than **0.05%** importance "
                                "in the baseline model and fall in the bottom 2% of all "
                                "features. They are unlikely to help and may introduce noise. "
                                "Consider dropping them in the **Column Types** panel above."
                            )
                            _useless_df = _useless.sort_values().reset_index()
                            _useless_df.columns = ['Column', 'Importance %']
                            _useless_df['Importance %'] = _useless_df['Importance %'].round(4)
                            st.dataframe(_useless_df, hide_index=True, use_container_width=True)
                            st.caption(
                                "To drop these columns: open **🔬 Column Types & Drop Suggestions** "
                                "above → check the drop box for each → Apply Changes → Re-analyze."
                            )
                except Exception:
                    pass
            # ─────────────────────────────────────────────────────────────────
        if st.session_state.baseline_model is not None:
            st.divider()
            st.subheader("📥 Download Training Data")

            # ── Preview: raw vs enhanced ──────────────────────────────────────
            _X_raw_prev = st.session_state.get('X_train')
            _X_enh_prev = st.session_state.get('X_train_enhanced')
            if _X_raw_prev is not None and _X_enh_prev is not None:
                _n_new_cols = _X_enh_prev.shape[1] - _X_raw_prev.shape[1]
                with st.expander(
                    f"🔍 Dataset preview after all transforms "
                    f"({_X_raw_prev.shape[1]} → {_X_enh_prev.shape[1]} columns, "
                    f"+{_n_new_cols} engineered)",
                    expanded=False,
                ):
                    _pv1, _pv2 = st.columns(2)
                    with _pv1:
                        st.caption(
                            f"**Baseline (raw)** — {_X_raw_prev.shape[0]:,} rows "
                            f"× {_X_raw_prev.shape[1]} cols"
                        )
                        st.dataframe(_X_raw_prev.head(8), use_container_width=True, height=260)
                    with _pv2:
                        st.caption(
                            f"**Enhanced (post-transform)** — {_X_enh_prev.shape[0]:,} rows "
                            f"× {_X_enh_prev.shape[1]} cols"
                        )
                        st.dataframe(_X_enh_prev.head(8), use_container_width=True, height=260)

                    # Highlight the newly added columns
                    _orig_cols = set(_X_raw_prev.columns)
                    _new_col_names = [c for c in _X_enh_prev.columns if c not in _orig_cols]
                    if _new_col_names:
                        st.caption(
                            "**Engineered columns added:** "
                            + ", ".join(f"`{c}`" for c in _new_col_names[:20])
                            + (f" (+{len(_new_col_names)-20} more)" if len(_new_col_names) > 20 else "")
                        )
            # ─────────────────────────────────────────────────────────────────

            # ── Download transformed datasets ──────────────────────────────────
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if st.session_state.get('X_train') is not None:
                    raw_csv = st.session_state.X_train.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download baseline X_train (raw)",
                        data=raw_csv,
                        file_name="X_train_baseline.csv",
                        mime="text/csv",
                        help="Original training features before any transforms",
                        key="dl_train_raw",
                    )
            with dl_col2:
                if st.session_state.get('X_train_enhanced') is not None:
                    enh_csv = st.session_state.X_train_enhanced.to_csv(index=False).encode('utf-8')
                    n_sel = len(st.session_state.get('selected_indices') or [])
                    st.download_button(
                        label="⬇️ Download enhanced X_train (post-transform)",
                        data=enh_csv,
                        file_name="X_train_enhanced.csv",
                        mime="text/csv",
                        help=f"Training features after applying {n_sel} selected transforms",
                        key="dl_train_enh",
                    )
            # ──────────────────────────────────────────────────────────────────

            # ── Export validation report (available immediately after training) ──
            st.divider()
            st.subheader("📥 Export Validation Report")
            st.caption("Report uses validation-set metrics. Upload a test CSV below for final held-out results.")
            try:
                from report_buttons import add_report_download_buttons
                _lbl_train = getattr(uploaded_train, "name", "dataset").replace(".csv", "")
                add_report_download_buttons(
                    st.session_state,
                    dataset_name=_lbl_train,
                    key_suffix="val",
                    report_stage="validation",
                )
            except ImportError:
                st.info("Place `report_generator.py` in the same directory to enable report export.")
            # ──────────────────────────────────────────────────────────────────
    else:
        # Show context-aware locked message
        if st.session_state.X_train is None:
            _render_locked_step("Upload training data in **①** first.")
        elif not st.session_state.suggestions:
            _render_locked_step("Click **Analyze Dataset** in **②** to generate transform suggestions.")
        else:
            _render_locked_step("Select at least one transform in **②** and click **Analyze Dataset**.")

    # =========================================================================
    # STEP 4: UPLOAD TEST DATA & COMPARE
    # =========================================================================
    st.divider()
    _step4_ready = (
        st.session_state.baseline_model is not None
        and st.session_state.enhanced_model is not None
    )
    _step4_has_results = bool(
        st.session_state.get('_test_baseline_metrics')
        or st.session_state.get('_test_enhanced_metrics')
    )
    if _step4_ready and _step4_has_results:
        _test_hdr_cols = st.columns([3, 1])
        _po_header = "④ Upload Test Data & Predict" if st.session_state.get('_predict_only') else "④ Upload Test Data & Compare"
        _test_hdr_cols[0].header(_po_header)
        _reset_label = "🔄 Reset Predictions" if st.session_state.get('_predict_only') else "🔄 Reset Test Results"
        if _test_hdr_cols[1].button(_reset_label,
                                     help="Clear test evaluation results"):
            st.session_state['_test_baseline_metrics'] = None
            st.session_state['_test_enhanced_metrics'] = None
            st.session_state['X_test_raw']             = None
            st.session_state['X_test_enhanced']        = None
            st.session_state['_test_file_sig']         = None
            st.session_state['_test_df_original']      = None
            st.session_state['_predict_only']          = False
            st.session_state.pop('_pred_dl_cols', None)
            st.rerun()

    else:
        st.header("④ Upload Test Data & Compare / Predict")

    if _step4_ready:
        uploaded_test = st.file_uploader(
            "Upload test CSV (with or without labels)",
            type=['csv'], key='test_upload',
            help="If the target column is absent, the app enters **predict-only** mode — "
                 "you can download predictions but evaluation metrics won't be computed.",
        )

        if uploaded_test is not None:
            # Detect when the user swaps to a different test file and clear stale results
            _new_test_sig = f"{uploaded_test.name}__{uploaded_test.size}"
            if st.session_state.get('_test_file_sig') != _new_test_sig:
                st.session_state['_test_file_sig']         = _new_test_sig
                st.session_state['_test_file_name']        = uploaded_test.name
                st.session_state['_test_baseline_metrics'] = None
                st.session_state['_test_enhanced_metrics'] = None
                st.session_state['X_test_raw']             = None
                st.session_state['X_test_enhanced']        = None
                st.session_state['_test_df_original']      = None
                st.session_state['_predict_only']          = False
                st.session_state.pop('_pred_dl_cols', None)

            try:
                test_df = pd.read_csv(uploaded_test, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_test.seek(0)
                test_df = pd.read_csv(uploaded_test, encoding='latin-1')
            test_df = sanitize_feature_names(test_df)
            target_col = st.session_state.target_col
            le = st.session_state.label_encoder
            n_classes = st.session_state.n_classes

            # ── Detect predict-only mode ──────────────────────────────────
            _predict_only = target_col not in test_df.columns

            if _predict_only:
                st.info(
                    f"🔮 **Predict-only mode** — target column `{target_col}` not found in test data. "
                    f"Predictions will be generated using the enhanced model, but evaluation "
                    f"metrics cannot be computed without ground-truth labels.",
                    icon=None,
                )
                y_test = None
                X_test = test_df.copy()
            else:
                # Check for labels in the test target that were never seen during training.
                # LabelEncoder.transform crashes hard on unseen values — surface a clear
                # error so the user knows which classes to remove or remap.
                _test_labels = set(test_df[target_col].astype(str).unique())
                _train_labels = set(le.classes_)
                _unseen = _test_labels - _train_labels
                if _unseen:
                    st.error(
                        f"**Test target contains class labels not seen during training: "
                        f"`{sorted(_unseen)}`**\n\n"
                        f"Training classes: `{sorted(_train_labels)}`\n\n"
                        f"Please either:\n"
                        f"- Remove rows with these labels from the test CSV, or\n"
                        f"- Map them to the closest training class before uploading."
                    )
                    st.stop()

                y_test = pd.Series(le.transform(test_df[target_col].astype(str)), name=target_col)
                X_test = test_df.drop(columns=[target_col])

            # Save the full test features (before any drops) so that ID
            # columns can be re-attached in the prediction download section.
            st.session_state['_test_df_original'] = X_test.copy()

            # Apply the same column drops the user made on the training set so
            # the test features are consistent with what the model was trained on.
            _applied_drops = st.session_state.get('_applied_drops') or []
            _test_drops = [c for c in _applied_drops if c in X_test.columns]
            if _test_drops:
                X_test = X_test.drop(columns=_test_drops)

            # ── Column alignment check ───────────────────────────────────
            _train_cols = set(st.session_state.get('baseline_train_cols') or [])
            if _train_cols:
                _test_feature_cols = set(X_test.columns)
                _missing_from_test = _train_cols - _test_feature_cols
                _extra_in_test     = _test_feature_cols - _train_cols
                if _missing_from_test:
                    st.warning(
                        f"⚠️ **{len(_missing_from_test)} training column(s) missing from test set** "
                        f"(will be zero-filled): "
                        + ", ".join(f"`{c}`" for c in sorted(_missing_from_test)[:10])
                        + (f" +{len(_missing_from_test)-10} more" if len(_missing_from_test) > 10 else "")
                    )
                elif _extra_in_test:
                    st.info(
                        f"ℹ️ {len(_extra_in_test)} extra column(s) in test set will be ignored."
                    )
                else:
                    st.success(f"✅ All {len(_train_cols)} training columns found in test set.")
            # ─────────────────────────────────────────────────────────────

            st.write(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} columns")

            _btn_label = "🔮 Generate Predictions" if _predict_only else "📊 Evaluate on Test Set"
            if st.button(_btn_label, type="primary"):
                # Enhanced test data
                X_test_enh = apply_fitted_to_test(X_test, st.session_state.fitted_params)
                st.session_state['X_test_enhanced'] = X_test_enh
                st.session_state['X_test_raw']      = X_test
                st.session_state['_predict_only']   = _predict_only

                if _predict_only:
                    with st.spinner("Generating predictions..."):
                        baseline_test = predict_on_set(
                            st.session_state.baseline_model, X_test,
                            st.session_state.baseline_train_cols, n_classes,
                            st.session_state.get('baseline_col_encoders'),
                        )
                        enhanced_test = predict_on_set(
                            st.session_state.enhanced_model, X_test_enh,
                            st.session_state.enhanced_train_cols, n_classes,
                            st.session_state.get('enhanced_col_encoders'),
                        )
                    st.session_state['_test_baseline_metrics'] = baseline_test
                    st.session_state['_test_enhanced_metrics'] = enhanced_test
                else:
                    with st.spinner("Evaluating..."):
                        baseline_test = evaluate_on_set(
                            st.session_state.baseline_model, X_test, y_test,
                            st.session_state.baseline_train_cols, n_classes,
                            st.session_state.get('baseline_col_encoders'),
                        )
                        enhanced_test = evaluate_on_set(
                            st.session_state.enhanced_model, X_test_enh, y_test,
                            st.session_state.enhanced_train_cols, n_classes,
                            st.session_state.get('enhanced_col_encoders'),
                        )
                    st.session_state['_test_baseline_metrics'] = baseline_test
                    st.session_state['_test_enhanced_metrics'] = enhanced_test
                # Reset threshold slider so it reinitialises from new test data
                st.session_state.pop('_pr_threshold_slider', None)

        # Show results if evaluation has been run
        if st.session_state.get('_test_baseline_metrics') and st.session_state.get('_test_enhanced_metrics'):
            baseline_test = st.session_state['_test_baseline_metrics']
            enhanced_test = st.session_state['_test_enhanced_metrics']

            # --- Results ---
            _is_predict_only = st.session_state.get('_predict_only', False)

            if _is_predict_only:
                st.subheader("🔮 Predict-Only Results")
                st.success(
                    f"Predictions generated for **{len(enhanced_test.get('_y_pred', []))}** rows. "
                    f"Download them below."
                )
                st.caption(
                    "Evaluation metrics (ROC-AUC, F1, confusion matrix, etc.) are not available "
                    "because the test set does not contain the target column. Upload a test set "
                    "with the target column to compare baseline vs. enhanced models."
                )

                # ── Threshold Tuning for Predict-Only (binary only) ───────
                _po_n_classes = st.session_state.get('n_classes', 0)
                _po_y_proba   = enhanced_test.get('_y_pred_proba')
                _po_y_pred    = enhanced_test.get('_y_pred')
                _po_le        = st.session_state.get('label_encoder')

                if (_po_n_classes == 2
                        and _po_y_proba is not None
                        and _po_y_proba.ndim == 2
                        and _po_y_proba.shape[1] >= 2
                        and _po_le is not None):
                    st.divider()
                    with st.expander(
                        "🎚️ Decision Threshold Tuning",
                        expanded=True,
                    ):
                        st.info(
                            "**Adjust the decision threshold**  \n"
                            "By default, the model predicts class 1 when the probability is ≥ 0.5 "
                            "— but 0.5 isn't always the best cutoff.  \n\n"
                            "• **Raising** the threshold → fewer positive predictions (higher precision).  \n"
                            "• **Lowering** it → more positive predictions (higher recall).  \n\n"
                            "Since no ground-truth labels are available, metrics like F1 or accuracy "
                            "cannot be computed — but you can see how the **predicted class distribution** "
                            "changes and pick the cutoff that fits your use case.  \n\n"
                            "💡 *The downloaded predictions will reflect the threshold you choose here.*",
                            icon="ℹ️",
                        )

                        # Initialise slider
                        if '_pr_threshold_slider' not in st.session_state:
                            st.session_state['_pr_threshold_slider'] = 0.50

                        def _po_set_threshold(val):
                            st.session_state['_pr_threshold_slider'] = val

                        _po_sl_col1, _po_sl_col2 = st.columns([0.65, 0.35])
                        with _po_sl_col1:
                            _po_custom_t = st.slider(
                                "Decision threshold",
                                min_value=0.01, max_value=0.99,
                                step=0.01,
                                key="_pr_threshold_slider",
                                help="Drag to change the cutoff. Predictions and download will update accordingly.",
                            )
                        with _po_sl_col2:
                            _po_quick = st.columns(3)
                            with _po_quick[0]:
                                st.button(
                                    "0.3", key="_po_btn_03",
                                    use_container_width=True,
                                    on_click=_po_set_threshold,
                                    args=(0.30,),
                                    help="High recall",
                                )
                            with _po_quick[1]:
                                st.button(
                                    "0.5", key="_po_btn_05",
                                    use_container_width=True,
                                    on_click=_po_set_threshold,
                                    args=(0.50,),
                                    help="Default",
                                )
                            with _po_quick[2]:
                                st.button(
                                    "0.7", key="_po_btn_07",
                                    use_container_width=True,
                                    on_click=_po_set_threshold,
                                    args=(0.70,),
                                    help="High precision",
                                )

                        # Compute predictions at the chosen threshold
                        _po_proba_1 = _po_y_proba[:, 1]
                        _po_preds_custom = (_po_proba_1 >= _po_custom_t).astype(int)
                        _po_preds_default = (_po_proba_1 >= 0.5).astype(int)

                        _po_class_names = list(_po_le.classes_)
                        _po_n_total = len(_po_preds_custom)
                        _po_n_pos = int(_po_preds_custom.sum())
                        _po_n_neg = _po_n_total - _po_n_pos
                        _po_n_pos_def = int(_po_preds_default.sum())
                        _po_n_neg_def = _po_n_total - _po_n_pos_def

                        # Class distribution metrics
                        _po_m1, _po_m2, _po_m3 = st.columns(3)
                        _po_m1.metric(
                            f"Predicted: {_po_class_names[1] if len(_po_class_names) > 1 else '1'}",
                            f"{_po_n_pos:,}  ({_po_n_pos / max(_po_n_total, 1) * 100:.1f}%)",
                            delta=f"{_po_n_pos - _po_n_pos_def:+,} vs default (0.5)",
                        )
                        _po_m2.metric(
                            f"Predicted: {_po_class_names[0]}",
                            f"{_po_n_neg:,}  ({_po_n_neg / max(_po_n_total, 1) * 100:.1f}%)",
                            delta=f"{_po_n_neg - _po_n_neg_def:+,} vs default (0.5)",
                        )
                        _po_m3.metric(
                            "Total rows",
                            f"{_po_n_total:,}",
                        )

                        # Probability distribution histogram
                        try:
                            import plotly.graph_objects as go

                            _fig_hist = go.Figure()
                            _fig_hist.add_trace(go.Histogram(
                                x=_po_proba_1,
                                nbinsx=50,
                                marker_color='#58a6ff',
                                opacity=0.7,
                                name='P(class 1)',
                            ))
                            # Threshold line
                            _fig_hist.add_vline(
                                x=_po_custom_t,
                                line_dash="dash",
                                line_color="#f0883e",
                                line_width=2,
                                annotation_text=f"threshold = {_po_custom_t:.2f}",
                                annotation_position="top",
                                annotation_font_color="#f0883e",
                            )
                            _fig_hist.update_layout(
                                xaxis_title='Predicted probability (class 1)',
                                yaxis_title='Count',
                                xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#30363d'),
                                yaxis=dict(showgrid=True, gridcolor='#30363d'),
                                margin=dict(t=40, b=40, l=50, r=20),
                                height=280,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#8b949e', size=11),
                                showlegend=False,
                            )
                            st.plotly_chart(_fig_hist, use_container_width=True)
                        except Exception:
                            pass

                        st.caption(
                            "💡 *The histogram shows how the model's predicted probabilities are distributed. "
                            "The orange line marks your chosen threshold — rows to the right are predicted "
                            "as the positive class.*"
                        )

            if not _is_predict_only:
                # --- Full evaluation results comparison ---
                st.subheader("Test Set Results")

                # Highlight headline metrics FIRST
                b_auc = baseline_test.get('roc_auc')
                e_auc = enhanced_test.get('roc_auc')
                if b_auc is not None and e_auc is not None:
                    delta_auc = e_auc - b_auc
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Baseline ROC-AUC", f"{b_auc:.4f}")
                    col2.metric("Enhanced ROC-AUC", f"{e_auc:.4f}", delta=f"{delta_auc:+.4f}")
                    pct = delta_auc / max(1.0 - b_auc, 0.001) * 100
                    _hdr_label = "Headroom Captured" if pct >= 0 else "Headroom Lost"
                    col3.metric(_hdr_label, f"{abs(pct):.1f}%", delta=f"{pct:+.1f}%")

                st.markdown("---")

                metrics_order = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss']
                comp_data = []
                for m in metrics_order:
                    bv = baseline_test.get(m)
                    ev = enhanced_test.get(m)
                    if bv is not None and ev is not None:
                        diff = ev - bv
                        # For log_loss, lower is better
                        better = diff < 0 if m == 'log_loss' else diff > 0
                        comp_data.append({
                            'Metric': m.replace('_', ' ').upper(),
                            'Baseline': f"{bv:.4f}",
                            'Enhanced': f"{ev:.4f}",
                            'Δ': f"{diff:+.4f}",
                            'Winner': '✅ Enhanced' if better else ('⚖️ Tie' if diff == 0 else '⬅️ Baseline'),
                        })

                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            # ── Precision-Recall Curve & Threshold Optimisation (binary, test set) ──
            _pr_n_classes = st.session_state.get('n_classes', 0)
            _pr_test_enh  = st.session_state.get('_test_enhanced_metrics') or {}
            _pr_test_base = st.session_state.get('_test_baseline_metrics') or {}
            _pr_data_enh  = _pr_test_enh.get('pr_data')
            _pr_data_base = _pr_test_base.get('pr_data')

            if not _is_predict_only and _pr_n_classes == 2 and _pr_data_enh is not None:
                st.divider()

                with st.expander(
                    "🎯 Precision-Recall Curve & Threshold Tuning",
                    expanded=False,
                ):
                    st.info(
                        "**How does this work?**  \n"
                        "By default, models predict class 1 when the probability is ≥ 0.5 — "
                        "but 0.5 isn't always the best cutoff.  \n\n"
                        "• **Raising** the threshold → model is pickier: fewer positives, but more of them correct (higher precision).  \n"
                        "• **Lowering** it → model is more inclusive: catches more true positives, but also more false alarms (higher recall).  \n\n"
                        "The PR curve shows this tradeoff at every threshold. Use the **optimal threshold finder** to pick "
                        "the cutoff that maximises the metric you care about most.  \n\n"
                        "💡 *Useful when the cost of a false positive is very different from the cost of a missed detection.  \n"
                        "Note: ROC-AUC is not listed in the threshold table because it measures ranking quality across **all** "
                        "thresholds at once — it stays the same no matter where you set the cutoff.*",
                        icon="ℹ️",
                    )
                    # ── PR Curve plot ─────────────────────────────────────────
                    _pr_enh_y_true = None
                    _pr_enh_y_proba = None
                    try:
                        import plotly.graph_objects as go

                        _fig_pr = go.Figure()

                        # Enhanced model curve
                        _fig_pr.add_trace(go.Scatter(
                            x=_pr_data_enh['recall'],
                            y=_pr_data_enh['precision'],
                            mode='lines',
                            name=f"Enhanced (AP={_pr_data_enh['avg_precision']:.3f})",
                            line=dict(color='#58a6ff', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(88,166,255,0.08)',
                        ))

                        # Baseline model curve (if available)
                        if _pr_data_base is not None:
                            _fig_pr.add_trace(go.Scatter(
                                x=_pr_data_base['recall'],
                                y=_pr_data_base['precision'],
                                mode='lines',
                                name=f"Baseline (AP={_pr_data_base['avg_precision']:.3f})",
                                line=dict(color='#8b949e', width=1.5, dash='dot'),
                            ))

                        # Default threshold marker (0.5)
                        _pr_enh_y_true  = _pr_test_enh.get('_y_true')
                        _pr_enh_y_proba = _pr_test_enh.get('_y_proba')
                        if _pr_enh_y_true is not None:
                            _m05 = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, 0.5)
                            _fig_pr.add_trace(go.Scatter(
                                x=[_m05['recall']],
                                y=[_m05['precision']],
                                mode='markers',
                                name='Default (t=0.5)',
                                marker=dict(color='#f0883e', size=10, symbol='diamond'),
                                showlegend=True,
                            ))

                        _fig_pr.update_layout(
                            xaxis_title='Recall',
                            yaxis_title='Precision',
                            xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#30363d'),
                            yaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#30363d'),
                            margin=dict(t=30, b=40, l=50, r=20),
                            height=360,
                            legend=dict(
                                yanchor='bottom', y=0.02,
                                xanchor='right', x=0.98,
                                bgcolor='rgba(0,0,0,0.3)',
                                font=dict(size=11),
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#8b949e', size=11),
                        )

                        st.plotly_chart(_fig_pr, use_container_width=True)
                    except Exception as _pr_plot_err:
                        st.warning(f"Could not render PR curve: {_pr_plot_err}")

                    # ── Optimal threshold finder ──────────────────────────────
                    if _pr_enh_y_true is not None and _pr_enh_y_proba is not None:
                        _optimal = _find_optimal_thresholds(_pr_enh_y_true, _pr_enh_y_proba)

                        st.markdown("**Optimal thresholds** *(enhanced model, test set)*")
                        _opt_rows = []
                        for _opt_m in ['f1', 'precision', 'recall', 'accuracy']:
                            _opt_rows.append({
                                'Optimise for': _opt_m.upper(),
                                'Best threshold': f"{_optimal[_opt_m]['threshold']:.3f}",
                                'Best value': f"{_optimal[_opt_m]['value']:.4f}",
                            })
                        st.dataframe(
                            pd.DataFrame(_opt_rows),
                            hide_index=True, use_container_width=True,
                        )

                        # ── Interactive threshold slider ──────────────────────
                        st.markdown("---")
                        st.markdown("**Explore a custom threshold**")

                        # Initialise slider value once; after that the widget
                        # manages its own state via the key.
                        if '_pr_threshold_slider' not in st.session_state:
                            st.session_state['_pr_threshold_slider'] = round(
                                _optimal['f1']['threshold'], 2,
                            )

                        # Button callbacks — on_click fires BEFORE widgets render
                        # on the next rerun, so setting the widget key here is safe.
                        def _pr_set_threshold(val):
                            st.session_state['_pr_threshold_slider'] = val

                        _pr_sl_col1, _pr_sl_col2 = st.columns([0.6, 0.4])
                        with _pr_sl_col1:
                            _custom_t = st.slider(
                                "Decision threshold",
                                min_value=0.01, max_value=0.99,
                                step=0.01,
                                key="_pr_threshold_slider",
                                help="Drag to see how metrics change at different thresholds.",
                            )
                        with _pr_sl_col2:
                            _quick_btns = st.columns(3)
                            with _quick_btns[0]:
                                st.button(
                                    "Max F1", key="_pr_btn_f1",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['f1']['threshold'], 2),),
                                )
                            with _quick_btns[1]:
                                st.button(
                                    "Max Prec", key="_pr_btn_prec",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['precision']['threshold'], 2),),
                                )
                            with _quick_btns[2]:
                                st.button(
                                    "Max Recall", key="_pr_btn_rec",
                                    use_container_width=True,
                                    on_click=_pr_set_threshold,
                                    args=(round(_optimal['recall']['threshold'], 2),),
                                )

                        _m_custom = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, _custom_t)
                        _m_default = _metrics_at_threshold(_pr_enh_y_true, _pr_enh_y_proba, 0.5)

                        _pr_mc1, _pr_mc2, _pr_mc3, _pr_mc4 = st.columns(4)
                        _pr_mc1.metric(
                            "F1", f"{_m_custom['f1']:.4f}",
                            delta=f"{_m_custom['f1'] - _m_default['f1']:+.4f} vs 0.5",
                        )
                        _pr_mc2.metric(
                            "Precision", f"{_m_custom['precision']:.4f}",
                            delta=f"{_m_custom['precision'] - _m_default['precision']:+.4f} vs 0.5",
                        )
                        _pr_mc3.metric(
                            "Recall", f"{_m_custom['recall']:.4f}",
                            delta=f"{_m_custom['recall'] - _m_default['recall']:+.4f} vs 0.5",
                        )
                        _pr_mc4.metric(
                            "Accuracy", f"{_m_custom['accuracy']:.4f}",
                            delta=f"{_m_custom['accuracy'] - _m_default['accuracy']:+.4f} vs 0.5",
                        )

                        # ── Mini confusion matrix at chosen threshold ─────────
                        try:
                            _y_t = np.asarray(_pr_enh_y_true)
                            _y_p = (np.asarray(_pr_enh_y_proba) >= _custom_t).astype(int)
                            _cm = confusion_matrix(_y_t, _y_p)
                            _tn, _fp, _fn, _tp = _cm.ravel()
                            _cm_col1, _cm_col2 = st.columns(2)
                            with _cm_col1:
                                st.caption(f"Confusion matrix at threshold **{_custom_t:.2f}**")
                                _cm_df = pd.DataFrame(
                                    _cm,
                                    index=[f"Actual 0", f"Actual 1"],
                                    columns=[f"Pred 0", f"Pred 1"],
                                )
                                st.dataframe(_cm_df, use_container_width=True)
                            with _cm_col2:
                                _total = len(_y_t)
                                _pos_rate = float(_y_p.sum()) / max(_total, 1) * 100
                                st.caption("Prediction summary")
                                st.markdown(
                                    f"- **Predicted positive rate**: {_pos_rate:.1f}%  \n"
                                    f"- **True positives**: {_tp:,} &nbsp;|&nbsp; **False positives**: {_fp:,}  \n"
                                    f"- **True negatives**: {_tn:,} &nbsp;|&nbsp; **False negatives**: {_fn:,}"
                                )
                        except Exception:
                            pass

                        st.caption(
                            "💡 *All metrics above are computed on the **held-out test set** — "
                            "these are unbiased estimates of real-world performance at your chosen threshold.*"
                        )

            # ── ⑤ Post-Training Analysis (shown after test evaluation) ──
            if not _is_predict_only and st.session_state.get('_suggestion_verdicts') is not None:
                st.divider()
                st.header("⑤ Analysis & Recommendations")

                _verdicts  = st.session_state['_suggestion_verdicts']
                _low_imp   = st.session_state.get('_low_imp_cols') or {}

                _VCOLOR = {'good': '#3fb950', 'marginal': '#f0883e', 'bad': '#f85149'}
                _VLABEL = {'good': 'Contributed', 'marginal': 'Marginal', 'bad': 'No contribution'}
                _TICON  = {
                    'numerical': '🔢', 'categorical': '🏷️', 'interaction': '🔗',
                    'row': '📊', 'date': '📅', 'text': '📝', 'imbalance': '⚖️',
                }

                # Exclude imbalance entries from counts — the imbalance verdict here is
                # based on *validation* metrics.  Once test results are available (this
                # block only renders post-test), Section ⑥ supersedes it with test F1.
                _n_good     = sum(1 for v in _verdicts if v['verdict'] == 'good'     and v.get('type') != 'imbalance')
                _n_marginal = sum(1 for v in _verdicts if v['verdict'] == 'marginal' and v.get('type') != 'imbalance')
                _n_bad      = sum(1 for v in _verdicts if v['verdict'] == 'bad'      and v.get('type') != 'imbalance')
                _n_remove   = _n_marginal + _n_bad

                # ── Compact summary banner ─────────────────────────────
                if _n_remove == 0 and not _low_imp:
                    st.success(
                        f"✅ All {len(_verdicts)} transforms contributed — "
                        f"the pipeline is well-optimized."
                    )
                else:
                    _parts = []
                    if _n_good:     _parts.append(f"<span style='color:#3fb950'>✅ {_n_good} contributed</span>")
                    if _n_marginal: _parts.append(f"<span style='color:#f0883e'>⚠️ {_n_marginal} marginal</span>")
                    if _n_bad:      _parts.append(f"<span style='color:#f85149'>❌ {_n_bad} not contributing</span>")
                    if _low_imp:    _parts.append(f"<span style='color:#8b949e'>🗑️ {len(_low_imp)} low-importance cols</span>")
                    st.markdown(
                        f"<div style='padding:10px 16px;background:#161b22;border:1px solid #30363d;"
                        f"border-radius:6px;font-size:0.88rem;margin-bottom:12px'>"
                        + " &nbsp;&nbsp;·&nbsp;&nbsp; ".join(_parts)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Transforms that need attention (only shown if any) ─
                # Imbalance is excluded here — its validation-based verdict is
                # superseded by the test-metrics assessment in Section ⑥ below.
                _problem_verdicts = [
                    v for v in _verdicts
                    if v['verdict'] in ('bad', 'marginal') and v.get('type') != 'imbalance'
                ]
                # Surface a brief note if the imbalance verdict would have appeared
                _imb_val_verdict = next(
                    (v for v in _verdicts if v.get('type') == 'imbalance'), None
                )
                if _imb_val_verdict and _imb_val_verdict['verdict'] in ('bad', 'marginal'):
                    st.info(
                        "⚖️ **Class reweighting** had a mixed/negative verdict on the "
                        "**validation** set — see the **Imbalance handling** row in "
                        "Section ⑥ below for the definitive test-set assessment.",
                        icon=None,
                    )
                if _problem_verdicts:
                    st.markdown(
                        "<p style='font-size:0.82rem;color:#8b949e;margin:0 0 6px 0'>"
                        "The following transforms did not pay off on the held-out test set. "
                        "Consider deselecting them and re-training.</p>",
                        unsafe_allow_html=True,
                    )
                    for _v in _problem_verdicts:
                        _color = _VCOLOR[_v['verdict']]
                        _label = _VLABEL[_v['verdict']]
                        _col_str = _v['column']
                        if _v.get('column_b'):
                            _col_str += f" × {_v['column_b']}"
                        _v_icon = '⚠️' if _v['verdict'] == 'marginal' else '❌'
                        st.markdown(
                            f"<div style='padding:7px 12px;margin:4px 0;background:#0d1117;"
                            f"border-left:3px solid {_color};border-radius:4px;font-size:0.82rem'>"
                            f"{_v_icon} <code style='font-size:0.82rem'>{_col_str}</code>"
                            f"&ensp;<span style='color:#79c0ff'>{_v['method']}</span>"
                            f"&ensp;—&ensp;<span style='color:{_color}'>{_label}</span><br>"
                            f"<span style='color:#6e7681;font-size:0.76rem'>{_v['reason']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Row-stat sub-feature notices ──────────────────
                    # Surface which individual stats have near-zero importance
                    # even though the parent transform is overall 'good'.
                    for _v in _verdicts:
                        _brs = _v.get('bad_row_stats')
                        if not _brs or _v.get('verdict') in ('bad', 'marginal'):
                            continue
                        _brs_names = ", ".join(f"`{s}`" for s in _brs)
                        st.markdown(
                            f"<div style='padding:7px 12px;margin:4px 0;background:#0d1117;"
                            f"border-left:3px solid #d29922;border-radius:4px;font-size:0.82rem'>"
                            f"⚠️ <span style='color:#79c0ff'>row_numeric_stats</span>"
                            f"&ensp;—&ensp;<span style='color:#d29922'>partial: "
                            f"{len(_brs)} of {len(_v['new_cols'])} stat(s) underperforming</span><br>"
                            f"<span style='color:#6e7681;font-size:0.76rem'>"
                            f"{_brs_names} has near-zero importance — auto-deselect will untick "
                            f"{'it' if len(_brs) == 1 else 'them'} while keeping the useful stats.</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Auto-deselect button ───────────────────────────
                    # Show whenever any bad/marginal transforms exist.
                    # Items without a resolved sug_idx are listed but can't be
                    # unticked automatically — show count in label so user knows.
                    # Auto-deselect button: imbalance is explicitly excluded — it is a
                    # model-level parameter, not a transform checkbox, and its effectiveness
                    # is assessed separately in Section ⑥ using test metrics.
                    _deselectable = [
                        v for v in _problem_verdicts
                        if v.get('sug_idx') is not None and v.get('type') != 'imbalance'
                    ]
                    # Also collect good date_features verdicts with bad sub-features
                    _date_subfeature_s5 = [
                        v for v in _verdicts
                        if v.get('bad_date_subfeatures') and v.get('sug_idx') is not None
                        and v.get('verdict') not in ('bad', 'marginal')
                    ]
                    # Also collect good row_numeric_stats verdicts with bad individual stats
                    _row_stat_s5 = [
                        v for v in _verdicts
                        if v.get('bad_row_stats') and v.get('sug_idx') is not None
                        and v.get('verdict') not in ('bad', 'marginal')
                    ]
                    _n_act = (
                        len(_deselectable)
                        + len(_date_subfeature_s5)
                        + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_s5)
                    )
                    _n_tot = (
                        len(_problem_verdicts)
                        + len(_date_subfeature_s5)
                        + sum(len(v.get('bad_row_stats') or []) for v in _row_stat_s5)
                    )
                    if _problem_verdicts or _date_subfeature_s5 or _row_stat_s5:
                        st.markdown("")
                        _btn_label = (
                            f"⚡ Auto-deselect {_n_act} bad/marginal transform(s) & re-train"
                            if _n_act == _n_tot
                            else f"⚡ Auto-deselect {_n_act} of {_n_tot} bad/marginal transform(s) & re-train"
                        )
                        if st.button(
                            _btn_label,
                            help="Unticks these transforms in Step ② and resets Step ③ — scroll up to re-train.",
                            type="primary",
                            key="_auto_deselect_step5",
                        ):
                            for _v in _deselectable:
                                st.session_state[f"_ck_persist_{_v['sug_idx']}"] = False
                            # Stage date sub-feature deselections in a non-widget key.
                            # ui_components applies them before st.checkbox is called on the next run.
                            _pending_s5 = st.session_state.get('_pending_date_deselect', {})
                            for _v in _date_subfeature_s5:
                                _si = _v['sug_idx']
                                _pending_s5[_si] = list(_v['bad_date_subfeatures'])
                            st.session_state['_pending_date_deselect'] = _pending_s5
                            # Stage row-stat sub-feature deselections
                            _pending_rs_s5 = st.session_state.get('_pending_row_stat_deselect', {})
                            for _v in _row_stat_s5:
                                _si = _v['sug_idx']
                                _pending_rs_s5[_si] = list(_v['bad_row_stats'])
                            st.session_state['_pending_row_stat_deselect'] = _pending_rs_s5
                            # Only clear model/training state — keep verdicts & FI visible
                            for _rk in [
                                'baseline_model', 'enhanced_model',
                                'baseline_train_cols', 'enhanced_train_cols',
                                'baseline_val_metrics', 'enhanced_val_metrics',
                                'baseline_col_encoders', 'enhanced_col_encoders',
                                'fitted_params', 'X_train_enhanced',
                                'X_test_raw', 'X_test_enhanced', '_test_df_original',
                                '_test_baseline_metrics', '_test_enhanced_metrics',
                            ]:
                                st.session_state[_rk] = None
                            st.session_state['_verdicts_stale'] = True
                            st.rerun()
                    st.markdown("")

                # ── Low-importance original columns (compact) ──────────
                if _low_imp:
                    _sorted_low = sorted(_low_imp.items(), key=lambda x: x[1])
                    _pills = " ".join(
                        f"<code style='font-size:0.78rem;background:#1c2128;padding:2px 7px;"
                        f"border-radius:3px;border:1px solid #30363d'>{c}&nbsp;{p:.2f}%</code>"
                        for c, p in _sorted_low
                    )
                    st.markdown(
                        f"<details style='margin-bottom:12px'>"
                        f"<summary style='cursor:pointer;color:#8b949e;font-size:0.83rem;"
                        f"user-select:none'>🗑️ {len(_low_imp)} low-importance original "
                        f"column(s) — near-zero baseline importance</summary>"
                        f"<div style='margin-top:8px;color:#6e7681;font-size:0.80rem'>"
                        f"Removing these may reduce noise and speed up training.<br><br>"
                        f"{_pills}</div></details>",
                        unsafe_allow_html=True,
                    )

                # ── Full verdict table (collapsed) ─────────────────────
                with st.expander(f"📋 All {len(_verdicts)} transform verdicts", expanded=False):
                    for _vk, _vl, _vi in [
                        ('good',     'Contributed',      '✅'),
                        ('marginal', 'Marginal',         '⚠️'),
                        ('bad',      'No contribution',  '❌'),
                    ]:
                        _grp = [v for v in _verdicts if v['verdict'] == _vk]
                        if not _grp:
                            continue
                        st.markdown(
                            f"<p style='font-size:0.8rem;font-weight:600;color:#8b949e;"
                            f"margin:10px 0 4px 0'>{_vi} {_vl} ({len(_grp)})</p>",
                            unsafe_allow_html=True,
                        )
                        for _v in _grp:
                            _col_str = _v['column']
                            if _v.get('column_b'):
                                _col_str += f" × {_v['column_b']}"
                            _ticon = _TICON.get(_v.get('type', ''), '•')
                            st.markdown(
                                f"<div style='padding:3px 10px;margin:2px 0;background:#0d1117;"
                                f"border-radius:4px;font-size:0.79rem;color:#c9d1d9'>"
                                f"{_ticon}&ensp;<code style='font-size:0.79rem'>{_col_str}</code>"
                                f"&ensp;<span style='color:#79c0ff'>{_v['method']}</span>"
                                f"&ensp;<span style='color:#6e7681'>— {_v['reason']}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # ── Recommendations ────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 💡 Recommendations")
                _rec_items = []

                # Use test metrics for recommendations (more reliable than val)
                _t_base_metrics = st.session_state.get('_test_baseline_metrics') or {}
                _t_enh_metrics  = st.session_state.get('_test_enhanced_metrics') or {}
                _tb_f1   = _t_base_metrics.get('f1')
                _te_f1   = _t_enh_metrics.get('f1')
                _tb_auc  = _t_base_metrics.get('roc_auc')
                _te_auc  = _t_enh_metrics.get('roc_auc')

                # 1. Underperforming transforms
                if _n_remove > 0:
                    _bad_names = [
                        (f"`{v['column']}`" + (f" × `{v['column_b']}`" if v.get('column_b') else "")
                         + f" — *{v['method']}*")
                        for v in _verdicts
                        if v['verdict'] in ('bad', 'marginal') and v.get('type') != 'imbalance'
                    ]
                    _rec_items.append((
                        "🔁",
                        f"**Re-run without {_n_remove} underperforming transform(s)**  \n"
                        f"Use the **⚡ Auto-deselect** button above, or untick {'it' if _n_remove == 1 else 'them'} "
                        f"manually in Step ②, then re-train for a leaner pipeline:  \n"
                        + "  \n".join(f"  - {n}" for n in _bad_names),
                        "normal",
                    ))
                else:
                    _rec_items.append((
                        "✅",
                        "**All transforms contributed** — the pipeline is already well-pruned.",
                        "good",
                    ))

                # 2. Low-importance columns
                if _low_imp:
                    _sorted_low_names = sorted(_low_imp.items(), key=lambda x: x[1])
                    _col_list_str = ", ".join(f"`{c}` ({p:.2f}%)" for c, p in _sorted_low_names)
                    _rec_items.append((
                        "🗑️",
                        f"**Consider dropping {len(_low_imp)} low-importance original column(s)**  \n"
                        f"{_col_list_str}",
                        "normal",
                    ))

                # 3. Imbalance effectiveness (using test F1)
                _imb_suggestion_exists = any(
                    s.get('type') == 'imbalance'
                    for s in st.session_state.get('suggestions', [])
                )
                if st.session_state.get('apply_imbalance') and _tb_f1 is not None and _te_f1 is not None:
                    _f1_delta  = _te_f1 - _tb_f1
                    _auc_delta = (_te_auc - _tb_auc) if (_tb_auc is not None and _te_auc is not None) else None
                    if _f1_delta > 0.02:
                        _imb_rating = "🟢 **Effective**"
                        if _auc_delta is None:
                            _auc_clause = "."
                        elif _auc_delta >= 0:
                            _auc_clause = f", and ROC-AUC by **+{_auc_delta:.4f}**."
                        else:
                            _auc_clause = f", though ROC-AUC decreased by **{abs(_auc_delta):.4f}**."
                        _imb_detail = (
                            f"Class reweighting improved test F1 by **+{_f1_delta:.4f}**"
                            + _auc_clause
                            + " Keep it enabled."
                        )
                    elif _f1_delta > 0:
                        _imb_rating = "🟡 **Marginal**"
                        _imb_detail = (
                            f"Class reweighting gave a small test F1 gain of **+{_f1_delta:.4f}**. "
                            "Consider a sampling strategy (e.g. SMOTE) for more impact."
                        )
                    else:
                        _imb_rating = "🔴 **Ineffective on test set**"
                        _imb_detail = (
                            f"Class reweighting did not improve test F1 (Δ = **{_f1_delta:+.4f}**). "
                            "Consider disabling it or using an oversampling approach instead."
                        )
                    _rec_items.append((
                        "⚖️",
                        f"**Imbalance handling** — {_imb_rating}  \n{_imb_detail}",
                        "normal",
                    ))
                elif _imb_suggestion_exists and not st.session_state.get('apply_imbalance') and _tb_f1 is not None:
                    _tb_prec = _t_base_metrics.get('precision')
                    _tb_rec  = _t_base_metrics.get('recall')
                    _WEAK_THRESHOLD = 0.80
                    _weak_metrics = {
                        k: v for k, v in [('F1', _tb_f1), ('Precision', _tb_prec), ('Recall', _tb_rec)]
                        if v is not None and v < _WEAK_THRESHOLD
                    }
                    if _weak_metrics:
                        _weak_str = ", ".join(f"**{k}={v:.3f}**" for k, v in _weak_metrics.items())
                        _rec_items.append((
                            "⚖️",
                            f"**Consider enabling imbalance handling**  \n"
                            f"Baseline test metrics are low: {_weak_str}. "
                            f"Go back to Step ② and tick the imbalance checkbox, then re-train.",
                            "normal",
                        ))
                    else:
                        _rec_items.append((
                            "⚖️",
                            f"**Imbalance handling not needed** — baseline metrics are strong "
                            f"(F1={_tb_f1:.3f}"
                            + (f", Precision={_tb_prec:.3f}" if _tb_prec is not None else "")
                            + (f", Recall={_tb_rec:.3f}" if _tb_rec is not None else "")
                            + f"). The model handles the class imbalance on its own.",
                            "good",
                        ))

                # 4. HP tuning headroom (using test AUC)
                if _tb_auc is not None and _te_auc is not None:
                    _headroom_pct = (_te_auc - _tb_auc) / max(1.0 - _tb_auc, 0.001) * 100
                    if _headroom_pct >= 20:
                        _rec_items.append((
                            "⚙️",
                            f"**HP tuning is likely worthwhile** ({_headroom_pct:.1f}% of gap captured)  \n"
                            f"Significant headroom remains. Consider a systematic HP search or "
                            f"increasing `n_estimators` / reducing `learning_rate`.",
                            "normal",
                        ))
                    elif _headroom_pct >= 5:
                        _rec_items.append((
                            "⚙️",
                            f"**Moderate HP tuning headroom** ({_headroom_pct:.1f}% of gap captured)  \n"
                            f"A light search on `num_leaves`, `learning_rate`, `min_child_samples` "
                            f"could further close the gap.",
                            "normal",
                        ))

                for _r_icon, _r_text, _r_type in _rec_items:
                    if _r_type == "good":
                        st.success(f"{_r_icon} {_r_text}")
                    else:
                        st.info(f"{_r_icon} {_r_text}")
            # ── End post-training analysis ────────────────────────────

            # ── Download Predictions ───────────────────────────────
            st.divider()
            st.subheader("📥 Download Predictions")

            # We need predictions from the enhanced model (primary) and
            # access to the original test columns (including dropped ID cols).
            _enh_metrics = st.session_state.get('_test_enhanced_metrics') or {}
            _y_pred      = _enh_metrics.get('_y_pred')
            _y_proba     = _enh_metrics.get('_y_pred_proba')
            _le          = st.session_state.get('label_encoder')
            _test_orig   = st.session_state.get('_test_df_original')  # before column drops

            if _y_pred is not None and _le is not None and _test_orig is not None:
                # ── Apply custom threshold when set ──────────────────────
                # The threshold slider (PR-curve tuning in evaluation mode,
                # or the predict-only threshold slider) lets users pick a
                # cutoff different from 0.5.  Re-derive hard predictions
                # here so the download reflects the chosen threshold.
                _n_classes   = st.session_state.get('n_classes', 2)
                _custom_threshold = st.session_state.get('_pr_threshold_slider')
                if (_custom_threshold is not None
                        and _n_classes == 2
                        and _y_proba is not None
                        and _y_proba.ndim == 2
                        and _y_proba.shape[1] >= 2):
                    _y_pred = (
                        _y_proba[:, 1] >= _custom_threshold
                    ).astype(int)

                # ── Build available columns ──────────────────────────────
                # Decode predictions back to original class labels
                _pred_labels = _le.inverse_transform(_y_pred)
                _class_names = list(_le.classes_)

                # Prediction columns that will be appended
                _pred_col_name = 'predicted_class'
                _prob_col_names = [f'prob_{c}' for c in _class_names]

                # Actual class column (available when test set had a target)
                _actual_col_name = 'actual_class'
                _y_true_raw = _enh_metrics.get('_y_true')
                _has_actual = _y_true_raw is not None
                if _has_actual:
                    _actual_labels = _le.inverse_transform(
                        np.array(_y_true_raw).astype(int)
                    )

                # Identify which original columns are ID columns (were
                # auto-dropped) vs regular feature columns.
                _applied_drops = st.session_state.get('_applied_drops') or []
                _skipped_info  = st.session_state.get('skipped_info') or {}
                _id_col_names  = set(_skipped_info.get('id_columns', {}).keys())

                # Columns from the original test set that were dropped
                # (these include IDs, constants, user-dropped cols).
                _dropped_cols = [c for c in _test_orig.columns
                                 if c in _applied_drops]
                # Columns that survived into the model
                _feature_cols = [c for c in _test_orig.columns
                                 if c not in _applied_drops]

                # ── Default selection logic ──────────────────────────────
                # Pre-select: ID columns + prediction + probabilities
                # Deselect: all feature columns (user already has those
                # in the raw/enhanced downloads and typically only wants
                # an ID + prediction mapping).
                _all_original_cols = list(_test_orig.columns)
                _all_pred_cols     = [_pred_col_name] + ([_actual_col_name] if _has_actual else []) + _prob_col_names

                # Group options for clarity
                _id_options = [c for c in _all_original_cols if c in _id_col_names]
                _other_dropped = [c for c in _dropped_cols if c not in _id_col_names]
                _non_dropped   = [c for c in _feature_cols]

                # Build the full option list with logical grouping
                _all_options = []
                _default_selection = []

                # ID columns first (pre-selected)
                for c in _id_options:
                    _all_options.append(f"🆔 {c}")
                    _default_selection.append(f"🆔 {c}")

                # Other dropped columns (not pre-selected)
                for c in _other_dropped:
                    _all_options.append(f"🗑️ {c}")

                # Feature columns (not pre-selected)
                for c in _non_dropped:
                    _all_options.append(c)

                # Prediction columns (pre-selected)
                for c in _all_pred_cols:
                    _lbl = f"🎯 {c}"
                    _all_options.append(_lbl)
                    _default_selection.append(_lbl)

                st.caption(
                    "Select which columns to include in the prediction export. "
                    "🆔 = ID column (auto-detected, dropped during training), "
                    "🗑️ = other dropped column, "
                    "🎯 = model prediction.  \n"
                    "By default only ID and prediction columns are selected."
                )

                # Show which threshold is active for the predictions
                if (_custom_threshold is not None
                        and _n_classes == 2
                        and abs(_custom_threshold - 0.5) > 1e-4):
                    st.info(
                        f"🎚️ Using custom decision threshold **{_custom_threshold:.2f}** "
                        f"(from threshold tuning above). Predicted classes reflect this cutoff."
                    )

                _selected = st.multiselect(
                    "Columns to include",
                    options=_all_options,
                    default=_default_selection,
                    key="_pred_dl_cols",
                )

                if _selected:
                    # ── Assemble the export DataFrame ────────────────────
                    _export_df = pd.DataFrame(index=range(len(_y_pred)))

                    for _opt in _selected:
                        # Strip prefix emoji labels to get the real column name
                        if _opt.startswith("🆔 "):
                            _real = _opt[2:].strip()
                            _export_df[_real] = _test_orig[_real].values
                        elif _opt.startswith("🗑️ "):
                            _real = _opt[2:].strip()
                            _export_df[_real] = _test_orig[_real].values
                        elif _opt.startswith("🎯 "):
                            _real = _opt[2:].strip()
                            if _real == _pred_col_name:
                                _export_df[_pred_col_name] = _pred_labels
                            elif _real == _actual_col_name and _has_actual:
                                _export_df[_actual_col_name] = _actual_labels
                            elif _real in _prob_col_names:
                                _ci = _prob_col_names.index(_real)
                                _export_df[_real] = _y_proba[:, _ci]
                        else:
                            # Regular feature column
                            if _opt in _test_orig.columns:
                                _export_df[_opt] = _test_orig[_opt].values

                    # Preview
                    st.dataframe(_export_df.head(10), use_container_width=True, height=250)
                    st.caption(f"Showing first 10 of {len(_export_df):,} rows  ·  {len(_export_df.columns)} columns selected")

                    _pred_csv = _export_df.to_csv(index=False).encode('utf-8')
                    _fname = (st.session_state.get('_test_file_name') or 'test').replace('.csv', '')
                    st.download_button(
                        label=f"⬇️ Download predictions ({len(_export_df.columns)} columns)",
                        data=_pred_csv,
                        file_name=f"{_fname}_predictions.csv",
                        mime="text/csv",
                        key="dl_pred_csv",
                        type="primary",
                    )
                else:
                    st.warning("Select at least one column to download.")

                # ── Collapsible: raw / enhanced test data ────────────────
                with st.expander("💾 Download raw / transformed test features", expanded=False):
                    _tdl1, _tdl2 = st.columns(2)
                    with _tdl1:
                        if st.session_state.get('X_test_raw') is not None:
                            _traw_csv = st.session_state.X_test_raw.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Test set (raw features)",
                                data=_traw_csv,
                                file_name="X_test_raw.csv",
                                mime="text/csv",
                                key="dl_test_raw",
                            )
                    with _tdl2:
                        if st.session_state.get('X_test_enhanced') is not None:
                            _tenh_csv = st.session_state.X_test_enhanced.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Test set (post-transform)",
                                data=_tenh_csv,
                                file_name="X_test_enhanced.csv",
                                mime="text/csv",
                                key="dl_test_enh",
                            )

                # ── Collapsible: baseline model predictions (predict-only) ───
                if _is_predict_only:
                    _base_metrics = st.session_state.get('_test_baseline_metrics') or {}
                    _base_y_pred  = _base_metrics.get('_y_pred')
                    _base_y_proba = _base_metrics.get('_y_pred_proba')
                    if _base_y_pred is not None and _le is not None:
                        with st.expander("📊 Compare: Baseline model predictions", expanded=False):
                            st.caption(
                                "These are predictions from the **baseline** model (no feature engineering). "
                                "Compare with the enhanced predictions above to see the effect of your transforms."
                            )
                            _base_pred_labels = _le.inverse_transform(_base_y_pred)
                            _base_export = pd.DataFrame()
                            # Include ID columns
                            for _idc in _id_options:
                                _real = _idc[2:].strip() if _idc.startswith("🆔 ") else _idc
                                if _real in _test_orig.columns:
                                    _base_export[_real] = _test_orig[_real].values
                            _base_export['predicted_class_baseline'] = _base_pred_labels
                            _base_export['predicted_class_enhanced'] = _pred_labels
                            if _base_y_proba is not None and _y_proba is not None:
                                for _ci, _cn in enumerate(_class_names):
                                    _base_export[f'prob_{_cn}_baseline'] = _base_y_proba[:, _ci]
                                    _base_export[f'prob_{_cn}_enhanced'] = _y_proba[:, _ci]
                            # Flag rows where baseline and enhanced disagree
                            _base_export['models_agree'] = (
                                _base_export['predicted_class_baseline'] == _base_export['predicted_class_enhanced']
                            )
                            _n_disagree = int((~_base_export['models_agree']).sum())
                            st.write(
                                f"Models **agree** on {len(_base_export) - _n_disagree:,} / "
                                f"{len(_base_export):,} rows "
                                f"(**{_n_disagree:,}** disagreements, "
                                f"{_n_disagree / max(len(_base_export), 1) * 100:.1f}%)."
                            )
                            st.dataframe(_base_export.head(10), use_container_width=True, height=250)
                            _base_csv = _base_export.to_csv(index=False).encode('utf-8')
                            _fname_b = (st.session_state.get('_test_file_name') or 'test').replace('.csv', '')
                            st.download_button(
                                label=f"⬇️ Download baseline vs enhanced comparison",
                                data=_base_csv,
                                file_name=f"{_fname_b}_baseline_vs_enhanced.csv",
                                mime="text/csv",
                                key="dl_pred_compare",
                            )
            else:
                _po_msg = "generating predictions" if st.session_state.get('_predict_only') else "evaluating on the test set"
                st.info(f"Predictions will be available after {_po_msg}.")

            # ── Export full report with test metrics ──────────────────
            if not _is_predict_only:
                st.divider()
                st.subheader("📥 Export Full Report")
                try:
                    from report_buttons import add_report_download_buttons
                    _lbl = (st.session_state.get('_test_file_name') or 'dataset').replace('.csv', '')
                    add_report_download_buttons(
                        st.session_state,
                        dataset_name=_lbl,
                        key_suffix="test",
                        report_stage="test",
                        test_baseline_metrics=baseline_test,
                        test_enhanced_metrics=enhanced_test,
                    )
                except ImportError:
                    st.info("Place `report_generator.py` in the same directory to enable report export.")
            # ─────────────────────────────────────────────────────────
    else:
        _render_locked_step("Train both models in **③** first.")
                    


if __name__ == '__main__':
    main()