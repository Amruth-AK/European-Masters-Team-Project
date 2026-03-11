"""
chat_component.py — LLM Chat Assistant for the Feature Engineering Recommender
===============================================================================

Drop this file in the same directory as recommend_app.py, then add two lines:

    1. At the top of recommend_app.py (with the other imports):
         from chat_component import render_chat_sidebar

    2. Inside main(), inside the `with st.sidebar:` block (around line 3783),
       after the Settings section ends:
         render_chat_sidebar(st.session_state)

The chat lives in the sidebar so it is accessible at every step of the workflow.

Requirements:
    pip install google-generativeai

Get a free API key at: https://aistudio.google.com/app/apikey
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Tool description — explains the app to the LLM regardless of session state
# ---------------------------------------------------------------------------

TOOL_DESCRIPTION = """
You are an expert ML assistant embedded in the "Feature Engineering Recommender" tool.

=== WHAT THIS TOOL DOES ===
This is an open-source Python/Streamlit application that helps practitioners improve
predictive performance on tabular classification datasets. It works as follows:

STEP ① — Upload Training Data
  The user uploads a CSV file and selects the target column.
  The tool automatically detects feature types (numeric vs categorical), missing values,
  class imbalance, and other dataset characteristics.

STEP ② — Analyze & Get Suggestions
  The tool computes ~50 meta-features describing the dataset (size, skewness, cardinality,
  correlations, class balance, etc.) and ~15 meta-features per column.
  These are fed into trained LightGBM meta-models (one per transform type) that predict
  which feature engineering transforms will improve AUC on THIS specific dataset.
  Each suggestion has two delta values: predicted_delta_raw is the model's actual predicted AUC 
  improvement (the meaningful number, typically between -0.1 and +0.1), and predicted_delta is a 
  z-score used only for cross-type ranking. Always quote predicted_delta_raw when discussing 
  predicted impact with the user."

  Available transform families:
    Numerical:    log_transform, sqrt_transform, polynomial_square,
                  polynomial_cube, reciprocal_transform, quantile_binning,
                  impute_median, missing_indicator
    Categorical:  frequency_encoding, target_encoding, onehot_encoding,
                  hashing_encoding, missing_indicator
    Interactions: product_interaction, division_interaction,
                  addition_interaction, abs_diff_interaction (num x num),
                  group_mean, group_std (num grouped by cat),
                  cat_concat (cat x cat)
    Row-level:    row_numeric_stats (mean/median/sum/std/min/max/range per row),
                  row_zero_stats, row_missing_stats

STEP ③ — Train Baseline & Enhanced Models
  Two LightGBM classifiers are trained:
    - Baseline: raw features, label-encoded categoricals
    - Enhanced: same features but with the selected transforms applied
  Both are evaluated on a held-out validation split.
  Metrics: ROC-AUC, Accuracy, F1, Precision, Recall, Log-loss.

STEP ④ — Test & Compare
  The user uploads a test CSV. Both models are evaluated on it side-by-side.

STEP ⑤ — Analysis & Recommendations
  Further actionable recommendations based on test results (e.g. handling class
  imbalance, further feature engineering). A full report can be exported.

=== META-MODEL BACKGROUND ===
The meta-models were trained on hundreds of Kaggle competition datasets.
For each dataset and transform, the actual AUC delta was measured, meta-features
were computed, and LightGBM models were trained to predict whether a given
transform helps on a new unseen dataset. This encodes the intuition of
experienced data scientists / Kaggle Grandmasters.

=== POST-TRAINING ANALYSIS ===
After training, the tool provides:
  - Feature importance breakdown: how much importance the baseline vs new (engineered)
    features carry in the enhanced model.  The top baseline and top new features are listed.
  - Suggestion verdicts: each applied transform is rated as 'good' (contributed meaningfully),
    'marginal' (minimal impact), or 'bad' (no contribution).  The verdict includes a reason
    based on feature importance.
  - Low-importance original columns: baseline columns with < 0.5% importance that the user
    may consider dropping.
  - Recommendations on imbalance handling, HP tuning headroom, and pipeline pruning.

=== HYPERPARAMETER TUNING ===
Users can:
  - Manually adjust LightGBM hyperparameters (num_leaves, learning_rate, etc.).
  - Run Optuna automated HP search, which finds optimal params via cross-validation.
  - Advisory-suggested HP overrides (e.g. for class imbalance or high cardinality).

=== YOUR ROLE ===
- Answer questions about the user's specific dataset and results (shown in context below).
- Explain WHY a recommended transform was suggested based on dataset characteristics.
- Explain suggestion verdicts: why a transform was rated good/marginal/bad.
- Advise on feature importance: which features matter, which engineered features paid off.
- Discuss hyperparameter tuning: what was changed, Optuna results, what to try next.
- Explain imbalance handling effectiveness based on test metrics.
- Suggest what the user should try next.
- Explain any metric or concept the user asks about.
- Be concise and practical.
- NEVER invent or estimate metric values — only quote numbers from the session context.
- If something hasn't been computed yet, say so clearly.
"""


# ---------------------------------------------------------------------------
# Context builder — reads correct session_state keys
# ---------------------------------------------------------------------------

def _build_context(state: dict) -> str:
    """Extract current session state into a structured context string."""
    import numpy as np
    from scipy.stats import skew as _skew

    lines = ["=== CURRENT SESSION STATE ==="]
    anything = False

    # ── Dataset basics ───────────────────────────────────────────────────────
    train_df  = state.get("train_df")
    X_train   = state.get("X_train")
    target    = state.get("target_col")
    n_classes = state.get("n_classes")

    df = train_df if train_df is not None else X_train
    if df is not None:
        anything = True
        lines.append("\n[Dataset — uploaded]")
        lines.append(f"  Rows:    {df.shape[0]}")
        lines.append(f"  Columns: {df.shape[1]}")
        if target:
            lines.append(f"  Target:  {target}")
        if n_classes:
            lines.append(f"  Classes: {n_classes}")
            # Task framing hint from column names
            _domain_hints = []
            _domain_kw = {
                'churn': 'customer churn', 'fraud': 'fraud detection', 'price': 'price prediction',
                'survived': 'survival', 'default': 'credit default', 'revenue': 'revenue',
                'age': 'demographics', 'income': 'income/financial', 'salary': 'salary',
                'click': 'click-through / CTR', 'purchase': 'purchase / e-commerce',
                'score': 'scoring', 'rating': 'ratings', 'loan': 'lending / credit',
                'medical': 'medical / health', 'patient': 'medical / health',
                'disease': 'medical / health', 'diagnosis': 'medical / health',
            }
            for col_name in list(df.columns) + ([target] if target else []):
                for kw, domain in _domain_kw.items():
                    if kw in str(col_name).lower() and domain not in _domain_hints:
                        _domain_hints.append(domain)
            if _domain_hints:
                lines.append(f"  Domain hints: {', '.join(_domain_hints[:3])}")

        try:
            num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
            cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != target]
            lines.append(f"  Numeric features:     {len(num_cols)}")
            lines.append(f"  Categorical features: {len(cat_cols)}")

            miss = df.isnull().mean()
            cols_with_missing = miss[miss > 0].drop(target, errors="ignore")
            if len(cols_with_missing):
                lines.append(f"  Columns with missing values: {len(cols_with_missing)}")
                for col, pct in cols_with_missing.nlargest(5).items():
                    lines.append(f"    - {col}: {pct:.1%} missing")
            else:
                lines.append("  Missing values: none")

            if target and target in df.columns:
                vc = df[target].value_counts(normalize=True)
                lines.append("  Class distribution:")
                for cls, pct in vc.items():
                    lines.append(f"    - class {cls}: {pct:.1%}")
        except Exception:
            pass

    # ── Column profiles (detected types from col_type_info) ──────────────────
    col_type_info = state.get("_col_type_info")
    X_ctx = state.get("X_train")
    if col_type_info and X_ctx is not None:
        anything = True
        lines.append("\n[Column profiles — per-column analysis]")
        lines.append(f"  {'Column':<22} {'Type':<12} {'n_unique':>8} {'Missing%':>9} {'Drop?':>6}")
        lines.append("  " + "-" * 62)
        for col_name, info in list(col_type_info.items())[:30]:   # cap at 30
            detected  = info.get('detected', '?')
            n_unique  = info.get('n_unique', '?')
            miss_pct  = f"{info.get('missing_pct', 0)*100:.0f}%"
            drop_flag = "⚠️ YES" if info.get('drop_suggested') else "no"
            lines.append(
                f"  {str(col_name)[:22]:<22} {detected:<12} {str(n_unique):>8} {miss_pct:>9} {drop_flag:>6}"
            )
        if len(col_type_info) > 30:
            lines.append(f"  … +{len(col_type_info)-30} more columns")

    # ── Numeric column summaries ──────────────────────────────────────────────
    if X_ctx is not None:
        try:
            _nc = X_ctx.select_dtypes(include="number").columns.tolist()
            _nc = [c for c in _nc if c != target]
            if _nc:
                lines.append("\n[Numeric column summaries — mean / std / min / max / skew / outlier%]")
                lines.append(f"  {'Column':<22} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'skew':>6} {'out%':>6}")
                lines.append("  " + "-" * 72)
                for _c in _nc[:20]:
                    s = X_ctx[_c].dropna()
                    if len(s) < 3:
                        continue
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    out_ratio = float(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean()) if iqr > 0 else 0.0
                    try:
                        from scipy.stats import skew as _skew_fn
                        sk = float(_skew_fn(s, nan_policy='omit'))
                    except Exception:
                        sk = 0.0
                    lines.append(
                        f"  {str(_c)[:22]:<22} "
                        f"{s.mean():>8.3g} {s.std():>8.3g} "
                        f"{s.min():>8.3g} {s.max():>8.3g} "
                        f"{sk:>6.2f} {out_ratio*100:>5.1f}%"
                    )
                if len(_nc) > 20:
                    lines.append(f"  … +{len(_nc)-20} more numeric columns")
        except Exception:
            pass

    # ── Sample rows ───────────────────────────────────────────────────────────
    if X_ctx is not None and target and target in (df.columns if df is not None else []):
        try:
            _sample_df = df.sample(min(5, len(df)), random_state=42)
            lines.append("\n[Sample rows — 5 random rows (all columns)]")
            lines.append("  " + " | ".join(f"{c[:12]:<12}" for c in _sample_df.columns[:8]))
            lines.append("  " + "-" * min(100, 16 * min(8, len(_sample_df.columns))))
            for _, row in _sample_df.iterrows():
                lines.append(
                    "  " + " | ".join(f"{str(v)[:12]:<12}" for v in list(row.values)[:8])
                )
            if len(_sample_df.columns) > 8:
                lines.append(f"  … +{len(_sample_df.columns)-8} more columns not shown")
        except Exception:
            pass

    # ── Quick Clean status ────────────────────────────────────────────────────
    if state.get('_qc_applied'):
        anything = True
        lines.append("\n[Quick Clean — applied]")
        if state.get('_qc_impute'):
            lines.append("  - Missing value imputation: numeric → median, categorical → mode")
        if state.get('_qc_outlier'):
            _qc_summary = state.get('_qc_outlier_summary', '')
            lines.append(f"  - Outlier capping: ±3 IQR on columns with >5% outliers{(' — ' + _qc_summary) if _qc_summary else ''}")
    elif state.get('_qc_X_clean') is not None:
        lines.append("\n[Quick Clean — preview computed but not yet applied]")

    # ── Applied drops from column panel ──────────────────────────────────────
    _applied_drops = state.get('_applied_drops') or []
    _applied_types = state.get('_applied_types') or {}
    if _applied_drops or _applied_types:
        anything = True
        lines.append("\n[Step 1 column changes]")
        if _applied_drops:
            lines.append(f"  Dropped columns ({len(_applied_drops)}): {', '.join(f'`{c}`' for c in _applied_drops)}")
        if _applied_types:
            lines.append(f"  Type overrides ({len(_applied_types)}):")
            for _c, _t in list(_applied_types.items())[:10]:
                lines.append(f"    - `{_c}` → {_t}")

    # ── Suggestions (ALL, not just top 10) ───────────────────────────────────
    suggestions = state.get("suggestions")
    if suggestions:
        anything = True
        sorted_s = sorted(
            [s for s in suggestions if s.get("predicted_delta") is not None],
            key=lambda s: s.get("predicted_delta", 0),
            reverse=True,
        )
        lines.append(f"\n[Suggestions — {len(suggestions)} total, sorted by predicted delta AUC]")
        lines.append(f"  {'Method':<28} {'Column':<22} {'ColB':<18} {'Delta':>8} {'Selected':>9}")
        lines.append("  " + "-" * 88)
        selected_idx_set = set(state.get("selected_indices") or [])
        for _i, s in enumerate(sorted_s):
            _orig_idx = suggestions.index(s) if s in suggestions else _i
            col   = s.get("column", "?")
            col_b = s.get("column_b") or "-"
            meth  = s.get("method", "?")
            delta = s.get("predicted_delta_raw", s.get("predicted_delta", 0))
            sel   = "✅" if _orig_idx in selected_idx_set else "☐"
            lines.append(
                f"  {meth[:28]:<28} {str(col)[:22]:<22} {str(col_b)[:18]:<18} {delta:>+8.4f} {sel:>9}"
            )

    # ── Selected transforms ───────────────────────────────────────────────────
    selected_indices = state.get("selected_indices")
    all_suggestions  = state.get("suggestions") or []
    if selected_indices is not None and all_suggestions:
        selected = [all_suggestions[i] for i in selected_indices if i < len(all_suggestions)]
        if selected:
            anything = True
            lines.append(f"\n[User-selected transforms — {len(selected)} chosen]")
            for s in selected:
                col   = s.get("column", "?")
                col_b = s.get("column_b")
                meth  = s.get("method", "?")
                delta = s.get("predicted_delta")
                entry = f"  - {meth} on '{col}'"
                if col_b:
                    entry += f" x '{col_b}'"
                if delta is not None:
                    entry += f"  (predicted delta AUC: {delta:+.4f})"
                lines.append(entry)

    # ── Feature column names ─────────────────────────────────────────────────
    X_train_ctx = state.get("X_train")
    target_ctx  = state.get("target_col")
    if X_train_ctx is not None:
        feat_cols = [c for c in X_train_ctx.columns if c != target_ctx]
        if feat_cols:
            preview = feat_cols[:20]
            suffix  = f" … +{len(feat_cols)-20} more" if len(feat_cols) > 20 else ""
            lines.append(f"\n[Feature columns — {len(feat_cols)} total]")
            lines.append("  " + ", ".join(preview) + suffix)

    # ── Dataset meta (baseline score, imbalance ratio, etc.) ─────────────────
    ds_meta = state.get("ds_meta")
    if ds_meta:
        anything = True
        lines.append("\n[Dataset meta-features (from analysis step)]")
        for k in ["n_rows", "n_cols", "n_numeric_cols", "n_cat_cols",
                  "missing_ratio", "class_imbalance_ratio",
                  "baseline_score", "relative_headroom"]:
            v = ds_meta.get(k)
            if v is not None:
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Advisories (class imbalance, HP suggestions, etc.) ───────────────────
    advisories = state.get("advisories")
    if advisories:
        anything = True
        lines.append(f"\n[Advisories — {len(advisories)} items]")
        for adv in advisories:
            title   = adv.get("title", "?")
            explain = adv.get("explain") or adv.get("message") or ""
            # Truncate long explanations
            short   = explain[:200].replace("\n", " ")
            lines.append(f"  - {title}: {short}")

    # ── Skipped / problematic columns (ID, constant, etc.) ─────────────────
    skipped_info = state.get("skipped_info")
    if skipped_info:
        anything = True
        id_cols    = skipped_info.get('id_columns', {})
        const_cols = skipped_info.get('constant_columns', {})
        if id_cols or const_cols:
            lines.append("\n[Auto-skipped columns]")
            if id_cols:
                lines.append(f"  ID columns ({len(id_cols)}): {', '.join(f'`{c}`' for c in list(id_cols)[:10])}")
            if const_cols:
                lines.append(f"  Constant columns ({len(const_cols)}): {', '.join(f'`{c}`' for c in list(const_cols)[:10])}")

    # ── Pre-analysis baseline estimate ────────────────────────────────────────
    _bs = state.get("_analyze_baseline_score")
    _bs_std = state.get("_analyze_baseline_std")
    if _bs is not None:
        anything = True
        lines.append("\n[Pre-analysis baseline estimate]")
        lines.append(f"  Quick baseline ROC-AUC: {_bs:.4f}" + (f" ± {_bs_std:.4f}" if _bs_std else ""))

    # ── Imbalance handling ────────────────────────────────────────────────────
    if state.get("apply_imbalance"):
        anything = True
        lines.append("\n[Class imbalance handling — ENABLED]")
        lines.append("  The enhanced model uses class reweighting (is_unbalance or class_weight='balanced').")
    elif state.get("apply_imbalance") is False and state.get("baseline_model") is not None:
        lines.append("\n[Class imbalance handling — DISABLED by user]")

    fitted_params = state.get("fitted_params")
    if fitted_params:
        lines.append(f"\n[Transforms applied to data — {len(fitted_params)} total]")
        for p in fitted_params:
            _m = p.get('method', '?')
            _c = p.get('column', '?')
            _cb = p.get('column_b')
            _nc = p.get('new_cols') or []
            entry = f"  - {_m} on `{_c}`"
            if _cb:
                entry += f" × `{_cb}`"
            if _nc:
                entry += f"  → new cols: {', '.join(f'`{c}`' for c in _nc[:3])}"
                if len(_nc) > 3:
                    entry += f" … +{len(_nc)-3} more"
            lines.append(entry)

    # ── Hyperparameter overrides ──────────────────────────────────────────────
    hp_overrides = state.get("_user_hp_overrides")
    if hp_overrides:
        try:
            from app_constants import BASE_PARAMS as _BP
        except ImportError:
            _BP = {}
        _changes = {k: v for k, v in hp_overrides.items() if v != _BP.get(k)}
        if _changes:
            anything = True
            lines.append(f"\n[Hyperparameter overrides — {len(_changes)} change(s) from defaults]")
            for k, v in _changes.items():
                lines.append(f"  {k}: {_BP.get(k)} → {v}")

    # ── Optuna tuning results ─────────────────────────────────────────────────
    optuna_params = state.get("_optuna_best_params")
    optuna_score  = state.get("_optuna_best_score")
    if optuna_params:
        anything = True
        lines.append("\n[Optuna HP tuning results]")
        if optuna_score is not None:
            lines.append(f"  Best CV AUC: {optuna_score:.4f}")
        for k, v in optuna_params.items():
            lines.append(f"  {k}: {v}")

    # ── Validation metrics ────────────────────────────────────────────────────
    baseline_val = state.get("baseline_val_metrics")
    enhanced_val = state.get("enhanced_val_metrics")
    if baseline_val or enhanced_val:
        anything = True
        lines.append("\n[Model performance — VALIDATION SET]")
        _append_metrics_table(lines, baseline_val, enhanced_val)

    # ── Feature importance breakdown ──────────────────────────────────────────
    _fi_orig_pct = state.get("_fi_orig_pct")
    _fi_new_pct  = state.get("_fi_new_pct")
    _fi_b_pct    = state.get("_fi_b_pct")
    _fi_e_new    = state.get("_fi_e_new")
    _fi_e_orig   = state.get("_fi_e_original")
    if _fi_orig_pct is not None and _fi_new_pct is not None:
        anything = True
        lines.append("\n[Feature importance — enhanced model]")
        lines.append(f"  Original features share: {_fi_orig_pct:.1f}%")
        lines.append(f"  New (engineered) features share: {_fi_new_pct:.1f}%")

        if _fi_b_pct is not None and len(_fi_b_pct) > 0:
            top_base = _fi_b_pct.nlargest(5)
            lines.append("  Top 5 baseline features:")
            for c, v in top_base.items():
                lines.append(f"    - {c}: {v:.2f}%")

        if _fi_e_new is not None and len(_fi_e_new) > 0:
            top_new = _fi_e_new.nlargest(5)
            lines.append("  Top 5 new (engineered) features:")
            for c, v in top_new.items():
                lines.append(f"    - {c}: {v:.2f}%")

    # ── Suggestion verdicts (post-training diagnosis) ─────────────────────────
    verdicts = state.get("_suggestion_verdicts")
    if verdicts:
        anything = True
        n_good     = sum(1 for v in verdicts if v['verdict'] == 'good')
        n_marginal = sum(1 for v in verdicts if v['verdict'] == 'marginal')
        n_bad      = sum(1 for v in verdicts if v['verdict'] == 'bad')
        lines.append(f"\n[Post-training transform verdicts — {len(verdicts)} evaluated]")
        lines.append(f"  ✅ Contributed: {n_good}   ⚠️ Marginal: {n_marginal}   ❌ No contribution: {n_bad}")
        for v in verdicts:
            _icon = {'good': '✅', 'marginal': '⚠️', 'bad': '❌'}.get(v['verdict'], '?')
            _col_str = v.get('column', '?')
            if v.get('column_b'):
                _col_str += f" × {v['column_b']}"
            lines.append(f"  {_icon} {v.get('method', '?')} on {_col_str}: {v.get('reason', '')[:120]}")

    low_imp = state.get("_low_imp_cols")
    if low_imp:
        anything = True
        lines.append(f"\n[Low-importance original columns — {len(low_imp)} columns]")
        for c, pct in sorted(low_imp.items(), key=lambda x: x[1]):
            lines.append(f"  - {c}: {pct:.2f}% importance")

    # ── Test metrics ──────────────────────────────────────────────────────────
    baseline_test = state.get("_test_baseline_metrics")
    enhanced_test = state.get("_test_enhanced_metrics")
    test_file     = state.get("_test_file_name")
    predict_only  = state.get("_predict_only")
    if baseline_test or enhanced_test:
        anything = True
        _test_label = "TEST SET"
        if test_file:
            _test_label += f" ({test_file})"
        if predict_only:
            _test_label += " — predict-only mode (no target in test set)"
        lines.append(f"\n[Model performance — {_test_label}]")
        _append_metrics_table(lines, baseline_test, enhanced_test)

    if not anything:
        lines.append("\n  Nothing computed yet — user has not uploaded a dataset.")

    return "\n".join(lines)


def _append_metrics_table(lines: list, baseline: dict, enhanced: dict) -> None:
    metrics = ["roc_auc", "accuracy", "f1", "precision", "recall", "log_loss"]
    lines.append(f"  {'Metric':<12}  {'Baseline':>10}  {'Enhanced':>10}  {'Delta':>8}")
    lines.append("  " + "-" * 48)
    for m in metrics:
        b_val = (baseline or {}).get(m)
        e_val = (enhanced or {}).get(m)
        b_str = f"{b_val:.4f}" if b_val is not None else "         —"
        e_str = f"{e_val:.4f}" if e_val is not None else "         —"
        if b_val is not None and e_val is not None:
            d_str = f"{e_val - b_val:+.4f}"
        else:
            d_str = "         —"
        lines.append(f"  {m:<12}  {b_str:>10}  {e_str:>10}  {d_str:>8}")


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _convert_history_for_gemini(history: list) -> list:
    """Convert {role, content} list to Gemini format, excluding the last user msg."""
    return [
        {
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        }
        for msg in history[:-1]
    ]


def _stream_response(chat_session, prompt: str):
    response = chat_session.send_message(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_chat_sidebar(state: dict) -> None:
    """
    Render the assistant chat panel inside the Streamlit sidebar.

    Call this from INSIDE the `with st.sidebar:` block in main(),
    after the Settings section.

    Parameters
    ----------
    state : dict
        Pass ``st.session_state`` directly.
    """
    st.divider()
    st.subheader("🤖 ML Assistant")

    # ── API key ───────────────────────────────────────────────────────────────
    st.text_input(
        "Google AI API key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
        key="google_api_key",
    )

    if not state.get("google_api_key"):
        st.caption("Enter your API key above to chat.")
        return

    # ── Lazy import ───────────────────────────────────────────────────────────
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("Run: `pip install google-generativeai`")
        return

    genai.configure(api_key=state["google_api_key"])

    if "chat_history" not in state:
        state["chat_history"] = []

    # Counter used to reset the text_area by changing its key (avoids the
    # StreamlitAPIException raised when setting a widget-bound state key directly).
    if "chat_input_counter" not in state:
        state["chat_input_counter"] = 0

    # ── Chat history display (last 6 messages) ────────────────────────────────
    recent = state["chat_history"][-6:]
    for msg in recent:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if len(state["chat_history"]) > 6:
        st.caption(f"↑ {len(state['chat_history']) - 6} earlier messages not shown")

    # ── Input ─────────────────────────────────────────────────────────────────
    # st.chat_input() is not supported inside the sidebar,
    # so we use a text_area + button instead.
    # The key includes the counter so incrementing it forces a fresh empty widget.
    input_key = f"chat_input_box_{state['chat_input_counter']}"
    user_input = st.text_area(
        "Message",
        placeholder="Ask about your dataset, transforms, or results…",
        label_visibility="collapsed",
        height=80,
        key=input_key,
    )

    col_send, col_clear = st.columns([2, 1])
    send  = col_send.button("Send ↩",  use_container_width=True, type="primary")
    clear = col_clear.button("Clear",  use_container_width=True)

    if clear:
        state["chat_history"] = []
        state["chat_input_counter"] += 1
        st.rerun()

    if send and user_input.strip():
        prompt = user_input.strip()
        state["chat_history"].append({"role": "user", "content": prompt})

        system_prompt = TOOL_DESCRIPTION + "\n\n" + _build_context(state)

        model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction=system_prompt,
        )
        chat_session = model.start_chat(
            history=_convert_history_for_gemini(state["chat_history"])
        )

        with st.chat_message("assistant"):
            response_text = st.write_stream(
                _stream_response(chat_session, prompt)
            )

        state["chat_history"].append({"role": "assistant", "content": response_text or ""})
        # Increment the counter so the text_area renders with a new key (= empty).
        state["chat_input_counter"] += 1
        st.rerun()