"""
report_generator.py — High-Quality Report Exporter
====================================================

Generates a self-contained, beautifully styled HTML report (and optionally
Markdown) from the Feature Engineering Recommender's analysis results.

Usage (from recommend_app.py):
    from report_generator import add_report_download_buttons

    # After validation training:
    add_report_download_buttons(
        st.session_state, dataset_name="my_dataset",
        key_suffix="val", report_stage="validation"
    )

    # After test evaluation:
    add_report_download_buttons(
        st.session_state, dataset_name="my_dataset",
        key_suffix="test", report_stage="test",
        test_baseline_metrics=baseline_test,
        test_enhanced_metrics=enhanced_test,
    )
"""

from __future__ import annotations

import base64
import datetime
import io
import platform
import textwrap
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":           "#0d1117",
    "surface":      "#161b22",
    "surface2":     "#21262d",
    "border":       "#30363d",
    "text":         "#e6edf3",
    "text_muted":   "#8b949e",
    "accent":       "#58a6ff",
    "accent2":      "#79c0ff",
    "green":        "#3fb950",
    "green_dim":    "#1a4429",
    "red":          "#f85149",
    "red_dim":      "#4a1414",
    "yellow":       "#d29922",
    "yellow_dim":   "#3d2f04",
    "purple":       "#a371f7",
    "tag_num":      "#1f4e79",
    "tag_cat":      "#1a3a2a",
    "tag_int":      "#3a1a3a",
    "tag_row":      "#2a2a1a",
}

SEVERITY_COLOR = {
    "high":   (PALETTE["red"],    PALETTE["red_dim"]),
    "medium": (PALETTE["yellow"], PALETTE["yellow_dim"]),
    "low":    (PALETTE["accent"], "#0d2033"),
}

TYPE_COLOR = {
    "numerical":    (PALETTE["accent"],  PALETTE["tag_num"]),
    "categorical":  (PALETTE["green"],   PALETTE["tag_cat"]),
    "interaction":  (PALETTE["purple"],  PALETTE["tag_int"]),
    "row":          (PALETTE["yellow"],  PALETTE["tag_row"]),
}

# Beginner-friendly explanations for advisory types
_ADV_TOOLTIPS = {
    "class imbalance":
        "The model sees far more examples of one outcome than another, which can make "
        "accuracy misleading — it could score high by predicting the majority class every time.",
    "high cardinality":
        "This column has many unique values (e.g. IDs or free text). Tree models can overfit "
        "to these, memorising patterns that won't generalise to new data.",
    "high missing":
        "A large fraction of values in this column are absent. This limits what the model "
        "can learn from it and may introduce bias if missingness is not random.",
    "outlier":
        "Extreme values can dominate some models, pulling predictions away from typical cases.",
    "low variance":
        "This column barely changes across rows, so it carries little information for the model.",
    "high correlation":
        "Two features carry nearly the same information. Keeping both adds noise without "
        "adding signal, and can slow training.",
    "small dataset":
        "With fewer rows, the model has less data to learn from. Validation metrics are "
        "less reliable and the model may not generalise well.",
    "high dimensionality":
        "Many features relative to the number of rows increases the risk of overfitting — "
        "the model may memorise training noise rather than learning real patterns.",
}

_ADV_ACTIONS = {
    "class imbalance":
        "Enable class reweighting in Step ② or use an oversampling technique (e.g. SMOTE).",
    "high cardinality":
        "Apply target encoding or hash encoding, or drop the column if it is an identifier.",
    "high missing":
        "Apply median imputation + a missing indicator flag, or consider dropping the column.",
    "outlier":
        "Apply a log transform or winsorisation to reduce the influence of extreme values.",
    "low variance":
        "Consider dropping this column — it is unlikely to improve model performance.",
    "high correlation":
        "Consider dropping one of the correlated columns to reduce redundancy.",
    "small dataset":
        "Use cross-validation instead of a single train/val split, and consider regularisation.",
    "high dimensionality":
        "Drop low-importance features. Use feature selection before training.",
}


def _match_advisory_hint(title: str, hint_dict: dict) -> str:
    t = title.lower()
    for kw, val in hint_dict.items():
        if kw in t:
            return val
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# DATA ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_report_data(
    session_state: Any,
    dataset_name: str = "dataset",
    report_stage: str = "validation",
    test_baseline_metrics: dict | None = None,
    test_enhanced_metrics: dict | None = None,
) -> dict:
    """
    Pull everything needed from Streamlit session_state into a plain dict.

    Parameters
    ----------
    session_state          : streamlit.session_state (or dict-like)
    dataset_name           : label used in report title / filename
    report_stage           : "validation" or "test"
    test_baseline_metrics  : metrics dict from test-set evaluation (test stage only)
    test_enhanced_metrics  : metrics dict from test-set evaluation (test stage only)
    """
    ss = session_state

    def _get(key):
        return ss.get(key) if hasattr(ss, "get") else getattr(ss, key, None)

    n_classes = _get("n_classes")
    X_train   = _get("X_train")

    n_rows = int(X_train.shape[0]) if X_train is not None else "—"
    n_cols = int(X_train.shape[1]) if X_train is not None else "—"

    # Truncate very long dataset names
    display_name = dataset_name
    if len(display_name) > 60:
        display_name = display_name[:57] + "\u2026"

    # Class distribution
    class_dist: dict = {}
    y_train = _get("y_train")
    if y_train is not None:
        try:
            vc = pd.Series(y_train).value_counts().sort_index()
            class_dist = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            pass
    if not class_dist:
        class_dist = _get("_report_class_dist") or {}

    # Feature type breakdown from col_type_info
    col_type_info: dict = _get("_col_type_info") or {}
    feature_type_counts: dict = {}
    missing_cols: list = []
    if col_type_info:
        for col, info in col_type_info.items():
            detected = info.get("detected", "Unknown")
            feature_type_counts[detected] = feature_type_counts.get(detected, 0) + 1
            mp = info.get("missing_pct", 0.0)
            if mp > 0:
                missing_cols.append((col, mp))
        missing_cols.sort(key=lambda x: -x[1])

    n_missing_cols  = len(missing_cols)
    max_missing     = missing_cols[0][1] if missing_cols else 0.0
    max_missing_col = missing_cols[0][0] if missing_cols else None

    # Software versions
    versions = {}
    for pkg in ["sklearn", "lightgbm", "pandas", "numpy"]:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    versions["python"] = platform.python_version()

    return {
        "dataset_name":          display_name,
        "dataset_name_raw":      dataset_name,
        "report_stage":          report_stage,
        "generated_at":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_rows":                n_rows,
        "n_cols":                n_cols,
        "target_col":            _get("target_col") or "—",
        "n_classes":             n_classes or "—",
        "task_type":             "Binary" if n_classes == 2 else "Multiclass",
        "class_dist":            class_dist,
        "feature_type_counts":   feature_type_counts,
        "n_missing_cols":        n_missing_cols,
        "max_missing":           max_missing,
        "max_missing_col":       max_missing_col,
        "missing_cols":          missing_cols,
        "suggestions":           _get("suggestions")          or [],
        "advisories":            _get("advisories")           or [],
        "skipped_info":          _get("skipped_info")         or {},
        "baseline_val_metrics":  _get("baseline_val_metrics") or {},
        "enhanced_val_metrics":  _get("enhanced_val_metrics") or {},
        "test_baseline_metrics": test_baseline_metrics or {},
        "test_enhanced_metrics": test_enhanced_metrics or {},
        "baseline_model":        _get("baseline_model"),
        "enhanced_model":        _get("enhanced_model"),
        "baseline_train_cols":   _get("baseline_train_cols")  or [],
        "enhanced_train_cols":   _get("enhanced_train_cols")  or [],
        "fitted_params":         _get("fitted_params")        or [],
        "versions":              versions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_metrics_comparison(baseline: dict, enhanced: dict) -> str:
    metrics_order = ["roc_auc", "accuracy", "f1", "precision", "recall"]
    labels, base_vals, enh_vals = [], [], []
    for m in metrics_order:
        bv = baseline.get(m)
        ev = enhanced.get(m)
        if bv is not None and ev is not None:
            labels.append(m.replace("_", " ").upper())
            base_vals.append(bv)
            enh_vals.append(ev)

    if not labels:
        return ""

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    bars_b = ax.bar(x - w/2, base_vals, w, label="Baseline",
                    color=PALETTE["text_muted"], alpha=0.85, zorder=3)
    bars_e = ax.bar(x + w/2, enh_vals,  w, label="Enhanced",
                    color=PALETTE["accent"],    alpha=0.90, zorder=3)

    for bar in [*bars_b, *bars_e]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=PALETTE["text_muted"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=PALETTE["text"])
    ax.set_ylim(max(0, min(base_vals + enh_vals) - 0.06),
                min(1.0, max(base_vals + enh_vals) + 0.06))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
    ax.spines[:].set_color(PALETTE["border"])
    ax.grid(axis="y", color=PALETTE["border"], linewidth=0.6, zorder=0)
    ax.legend(fontsize=8, facecolor=PALETTE["surface2"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
    fig.tight_layout(pad=1.2)
    return _fig_to_b64(fig)


def _chart_feature_importance(model, col_names: list, title: str,
                               color: str, top_n: int = 15) -> str:
    if model is None or not col_names:
        return ""
    try:
        fi = pd.Series(model.feature_importances_, index=col_names)
        fi = fi.sort_values(ascending=True).tail(top_n)
    except Exception:
        return ""

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(fi) * 0.28)),
                           facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    bars = ax.barh(fi.index, fi.values, color=color, alpha=0.85, zorder=3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + fi.values.max() * 0.01, bar.get_y() + bar.get_height()/2,
                f"{int(w)}", va="center", fontsize=7, color=PALETTE["text_muted"])

    ax.set_title(title, fontsize=9, color=PALETTE["text"], pad=6)
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=7.5)
    ax.spines[:].set_color(PALETTE["border"])
    ax.grid(axis="x", color=PALETTE["border"], linewidth=0.5, zorder=0)
    fig.tight_layout(pad=1.0)
    return _fig_to_b64(fig)


def _chart_delta_distribution(suggestions: list) -> str:
    if not suggestions:
        return ""
    deltas = [s.get("predicted_delta_raw", s.get("predicted_delta", 0))
              for s in suggestions]
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])
    ax.hist(deltas, bins=30, color=PALETTE["accent"], alpha=0.75,
            edgecolor=PALETTE["border"], zorder=3)
    ax.axvline(0, color=PALETTE["red"], linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xlabel("Predicted \u0394 ROC-AUC", fontsize=8, color=PALETTE["text_muted"])
    ax.set_ylabel("Count",                   fontsize=8, color=PALETTE["text_muted"])
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=7.5)
    ax.spines[:].set_color(PALETTE["border"])
    ax.grid(axis="y", color=PALETTE["border"], linewidth=0.5, zorder=0)
    fig.tight_layout(pad=1.0)
    return _fig_to_b64(fig)


def _chart_class_balance(class_dist: dict) -> str:
    if not class_dist:
        return ""
    try:
        total  = sum(class_dist.values())
        labels = list(class_dist.keys())
        fracs  = [class_dist[l] / max(total, 1) for l in labels]
        counts = [class_dist[l] for l in labels]

        fig, ax = plt.subplots(figsize=(5, max(1.6, len(labels) * 0.4)),
                               facecolor=PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])
        palette = [PALETTE["accent"], PALETTE["green"], PALETTE["purple"],
                   PALETTE["yellow"], PALETTE["red"]]
        bars = ax.barh(labels, fracs,
                       color=[palette[i % len(palette)] for i in range(len(labels))],
                       alpha=0.80, zorder=3)
        for bar, cnt, frac in zip(bars, counts, fracs):
            ax.text(frac + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{cnt:,}  ({frac*100:.1f}%)",
                    va="center", fontsize=7.5, color=PALETTE["text_muted"])
        ax.set_xlim(0, 1.35)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
        ax.spines[:].set_color(PALETTE["border"])
        ax.grid(axis="x", color=PALETTE["border"], linewidth=0.4, zorder=0)
        fig.tight_layout(pad=0.8)
        return _fig_to_b64(fig)
    except Exception:
        return ""


def _chart_confusion_matrix(cm_data, class_labels, title: str) -> str:
    if not cm_data:
        return ""
    try:
        cm = np.array(cm_data)
        n  = cm.shape[0]
        labels = class_labels or [str(i) for i in range(n)]

        fig, ax = plt.subplots(figsize=(max(3.5, n * 1.1), max(3.0, n * 0.9)),
                               facecolor=PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=8, color=PALETTE["text"])
        ax.set_yticklabels(labels, fontsize=8, color=PALETTE["text"])
        ax.set_xlabel("Predicted", fontsize=8, color=PALETTE["text_muted"])
        ax.set_ylabel("Actual",    fontsize=8, color=PALETTE["text_muted"])
        ax.set_title(title, fontsize=9, color=PALETTE["text"], pad=6)

        for i in range(n):
            for j in range(n):
                txt_col = "white" if cm_norm[i, j] > 0.5 else PALETTE["text"]
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=9, color=txt_col, fontweight="bold")

        ax.tick_params(colors=PALETTE["text_muted"])
        ax.spines[:].set_color(PALETTE["border"])
        fig.tight_layout(pad=0.8)
        return _fig_to_b64(fig)
    except Exception:
        return ""


def _chart_roc_curves(baseline_roc, enhanced_roc) -> str:
    if not baseline_roc and not enhanced_roc:
        return ""
    try:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])

        if baseline_roc:
            auc_b = baseline_roc.get("auc", 0)
            ax.plot(baseline_roc["fpr"], baseline_roc["tpr"],
                    color=PALETTE["text_muted"], linewidth=1.6,
                    label=f"Baseline (AUC={auc_b:.3f})")
        if enhanced_roc:
            auc_e = enhanced_roc.get("auc", 0)
            ax.plot(enhanced_roc["fpr"], enhanced_roc["tpr"],
                    color=PALETTE["accent"], linewidth=2.0,
                    label=f"Enhanced (AUC={auc_e:.3f})")

        ax.plot([0, 1], [0, 1], color=PALETTE["border"], linewidth=1,
                linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=8, color=PALETTE["text_muted"])
        ax.set_ylabel("True Positive Rate",  fontsize=8, color=PALETTE["text_muted"])
        ax.set_title("ROC Curve", fontsize=9, color=PALETTE["text"], pad=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=7.5)
        ax.spines[:].set_color(PALETTE["border"])
        ax.grid(color=PALETTE["border"], linewidth=0.4, zorder=0)
        ax.legend(fontsize=7.5, facecolor=PALETTE["surface2"],
                  edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
        fig.tight_layout(pad=0.8)
        return _fig_to_b64(fig)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def _tag(label: str, text_col: str, bg_col: str) -> str:
    return (f'<span style="background:{bg_col};color:{text_col};padding:2px 8px;'
            f'border-radius:3px;font-size:0.72rem;font-family:monospace;'
            f'font-weight:600;letter-spacing:0.04em">{label}</span>')


def _metric_card(label: str, value: str, delta: str | None = None,
                 positive_good: bool = True, sub: str = "") -> str:
    delta_html = ""
    if delta is not None:
        try:
            dv = float(delta.replace("+", ""))
            color = PALETTE["green"] if (dv >= 0) == positive_good else PALETTE["red"]
        except ValueError:
            color = PALETTE["text_muted"]
        delta_html = (f'<div style="font-size:0.78rem;color:{color};'
                      f'margin-top:3px;font-weight:600">{delta}</div>')
    sub_html = (f'<div style="font-size:0.72rem;color:{PALETTE["text_muted"]};'
                f'margin-top:2px">{sub}</div>') if sub else ""
    return f"""
<div style="background:{PALETTE['surface2']};border:1px solid {PALETTE['border']};
     border-radius:8px;padding:16px 20px;min-width:120px;flex:1">
  <div style="font-size:0.72rem;color:{PALETTE['text_muted']};text-transform:uppercase;
       letter-spacing:0.08em;margin-bottom:4px">{label}</div>
  <div style="font-size:1.45rem;font-weight:700;color:{PALETTE['text']};
       font-family:monospace;word-break:break-all">{value}</div>
  {delta_html}{sub_html}
</div>"""


def _section_header(title: str, subtitle: str = "") -> str:
    sub_html = (f'<p style="margin:4px 0 0;color:{PALETTE["text_muted"]};'
                f'font-size:0.82rem">{subtitle}</p>') if subtitle else ""
    return f"""
<div style="margin:40px 0 18px;padding-bottom:10px;border-bottom:1px solid {PALETTE['border']}">
  <h2 style="margin:0;font-size:1.1rem;color:{PALETTE['accent']};
       font-weight:600;letter-spacing:0.03em">{title}</h2>
  {sub_html}
</div>"""


def _chart_block(b64: str, caption: str = "") -> str:
    if not b64:
        return ""
    cap = (f'<p style="text-align:center;font-size:0.75rem;'
           f'color:{PALETTE["text_muted"]};margin:6px 0 0">{caption}</p>') if caption else ""
    return f"""
<div style="background:{PALETTE['surface']};border:1px solid {PALETTE['border']};
     border-radius:8px;padding:16px;margin:12px 0">
  <img src="data:image/png;base64,{b64}" style="width:100%;height:auto;display:block">
  {cap}
</div>"""


def _info_box(text: str, color: str = "") -> str:
    c = color or PALETTE["accent"]
    return f"""
<div style="background:{PALETTE['surface2']};border-left:3px solid {c};
     border-radius:0 6px 6px 0;padding:12px 16px;margin:10px 0;
     font-size:0.82rem;color:{PALETTE['text']};line-height:1.65">{text}</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_executive_summary(d: dict) -> str:
    n_rows    = d["n_rows"]
    n_cols    = d["n_cols"]
    stage     = d["report_stage"]
    n_applied = len(d.get("fitted_params") or [])
    n_sugg    = len(d.get("suggestions") or [])
    advisories = d.get("advisories") or []
    n_high    = sum(1 for a in advisories if a.get("severity") == "high")
    n_medium  = sum(1 for a in advisories if a.get("severity") == "medium")

    rows_str = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)

    b_met = d.get("baseline_val_metrics") or {}
    e_met = d.get("enhanced_val_metrics") or {}
    if stage == "test":
        bt = d.get("test_baseline_metrics") or {}
        et = d.get("test_enhanced_metrics") or {}
        primary_b = bt.get("roc_auc") if bt else b_met.get("roc_auc")
        primary_e = et.get("roc_auc") if et else e_met.get("roc_auc")
        eval_label = "test-set"
    else:
        primary_b = b_met.get("roc_auc")
        primary_e = e_met.get("roc_auc")
        eval_label = "validation"

    s1 = (f"This is a <strong>{d['task_type']} classification</strong> task targeting "
          f"<code>{d['target_col']}</code> on a dataset of "
          f"<strong>{rows_str} rows &times; {n_cols} features</strong>.")

    if primary_b is not None and primary_e is not None:
        delta = primary_e - primary_b
        dc = PALETTE["green"] if delta > 0 else PALETTE["red"]
        word = "improved" if delta > 0 else "decreased"
        s2 = (f"The baseline {eval_label} ROC-AUC was "
              f"<strong style='font-family:monospace'>{primary_b:.4f}</strong>; "
              f"after applying <strong>{n_applied} feature engineering transform(s)</strong> "
              f"the enhanced model {word} to "
              f"<strong style='font-family:monospace;color:{dc}'>{primary_e:.4f}</strong> "
              f"<span style='color:{dc}'>({delta:+.4f})</span>.")
    elif n_applied == 0:
        s2 = "No feature engineering transforms were applied in this run."
    else:
        s2 = "Model metrics were not yet available at report generation time."

    if n_high > 0 or n_medium > 0:
        parts = []
        if n_high:   parts.append(f"<span style='color:{PALETTE['red']}'>{n_high} high-severity</span>")
        if n_medium: parts.append(f"<span style='color:{PALETTE['yellow']}'>{n_medium} medium-severity</span>")
        s3 = f"{' and '.join(parts)} data quality issue(s) were flagged &mdash; review Section ③ before deploying."
    elif advisories:
        s3 = f"No critical issues found; {len(advisories)} low-severity note(s) listed in Section ③."
    else:
        s3 = "No data quality issues were detected."

    s4 = (f"The feature engineering engine evaluated <strong>{n_sugg} candidate transforms</strong>, "
          f"of which <strong>{n_applied} were applied</strong>. See Section ⑤ for a breakdown by type."
          ) if n_sugg > 0 else ""

    body = " ".join(filter(None, [s1, s2, s3, s4]))

    edge_notes = []
    if isinstance(n_rows, int) and n_rows < 200:
        edge_notes.append(
            f'<div style="color:{PALETTE["yellow"]};font-size:0.80rem;margin-top:8px">'
            f'&ensp;&#9888; <strong>Very small dataset</strong> ({n_rows} rows) &mdash; '
            f'validation metrics may be unreliable. Consider cross-validation.</div>')
    if primary_e is not None and primary_e >= 0.999:
        edge_notes.append(
            f'<div style="color:{PALETTE["red"]};font-size:0.80rem;margin-top:8px">'
            f'&ensp;&#9888; <strong>Suspiciously high ROC-AUC ({primary_e:.4f})</strong> &mdash; '
            f'check for data leakage (a column derived from the target, or future information).</div>')

    return f"""
{_section_header("&#9312; Executive Summary")}
<div style="background:{PALETTE['surface2']};border:1px solid {PALETTE['border']};
     border-radius:8px;padding:18px 22px;line-height:1.75;font-size:0.88rem">
  {body}{"".join(edge_notes)}
</div>"""


def _render_overview(d: dict) -> str:
    rows_str = f"{d['n_rows']:,}" if isinstance(d["n_rows"], int) else str(d["n_rows"])

    cards = "".join([
        _metric_card("Rows",     rows_str),
        _metric_card("Features", str(d["n_cols"])),
        _metric_card("Target",   d["target_col"]),
        _metric_card("Task",     d["task_type"]),
        _metric_card("Classes",  str(d["n_classes"])),
    ])
    cards_html = f'<div style="display:flex;flex-wrap:wrap;gap:10px">{cards}</div>'

    # ── Class Balance ─────────────────────────────────────────────────────────
    class_dist = d.get("class_dist") or {}
    class_html = ""
    if class_dist:
        total  = sum(class_dist.values())
        counts = sorted(class_dist.values(), reverse=True)
        ratio  = counts[0] / max(counts[-1], 1) if len(counts) >= 2 else 1.0

        if ratio > 3:
            bc = PALETTE["red"];    bl = f"Imbalanced ({ratio:.1f}:1)"
            bn = ("The model sees significantly more examples of one class. "
                  "Accuracy may be misleading &mdash; check F1 and ROC-AUC.")
        elif ratio > 1.5:
            bc = PALETTE["yellow"]; bl = f"Mildly imbalanced ({ratio:.1f}:1)"
            bn = "A slight class imbalance exists. Monitor precision/recall per class."
        else:
            bc = PALETTE["green"];  bl = "Balanced"
            bn = "Classes are approximately equally represented."

        b64 = _chart_class_balance(class_dist)
        chart_part = _chart_block(b64) if b64 else ""

        trows = "".join(
            f'<tr><td style="padding:5px 10px;font-family:monospace;color:{PALETTE["text"]}">{cls}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["text_muted"]}">{cnt:,}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["text_muted"]}">{cnt/max(total,1)*100:.1f}%</td></tr>'
            for cls, cnt in class_dist.items()
        )

        class_html = f"""
<div style="margin-top:24px">
  <div style="font-size:0.78rem;font-weight:600;color:{PALETTE['text_muted']};
       text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">Class Balance</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
    <div style="flex:1;min-width:220px">
      <div style="background:{PALETTE['surface2']};border-left:3px solid {bc};
           border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:10px">
        <span style="color:{bc};font-weight:700;font-size:0.85rem">{bl}</span>
        <p style="margin:4px 0 0;color:{PALETTE['text_muted']};font-size:0.78rem">{bn}</p>
      </div>
      <table style="border-collapse:collapse;font-size:0.80rem;width:100%">
        <tr style="background:{PALETTE['surface2']};border-bottom:1px solid {PALETTE['border']}">
          <th style="padding:5px 10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.72rem;text-transform:uppercase">Class</th>
          <th style="padding:5px 10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.72rem;text-transform:uppercase">Count</th>
          <th style="padding:5px 10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.72rem;text-transform:uppercase">Share</th>
        </tr>
        {trows}
      </table>
    </div>
    <div style="flex:1;min-width:260px">{chart_part}</div>
  </div>
</div>"""

    # ── Feature Types ─────────────────────────────────────────────────────────
    ftc = d.get("feature_type_counts") or {}
    ft_html = ""
    if ftc:
        type_colors = {
            "Numerical": PALETTE["accent"],  "Binary": PALETTE["accent2"],
            "Categorical": PALETTE["green"], "DateTime": PALETTE["yellow"],
            "Date": PALETTE["yellow"],       "Day-of-Week": PALETTE["yellow"],
            "Free Text": PALETTE["purple"],  "ID": PALETTE["red"],
            "Constant": PALETTE["red"],      "Time only": PALETTE["yellow"],
        }
        pills = "".join(
            f'<span style="background:{PALETTE["surface2"]};'
            f'border:1px solid {type_colors.get(t, PALETTE["border"])};'
            f'color:{type_colors.get(t, PALETTE["text_muted"])};'
            f'border-radius:20px;padding:4px 12px;font-size:0.78rem;'
            f'font-family:monospace;white-space:nowrap">{t}: {cnt}</span>'
            for t, cnt in sorted(ftc.items(), key=lambda x: -x[1])
        )
        ft_html = f"""
<div style="margin-top:20px">
  <div style="font-size:0.78rem;font-weight:600;color:{PALETTE['text_muted']};
       text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Feature Types</div>
  <div style="display:flex;flex-wrap:wrap;gap:8px">{pills}</div>
</div>"""

    # ── Missing Values ────────────────────────────────────────────────────────
    n_miss = d.get("n_missing_cols", 0)
    miss_html = ""
    if n_miss > 0:
        top5 = (d.get("missing_cols") or [])[:5]
        extra = n_miss - 5
        trows = "".join(
            f'<tr style="border-bottom:1px solid {PALETTE["border"]}">'
            f'<td style="padding:5px 10px;font-family:monospace;font-size:0.79rem;'
            f'color:{PALETTE["text"]}">{col}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["red"] if mp>0.3 else PALETTE["yellow"]};'
            f'font-family:monospace;font-size:0.79rem">{mp*100:.1f}%</td></tr>'
            for col, mp in top5
        ) + (
            f'<tr><td colspan="2" style="padding:5px 10px;color:{PALETTE["text_muted"]};'
            f'font-size:0.78rem">&hellip; and {extra} more columns</td></tr>'
            if extra > 0 else ""
        )
        miss_html = f"""
<div style="margin-top:20px">
  <div style="font-size:0.78rem;font-weight:600;color:{PALETTE['text_muted']};
       text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Missing Values</div>
  <div style="background:{PALETTE['surface2']};border:1px solid {PALETTE['border']};
       border-radius:8px;padding:14px 18px">
    <p style="margin:0 0 10px;font-size:0.82rem;color:{PALETTE['text']}">
      <strong>{n_miss} of {d['n_cols']} feature(s)</strong> have missing values.
      Worst: <code>{d.get('max_missing_col','')}</code>
      <span style="color:{PALETTE['red'] if d.get('max_missing',0)>0.3 else PALETTE['yellow']}">
        ({d.get('max_missing',0)*100:.1f}% missing)</span>.
    </p>
    <table style="border-collapse:collapse;font-size:0.80rem">
      <tr style="background:{PALETTE['surface']};border-bottom:1px solid {PALETTE['border']}">
        <th style="padding:5px 10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.72rem;text-transform:uppercase">Column</th>
        <th style="padding:5px 10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.72rem;text-transform:uppercase">Missing %</th>
      </tr>{trows}
    </table>
  </div>
</div>"""
    elif ftc:
        miss_html = f"""
<div style="margin-top:16px">
  <div style="background:{PALETTE['surface2']};border-left:3px solid {PALETTE['green']};
       border-radius:0 6px 6px 0;padding:10px 14px;font-size:0.82rem;color:{PALETTE['text']}">
    &#10003; <strong>No missing values</strong> &mdash; all features are fully populated.
  </div>
</div>"""

    return f"""
{_section_header("&#9313; Dataset Profile",
    "Size, target distribution, feature types, and data completeness")}
{cards_html}{class_html}{ft_html}{miss_html}"""


def _render_advisories(advisories: list) -> str:
    if not advisories:
        return ""

    icons = {"high": "&#9888;", "medium": "&#9670;", "low": "&#9432;"}

    def _card(adv):
        sev      = adv.get("severity", "low")
        tc, bg   = SEVERITY_COLOR.get(sev, SEVERITY_COLOR["low"])
        icon     = icons.get(sev, "&bull;")
        title    = adv.get("title", "")
        code_blk = ""
        if adv.get("code_hint"):
            code_blk = (
                f'<pre style="background:{PALETTE["bg"]};border:1px solid {PALETTE["border"]};'
                f'border-radius:4px;padding:10px;margin:8px 0 0;font-size:0.76rem;'
                f'color:{PALETTE["text_muted"]};overflow-x:auto;white-space:pre-wrap">'
                f'{adv["code_hint"]}</pre>')
        tooltip = _match_advisory_hint(title, _ADV_TOOLTIPS)
        action  = _match_advisory_hint(title, _ADV_ACTIONS)
        tip_html = (f'<p style="margin:6px 0 0;color:{PALETTE["text_muted"]};font-size:0.78rem;'
                    f'font-style:italic">{tooltip}</p>') if tooltip else ""
        act_html = (f'<p style="margin:4px 0 0;color:{PALETTE["text_muted"]};font-size:0.78rem">'
                    f'<strong>Suggested action:</strong> {action}</p>') if action else ""
        return f"""
<div style="background:{bg};border:1px solid {tc}44;border-radius:8px;padding:14px 16px;margin:8px 0">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <span style="color:{tc};font-size:1.1rem">{icon}</span>
    <strong style="color:{tc};font-size:0.88rem">{title}</strong>
    {_tag(sev.upper(), tc, bg)}
  </div>
  <p style="margin:6px 0 0;color:{PALETTE['text']};font-size:0.82rem;line-height:1.5">
    {adv.get('detail','')}
  </p>{tip_html}{act_html}{code_blk}
</div>"""

    high   = [a for a in advisories if a.get("severity") == "high"]
    medium = [a for a in advisories if a.get("severity") == "medium"]
    low    = [a for a in advisories if a.get("severity") == "low"]

    parts = [_card(a) for a in high]

    if medium:
        parts.append(f"""
<details open>
  <summary style="cursor:pointer;font-size:0.82rem;color:{PALETTE['yellow']};
       padding:8px 0;user-select:none">
    &#9654; {len(medium)} medium-severity issue(s) &mdash; click to collapse
  </summary>
  {"".join(_card(a) for a in medium)}
</details>""")

    if low:
        parts.append(f"""
<details>
  <summary style="cursor:pointer;font-size:0.82rem;color:{PALETTE['accent']};
       padding:8px 0;user-select:none">
    &#9654; {len(low)} low-severity note(s) &mdash; click to expand
  </summary>
  {"".join(_card(a) for a in low)}
</details>""")

    return f"""
{_section_header("&#9314; Data Quality Advisories",
    f"{len(advisories)} issue(s) &mdash; high-severity shown first; medium &amp; low collapsible")}
{"".join(parts)}"""


def _render_model_perf_block(
    label: str, baseline: dict, enhanced: dict,
    n_classes, section_tag: str, subtitle: str = "",
) -> str:
    if not baseline and not enhanced:
        return ""

    metrics_order = ["roc_auc", "accuracy", "f1", "precision", "recall", "log_loss"]
    cards = []
    for m in metrics_order:
        bv = baseline.get(m);  ev = enhanced.get(m)
        if bv is None and ev is None:
            continue
        lbl = m.replace("_", " ").upper()
        if ev is not None and bv is not None:
            delta    = ev - bv
            dstr     = f"{delta:+.4f}"
            pos_good = m != "log_loss"
        else:
            dstr = None; pos_good = True
        val_str = (f"{ev:.4f}" if ev is not None
                   else (f"{bv:.4f}" if bv is not None else "—"))
        cards.append(_metric_card(lbl, val_str, dstr, pos_good))

    cards_html  = f'<div style="display:flex;flex-wrap:wrap;gap:10px">{"".join(cards)}</div>'
    bar_b64     = _chart_metrics_comparison(baseline, enhanced)
    bar_html    = _chart_block(bar_b64, "Baseline vs Enhanced &mdash; validation metrics")

    # Confusion matrices
    cm_base = _chart_confusion_matrix(
        baseline.get("confusion_matrix"), baseline.get("y_classes"),
        "Baseline &mdash; Confusion Matrix")
    cm_enh  = _chart_confusion_matrix(
        enhanced.get("confusion_matrix"), enhanced.get("y_classes"),
        "Enhanced &mdash; Confusion Matrix")
    cm_html = ""
    if cm_base or cm_enh:
        cols = []
        if cm_base: cols.append(f'<div style="flex:1;min-width:250px">{_chart_block(cm_base)}</div>')
        if cm_enh:  cols.append(f'<div style="flex:1;min-width:250px">{_chart_block(cm_enh)}</div>')
        cm_html = (
            _info_box(
                "A confusion matrix shows how often the model correctly predicted each class "
                "(diagonal cells) vs. which classes it confused (off-diagonal). "
                "Darker = more predictions. Numbers are raw row counts.",
                PALETTE["accent"])
            + f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(cols)}</div>'
        )

    # ROC curve (binary only)
    roc_html = ""
    if isinstance(n_classes, int) and n_classes == 2:
        roc_b64 = _chart_roc_curves(baseline.get("roc_data"), enhanced.get("roc_data"))
        if roc_b64:
            roc_html = (
                _info_box(
                    "The ROC curve plots the trade-off between true positive rate (sensitivity) "
                    "and false positive rate at every decision threshold. A larger area under the "
                    "curve (AUC) means the model ranks positives above negatives more reliably.",
                    PALETTE["purple"])
                + _chart_block(roc_b64, "ROC Curve &mdash; Baseline vs Enhanced")
            )

    return f"""
{_section_header(f"{section_tag} Model Performance &mdash; {label}", subtitle)}
{cards_html}{bar_html}{cm_html}{roc_html}"""


def _render_model_comparison(d: dict) -> str:
    n_classes = d.get("n_classes")

    val_html = _render_model_perf_block(
        "Validation Set",
        d.get("baseline_val_metrics") or {},
        d.get("enhanced_val_metrics") or {},
        n_classes, "&#9315;",
        "Metrics on a held-out 20% validation split used during training"
    )

    # Feature importance charts (shown once, after validation block)
    fi_base = _chart_feature_importance(
        d.get("baseline_model"), d.get("baseline_train_cols", []),
        "Baseline &mdash; Top Feature Importances", PALETTE["text_muted"])
    fi_enh = _chart_feature_importance(
        d.get("enhanced_model"), d.get("enhanced_train_cols", []),
        "Enhanced &mdash; Top Feature Importances", PALETTE["accent"])

    fi_html = ""
    if fi_base or fi_enh:
        cols = []
        if fi_base: cols.append(f'<div style="flex:1;min-width:300px">{_chart_block(fi_base)}</div>')
        if fi_enh:  cols.append(f'<div style="flex:1;min-width:300px">{_chart_block(fi_enh)}</div>')
        fi_html = (
            _info_box(
                "Feature importance shows which columns the model relied on most to make "
                "predictions. Higher = more splits on that feature across all decision trees.",
                PALETTE["green"])
            + f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(cols)}</div>'
        )

    # Test section (only when test metrics present)
    test_base = d.get("test_baseline_metrics") or {}
    test_enh  = d.get("test_enhanced_metrics")  or {}
    test_html = ""
    if test_base or test_enh:
        test_html = _render_model_perf_block(
            "Test Set",
            test_base, test_enh, n_classes, "&#9315;b",
            "Metrics on the held-out test CSV &mdash; the most reliable performance estimate"
        )

    return val_html + fi_html + test_html


def _render_suggestions(suggestions: list) -> str:
    if not suggestions:
        return f"""
{_section_header("&#9316; Feature Engineering Suggestions")}
{_info_box("No feature engineering suggestions were generated. This may indicate the dataset "
           "is already well-prepared, or that too few columns met the evaluation criteria.")}"""

    groups: dict = {}
    for s in suggestions:
        groups.setdefault(s.get("type", "other"), []).append(s)

    intro = _info_box(
        "Feature engineering creates new input columns derived from existing ones, helping the "
        "model detect patterns it would otherwise miss. The table below summarises which transform "
        "types were evaluated, how many showed a positive predicted improvement, and the single "
        "best opportunity in each category. Full details are available in the HTML report.",
        PALETTE["accent"]
    )

    rows = []
    for typ in ["numerical", "categorical", "interaction", "row", "other"]:
        items = groups.get(typ)
        if not items:
            continue
        tc, bg    = TYPE_COLOR.get(typ, (PALETTE["text_muted"], PALETTE["surface2"]))
        n_pos     = sum(1 for s in items
                        if s.get("predicted_delta_raw", s.get("predicted_delta", 0)) > 0)
        best      = max(items, key=lambda s: s.get("predicted_delta_raw", s.get("predicted_delta", 0)))
        bd        = best.get("predicted_delta_raw", best.get("predicted_delta", 0))
        bc        = best.get("column", "")
        bc2       = best.get("column_b", "")
        bc_disp   = bc + (f" &times; {bc2}" if bc2 else "")
        if typ == "row":
            bc_disp = "(all numeric cols)"
        bm        = best.get("method", "")
        dc        = PALETTE["green"] if bd > 0 else PALETTE["red"]

        rows.append(f"""
<tr style="border-bottom:1px solid {PALETTE['border']}">
  <td style="padding:10px 12px">{_tag(typ.upper(), tc, bg)}</td>
  <td style="padding:10px 12px;text-align:center;color:{PALETTE['text']};font-size:0.82rem">{len(items)}</td>
  <td style="padding:10px 12px;text-align:center;color:{PALETTE['green'] if n_pos>0 else PALETTE['text_muted']};font-size:0.82rem">{n_pos}</td>
  <td style="padding:10px 12px;font-family:monospace;font-size:0.80rem;color:{PALETTE['text']}">{bc_disp}</td>
  <td style="padding:10px 12px;color:{PALETTE['accent2']};font-size:0.80rem">{bm}</td>
  <td style="padding:10px 12px;font-family:monospace;font-weight:700;font-size:0.85rem;color:{dc}">{bd:+.4f}</td>
</tr>""")

    table = f"""
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
  <thead>
    <tr style="background:{PALETTE['surface2']};border-bottom:2px solid {PALETTE['border']}">
      <th style="padding:10px 12px;text-align:left;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Type</th>
      <th style="padding:10px 12px;text-align:center;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Evaluated</th>
      <th style="padding:10px 12px;text-align:center;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Positive &Delta;</th>
      <th style="padding:10px 12px;text-align:left;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Best Column(s)</th>
      <th style="padding:10px 12px;text-align:left;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Method</th>
      <th style="padding:10px 12px;text-align:left;color:{PALETTE['text_muted']};font-size:0.73rem;text-transform:uppercase;letter-spacing:0.06em">Best Predicted &Delta;</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>"""

    delta_chart = _chart_delta_distribution(suggestions)
    chart_html  = _chart_block(delta_chart,
        "Distribution of predicted AUC improvements &mdash; positive = potential gain")

    return f"""
{_section_header("&#9316; Feature Engineering Suggestions",
    f"{len(suggestions)} transforms evaluated &mdash; summary by category")}
{intro}{table}{chart_html}"""


def _render_applied_transforms(fitted_params: list) -> str:
    if not fitted_params:
        return f"""
{_section_header("&#9317; Applied Transforms")}
{_info_box("No transforms were applied. Select suggestions in Step &#9313; and re-train "
           "to see a breakdown here.")}"""

    rows = []
    for i, p in enumerate(fitted_params, 1):
        col = p.get("column", "")
        if p.get("column_b"):
            col += f" &times; {p['column_b']}"
        method = p.get("method", "")
        typ    = p.get("type", "")
        tc, bg = TYPE_COLOR.get(typ, (PALETTE["text_muted"], PALETTE["surface2"]))
        rows.append(f"""
<tr style="border-bottom:1px solid {PALETTE['border']}">
  <td style="padding:8px 10px;color:{PALETTE['text_muted']};font-family:monospace;font-size:0.8rem">{i}</td>
  <td style="padding:8px 10px">{_tag(typ.upper(), tc, bg)}</td>
  <td style="padding:8px 10px;color:{PALETTE['text']};font-family:monospace;font-size:0.8rem">{col}</td>
  <td style="padding:8px 10px;color:{PALETTE['accent2']};font-size:0.8rem">{method}</td>
</tr>""")

    caption = _info_box(
        "These are the exact transformations <strong>fitted on training data only</strong> and "
        "applied identically to train and test sets. Feature statistics (medians, encoding maps, "
        "&hellip;) were learned from training data, preventing data leakage.",
        PALETTE["text_muted"]
    )

    return f"""
{_section_header("&#9317; Applied Transforms",
    f"{len(fitted_params)} transform(s) fitted and applied")}
{caption}
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse">
  <thead>
    <tr style="background:{PALETTE['surface2']};border-bottom:2px solid {PALETTE['border']}">
      <th style="padding:10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">#</th>
      <th style="padding:10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">Type</th>
      <th style="padding:10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">Column(s)</th>
      <th style="padding:10px;text-align:left;color:{PALETTE['text_muted']};font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">Method</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>"""


def _render_reproducibility(d: dict) -> str:
    versions = d.get("versions") or {}
    ver_pills = "".join(
        f'<span style="background:{PALETTE["surface2"]};border:1px solid {PALETTE["border"]};'
        f'border-radius:4px;padding:3px 10px;font-family:monospace;font-size:0.76rem;'
        f'color:{PALETTE["text_muted"]};margin-right:6px;margin-bottom:4px;display:inline-block">'
        f'{pkg}=={ver}</span>'
        for pkg, ver in versions.items()
    )

    n_applied = len(d.get("fitted_params") or [])
    code = textwrap.dedent(f"""\
        # Re-apply the {n_applied} fitted transform(s) to a new dataset
        from transforms import apply_fitted_to_test

        X_new_enhanced = apply_fitted_to_test(X_new, fitted_params)
        # predictions = enhanced_model.predict(X_new_enhanced)""")

    return f"""
{_section_header("&#9318; Reproducibility",
    "Software environment and how to re-apply these transforms to new data")}
<div style="background:{PALETTE['surface2']};border:1px solid {PALETTE['border']};
     border-radius:8px;padding:16px 20px">
  <div style="font-size:0.78rem;color:{PALETTE['text_muted']};text-transform:uppercase;
       letter-spacing:0.08em;margin-bottom:10px">Software Versions</div>
  <div>{ver_pills}</div>
  <div style="font-size:0.78rem;color:{PALETTE['text_muted']};text-transform:uppercase;
       letter-spacing:0.08em;margin:16px 0 8px">Applying Transforms to New Data</div>
  <pre style="background:{PALETTE['bg']};border:1px solid {PALETTE['border']};
       border-radius:4px;padding:12px 14px;font-size:0.78rem;
       color:{PALETTE['accent2']};overflow-x:auto;margin:0">{code}</pre>
  <p style="margin:12px 0 0;font-size:0.78rem;color:{PALETTE['text_muted']}">
    Random seed: <code>42</code> &nbsp;&middot;&nbsp;
    Generated: <code>{d['generated_at']}</code> &nbsp;&middot;&nbsp;
    Dataset: <code>{d['dataset_name']}</code>
  </p>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background:  {PALETTE['bg']};
  color:       {PALETTE['text']};
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  font-size:   14px;
  line-height: 1.6;
}}

.page {{
  max-width: 960px;
  margin:    0 auto;
  padding:   40px 32px 80px;
}}

.report-header {{
  background:    linear-gradient(135deg, {PALETTE['surface']} 0%, {PALETTE['surface2']} 100%);
  border-bottom: 2px solid {PALETTE['accent']}44;
  padding:       36px 32px;
}}

.report-header h1 {{
  font-size:    1.75rem;
  font-weight:  700;
  color:        {PALETTE['text']};
  letter-spacing: -0.02em;
  word-break:   break-word;
  overflow-wrap: anywhere;
  max-width:    620px;
  margin-top:   8px;
}}

.report-header .subtitle {{
  color:     {PALETTE['text_muted']};
  font-size: 0.85rem;
  margin-top: 6px;
}}

.meta-row {{
  display:    flex;
  gap:        8px;
  margin-top: 14px;
  flex-wrap:  wrap;
}}

.meta-pill {{
  background:    {PALETTE['surface2']};
  border:        1px solid {PALETTE['border']};
  border-radius: 20px;
  padding:       3px 12px;
  font-size:     0.76rem;
  color:         {PALETTE['text_muted']};
  font-family:   'IBM Plex Mono', monospace;
}}

.stage-badge {{
  border-radius:  4px;
  padding:        3px 10px;
  font-size:      0.72rem;
  font-family:    monospace;
  font-weight:    600;
  letter-spacing: 0.06em;
  display:        inline-block;
}}

.stage-validation {{ background:{PALETTE['tag_num']}; color:{PALETTE['accent']}; border:1px solid {PALETTE['accent']}66; }}
.stage-test        {{ background:{PALETTE['green_dim']}; color:{PALETTE['green']}; border:1px solid {PALETTE['green']}66; }}

.footer {{
  margin-top:  60px;
  padding-top: 20px;
  border-top:  1px solid {PALETTE['border']};
  color:       {PALETTE['text_muted']};
  font-size:   0.75rem;
  text-align:  center;
}}

details > summary {{ list-style: none; }}
details > summary::-webkit-details-marker {{ display: none; }}

@media print {{
  body {{ background: white; color: #111; }}
  .page {{ padding: 20px; }}
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(d: dict) -> bytes:
    """Build a fully self-contained HTML report. Returns UTF-8 encoded bytes."""
    exec_summary = _render_executive_summary(d)
    overview     = _render_overview(d)
    advisories   = _render_advisories(d.get("advisories") or [])
    model_cmp    = _render_model_comparison(d)
    suggestions  = _render_suggestions(d.get("suggestions") or [])
    transforms   = _render_applied_transforms(d.get("fitted_params") or [])
    repro        = _render_reproducibility(d)

    n_suggestions = len(d.get("suggestions") or [])
    n_applied     = len(d.get("fitted_params") or [])
    bm = (d.get("baseline_val_metrics") or {}).get("roc_auc")
    em = (d.get("enhanced_val_metrics") or {}).get("roc_auc")
    auc_delta = f"{em - bm:+.4f}" if bm is not None and em is not None else "&mdash;"

    stage       = d.get("report_stage", "validation")
    stage_label = "VALIDATION REPORT" if stage == "validation" else "FINAL TEST REPORT"
    stage_cls   = f"stage-badge stage-{stage}"

    rows_str = f"{d['n_rows']:,}" if isinstance(d["n_rows"], int) else str(d["n_rows"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Report &mdash; {d['dataset_name']} ({stage_label})</title>
  <style>{_CSS}</style>
</head>
<body>

<div class="report-header">
  <div style="max-width:960px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
      <div>
        <span class="{stage_cls}">{stage_label}</span>
        <h1>{d['dataset_name']}</h1>
        <div class="subtitle">Automated ML analysis &mdash; {d['task_type']} Classification</div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div style="font-size:0.72rem;color:{PALETTE['text_muted']}">Generated</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:{PALETTE['text']}">{d['generated_at']}</div>
      </div>
    </div>
    <div class="meta-row">
      <span class="meta-pill">&#128203; {rows_str} rows &times; {d['n_cols']} features</span>
      <span class="meta-pill">&#127919; target: {d['target_col']}</span>
      <span class="meta-pill">&#128290; {d['n_classes']} classes</span>
      <span class="meta-pill">&#128161; {n_suggestions} suggestions</span>
      <span class="meta-pill">&#9881; {n_applied} transforms applied</span>
      <span class="meta-pill" style="color:{PALETTE['green'] if isinstance(auc_delta,str) and auc_delta.startswith('+') else PALETTE['red']}">
        &Delta; AUC {auc_delta}
      </span>
    </div>
  </div>
</div>

<div class="page">
  {exec_summary}
  {overview}
  {advisories}
  {model_cmp}
  {suggestions}
  {transforms}
  {repro}
  <div class="footer">
    Generated by <strong>Feature Engineering Recommender</strong> &nbsp;&middot;&nbsp;
    {d['generated_at']} &nbsp;&middot;&nbsp; {stage_label} &nbsp;&middot;&nbsp; {d['dataset_name']}
  </div>
</div>
</body>
</html>"""

    return html.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(d: dict) -> bytes:
    """Lightweight Markdown version (no charts). Returns UTF-8 bytes."""
    stage    = d.get("report_stage", "validation")
    label    = "Validation Report" if stage == "validation" else "Final Test Report"
    rows_str = f"{d['n_rows']:,}" if isinstance(d["n_rows"], int) else str(d["n_rows"])

    bm = (d.get("baseline_val_metrics") or {}).get("roc_auc")
    em = (d.get("enhanced_val_metrics") or {}).get("roc_auc")
    auc_delta = f"{em - bm:+.4f}" if bm is not None and em is not None else "—"

    n_applied = len(d.get("fitted_params") or [])
    n_sugg    = len(d.get("suggestions") or [])

    out = [
        f"# Feature Engineering Report — {d['dataset_name']}",
        f"",
        f"**{label}** · Generated: {d['generated_at']}",
        f"",
        f"| | |",
        f"|---|---|",
        f"| Dataset | `{d['dataset_name']}` |",
        f"| Rows × Features | {rows_str} × {d['n_cols']} |",
        f"| Target | `{d['target_col']}` |",
        f"| Task | {d['task_type']} Classification |",
        f"| Classes | {d['n_classes']} |",
        f"| Transforms applied | {n_applied} |",
        f"| Δ Val AUC | {auc_delta} |",
        f"",
        f"---",
        f"",
        f"## ① Executive Summary",
        f"",
    ]

    # Narrative
    b_met = d.get("baseline_val_metrics") or {}
    e_met = d.get("enhanced_val_metrics") or {}
    if stage == "test":
        bt = d.get("test_baseline_metrics") or {}
        et = d.get("test_enhanced_metrics") or {}
        pb = bt.get("roc_auc") or b_met.get("roc_auc")
        pe = et.get("roc_auc") or e_met.get("roc_auc")
    else:
        pb = b_met.get("roc_auc")
        pe = e_met.get("roc_auc")

    summary = [f"**{d['task_type']} classification** on `{d['target_col']}` ({rows_str} rows × {d['n_cols']} features)."]
    if pb is not None and pe is not None:
        delta = pe - pb
        word  = "improved" if delta > 0 else "decreased"
        summary.append(f"Baseline ROC-AUC: **{pb:.4f}**. Enhanced {word} to **{pe:.4f}** ({delta:+.4f}) after {n_applied} transforms.")
    advisories = d.get("advisories") or []
    n_high = sum(1 for a in advisories if a.get("severity") == "high")
    if n_high:
        summary.append(f"⚠ **{n_high} high-severity data quality issue(s)** — review before deployment.")
    out += [" ".join(summary), "", "---", "", "## ② Dataset Profile", ""]

    class_dist = d.get("class_dist") or {}
    if class_dist:
        total = sum(class_dist.values())
        out += ["**Class Balance:**", "", "| Class | Count | Share |", "|-------|-------|-------|"]
        for cls, cnt in class_dist.items():
            out.append(f"| `{cls}` | {cnt:,} | {cnt/max(total,1)*100:.1f}% |")
        out.append("")

    ftc = d.get("feature_type_counts") or {}
    if ftc:
        out.append("**Feature Types:** " + ", ".join(f"{t}: {c}" for t, c in sorted(ftc.items(), key=lambda x: -x[1])))
        out.append("")

    n_miss = d.get("n_missing_cols", 0)
    if n_miss > 0:
        out.append(f"**Missing Values:** {n_miss} column(s) with missing values. "
                   f"Worst: `{d.get('max_missing_col')}` ({d.get('max_missing', 0)*100:.1f}%).")
    elif ftc:
        out.append("**Missing Values:** None — dataset fully populated.")
    out += ["", "---", "", "## ③ Data Quality Advisories", ""]

    if advisories:
        for adv in sorted(advisories, key=lambda a: {"high":0,"medium":1,"low":2}.get(a.get("severity","low"),3)):
            icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(adv.get("severity","low"), "•")
            out += [f"### {icon} {adv.get('title','')} `{adv.get('severity','').upper()}`", ""]
            out.append(adv.get("detail", ""))
            tip = _match_advisory_hint(adv.get("title",""), _ADV_TOOLTIPS)
            act = _match_advisory_hint(adv.get("title",""), _ADV_ACTIONS)
            if tip: out += ["", f"*{tip}*"]
            if act: out += ["", f"**Suggested action:** {act}"]
            out.append("")
    else:
        out += ["No data quality issues detected.", ""]

    out += ["---", "", "## ④ Model Performance", ""]

    def _md_table(baseline, enhanced, cap):
        if not baseline and not enhanced:
            return []
        rows = [f"**{cap}**", "", "| Metric | Baseline | Enhanced | Δ |", "|--------|----------|----------|---|"]
        for m in ["roc_auc","accuracy","f1","precision","recall","log_loss"]:
            bv = baseline.get(m); ev = enhanced.get(m)
            if bv is None and ev is None: continue
            lbl  = m.replace("_"," ").upper()
            bstr = f"{bv:.4f}" if bv is not None else "—"
            estr = f"{ev:.4f}" if ev is not None else "—"
            dstr = (f"{ev-bv:+.4f}" if bv is not None and ev is not None else "—")
            rows.append(f"| {lbl} | {bstr} | {estr} | {dstr} |")
        rows.append("")
        return rows

    out += _md_table(b_met, e_met, "Validation Set")
    if d.get("test_baseline_metrics") or d.get("test_enhanced_metrics"):
        out += _md_table(d.get("test_baseline_metrics") or {}, d.get("test_enhanced_metrics") or {}, "Test Set")

    out.append("> Charts (confusion matrix, ROC curve, feature importances) are in the HTML report.")
    out += ["", "---", "", "## ⑤ Feature Engineering Suggestions", ""]

    suggestions = d.get("suggestions") or []
    if suggestions:
        groups: dict = {}
        for s in suggestions:
            groups.setdefault(s.get("type","other"), []).append(s)
        out += ["| Type | Evaluated | Positive Δ | Best Column | Best Method | Best Predicted Δ |",
                "|------|-----------|------------|-------------|-------------|-----------------|"]
        for typ in ["numerical","categorical","interaction","row","other"]:
            items = groups.get(typ)
            if not items: continue
            n_pos = sum(1 for s in items if s.get("predicted_delta_raw", s.get("predicted_delta",0)) > 0)
            best  = max(items, key=lambda s: s.get("predicted_delta_raw", s.get("predicted_delta",0)))
            bd    = best.get("predicted_delta_raw", best.get("predicted_delta",0))
            bc    = best.get("column","") + (" × " + best.get("column_b","") if best.get("column_b") else "")
            if typ == "row": bc = "(all numeric)"
            out.append(f"| {typ.upper()} | {len(items)} | {n_pos} | `{bc}` | {best.get('method','')} | {bd:+.4f} |")
    else:
        out.append("No suggestions generated.")
    out += ["", "---", "", "## ⑥ Applied Transforms", ""]

    fitted = d.get("fitted_params") or []
    if fitted:
        out += ["| # | Type | Column(s) | Method |", "|---|------|-----------|--------|"]
        for i, p in enumerate(fitted, 1):
            col = p.get("column","") + (" × " + p.get("column_b","") if p.get("column_b") else "")
            out.append(f"| {i} | {p.get('type','').upper()} | `{col}` | {p.get('method','')} |")
    else:
        out.append("No transforms applied.")
    out += ["", "---", "", "## ⑦ Reproducibility", ""]

    versions = d.get("versions") or {}
    if versions:
        out.append("**Software:** " + " | ".join(f"`{k}=={v}`" for k, v in versions.items()))
    out += [f"", f"**Random seed:** 42 · **Dataset:** `{d['dataset_name']}` · **Generated:** {d['generated_at']}", ""]

    return "\n".join(out).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(d: dict) -> bytes | None:
    """Convert HTML report to PDF via WeasyPrint. Returns None if not installed."""
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        return None
    html_bytes = generate_html_report(d)
    return HTML(string=html_bytes.decode("utf-8")).write_pdf(
        stylesheets=[CSS(string="@page { margin: 15mm 12mm; size: A4; }")]
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def add_report_download_buttons(
    session_state: Any,
    dataset_name: str = "dataset",
    key_suffix: str = "",
    report_stage: str = "validation",
    test_baseline_metrics: dict | None = None,
    test_enhanced_metrics: dict | None = None,
) -> None:
    """
    Drop this call in your Streamlit app to add HTML, Markdown, and PDF
    download buttons.

    Parameters
    ----------
    session_state          : streamlit.session_state
    dataset_name           : label used in the report title and filename
    key_suffix             : unique string per call site (prevents duplicate-key errors)
    report_stage           : "validation" or "test"
    test_baseline_metrics  : pass when report_stage="test"
    test_enhanced_metrics  : pass when report_stage="test"
    """
    try:
        import streamlit as st
    except ImportError:
        return

    report_data = build_report_data(
        session_state,
        dataset_name=dataset_name,
        report_stage=report_stage,
        test_baseline_metrics=test_baseline_metrics,
        test_enhanced_metrics=test_enhanced_metrics,
    )

    html_bytes = generate_html_report(report_data)
    md_bytes   = generate_markdown_report(report_data)

    stage_slug = "test" if report_stage == "test" else "validation"
    safe_name  = dataset_name.replace(" ", "_")[:40]
    ts         = datetime.datetime.now().strftime("%Y%m%d")
    base_fn    = f"report_{safe_name}_{stage_slug}_{ts}"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Download HTML Report",
            data=html_bytes,
            file_name=f"{base_fn}.html",
            mime="text/html",
            help="Self-contained HTML with all charts — open in any browser",
            use_container_width=True,
            key=f"dl_html_report_{key_suffix}",
        )

    with col2:
        st.download_button(
            label="📝 Download Markdown",
            data=md_bytes,
            file_name=f"{base_fn}.md",
            mime="text/markdown",
            help="Markdown version — paste into GitHub, Notion, or Confluence",
            use_container_width=True,
            key=f"dl_md_report_{key_suffix}",
        )

    with col3:
        pdf_bytes = generate_pdf_report(report_data)
        if pdf_bytes:
            st.download_button(
                label="📑 Download PDF",
                data=pdf_bytes,
                file_name=f"{base_fn}.pdf",
                mime="application/pdf",
                help="PDF export via WeasyPrint",
                use_container_width=True,
                key=f"dl_pdf_report_{key_suffix}",
            )
        else:
            st.button(
                "📑 PDF unavailable",
                disabled=True,
                use_container_width=True,
                key=f"dl_pdf_disabled_{key_suffix}",
                help="Run `pip install weasyprint` to enable PDF export",
            )