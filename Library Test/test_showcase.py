"""
ML Compass Library — Brief Showcase
=================================

A concise end-to-end demonstration of the mlcompass pipeline using the
Booking Cancellation dataset. Shows how to go from raw CSV to a full
HTML report in ~60 lines of pipeline code.

Usage:  python test_showcase.py
"""

import os, sys, logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("showcase")

# -- path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlcompass.recommendation.meta_models import load_meta_models
from mlcompass.analysis.profiling import get_column_type_info, detect_problematic_columns
from mlcompass.recommendation.engine import generate_suggestions, deduplicate_suggestions, recommended_top_k
from mlcompass.transforms.applicator import fit_and_apply_suggestions, apply_fitted_to_test
from mlcompass.transforms.helpers import ensure_numeric_target, sanitize_feature_names
from mlcompass.evaluation.training import prepare_data_for_model, train_lgbm_model
from mlcompass.evaluation.metrics import evaluate_on_set
from mlcompass.reporting.generator import build_report_data, generate_html_report

# -- configuration ------------------------------------------------------------
DATASETS = os.path.join(os.path.dirname(__file__), "Datasets")
TRAIN_CSV = os.path.join(DATASETS, "Booking_Cancelation_train.csv")
TEST_CSV  = os.path.join(DATASETS, "Booking_Cancelation_test.csv")
TARGET    = "booking status"
TOP_K     = None                     # resolved dynamically via recommended_top_k()

# =============================================================================
# 1. Load data
# =============================================================================
log.info("Loading datasets ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_full = train_df.drop(columns=[TARGET])
y_full = train_df[TARGET]
y_full = ensure_numeric_target(y_full)

log.info(f"  Train shape : {X_full.shape}  |  Classes : {y_full.nunique()}")

# =============================================================================
# 2. Infer column types & drop useless columns
# =============================================================================
log.info("Inferring column types ...")
col_info = get_column_type_info(X_full)
for name, info in col_info.items():
    log.info(f"  {name:30s}  ->  {info['detected']:12s}  (drop={info['drop_suggested']})")

drop_cols = [c for c, info in col_info.items() if info["drop_suggested"]]
if drop_cols:
    log.info(f"Dropping {len(drop_cols)} column(s): {drop_cols}")
    X_full = X_full.drop(columns=drop_cols)

# =============================================================================
# 3. Analyze dataset (problematic-column scan)
# =============================================================================
log.info("Detecting problematic columns ...")
problems = detect_problematic_columns(X_full)
for category, cols in problems.items():
    if cols:
        log.info(f"  {category}: {cols}")

# =============================================================================
# 4. Train / validation split & baseline model
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

X_tr_prep, X_vl_prep, col_encs = prepare_data_for_model(X_train, X_val)
n_classes = y_full.nunique()

log.info("Training baseline model ...")
base_model, base_cols, base_encs = train_lgbm_model(
    X_tr_prep, y_train, X_vl_prep, y_val, n_classes=n_classes
)
base_val = evaluate_on_set(base_model, X_vl_prep, y_val, base_cols, n_classes, base_encs)
log.info(f"  Baseline ROC-AUC : {base_val['roc_auc']:.4f}")

# =============================================================================
# 5. Generate & select suggestions
# =============================================================================
log.info("Loading meta-models ...")
meta_models = load_meta_models()

log.info("Generating suggestions ...")
suggestions, skipped, advisories, ds_meta = generate_suggestions(
    X_full, y_full, meta_models,
    baseline_score=base_val["roc_auc"],
    baseline_std=0.02,
)
suggestions = deduplicate_suggestions(suggestions)
log.info(f"  Total suggestions (after dedup) : {len(suggestions)}")

TOP_K = recommended_top_k(X_full)
selected = suggestions[:TOP_K]
log.info(f"  Selected top {TOP_K} suggestions (auto-sized for {X_full.shape[0]:,} rows × {X_full.shape[1]} cols):")
for i, s in enumerate(selected):
    log.info(f"    {i+1:2d}. [{s['type']:12s}] {s['method']:25s}  "
             f"col={s.get('column','—'):20s}  delta={s['predicted_delta']:+.4f}")

# -- Custom suggestions (manually crafted, not from meta-models) --------------
custom_suggestions = [
    # 1. Row-level: zero statistics across all numeric columns
    {
        'type': 'row',
        'column': '(all numeric cols)',
        'column_b': None,
        'method': 'row_zero_stats',
    },
    # 2. Date: cyclical month encoding for reservation date
    {
        'type': 'date',
        'column': 'date of reservation',
        'column_b': None,
        'method': 'date_cyclical_month',
    },
    # 3. Interaction: mean of "average price" grouped by "type of meal"
    {
        'type': 'interaction',
        'column': 'average price',
        'column_b': 'type of meal',
        'method': 'group_mean',
    },
    # 4. Numerical: log transform on "lead time" (right-skewed booking horizon)
    {
        'type': 'numerical',
        'column': 'lead time',
        'column_b': None,
        'method': 'log_transform',
    },
    # 5. Categorical: target encoding on "market segment type"
    {
        'type': 'categorical',
        'column': 'market segment type',
        'column_b': None,
        'method': 'target_encoding',
    },
]

selected.extend(custom_suggestions)
log.info(f"  Added {len(custom_suggestions)} custom suggestions → total selected: {len(selected)}")
for i, s in enumerate(custom_suggestions, start=1):
    log.info(f"    [custom {i}] [{s['type']:12s}] {s['method']:25s}  col={s.get('column','—')}")

# =============================================================================
# 6. Apply suggestions
# =============================================================================
log.info("Applying suggestions to training data ...")
X_enhanced, fitted_params = fit_and_apply_suggestions(X_full, y_full, selected)
log.info(f"  Enhanced shape : {X_enhanced.shape}  (was {X_full.shape})")

# =============================================================================
# 7. Train enhanced model
# =============================================================================
X_enh_tr, X_enh_vl, y_enh_tr, y_enh_vl = train_test_split(
    X_enhanced, y_full, test_size=0.2, random_state=42, stratify=y_full
)
X_enh_tr_p, X_enh_vl_p, enh_col_encs = prepare_data_for_model(X_enh_tr, X_enh_vl)

log.info("Training enhanced model ...")
enh_model, enh_cols, enh_encs = train_lgbm_model(
    X_enh_tr_p, y_enh_tr, X_enh_vl_p, y_enh_vl, n_classes=n_classes
)
enh_val = evaluate_on_set(enh_model, X_enh_vl_p, y_enh_vl, enh_cols, n_classes, enh_encs)
log.info(f"  Enhanced ROC-AUC : {enh_val['roc_auc']:.4f}")

# =============================================================================
# 8. Feature importances & comparison
# =============================================================================
base_imp = dict(zip(base_cols, base_model.feature_importances_))
enh_imp  = dict(zip(enh_cols,  enh_model.feature_importances_))

log.info("Baseline — top 10 features by importance:")
for feat, imp in sorted(base_imp.items(), key=lambda x: -x[1])[:10]:
    log.info(f"    {feat:35s}  {imp}")

log.info("Enhanced — top 10 features by importance:")
for feat, imp in sorted(enh_imp.items(), key=lambda x: -x[1])[:10]:
    log.info(f"    {feat:35s}  {imp}")

# =============================================================================
# 9. Evaluate on test set
# =============================================================================
log.info("Evaluating on test set ...")

X_test = test_df.drop(columns=[TARGET])
y_test = ensure_numeric_target(test_df[TARGET])

# Drop the same columns we dropped from training
if drop_cols:
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Baseline on test
X_test_prep_base, _, base_test_encs = prepare_data_for_model(X_test, X_test)
test_base = evaluate_on_set(base_model, X_test_prep_base, y_test, base_cols, n_classes, base_test_encs)
log.info(f"  Baseline  test ROC-AUC : {test_base['roc_auc']:.4f}")

# Enhanced on test
X_test_enh = apply_fitted_to_test(X_test, fitted_params)
X_test_prep_enh, _, enh_test_encs = prepare_data_for_model(X_test_enh, X_test_enh)
test_enh = evaluate_on_set(enh_model, X_test_prep_enh, y_test, enh_cols, n_classes, enh_test_encs)
log.info(f"  Enhanced  test ROC-AUC : {test_enh['roc_auc']:.4f}")

# =============================================================================
# 10. Compare baseline vs enhanced (validation + test)
# =============================================================================
log.info("=" * 60)
log.info("MODEL COMPARISON  —  VALIDATION")
log.info("=" * 60)
for metric in ["accuracy", "roc_auc", "f1", "precision", "recall", "log_loss"]:
    b = base_val.get(metric, 0)
    e = enh_val.get(metric, 0)
    delta = e - b
    arrow = "+" if delta >= 0 else ""
    log.info(f"  {metric:12s}   baseline={b:.4f}   enhanced={e:.4f}   ({arrow}{delta:.4f})")

log.info("=" * 60)
log.info("MODEL COMPARISON  —  TEST")
log.info("=" * 60)
for metric in ["accuracy", "roc_auc", "f1", "precision", "recall", "log_loss"]:
    b = test_base.get(metric, 0)
    e = test_enh.get(metric, 0)
    delta = e - b
    arrow = "+" if delta >= 0 else ""
    log.info(f"  {metric:12s}   baseline={b:.4f}   enhanced={e:.4f}   ({arrow}{delta:.4f})")

# =============================================================================
# 11. Generate HTML report (with test results)
# =============================================================================
log.info("Generating HTML report ...")

session_state = {
    "X_train": X_full,
    "y_train": y_full,
    "target_col": TARGET,
    "n_classes": n_classes,
    "_col_type_info": col_info,
    "suggestions": suggestions,
    "selected_indices": list(range(TOP_K)),
    "advisories": advisories,
    "skipped_info": skipped,
    "ds_meta": ds_meta,
    "baseline_val_metrics": base_val,
    "enhanced_val_metrics": enh_val,
    "baseline_model": base_model,
    "enhanced_model": enh_model,
    "baseline_train_cols": base_cols,
    "enhanced_train_cols": enh_cols,
    "fitted_params": fitted_params,
}

report_data = build_report_data(
    session_state,
    dataset_name="Booking Cancellation",
    report_stage="test",
    test_baseline_metrics=test_base,
    test_enhanced_metrics=test_enh,
)
html_bytes = generate_html_report(report_data)

out_path = os.path.join(os.path.dirname(__file__), "showcase_report.html")
with open(out_path, "wb") as f:
    f.write(html_bytes)
log.info(f"  Report saved to {out_path}  ({len(html_bytes):,} bytes)")

# =============================================================================
# Done
# =============================================================================
log.info("=" * 60)
log.info("SHOWCASE COMPLETE")
log.info("=" * 60)
