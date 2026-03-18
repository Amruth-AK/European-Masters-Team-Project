"""
mlcompass — ML feature engineering recommendation library
========================================================

Provides programmatic access to meta-feature extraction, transform suggestion
generation, model training/evaluation, and report generation.
"""

__version__ = "0.1.0"

from mlcompass.analysis.meta_features import (
    get_dataset_meta,
    get_row_dataset_meta,
    get_numeric_column_meta,
    get_categorical_column_meta,
    get_pair_meta_features,
    get_baseline_importances,
    should_test_numerical,
    should_test_categorical,
)
from mlcompass.analysis.profiling import (
    get_column_type_info,
    detect_problematic_columns,
    detect_date_columns,
    detect_date_has_hour,
    _override_options_for,
    _validate_col_override,
    _apply_type_reassignments,
)
from mlcompass.analysis.advisories import generate_dataset_advisories

from mlcompass.recommendation.engine import generate_suggestions, deduplicate_suggestions, recommended_top_k
from mlcompass.recommendation.meta_models import load_meta_models, build_feature_vector
from mlcompass.recommendation.verdicts import _compute_suggestion_verdicts

from mlcompass.transforms.applicator import fit_and_apply_suggestions, apply_fitted_to_test
from mlcompass.transforms.helpers import (
    ensure_numeric_target, sanitize_feature_names,
    _apply_date_features, _apply_date_cyclical, _apply_text_stats,
    _to_datetime_safe,
)
from mlcompass.transforms.detection import (
    detect_dow_columns, detect_text_columns,
    _DOW_TO_INT, _DOW_ALL,
)

from mlcompass.evaluation.training import train_lgbm_model, prepare_data_for_model
from mlcompass.evaluation.metrics import (
    evaluate_on_set, predict_on_set,
    _metrics_at_threshold, _find_optimal_thresholds,
)

from mlcompass.reporting.generator import (
    build_report_data,
    generate_html_report,
    generate_markdown_report,
    generate_pdf_report,
)
