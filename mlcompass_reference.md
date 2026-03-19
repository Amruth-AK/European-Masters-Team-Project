# mlcompass — Complete Reference

**Description:** ML feature engineering recommendation library — meta-model driven transform suggestions for classification datasets.

**Dependencies:** numpy ≥1.23, pandas ≥1.5, scikit-learn ≥1.2, scipy ≥1.9, lightgbm ≥3.3
**Optional:** matplotlib ≥3.6 (for report generation — install with `pip install mlcompass[reports]`)

---

## Overview

When a user installs mlcompass, they get a complete pipeline for **automated feature engineering recommendations** for classification tasks. The library analyzes a dataset, extracts meta-features at the dataset, column, and column-pair levels, runs pre-trained meta-models to predict which transforms will improve model performance, and then applies those transforms. It also trains LightGBM baselines, evaluates results, and generates styled reports.

The typical workflow is:

1. **Load meta-models** → `load_meta_models()`
2. **Profile the dataset** → `get_column_type_info()`, `detect_problematic_columns()`
3. **Train a baseline model** → `train_lgbm_model()`, `evaluate_on_set()`
4. **Generate suggestions** → `generate_suggestions()`, `deduplicate_suggestions()`
5. **Apply selected transforms** → `fit_and_apply_suggestions()`
6. **Train an enhanced model** → `train_lgbm_model()`, `evaluate_on_set()`
7. **Diagnose transform impact** → `_compute_suggestion_verdicts()`
8. **Apply transforms to test data** → `apply_fitted_to_test()`
9. **Generate reports** → `build_report_data()`, `generate_html_report()`

---

## Module: `mlcompass.recommendation.meta_models`

### `load_meta_models(model_dir=None)`

Load all pre-trained meta-models from disk. These are LightGBM regressors that predict how much a given transform will improve model performance.

**Parameters:**
- `model_dir` *(str or path-like, optional)* — Directory containing subdirectories (`numerical/`, `categorical/`, `interaction/`, `row/`) each with a `*_config.json` and `*_regressor.txt`. If `None`, loads bundled models from `mlcompass.data.meta_models`.

**Returns:** `dict` — Keys are `'numerical'`, `'categorical'`, `'interaction'`, `'row'`. Each value is a dict with keys `'booster'` (LightGBM Booster), `'config'` (dict), `'feature_names'` (list), `'method_vocab'` (list).

---

### `build_feature_vector(meta_dict, method, config)`

Build a single feature vector matching the training schema of a meta-model. Used internally by `generate_suggestions()`.

**Parameters:**
- `meta_dict` *(dict)* — Combined dataset + column/pair meta-features.
- `method` *(str)* — The transform method name (e.g. `'log_transform'`, `'target_encoding'`).
- `config` *(dict)* — The config dict from `load_meta_models()`.

**Returns:** `pd.DataFrame` — A single-row DataFrame ready for `booster.predict()`.

---

## Module: `mlcompass.analysis.meta_features`

### `get_dataset_meta(X, y)`

Compute dataset-level meta-features: shape, class balance, correlations, and a landmarking score.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `y` *(pd.Series)* — Target variable.

**Returns:** `dict` with keys: `n_rows`, `n_cols`, `n_numeric_cols`, `n_cat_cols`, `cat_ratio`, `missing_ratio`, `row_col_ratio`, `n_classes`, `class_imbalance_ratio`, `avg_feature_corr`, `max_feature_corr`, `avg_target_corr`, `max_target_corr`, `landmarking_score`.

---

### `get_row_dataset_meta(X)`

Compute row-level dataset meta-features used by the row meta-model. These are dataset-level aggregates (not per-column).

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.

**Returns:** `dict` with keys: `n_numeric_cols_used`, `avg_numeric_mean`, `avg_numeric_std`, `avg_missing_pct`, `max_missing_pct`, `avg_row_variance`, `pct_rows_with_any_missing`, `pct_cells_zero`, `pct_rows_with_any_zero`, `numeric_col_corr_mean`, `numeric_col_corr_max`, `avg_row_entropy`, `numeric_range_ratio`.

---

### `get_numeric_column_meta(series, y, importance, importance_rank_pct)`

Compute meta-features for a single numeric column.

**Parameters:**
- `series` *(pd.Series)* — The numeric column.
- `y` *(pd.Series)* — Target variable.
- `importance` *(float)* — Baseline feature importance for this column.
- `importance_rank_pct` *(float)* — Percentile rank of this column's importance (0–1).

**Returns:** `dict` with keys: `null_pct`, `unique_ratio`, `outlier_ratio`, `skewness`, `kurtosis_val`, `coeff_variation`, `zeros_ratio`, `entropy`, `is_binary`, `range_iqr_ratio`, `baseline_feature_importance`, `importance_rank_pct`, `spearman_corr_target`, `mutual_info_score`, `shapiro_p_value`, `bimodality_coefficient`, `pct_negative`, `pct_in_0_1_range`.

---

### `get_categorical_column_meta(series, y, importance, importance_rank_pct)`

Compute meta-features for a single categorical column.

**Parameters:**
- `series` *(pd.Series)* — The categorical column.
- `y` *(pd.Series)* — Target variable.
- `importance` *(float)* — Baseline feature importance.
- `importance_rank_pct` *(float)* — Percentile rank of importance.

**Returns:** `dict` with keys: `null_pct`, `n_unique`, `unique_ratio`, `entropy`, `normalized_entropy`, `is_binary`, `is_low_cardinality`, `is_high_cardinality`, `top_category_dominance`, `top3_category_concentration`, `rare_category_pct`, `conditional_entropy`, `baseline_feature_importance`, `importance_rank_pct`, `mutual_info_score`, `pps_score`.

---

### `get_pair_meta_features(col_a, col_b, y, imp_a, imp_b)`

Compute order-invariant pair-level meta-features for interaction suggestions. Handles num+num, num+cat, and cat+cat pairs using sentinel encoding.

**Parameters:**
- `col_a` *(pd.Series)* — First column.
- `col_b` *(pd.Series)* — Second column.
- `y` *(pd.Series)* — Target variable.
- `imp_a` *(float)* — Baseline importance of col_a.
- `imp_b` *(float)* — Baseline importance of col_b.

**Returns:** `dict` with 35 features including `n_numerical_cols`, `pearson_corr`, `spearman_corr`, `mutual_info_pair`, `mic_score`, `scale_ratio`, `sum_importance`, `eta_squared`, `anova_f_stat`, `cramers_v`, and more.

---

### `get_baseline_importances(X, y)`

Train a quick LightGBM classifier and extract feature importances as a baseline reference.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `y` *(pd.Series)* — Target variable.

**Returns:** `pd.Series` — Feature importances indexed by column name.

---

### `should_test_numerical(method, col_meta, series)`

Determine whether a numerical transform method is applicable to a given column based on its meta-features.

**Parameters:**
- `method` *(str)* — Transform name (e.g. `'log_transform'`).
- `col_meta` *(dict)* — Output of `get_numeric_column_meta()`.
- `series` *(pd.Series)* — The column data.

**Returns:** `bool` — `True` if the method should be tested on this column.

---

### `should_test_categorical(method, col_meta)`

Determine whether a categorical transform method is applicable.

**Parameters:**
- `method` *(str)* — Transform name (e.g. `'onehot_encoding'`).
- `col_meta` *(dict)* — Output of `get_categorical_column_meta()`.

**Returns:** `bool` — `True` if applicable.

---

## Module: `mlcompass.analysis.profiling`

### `get_column_type_info(X)`

Run all column-type detectors (date, day-of-week, text, ID, constant, binary, numerical, categorical) and return a per-column summary.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.

**Returns:** `dict[str, dict]` — Keyed by column name. Each value has: `detected` (type string), `icon`, `n_unique`, `missing_pct`, `sample` (first non-null value), `drop_suggested` (bool), `drop_reason`, `is_numeric`.

---

### `detect_problematic_columns(X, known_date_cols=None, known_text_cols=None)`

Detect columns that should be excluded or handled carefully: IDs, constants, binary numerics, and high-missing columns.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `known_date_cols` *(set, optional)* — Columns already identified as dates.
- `known_text_cols` *(set, optional)* — Columns already identified as text.

**Returns:** `dict` with keys: `'id_columns'`, `'constant_columns'`, `'binary_num_columns'`, `'high_missing_columns'`. Each maps column name → reason string.

---

### `detect_date_columns(X)`

Auto-detect columns containing date/datetime/time values by attempting to parse them.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.

**Returns:** `dict[str, dict]` — Keyed by column name. Each value has `'parse_rate'` (float) and `'col_type'` (`'date'`, `'datetime'`, or `'time'`).

---

### `detect_date_has_hour(X, col)`

Check whether a detected date column carries a time component.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `col` *(str)* — Column name.

**Returns:** `bool`

---

## Module: `mlcompass.analysis.advisories`

### `generate_dataset_advisories(X, y)`

Inspect dataset-level properties and return advisory warnings for small datasets, high dimensionality, and high missingness.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `y` *(pd.Series)* — Target variable.

**Returns:** `list[dict]` — Each advisory has keys: `'category'`, `'severity'` (`'low'`/`'medium'`), `'title'`, `'detail'`, `'suggested_params'` (dict or None). Note: `'high'` severity advisories (e.g., data leakage warnings) are added later by `generate_suggestions()` in the recommendation engine, not by this function.

---

## Module: `mlcompass.recommendation.engine`

### `generate_suggestions(X, y, meta_models, baseline_score, baseline_std, progress_cb=None, type_reassignments=None, real_n_rows=None, include_imbalance=True)`

The main suggestion engine. Runs meta-models on all applicable (column, method) combinations and returns ranked suggestions.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix (training data).
- `y` *(pd.Series)* — Target variable.
- `meta_models` *(dict)* — Output of `load_meta_models()`.
- `baseline_score` *(float)* — Baseline model score (e.g. ROC-AUC).
- `baseline_std` *(float)* — Standard deviation of baseline score.
- `progress_cb` *(callable, optional)* — Progress callback accepting a float 0.0–1.0.
- `type_reassignments` *(dict, optional)* — Manual column type overrides (column → type string).
- `real_n_rows` *(int, optional)* — When using a subsample, pass the full dataset's row count.
- `include_imbalance` *(bool, default True)* — If `True`, a `class_weight_balance` suggestion is injected when the class imbalance ratio ≥ `_IMBALANCE_MODERATE` (5:1). Set to `False` to suppress imbalance suggestions entirely. Note: even when `True`, datasets with low imbalance (ratio < 5:1) will **not** receive an imbalance suggestion.

**Returns:** `tuple` of:
1. `suggestions` *(list[dict])* — Ranked list of suggestion dicts, each with keys: `type`, `column`, `column_b`, `method`, `predicted_delta`, `description`, `meta`, and type-specific extras.
2. `skipped_info` *(dict)* — From `detect_problematic_columns()`.
3. `advisories` *(list[dict])* — Dataset advisories including leakage warnings.
4. `ds_meta` *(dict)* — Dataset-level meta-features.

---

### `deduplicate_suggestions(suggestions)`

For single-column transforms, keep only the best in-place method per column. Additive transforms (e.g. polynomial, missing indicator) are always kept. For interactions, keep only the best method per pair.

**Parameters:**
- `suggestions` *(list[dict])* — Output of `generate_suggestions()`.

**Returns:** `list[dict]` — Deduplicated and sorted suggestions.

---

### `recommended_top_k(X)`

Compute a sensible default for the number of suggestions to apply, scaled by dataset rows and columns. Use this to decide how many top-ranked suggestions to pass to `fit_and_apply_suggestions()` when calling the library programmatically.

**Parameters:**
- `X` *(pd.DataFrame or None)* — Training feature matrix. When `None`, returns a safe fallback of `10`.

**Returns:** `int` — Recommended number of top suggestions to apply.

**Scaling logic:**

| Rows | Base k |
|------|--------|
| < 500 | 3 |
| < 2 000 | 5 |
| < 10 000 | 8 |
| < 50 000 | 12 |
| ≥ 50 000 | 15 |

Column adjustment: +1 if `n_cols > 15`, +3 if `n_cols > 30` (capped at 20).

**Example (programmatic usage):**

```python
from mlcompass import (
    load_meta_models, generate_suggestions, deduplicate_suggestions,
    recommended_top_k, fit_and_apply_suggestions,
)

meta_models = load_meta_models()
suggestions, _, _, _ = generate_suggestions(X_train, y_train, meta_models,
                                            baseline_score=0.75, baseline_std=0.02)
suggestions = deduplicate_suggestions(suggestions)

# Use recommended_top_k as the default number of suggestions to apply
k = recommended_top_k(X_train)
top_suggestions = [s for s in suggestions[:k] if s.get('predicted_delta_raw', 0) > 0]
X_enhanced, fitted_params = fit_and_apply_suggestions(X_train, y_train, top_suggestions)
```

---

## Module: `mlcompass.transforms.applicator`

### `fit_and_apply_suggestions(X_train, y_train, suggestions)`

Apply selected suggestions to training data. Fits all necessary statistics (medians, encoding maps, bin edges, etc.) from training data only, then transforms.

**Parameters:**
- `X_train` *(pd.DataFrame)* — Training features.
- `y_train` *(pd.Series)* — Training target.
- `suggestions` *(list[dict])* — Selected suggestion dicts (subset of `generate_suggestions()` output).

**Returns:** `tuple` of:
1. `X_enhanced` *(pd.DataFrame)* — Transformed training data.
2. `fitted_params` *(list[dict])* — Fitted parameters for each transform, needed to replay on test data.

**Supported transforms:**

| Type | Method | Description |
|------|--------|-------------|
| Numerical | `log_transform` | Log transform — reduces right skew |
| Numerical | `sqrt_transform` | Square root transform — mild skew reduction |
| Numerical | `polynomial_square` | Adds squared feature (new column) |
| Numerical | `polynomial_cube` | Adds cubed feature (new column) |
| Numerical | `reciprocal_transform` | Adds reciprocal 1/x feature (new column) |
| Numerical | `quantile_binning` | Discretizes into 5 quantile bins |
| Numerical | `impute_median` | Fills missing with training median |
| Numerical | `missing_indicator` | Adds binary `_is_na` column |
| Categorical | `frequency_encoding` | Replaces categories with their frequency |
| Categorical | `target_encoding` | Replaces categories with smoothed target mean (5-fold OOF) |
| Categorical | `onehot_encoding` | One-hot encodes (drop first) |
| Categorical | `hashing_encoding` | Hash encodes into 32 buckets |
| Categorical | `missing_indicator` | Adds binary `_is_na` column |
| Interaction | `product_interaction` | A × B (new column) |
| Interaction | `division_interaction` | A / |B| (new column) |
| Interaction | `addition_interaction` | A + B (new column) |
| Interaction | `abs_diff_interaction` | |A − B| (new column) |
| Interaction | `group_mean` | Mean of numeric grouped by category (new column) |
| Interaction | `group_std` | Std of numeric grouped by category (new column) |
| Interaction | `cat_concat` | Concatenates two categories → label-encoded (new column) |
| Row | `row_numeric_stats` | Row mean, median, sum, std, min, max, range |
| Row | `row_zero_stats` | Row zero count and percentage |
| Row | `row_missing_stats` | Row missing count and percentage |
| Date | `date_features` | Extracts year, month, day, dayofweek, quarter, is_weekend, weekofyear, days_since_min, hour |
| Date | `date_cyclical_month` | Sin/cos of month (1–12) |
| Date | `date_cyclical_dow` | Sin/cos of day-of-week (0–6) |
| Date | `date_cyclical_dom` | Sin/cos of day-of-month (1–31) |
| Date | `date_cyclical_hour` | Sin/cos of hour (0–23) |
| Date | `dow_ordinal` | Maps day-of-week text → 0–6 integer |
| Date | `dow_cyclical` | Sin/cos of day-of-week from text column |
| Text | `text_stats` | Word count, char count, avg word length, uppercase %, digit %, punctuation % |
| Text | `text_tfidf` | Top-20 TF-IDF unigram features |

---

### `apply_fitted_to_test(X_test, fitted_params_list)`

Apply pre-fitted transforms to test data using statistics learned from training.

**Parameters:**
- `X_test` *(pd.DataFrame)* — Test features.
- `fitted_params_list` *(list[dict])* — Output of `fit_and_apply_suggestions()`.

**Returns:** `pd.DataFrame` — Transformed test data.

---

## Module: `mlcompass.transforms.helpers`

### `ensure_numeric_target(y)`

Convert a target series to numeric values using label encoding if needed.

**Parameters:**
- `y` *(pd.Series)* — Target variable.

**Returns:** `pd.Series` — Numeric target.

---

### `sanitize_feature_names(df)`

Clean column names by replacing special characters (`[]{}":,`) with underscores and resolving duplicates.

**Parameters:**
- `df` *(pd.DataFrame)* — DataFrame whose columns to sanitize.

**Returns:** `pd.DataFrame` — Same DataFrame with cleaned column names.

---

## Module: `mlcompass.transforms.detection`

### `detect_dow_columns(X, already_date_cols=None)`

Detect columns containing day-of-week labels (Monday, Mon, Mo, etc.).

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `already_date_cols` *(set, optional)* — Columns to skip (already identified as dates).

**Returns:** `set[str]` — Column names containing day-of-week values.

---

### `detect_text_columns(X, date_cols=None, dow_cols=None)`

Detect columns that appear to contain free-form text (avg length > 30 chars, unique ratio > 10%).

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `date_cols` *(set, optional)* — Columns to skip.
- `dow_cols` *(set, optional)* — Columns to skip.

**Returns:** `dict[str, str]` — Column name → description string.

---

## Module: `mlcompass.evaluation.training`

### `prepare_data_for_model(X_train, X_val)`

Encode categoricals (label encoding) and fill missing values for LightGBM training.

**Parameters:**
- `X_train` *(pd.DataFrame)* — Training features.
- `X_val` *(pd.DataFrame)* — Validation features.

**Returns:** `tuple` of:
1. `X_tr` *(pd.DataFrame)* — Encoded training data.
2. `X_vl` *(pd.DataFrame)* — Encoded validation data.
3. `col_encoders` *(dict)* — Per-column dict mapping column → `{'encoder': LabelEncoder or None, 'median': float}`.

---

### `train_lgbm_model(X_train, y_train, X_val, y_val, n_classes, apply_imbalance=False, imbalance_strategy='none', base_params=None)`

Train a LightGBM classifier with early stopping on the validation set.

**Parameters:**
- `X_train` *(pd.DataFrame)* — Training features.
- `y_train` *(pd.Series)* — Training target.
- `X_val` *(pd.DataFrame)* — Validation features.
- `y_val` *(pd.Series)* — Validation target.
- `n_classes` *(int)* — Number of target classes.
- `apply_imbalance` *(bool)* — Whether to apply class reweighting.
- `imbalance_strategy` *(str)* — One of `'none'`, `'binary'`, `'multiclass_moderate'`, `'low'`.
- `base_params` *(dict, optional)* — LightGBM hyperparameters (defaults to `BASE_PARAMS`).

**Returns:** `tuple` of:
1. `model` — Trained `LGBMClassifier`.
2. `train_columns` *(list[str])* — Column names used for training.
3. `col_encoders` *(dict)* — Encoding information for each column.

---

## Module: `mlcompass.evaluation.metrics`

### `metrics_at_threshold(y_true, y_proba, threshold)`

Compute classification metrics using a custom probability threshold for binary problems.

**Parameters:**
- `y_true` *(array-like)* — Ground-truth binary labels (0/1).
- `y_proba` *(array-like)* — Predicted probabilities for the positive class.
- `threshold` *(float)* — Decision boundary (0–1).

**Returns:** `dict` with keys: `threshold`, `accuracy`, `f1`, `precision`, `recall`.

---

### `find_optimal_thresholds(y_true, y_proba, n_steps=200)`

Sweep thresholds and return the optimal one for each metric. Useful for finding the threshold that maximises F1, precision, recall, or accuracy on a labelled set — then passing that threshold to `predict_on_set()` or `evaluate_on_set()` for deployment or test-set evaluation.

**Parameters:**
- `y_true` *(array-like)* — Ground-truth binary labels (0/1).
- `y_proba` *(array-like)* — Predicted probabilities for the positive class.
- `n_steps` *(int, default 200)* — Number of threshold steps to evaluate.

**Returns:** `dict` — Keys are `'f1'`, `'precision'`, `'recall'`, `'accuracy'`. Each value is a dict with `'threshold'` (float) and `'value'` (float) representing the threshold that maximises that metric and the metric value achieved.

---

### `evaluate_on_set(model, X, y, train_columns, n_classes, col_encoders=None, threshold=None)`

Evaluate a trained model on a dataset. Computes accuracy, ROC-AUC, F1, precision, recall, log loss, confusion matrix, ROC curve data, precision-recall curve data, and (for binary problems) optimal thresholds per metric.

**Parameters:**
- `model` — Trained LightGBM model.
- `X` *(pd.DataFrame)* — Features to evaluate on.
- `y` *(pd.Series)* — True labels.
- `train_columns` *(list[str])* — Column names the model was trained on.
- `n_classes` *(int)* — Number of target classes.
- `col_encoders` *(dict, optional)* — From `prepare_data_for_model()`.
- `threshold` *(float, optional)* — Custom probability threshold for binary classification (0–1). When provided and `n_classes == 2`, predictions are derived from `y_pred_proba[:, 1] >= threshold` instead of `model.predict()`. All downstream metrics (accuracy, F1, precision, recall, confusion matrix) reflect the custom threshold. Ignored for multiclass problems.

**Returns:** `dict` with keys: `accuracy`, `roc_auc`, `f1`, `precision`, `recall`, `log_loss`, `confusion_matrix`, `y_classes`, `roc_data` (fpr/tpr/auc), `pr_data` (precision/recall/thresholds/avg_precision), `optimal_thresholds` (binary only — per-metric optimal thresholds from `find_optimal_thresholds()`; `None` for multiclass), `_y_pred`, `_y_pred_proba`.

---

### `predict_on_set(model, X, train_columns, n_classes, col_encoders=None, threshold=None)`

Generate predictions without ground-truth labels (predict-only mode).

**Parameters:**
- `model` — Trained LightGBM model.
- `X` *(pd.DataFrame)* — Features.
- `train_columns` *(list[str])* — Training column names.
- `n_classes` *(int)* — Number of classes.
- `col_encoders` *(dict, optional)* — Encoding info.
- `threshold` *(float, optional)* — Custom probability threshold for binary classification (0–1). When provided and `n_classes == 2`, predictions are derived from `y_pred_proba[:, 1] >= threshold` instead of `model.predict()`. Ignored for multiclass problems.

**Returns:** `dict` with keys: `_y_pred`, `_y_pred_proba`.

---

## Module: `mlcompass.recommendation.verdicts`

### `_compute_suggestion_verdicts(fitted_params, suggestions, selected_indices, enhanced_model, enhanced_train_cols, baseline_model, baseline_train_cols, baseline_val_metrics, enhanced_val_metrics, apply_imbalance=False)`

Attribute each applied suggestion's impact using feature importances. Produces a verdict (`'good'`, `'marginal'`, `'bad'`) for each transform and optionally for class reweighting.

**Parameters:**
- `fitted_params` *(list[dict])* — From `fit_and_apply_suggestions()`.
- `suggestions` *(list[dict])* — Full suggestion list.
- `selected_indices` *(list[int])* — Indices of selected suggestions.
- `enhanced_model` — Trained enhanced model.
- `enhanced_train_cols` *(list[str])* — Enhanced model's training columns.
- `baseline_model` — Trained baseline model.
- `baseline_train_cols` *(list[str])* — Baseline model's training columns.
- `baseline_val_metrics` *(dict)* — Baseline validation metrics.
- `enhanced_val_metrics` *(dict)* — Enhanced validation metrics.
- `apply_imbalance` *(bool)* — Whether class reweighting was applied.

**Returns:** `tuple` of:
1. `verdicts` *(list[dict])* — Per-transform verdicts with `verdict`, `reason`, `new_cols`, `bad_date_subfeatures`, `bad_row_stats`.
2. `low_imp_orig` *(dict[str, float])* — Original columns with < 0.5% baseline importance.

---

## Module: `mlcompass.reporting.generator`

### `build_report_data(session_state, dataset_name="dataset", report_stage="validation", test_baseline_metrics=None, test_enhanced_metrics=None)`

Assemble all data needed for report generation from a Streamlit session state (or dict-like object).

**Parameters:**
- `session_state` — Dict-like object containing all analysis results.
- `dataset_name` *(str)* — Label for the report title.
- `report_stage` *(str)* — `"validation"` or `"test"`.
- `test_baseline_metrics` *(dict, optional)* — Test-set baseline metrics.
- `test_enhanced_metrics` *(dict, optional)* — Test-set enhanced metrics.

**Returns:** `dict` — Complete report data bundle.

---

### `generate_html_report(d)`

Build a fully self-contained, styled HTML report with embedded charts (metrics comparison, feature importances, confusion matrices, ROC curves, class balance, delta distribution).

**Parameters:**
- `d` *(dict)* — Output of `build_report_data()`.

**Returns:** `bytes` — UTF-8 encoded HTML.

**Requires:** `matplotlib` (install with `pip install mlcompass[reports]`).

---

### `generate_markdown_report(d)`

Generate a lightweight Markdown version of the report (no charts).

**Parameters:**
- `d` *(dict)* — Output of `build_report_data()`.

**Returns:** `bytes` — UTF-8 encoded Markdown.

---

### `generate_pdf_report(d)`

Convert the HTML report to PDF via WeasyPrint.

**Parameters:**
- `d` *(dict)* — Output of `build_report_data()`.

**Returns:** `bytes | None` — PDF bytes, or `None` if WeasyPrint is not installed.

---

## Module: `mlcompass.analysis.profiling` (additional helpers)

### `_override_options_for(info)`

Return valid type-override options for a column (used in UI for manual column type reassignment).

**Parameters:**
- `info` *(dict)* — Column info dict from `get_column_type_info()`.

**Returns:** `list[str]` — e.g. `['Auto', 'Numerical', 'Categorical', 'Binary', 'Date', ...]`

---

### `_validate_col_override(col, override_type, info, X)`

Check whether a manual type override looks correct.

**Parameters:**
- `col` *(str)* — Column name.
- `override_type` *(str)* — The type the user wants to assign.
- `info` *(dict)* — Column info dict.
- `X` *(pd.DataFrame)* — Feature matrix.

**Returns:** `tuple(severity, message) | None` — `None` if override is valid; otherwise `('warning', msg)` or `('error', msg)`.

---

### `_apply_type_reassignments(X, type_reassignments, date_col_map, text_col_map, dow_cols, skipped_info)`

Apply user-specified column type overrides to all detection dictionaries.

**Parameters:**
- `X` *(pd.DataFrame)* — Feature matrix.
- `type_reassignments` *(dict)* — Column → new type string.
- `date_col_map`, `text_col_map`, `dow_cols`, `skipped_info` — Detection outputs to update.

**Returns:** `tuple` — Updated `(X, date_col_map, text_col_map, dow_cols, skipped_info)`.

---

## Constants (from `mlcompass.constants`)

| Constant | Description |
|----------|-------------|
| `DATASET_FEATURES` | List of 17 dataset-level meta-feature names |
| `NUMERICAL_COLUMN_FEATURES` | List of 18 numeric column meta-feature names |
| `CATEGORICAL_COLUMN_FEATURES` | List of 16 categorical column meta-feature names |
| `INTERACTION_PAIR_FEATURES` | List of 35 pair-level meta-feature names |
| `ROW_DATASET_FEATURES` | List of 13 row-level dataset meta-feature names |
| `NUMERICAL_METHODS` | 8 numerical transform methods |
| `CATEGORICAL_METHODS` | 5 categorical transform methods |
| `INTERACTION_METHODS_NUM_NUM` | 4 numeric interaction methods |
| `INTERACTION_METHODS_CAT_NUM` | 2 categorical-numeric interaction methods |
| `INTERACTION_METHODS_CAT_CAT` | 1 categorical-categorical interaction method |
| `ROW_FAMILIES` | 3 row-level transform families |
| `BASE_PARAMS` | Default LightGBM hyperparameters |
| `METHOD_DESCRIPTIONS` | Human-readable descriptions for all ~30 transform methods |
| `_SUGGESTION_GROUPS` | 9 UI groupings for suggestions (imbalance, missing, skewed, nonlinear, categorical, interactions, row patterns, date, text) |
| `_CUSTOM_METHODS` | Methods available for custom steps, by type |

---

## Protocol: `ProgressCallback`

Defined in `mlcompass._compat`. Any callable accepting a single `float` (0.0–1.0) satisfies this protocol. Works with Streamlit progress bars, tqdm updaters, or plain functions.

```python
def my_progress(fraction: float) -> None:
    print(f"{fraction*100:.0f}% done")
```