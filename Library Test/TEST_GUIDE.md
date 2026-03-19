# Comprehensive ML Compass Test Suite

## Overview

The `test_detailed.py` file contains a comprehensive test suite for the **mlcompass** Python library. This test suite is designed to be run **manually** by the user (not auto-executed) and provides extensive logging and print statements for visibility into all function calls and their return values.

**File Location:** `Library Test/test_detailed.py`

## Test Coverage

The test suite contains **40 test functions** organized into the following categories:

### 1. Data Loading & Setup (Section 1)
- Loads all 5 datasets from the `Datasets/` directory
- Displays dataset shapes, dtypes, missing value counts
- Separates X and y for each dataset
- **Test:** `test_data_loading()`

### 2. Meta-Model Loading (Section 2)
- Loads pre-trained meta-models using `load_meta_models()`
- Inspects booster types, configs, feature names, method vocabularies
- **Test:** `test_load_meta_models()`

### 3. Column Type Detection (Section 3)
- Tests `get_column_type_info()` on all datasets
- Logs detected types, icons, unique values, missing percentages
- Verifies date columns detected in Booking and customer_conversion
- Verifies text columns detected where expected
- **Test:** `test_get_column_type_info()`

### 4. Problematic Column Detection (Section 4)
- Tests `detect_problematic_columns()` on all datasets
- Identifies ID columns, constant columns, binary numerics, high missing columns
- Verifies expected ID columns are detected for each dataset
- **Test:** `test_detect_problematic_columns()`

### 5. Date Detection (Section 5)
- Tests `detect_date_columns()` and `detect_date_has_hour()` on Booking and customer_conversion
- Logs parse rates and column types
- **Test:** `test_detect_date_columns()`

### 6. Day-of-Week Detection (Section 6)
- Tests `detect_dow_columns()` on all datasets
- **Test:** `test_detect_dow_columns()`

### 7. Text Column Detection (Section 7)
- Tests `detect_text_columns()` on customer_conversion and recipe_reviews
- **Test:** `test_detect_text_columns()`

### 8. Dataset Advisories (Section 8)
- Tests `generate_dataset_advisories()` on all datasets
- Logs category, severity, title, details, and suggested parameters
- **Test:** `test_generate_dataset_advisories()`

### 9. Dataset-Level Meta-Features (Section 9)
- Tests `get_dataset_meta()` on all datasets
- Logs all 14 meta-feature values
- **Test:** `test_get_dataset_meta()`

### 10. Row-Level Meta-Features (Section 10)
- Tests `get_row_dataset_meta()` on all datasets
- Logs all 13 row-level meta-feature values
- **Test:** `test_get_row_dataset_meta()`

### 11. Numeric Column Meta-Features (Section 11)
- Tests `get_numeric_column_meta()` on numeric columns from each dataset
- Includes baseline importance computation
- Logs all 18 meta-feature values
- **Test:** `test_get_numeric_column_meta()`

### 12. Categorical Column Meta-Features (Section 12)
- Tests `get_categorical_column_meta()` on categorical columns
- Logs all 16 meta-feature values
- **Test:** `test_get_categorical_column_meta()`

### 13. Pair Meta-Features (Section 13)
- Tests `get_pair_meta_features()` on num+num, num+cat, and cat+cat pairs
- Logs all ~30 interaction features
- **Test:** `test_get_pair_meta_features()`

### 14. Baseline Importances (Section 14)
- Tests `get_baseline_importances()` on all datasets
- Logs full importance series and top 5 features
- **Test:** `test_get_baseline_importances()`

### 15. Should-Test Filters (Section 15)
- Tests `should_test_numerical()` and `should_test_categorical()` filters
- **Test:** `test_should_test_filters()`

### 16. Build Feature Vector (Section 16)
- Tests `build_feature_vector()` for different methods
- Logs feature vector shapes and column names
- **Test:** `test_build_feature_vector()`

### 17. Target Encoding Helper (Section 17)
- Tests `ensure_numeric_target()` with string and numeric targets
- **Test:** `test_ensure_numeric_target()`

### 18. Feature Name Sanitization (Section 18)
- Tests `sanitize_feature_names()` with special characters
- **Test:** `test_sanitize_feature_names()`

### 19. Suggestion Generation (Section 19)
- Tests `generate_suggestions()` on all datasets
- Includes progress callback logging
- Logs total suggestions and first 10 with all fields
- Logs skipped info, advisories, and dataset meta
- **Test:** `test_generate_suggestions()`

### 20. Suggestion Deduplication (Section 20)
- Tests `deduplicate_suggestions()`
- Logs before/after counts and duplicate removal info
- **Test:** `test_deduplicate_suggestions()`

### 21. Fit and Apply Suggestions (Section 21)
- Tests `fit_and_apply_suggestions()` on 2+ datasets
- Logs enhanced X shapes, new columns, fitted_params structure
- **Test:** `test_fit_and_apply_suggestions()`

### 22. Data Preparation (Section 22)
- Tests `prepare_data_for_model()`
- Logs shapes and column encoder structures
- **Test:** `test_prepare_data_for_model()`

### 23. Model Training (Section 23)
- Tests `train_lgbm_model()` on all datasets
- Trains baseline models with and without imbalance handling
- Logs model type, train columns, col encoders
- **Test:** `test_train_lgbm_model()`

### 24. Model Evaluation (Section 24)
- Tests `evaluate_on_set()` on all datasets
- Logs ALL metrics: accuracy, roc_auc, f1, precision, recall, log_loss
- Logs confusion_matrix shape, roc_data and pr_data keys
- **Test:** `test_evaluate_on_set()`

### 25. Predict-Only Mode (Section 25)
- Tests `predict_on_set()` with student_dropout_test_no_Labels.csv
- Trains model on train set, predicts on test set without labels
- Logs predictions and prediction probabilities shapes
- **Test:** `test_predict_on_set()`

### 26. Apply Fitted to Test (Section 26)
- Tests `apply_fitted_to_test()` on 2+ datasets
- Logs resulting shapes and column alignment
- **Test:** `test_apply_fitted_to_test()`

### 27. Suggestion Verdicts (Section 27)
- Tests `_compute_suggestion_verdicts()`
- Logs verdict type, reason, new_cols for each suggestion
- Logs low_imp_orig results
- **Test:** `test_compute_suggestion_verdicts()`

### 28. Report Generation (Section 28)
- Tests `build_report_data()`, `generate_html_report()`, `generate_markdown_report()`
- Logs report byte sizes
- **Test:** `test_report_generation()`

### 29. Override Options (Section 29)
- Tests `_override_options_for()` with different column type infos
- **Test:** `test_override_options()`

### 30. Validate Column Override (Section 30)
- Tests `_validate_col_override()` with valid and invalid overrides
- **Test:** `test_validate_col_override()`

### 31. Constants Validation (Section 31)
- Imports and validates all constant lists (DATASET_FEATURES, NUMERICAL_METHODS, etc.)
- Verifies all constants are non-empty
- **Test:** `test_constants_validation()`

### 32-35. End-to-End Pipelines (Sections 32-35)
Complete end-to-end pipelines testing the entire workflow:

1. **Student Dropout Dataset** - Complete pipeline with all steps logged
   - Data loading → Profiling → Baseline training → Suggestion generation → Deduplication → Fit/Apply → Enhanced training → Verdicts → Test application → Evaluation → Report generation
   - **Test:** `test_end_to_end_student_dropout()`

2. **Booking Dataset** - With date column detection
   - **Test:** `test_end_to_end_booking()`

3. **Customer Conversion Dataset** - With text, dates, and missing values
   - **Test:** `test_end_to_end_customer_conversion()`

4. **Recipe Reviews Dataset** - With text and multiclass classification
   - **Test:** `test_end_to_end_recipe_reviews()`

## How to Run

### Run All Tests
```bash
cd "Library Test"
python3 test_detailed.py
```

### View Logs
Logs are printed to stdout with timestamps and log levels (INFO, ERROR, etc.). Each test section is clearly marked with:
```
================================================================================
TEST: Test Name
================================================================================
```

And results with:
```
PASSED: Description
```
or
```
FAILED: Error description
```

### Test Summary
At the end of execution, you'll see a summary:
```
================================================================================
TEST SUMMARY
================================================================================
✓ Data Loading: PASSED
✓ Load Meta-Models: PASSED
...
================================================================================
TOTAL: 40/40 tests passed (100%)
================================================================================
```

## Dataset Information

### Datasets Used
- **Booking_Cancelation**: Binary classification (hotel booking cancellation)
  - Target: `booking status` | ID: `Booking_ID`
  - Special: Date column `date of reservation`

- **Customer_Conversion**: Binary classification (customer conversion)
  - Target: `converted` | ID: `user_id`
  - Special: Text columns (`product_desc`, `review_text`), Date columns (`last_active`, `signup_date`), Missing values

- **Recipe_Reviews**: Multiclass classification (star ratings 1-5)
  - Target: `stars` | IDs: `Column0`, `comment_id`, `user_id`
  - Special: Text column `text`, Timestamp column `created_at`

- **Student_Dropout**: Binary classification (student dropout)
  - Target: `Dropout` | ID: `Student_ID`
  - Special: Missing values

## Key Features of the Test Suite

1. **Extensive Logging**: Every function return value is logged with detailed information
2. **Structured Output**: Clear section headers and organized results
3. **Error Handling**: All tests wrapped in try/except with detailed error logging
4. **Data Shape Logging**: For large objects, shapes and first-few items are logged instead of full contents
5. **Progress Callbacks**: Suggestion generation includes progress reporting
6. **Multi-Dataset Testing**: Tests run on multiple datasets to ensure robustness
7. **Manual Execution**: Not auto-executed; user runs manually to see all output
8. **End-to-End Coverage**: Complete pipeline tests verify the full workflow


## Dependencies

The test suite requires:
- Python 3.9+
- pandas
- numpy
- scikit-learn
- lightgbm
- mlcompass (the library being tested)

All imports are at the top of the test file for easy verification.

## Notes

- Tests are designed to be informative rather than strict; many use reasonable defaults
- Some tests sample data to save time (especially for large datasets)
- All logging uses Python's built-in `logging` module configured at INFO level
- Test functions can be run individually by importing and calling them directly
- No external test framework (pytest, unittest) is required - pure Python execution

## Troubleshooting

If you encounter errors:

1. **Import errors**: Ensure mlcompass is installed in the parent directory
2. **File not found errors**: Verify datasets are in `Datasets/` directory (note the typo in original naming)
3. **Memory errors**: Comment out large dataset tests 
4. **Timeout errors**: Increase limits or reduce dataset sampling

See individual test function docstrings for specific test details.
