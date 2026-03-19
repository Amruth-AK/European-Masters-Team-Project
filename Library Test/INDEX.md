# ML Compass Test Suite - Complete Index

## Quick Navigation

### Getting Started
- **[TEST_GUIDE.md](TEST_GUIDE.md)** - Comprehensive documentation of all tests

### Main Test File
- **[test_detailed.py](test_detailed.py)** - 2,209 lines, 40 test functions

## File Locations

```
Library Test/
├── test_detailed.py              ← Main comprehensive test suite
├── test_showcase.py              ← End-to-end demo pipeline
├── TEST_GUIDE.md                  ← Detailed test documentation
├── INDEX.md                       ← This file
├── Datasets/                       ← All datasets (9 CSV files)
│   ├── Booking_Cancelation_train.csv
│   ├── Booking_Cancelation_test.csv
│   ├── customer_conversion_train.csv
│   ├── customer_conversion_test.csv
│   ├── Recipe_reviews_train.csv
│   ├── Recipe_reviews_test.csv
│   ├── student_dropout_train.csv
│   ├── student_dropout_test.csv
│   └── student_dropout_test_no_Labels.csv
└── logs/                          ← Test output logs
```

## Test Functions (40 total)

### Section 1: Data Loading
1. `test_data_loading()` - Load all 5 datasets and inspect structure

### Section 2: Meta-Models
2. `test_load_meta_models()` - Load and inspect meta-models

### Section 3-10: Analysis & Detection
3. `test_get_column_type_info()` - Column type detection on all datasets
4. `test_detect_problematic_columns()` - ID/constant/missing column detection
5. `test_detect_date_columns()` - Date column detection
6. `test_detect_dow_columns()` - Day-of-week column detection
7. `test_detect_text_columns()` - Text column detection
8. `test_generate_dataset_advisories()` - Dataset advisory generation
9. `test_get_dataset_meta()` - Dataset-level meta-features (14 features)
10. `test_get_row_dataset_meta()` - Row-level meta-features (13 features)

### Section 11-13: Column & Pair Analysis
11. `test_get_numeric_column_meta()` - Numeric column features (18 features)
12. `test_get_categorical_column_meta()` - Categorical column features (16 features)
13. `test_get_pair_meta_features()` - Pair interaction features (~30 features)

### Section 14-18: Meta-Features & Helpers
14. `test_get_baseline_importances()` - Baseline feature importances
15. `test_should_test_filters()` - Method filtering logic
16. `test_build_feature_vector()` - Feature vector construction
17. `test_ensure_numeric_target()` - Target encoding helper
18. `test_sanitize_feature_names()` - Feature name sanitization

### Section 19-27: Suggestions & Pipeline
19. `test_generate_suggestions()` - Generate transform suggestions
20. `test_deduplicate_suggestions()` - Remove duplicate suggestions
21. `test_fit_and_apply_suggestions()` - Fit and apply transforms
22. `test_prepare_data_for_model()` - Data preparation for training
23. `test_train_lgbm_model()` - LightGBM model training
24. `test_evaluate_on_set()` - Model evaluation and metrics
25. `test_predict_on_set()` - Prediction on unlabeled data
26. `test_apply_fitted_to_test()` - Apply transforms to test data
27. `test_compute_suggestion_verdicts()` - Suggestion impact analysis

### Section 28-31: Reporting & Configuration
28. `test_report_generation()` - HTML/Markdown report generation
29. `test_override_options()` - Column type override options
30. `test_validate_col_override()` - Override validation
31. `test_constants_validation()` - Validate constants definitions

### Section 32-35: End-to-End Pipelines
32. `test_end_to_end_student_dropout()` - Complete pipeline: Student Dropout
33. `test_end_to_end_booking()` - Complete pipeline: Booking (with dates)
34. `test_end_to_end_customer_conversion()` - Complete pipeline: Customer Conversion (text + dates + missing)
35. `test_end_to_end_recipe_reviews()` - Complete pipeline: Recipe Reviews (multiclass + text)

### Utility Functions (2)
- `main()` - Test runner with summary statistics
- Module-level: Logging configuration and imports

## Quick Command Reference

```bash
# Navigate to test directory
cd "Library Test"

# Run all tests
python3 test_detailed.py

# Save output to file
python3 test_detailed.py > test_results.log 2>&1

# View results as they happen
python3 test_detailed.py | grep "PASSED\|FAILED\|TOTAL"

# Check specific test section
grep "TEST: Generate Suggestions" test_results.log -A 10

# Count test results
grep "PASSED" test_results.log | wc -l
grep "FAILED" test_results.log | wc -l
```

## Dataset Reference

| Dataset | Type | Train Rows | Test Rows | Features | Target |
|---------|------|-----------|-----------|----------|--------|
| **Student Dropout** | Binary | 7,001 | 3,001 | ~34 | Dropout |
| **Booking Cancelation** | Binary | 25,400 | 10,887 | ~32 | booking status |
| **Customer Conversion** | Binary | 2,801 | 1,201 | ~20+ | converted |
| **Recipe Reviews** | Multiclass (5) | 12,817 | 5,488 | ~15 | stars |

Special Notes:
- All train/test datasets included
- Student Dropout has test_no_Labels.csv variant for predict-only testing
- Booking has date columns (detected automatically)
- Customer Conversion has text columns and missing values
- Recipe Reviews is multiclass with text column

## Import Structure

```python
# Data loading & setup
from test_detailed import test_data_loading

# Analysis functions
from test_detailed import (
    test_get_column_type_info,
    test_detect_problematic_columns,
    test_detect_date_columns,
    test_get_dataset_meta,
    # ... etc
)

# Example: Run a single test
loaded_data = test_data_loading()
test_get_column_type_info(loaded_data)
```

## Logging Output Format

Each test produces logs in this format:
```
[TIMESTAMP] [LEVEL] [MESSAGE]
```

Example:
```
2024-03-14 10:15:32,123 [INFO] ================================================================================
2024-03-14 10:15:32,124 [INFO] TEST: Data Loading & Setup
2024-03-14 10:15:32,125 [INFO] ================================================================================
2024-03-14 10:15:32,500 [INFO] Train shape: (7001, 34)
2024-03-14 10:15:32,750 [INFO] PASSED: All datasets loaded successfully
```

## Expected Test Results

When all tests pass:
```
================================================================================
TEST SUMMARY
================================================================================
✓ Data Loading: PASSED
✓ Load Meta-Models: PASSED
✓ Column Type Detection: PASSED
...
✓ E2E: Recipe Reviews: PASSED
================================================================================
TOTAL: 40/40 tests passed (100%)
================================================================================
```

## Features Tested per Category

### Data Analysis (12 tests)
- ✓ Column type detection (numeric, categorical, date, text)
- ✓ Problem column detection (IDs, constants, high-missing)
- ✓ Dataset meta-features (14 core features)
- ✓ Row-level features (13 features)
- ✓ Numeric column features (18 features)
- ✓ Categorical column features (16 features)
- ✓ Pair interaction features (~30 features)

### Feature Engineering (8 tests)
- ✓ Baseline feature importances
- ✓ Method selection filters
- ✓ Suggestion generation
- ✓ Suggestion deduplication
- ✓ Transform fitting and application
- ✓ Feature vector building
- ✓ Feature name sanitization
- ✓ Target encoding

### Model Training & Evaluation (5 tests)
- ✓ Data preparation for models
- ✓ LightGBM model training (with imbalance handling)
- ✓ Model evaluation (7 metrics: accuracy, ROC-AUC, F1, precision, recall, log_loss, confusion_matrix)
- ✓ Prediction on unlabeled data
- ✓ Transform application to test sets

### Reporting & Configuration (3 tests)
- ✓ Report data building
- ✓ HTML report generation
- ✓ Markdown report generation
- ✓ Column type overrides
- ✓ Constants validation

### End-to-End (4 tests)
- ✓ Complete pipeline with Student Dropout dataset
- ✓ Complete pipeline with Booking dataset (date handling)
- ✓ Complete pipeline with Customer Conversion (text + dates + missing)
- ✓ Complete pipeline with Recipe Reviews (multiclass + text)

## Performance Metrics Logged

- Accuracy
- ROC-AUC
- F1 Score
- Precision
- Recall
- Log Loss
- Confusion Matrix
- ROC Data
- PR Data

## Estimated Execution Times

- **Data Loading Only**: 1 minute
- **Basic Tests (1-20)**: 5 minutes
- **Full Test Suite**: 10-15 minutes
  - Depends on hardware (CPU cores, RAM)
  - First run slower due to compilation
  - Subsequent runs faster due to caching

## System Requirements

- Python 3.9+
- 4GB RAM (2GB minimum with sampling)
- 2GB disk space (for datasets + cache)
- Multi-core CPU recommended

## Dependencies

All dependencies are automatically managed by mlcompass:
- pandas
- numpy
- scikit-learn
- lightgbm
- mlcompass (the library being tested)

## Documentation Files

| File | Size | Purpose |
|------|------|---------|
| test_detailed.py | 86 KB | Main test suite (2,209 lines) |
| test_showcase.py | — | End-to-end demo pipeline |
| TEST_GUIDE.md | 12 KB | Detailed test documentation |
| INDEX.md | This file | Complete file and function index |

## Troubleshooting Checklist

- [ ] Python 3.9+ installed? → `python3 --version`
- [ ] In correct directory? → `pwd` should show `Library Test`
- [ ] Datasets present? → `ls Datasets/` should show 9 files
- [ ] mlcompass available? → `ls ../mlcompass/` should show files
- [ ] Syntax valid? → `python3 -m py_compile test_detailed.py`
- [ ] RAM sufficient? → `free -h` should show >2GB available
- [ ] Dependencies installed? → `pip3 install pandas numpy scikit-learn lightgbm`

## Next Steps

1. **Read TEST_GUIDE.md** - Understand what's being tested and how to run the tests
2. **Run tests** - `python3 test_detailed.py`
3. **Review output** - Look for any FAILED tests

## Support

For help, consult:
1. **TEST_GUIDE.md** - For detailed test descriptions
2. **test_detailed.py** - For inline documentation
3. Comments in the test file are extensive and helpful

---

**Created**: March 14, 2024
**Test Suite Version**: 1.0
**ML Compass Version**: 0.1.0
**Status**: Ready for execution
