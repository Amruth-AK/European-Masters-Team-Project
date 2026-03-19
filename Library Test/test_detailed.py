"""
Comprehensive Detailed Test Suite for ML Compass Library
======================================================

This test file provides extensive coverage of the mlcompass Python library,
with correct API signatures, comprehensive logging to console AND file,
and full end-to-end pipeline testing.

Each test function:
  - Logs to both console and test_detailed.log file
  - Wraps operations in try/except with full tracebacks
  - Logs ALL return values extensively (types, shapes, keys, sample values)
  - Tests correct function signatures per actual API
  - Uses encoding='latin-1' for Recipe CSV files

Usage: python test_detailed.py
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup logging: both file and console
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_detailed.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add mlcompass to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlcompass
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
)
from mlcompass.transforms.detection import (
    detect_dow_columns,
    detect_text_columns,
)
from mlcompass.transforms.helpers import (
    ensure_numeric_target,
    sanitize_feature_names,
)
from mlcompass.analysis.advisories import generate_dataset_advisories
from mlcompass.recommendation.engine import (
    generate_suggestions,
    deduplicate_suggestions,
)
from mlcompass.recommendation.meta_models import load_meta_models, build_feature_vector
from mlcompass.recommendation.verdicts import _compute_suggestion_verdicts
from mlcompass.evaluation.training import (
    prepare_data_for_model,
    train_lgbm_model,
)
from mlcompass.evaluation.metrics import (
    evaluate_on_set,
    predict_on_set,
)
from mlcompass.transforms.applicator import (
    fit_and_apply_suggestions,
    apply_fitted_to_test,
)
from mlcompass.reporting.generator import (
    build_report_data,
    generate_html_report,
    generate_markdown_report,
)
from mlcompass import constants

# Dataset paths
DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'Datasets')


# ============================================================================
# TEST 1: DATA LOADING FROM ALL DATASETS
# ============================================================================

def test_load_all_datasets():
    """Load all 5 datasets (4 train/test pairs + no-labels)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Load All Datasets")
    try:
        datasets = {
            'booking': {
                'train': os.path.join(DATASETS_DIR, 'Booking_Cancelation_train.csv'),
                'test': os.path.join(DATASETS_DIR, 'Booking_Cancelation_test.csv'),
                'target': 'booking status',
                'id_col': 'Booking_ID',
                'date_col': 'date of reservation',
            },
            'customer_conversion': {
                'train': os.path.join(DATASETS_DIR, 'customer_conversion_train.csv'),
                'test': os.path.join(DATASETS_DIR, 'customer_conversion_test.csv'),
                'target': 'converted',
                'id_col': 'user_id',
                'text_cols': ['product_desc', 'review_text'],
                'date_cols': ['last_active', 'signup_date'],
            },
            'recipe_reviews': {
                'train': os.path.join(DATASETS_DIR, 'Recipe_reviews_train.csv'),
                'test': os.path.join(DATASETS_DIR, 'Recipe_reviews_test.csv'),
                'target': 'stars',
                'id_cols': ['Column0', 'comment_id', 'user_id'],
                'text_col': 'text',
                'encoding': 'latin-1',
            },
            'student_dropout': {
                'train': os.path.join(DATASETS_DIR, 'student_dropout_train.csv'),
                'test': os.path.join(DATASETS_DIR, 'student_dropout_test.csv'),
                'test_no_labels': os.path.join(DATASETS_DIR, 'student_dropout_test_no_Labels.csv'),
                'target': 'Dropout',
                'id_col': 'Student_ID',
            },
        }

        loaded_data = {}
        for dataset_name, paths in datasets.items():
            logger.info(f"\n--- Loading {dataset_name} ---")

            encoding = paths.get('encoding', 'utf-8')
            train_df = pd.read_csv(paths['train'], encoding=encoding)
            test_df = pd.read_csv(paths['test'], encoding=encoding)

            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Train dtypes:\n{train_df.dtypes}")
            logger.info(f"Test shape: {test_df.shape}")
            missing = train_df.isnull().sum()
            if missing.sum() > 0:
                logger.info(f"Train missing:\n{missing[missing > 0]}")

            # Separate X and y
            target = paths.get('target')
            if target and target in train_df.columns:
                X_train = train_df.drop(columns=[target])
                y_train = train_df[target]
                logger.info(f"Target: {target}, y_train shape: {y_train.shape}")
                logger.info(f"  Unique values: {y_train.nunique()}, Value counts:\n{y_train.value_counts()}")
            else:
                X_train = train_df
                y_train = None

            if 'test_no_labels' in paths:
                test_no_labels_df = pd.read_csv(paths['test_no_labels'], encoding=encoding)
                logger.info(f"Test (no labels) shape: {test_no_labels_df.shape}")
                # Also split test into X_test/y_test if target exists
                if target and target in test_df.columns:
                    X_test = test_df.drop(columns=[target])
                    y_test = test_df[target]
                else:
                    X_test = test_df
                    y_test = None
                loaded_data[dataset_name] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'test': test_df,
                    'test_no_labels': test_no_labels_df,
                }
            else:
                if target and target in test_df.columns:
                    X_test = test_df.drop(columns=[target])
                    y_test = test_df[target]
                else:
                    X_test = test_df
                    y_test = None
                loaded_data[dataset_name] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                }

            logger.info(f"✓ {dataset_name} loaded")

        logger.info("\nPASSED: All datasets loaded")
        return loaded_data

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 2: LOAD META-MODELS
# ============================================================================

def test_load_meta_models():
    """Load meta-models, inspect structure."""
    logger.info("=" * 80)
    logger.info("TEST 2: Load Meta-Models")
    try:
        meta_models = load_meta_models()
        logger.info(f"Meta-models loaded, keys: {list(meta_models.keys())}")

        for model_type, model_data in meta_models.items():
            logger.info(f"\nModel type: {model_type}")
            logger.info(f"  Data keys: {list(model_data.keys())}")
            if 'booster' in model_data:
                logger.info(f"  Booster type: {type(model_data['booster'])}")
            if 'config' in model_data:
                logger.info(f"  Config keys: {list(model_data['config'].keys())}")
            if 'feature_names' in model_data:
                logger.info(f"  Feature names count: {len(model_data['feature_names'])}")
            if 'method_vocab' in model_data:
                logger.info(f"  Method vocab count: {len(model_data['method_vocab'])}")

        assert len(meta_models) > 0, "Meta-models should not be empty"
        logger.info("\nPASSED: Meta-models loaded")
        return meta_models

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 3: GET COLUMN TYPE INFO
# ============================================================================

def test_get_column_type_info(loaded_data):
    """Test column type detection on all datasets."""
    logger.info("=" * 80)
    logger.info("TEST 3: Get Column Type Info")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']

            col_info = get_column_type_info(X_train)
            logger.info(f"Columns analyzed: {len(col_info)}")

            for col_name, col_data in list(col_info.items())[:3]:
                logger.info(f"\n  Column: {col_name}")
                logger.info(f"    Detected type: {col_data.get('detected')}")
                logger.info(f"    N unique: {col_data.get('n_unique')}")
                logger.info(f"    Missing %: {col_data.get('missing_pct', 0):.2f}")
                logger.info(f"    Is numeric: {col_data.get('is_numeric')}")

        logger.info("\nPASSED: Column type detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 4: DETECT PROBLEMATIC COLUMNS
# ============================================================================

def test_detect_problematic_columns(loaded_data):
    """Test detection of ID columns, constants, etc."""
    logger.info("=" * 80)
    logger.info("TEST 4: Detect Problematic Columns")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']

            problematic = detect_problematic_columns(X_train)
            logger.info(f"Problematic keys: {list(problematic.keys())}")
            for key, cols in problematic.items():
                logger.info(f"  {key}: {cols}")

        logger.info("\nPASSED: Problematic column detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 5: DETECT DATE COLUMNS
# ============================================================================

def test_detect_date_columns(loaded_data):
    """Test date column detection."""
    logger.info("=" * 80)
    logger.info("TEST 5: Detect Date Columns")
    try:
        for dataset_name in ['booking', 'customer_conversion']:
            if dataset_name not in loaded_data:
                continue

            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = loaded_data[dataset_name]['X_train']

            date_cols = detect_date_columns(X_train)
            logger.info(f"Detected date columns: {list(date_cols.keys())}")

            for col_name, col_data in date_cols.items():
                logger.info(f"\n  Column: {col_name}")
                logger.info(f"    Parse rate: {col_data.get('parse_rate')}")
                logger.info(f"    Col type: {col_data.get('col_type')}")

        logger.info("\nPASSED: Date detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 6: DETECT DATE HAS HOUR
# ============================================================================

def test_detect_date_has_hour(loaded_data):
    """Test detect_date_has_hour with correct signature: (X, col)."""
    logger.info("=" * 80)
    logger.info("TEST 6: Detect Date Has Hour")
    try:
        for dataset_name in ['booking', 'customer_conversion']:
            if dataset_name not in loaded_data:
                continue

            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = loaded_data[dataset_name]['X_train']

            date_cols = detect_date_columns(X_train)
            if not date_cols:
                logger.info("No date columns detected, skipping")
                continue

            for col_name in list(date_cols.keys())[:2]:
                has_hour = detect_date_has_hour(X_train, col_name)
                logger.info(f"  {col_name} has hour: {has_hour}, type: {type(has_hour)}")

        logger.info("\nPASSED: Date has hour detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 7: DETECT DOW COLUMNS
# ============================================================================

def test_detect_dow_columns(loaded_data):
    """Test day-of-week detection."""
    logger.info("=" * 80)
    logger.info("TEST 7: Detect DOW Columns")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']

            dow_cols = detect_dow_columns(X_train)
            logger.info(f"DOW columns: {dow_cols}, type: {type(dow_cols)}")

        logger.info("\nPASSED: DOW detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 8: DETECT TEXT COLUMNS
# ============================================================================

def test_detect_text_columns(loaded_data):
    """Test text column detection."""
    logger.info("=" * 80)
    logger.info("TEST 8: Detect Text Columns")
    try:
        for dataset_name in ['customer_conversion', 'recipe_reviews']:
            if dataset_name not in loaded_data:
                continue

            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = loaded_data[dataset_name]['X_train']

            text_cols = detect_text_columns(X_train)
            logger.info(f"Text columns: {text_cols}, type: {type(text_cols)}")

        logger.info("\nPASSED: Text detection works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 9: GENERATE DATASET ADVISORIES
# ============================================================================

def test_generate_dataset_advisories(loaded_data):
    """Test advisory generation."""
    logger.info("=" * 80)
    logger.info("TEST 9: Generate Dataset Advisories")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            advisories = generate_dataset_advisories(X_train, y_train)
            logger.info(f"Advisories: {len(advisories)} total")

            for i, adv in enumerate(advisories[:2]):
                logger.info(f"\n  Advisory {i+1}: {adv.get('category')} / {adv.get('severity')}")
                logger.info(f"    Title: {adv.get('title')}")

        logger.info("\nPASSED: Advisory generation works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 10: GET DATASET META
# ============================================================================

def test_get_dataset_meta(loaded_data):
    """Test dataset-level meta-features."""
    logger.info("=" * 80)
    logger.info("TEST 10: Get Dataset Meta")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            ds_meta = get_dataset_meta(X_train, y_train)
            logger.info(f"Keys ({len(ds_meta)}): {list(ds_meta.keys())}")
            logger.info(f"Sample values: n_rows={ds_meta.get('n_rows')}, n_cols={ds_meta.get('n_cols')}")

        logger.info("\nPASSED: Dataset meta works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 11: GET ROW DATASET META
# ============================================================================

def test_get_row_dataset_meta(loaded_data):
    """Test row-level meta-features: only takes X, returns ~13 keys."""
    logger.info("=" * 80)
    logger.info("TEST 11: Get Row Dataset Meta")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']

            row_meta = get_row_dataset_meta(X_train)
            logger.info(f"Keys ({len(row_meta)}): {list(row_meta.keys())}")
            logger.info(f"Return type: {type(row_meta)}")

            for key, value in list(row_meta.items())[:3]:
                logger.info(f"  {key}: {value} ({type(value).__name__})")

        logger.info("\nPASSED: Row meta works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 12: GET BASELINE IMPORTANCES
# ============================================================================

def test_get_baseline_importances(loaded_data):
    """Test baseline importance extraction."""
    logger.info("=" * 80)
    logger.info("TEST 12: Get Baseline Importances")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            baseline_imp = get_baseline_importances(X_train, y_train)
            logger.info(f"Return type: {type(baseline_imp)}")
            logger.info(f"Shape: {baseline_imp.shape}")
            logger.info(f"Sample (first 3): {baseline_imp.head(3).to_dict()}")

        logger.info("\nPASSED: Baseline importances work")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 13: GET NUMERIC COLUMN META
# ============================================================================

def test_get_numeric_column_meta(loaded_data):
    """Test numeric column meta-features: needs series, y, importance, rank_pct."""
    logger.info("=" * 80)
    logger.info("TEST 13: Get Numeric Column Meta")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            baseline_imp = get_baseline_importances(X_train, y_train)
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if not numeric_cols:
                logger.info("No numeric columns")
                continue

            col_name = numeric_cols[0]
            logger.info(f"Column: {col_name}")

            col_meta = get_numeric_column_meta(
                X_train[col_name],
                y_train,
                importance=baseline_imp.get(col_name, 0.0),
                importance_rank_pct=0.5
            )

            logger.info(f"Return type: {type(col_meta)}")
            logger.info(f"Keys ({len(col_meta)}): {list(col_meta.keys())}")
            logger.info(f"Sample: {dict(list(col_meta.items())[:3])}")

        logger.info("\nPASSED: Numeric column meta works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 14: GET CATEGORICAL COLUMN META
# ============================================================================

def test_get_categorical_column_meta(loaded_data):
    """Test categorical column meta-features."""
    logger.info("=" * 80)
    logger.info("TEST 14: Get Categorical Column Meta")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            baseline_imp = get_baseline_importances(X_train, y_train)
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            if not cat_cols:
                logger.info("No categorical columns")
                continue

            col_name = cat_cols[0]
            logger.info(f"Column: {col_name}")

            col_meta = get_categorical_column_meta(
                X_train[col_name],
                y_train,
                importance=baseline_imp.get(col_name, 0.0),
                importance_rank_pct=0.5
            )

            logger.info(f"Return type: {type(col_meta)}")
            logger.info(f"Keys ({len(col_meta)}): {list(col_meta.keys())}")

        logger.info("\nPASSED: Categorical column meta works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 15: GET PAIR META-FEATURES
# ============================================================================

def test_get_pair_meta_features(loaded_data):
    """Test pair meta-features: (col_a, col_b, y, imp_a, imp_b)."""
    logger.info("=" * 80)
    logger.info("TEST 15: Get Pair Meta-Features")
    try:
        for dataset_name, data in loaded_data.items():
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            X_train = data['X_train']
            y_train = data['y_train']

            if X_train.shape[1] < 2:
                logger.info("Not enough columns")
                continue

            baseline_imp = get_baseline_importances(X_train, y_train)

            col_a = X_train.iloc[:, 0]
            col_b = X_train.iloc[:, 1]
            imp_a = baseline_imp.get(col_a.name, 0.0)
            imp_b = baseline_imp.get(col_b.name, 0.0)

            logger.info(f"Pair: {col_a.name} + {col_b.name}")

            pair_meta = get_pair_meta_features(col_a, col_b, y_train, imp_a, imp_b)

            logger.info(f"Return type: {type(pair_meta)}")
            logger.info(f"Keys ({len(pair_meta)}): {list(pair_meta.keys())}")

        logger.info("\nPASSED: Pair meta-features work")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 16: SHOULD TEST NUMERICAL
# ============================================================================

def test_should_test_numerical(loaded_data):
    """Test should_test_numerical."""
    logger.info("=" * 80)
    logger.info("TEST 16: Should Test Numerical")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if not numeric_cols:
            logger.info("No numeric columns")
            return

        col_name = numeric_cols[0]
        col_meta = get_numeric_column_meta(X_train[col_name], y_train, importance=0.5, importance_rank_pct=0.5)

        result = should_test_numerical('dummy_method', col_meta, X_train[col_name])
        logger.info(f"should_test_numerical result: {result}, type: {type(result)}")

        logger.info("\nPASSED: Should test numerical works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 17: SHOULD TEST CATEGORICAL
# ============================================================================

def test_should_test_categorical(loaded_data):
    """Test should_test_categorical."""
    logger.info("=" * 80)
    logger.info("TEST 17: Should Test Categorical")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        if not cat_cols:
            logger.info("No categorical columns")
            return

        col_name = cat_cols[0]
        col_meta = get_categorical_column_meta(X_train[col_name], y_train, importance=0.5, importance_rank_pct=0.5)

        result = should_test_categorical('dummy_method', col_meta)
        logger.info(f"should_test_categorical result: {result}, type: {type(result)}")

        logger.info("\nPASSED: Should test categorical works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 18: BUILD FEATURE VECTOR
# ============================================================================

def test_build_feature_vector(loaded_data, meta_models):
    """Test build_feature_vector: (meta_dict, method, config)."""
    logger.info("=" * 80)
    logger.info("TEST 18: Build Feature Vector")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        # Get a sample meta dict
        col_info = get_column_type_info(X_train)
        col_name = X_train.columns[0]
        col_meta = col_info[col_name]

        # Get a meta-model to extract config
        model_type = list(meta_models.keys())[0]
        config = meta_models[model_type].get('config', {})
        method = model_type

        # Create a minimal meta dict
        meta_dict = col_meta.copy()

        fv = build_feature_vector(meta_dict, method, config)
        logger.info(f"Return type: {type(fv)}")
        logger.info(f"Shape: {fv.shape}")
        logger.info(f"Columns (first 5): {fv.columns[:5].tolist()}")

        logger.info("\nPASSED: Build feature vector works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 19: ENSURE NUMERIC TARGET
# ============================================================================

def test_ensure_numeric_target(loaded_data):
    """Test ensure_numeric_target: y -> pd.Series."""
    logger.info("=" * 80)
    logger.info("TEST 19: Ensure Numeric Target")
    try:
        data = loaded_data['student_dropout']
        y_train = data['y_train']

        y_numeric = ensure_numeric_target(y_train)
        logger.info(f"Input type: {type(y_train)}, Output type: {type(y_numeric)}")
        logger.info(f"Input dtype: {y_train.dtype}, Output dtype: {y_numeric.dtype}")
        logger.info(f"Sample output: {y_numeric.head(3).tolist()}")

        assert isinstance(y_numeric, pd.Series), "Should return pd.Series"
        logger.info("\nPASSED: Ensure numeric target works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 20: SANITIZE FEATURE NAMES
# ============================================================================

def test_sanitize_feature_names(loaded_data):
    """Test sanitize_feature_names."""
    logger.info("=" * 80)
    logger.info("TEST 20: Sanitize Feature Names")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']

        X_sanitized = sanitize_feature_names(X_train)
        logger.info(f"Input columns: {X_train.columns.tolist()[:3]}")
        logger.info(f"Output columns: {X_sanitized.columns.tolist()[:3]}")
        logger.info(f"Shape preserved: {X_train.shape == X_sanitized.shape}")

        logger.info("\nPASSED: Sanitize feature names works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 21: OVERRIDE OPTIONS FOR
# ============================================================================

def test_override_options_for():
    """Test _override_options_for: info -> list[str]."""
    logger.info("=" * 80)
    logger.info("TEST 21: Override Options For")
    try:
        # Create sample info dict
        info = {
            'detected': 'numeric',
            'n_unique': 50,
            'missing_pct': 5.0,
            'is_numeric': True,
        }

        options = _override_options_for(info)
        logger.info(f"Return type: {type(options)}")
        logger.info(f"Result: {options}")

        assert isinstance(options, list), "Should return list"
        logger.info("\nPASSED: Override options for works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 22: VALIDATE COL OVERRIDE
# ============================================================================

def test_validate_col_override(loaded_data):
    """Test _validate_col_override: (col, type, info, X)."""
    logger.info("=" * 80)
    logger.info("TEST 22: Validate Col Override")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']

        col_info = get_column_type_info(X_train)
        col_name = X_train.columns[0]
        info = col_info[col_name]

        result = _validate_col_override(col_name, 'numeric', info, X_train)
        logger.info(f"Return value: {result}, type: {type(result)}")

        logger.info("\nPASSED: Validate col override works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 23: VERIFY CONSTANTS
# ============================================================================

def test_constants():
    """Test that all constants are present and non-empty."""
    logger.info("=" * 80)
    logger.info("TEST 23: Verify Constants")
    try:
        constant_names = [
            'DATASET_FEATURES',
            'NUMERICAL_COLUMN_FEATURES',
            'CATEGORICAL_COLUMN_FEATURES',
            'INTERACTION_PAIR_FEATURES',
            'ROW_DATASET_FEATURES',
            'NUMERICAL_METHODS',
            'CATEGORICAL_METHODS',
            'INTERACTION_METHODS_NUM_NUM',
            'INTERACTION_METHODS_CAT_NUM',
            'INTERACTION_METHODS_CAT_CAT',
            'ROW_FAMILIES',
            'BASE_PARAMS',
            'METHOD_DESCRIPTIONS',
        ]

        for const_name in constant_names:
            const_val = getattr(constants, const_name, None)
            assert const_val is not None, f"{const_name} should exist"
            if isinstance(const_val, (list, dict)):
                assert len(const_val) > 0, f"{const_name} should not be empty"
            logger.info(f"  {const_name}: OK ({type(const_val).__name__}, size={len(const_val) if hasattr(const_val, '__len__') else 'N/A'})")

        logger.info("\nPASSED: All constants verified")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 24: GENERATE SUGGESTIONS
# ============================================================================

def test_generate_suggestions(loaded_data, meta_models):
    """Test generate_suggestions: complex signature."""
    logger.info("=" * 80)
    logger.info("TEST 24: Generate Suggestions")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        # Baseline score and std
        baseline_score = 0.75
        baseline_std = 0.05

        suggestions, skipped_info, advisories, ds_meta = generate_suggestions(
            X_train,
            y_train,
            meta_models,
            baseline_score=baseline_score,
            baseline_std=baseline_std,
            progress_cb=None,
            type_reassignments=None,
            real_n_rows=None
        )

        logger.info(f"Suggestions: {len(suggestions)}")
        logger.info(f"Skipped info type: {type(skipped_info)}")
        logger.info(f"Advisories: {len(advisories)}")
        logger.info(f"DS meta keys: {list(ds_meta.keys())[:5]}")

        logger.info(f"Return types: tuple of (list, dict, list, dict)")
        logger.info("\nPASSED: Generate suggestions works")

        return suggestions, advisories

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 25: DEDUPLICATE SUGGESTIONS
# ============================================================================

def test_deduplicate_suggestions(suggestions):
    """Test deduplicate_suggestions."""
    logger.info("=" * 80)
    logger.info("TEST 25: Deduplicate Suggestions")
    try:
        if not suggestions:
            logger.info("No suggestions to dedup")
            return []

        dedup_suggestions = deduplicate_suggestions(suggestions)
        logger.info(f"Before: {len(suggestions)}, After: {len(dedup_suggestions)}")
        logger.info(f"Return type: {type(dedup_suggestions)}")

        logger.info("\nPASSED: Deduplicate suggestions works")
        return dedup_suggestions

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 26: PREPARE DATA FOR MODEL
# ============================================================================

def test_prepare_data_for_model(loaded_data):
    """Test prepare_data_for_model: (X_train, X_val) -> (X_tr, X_vl, col_encoders)."""
    logger.info("=" * 80)
    logger.info("TEST 26: Prepare Data For Model")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']

        # Split into train/val
        X_tr, X_vl = train_test_split(X_train, test_size=0.2, random_state=42)
        logger.info(f"Train split: {X_tr.shape}, Val split: {X_vl.shape}")

        X_tr_prep, X_vl_prep, col_encoders = prepare_data_for_model(X_tr, X_vl)

        logger.info(f"Return type: tuple of 3 items")
        logger.info(f"X_tr_prep shape: {X_tr_prep.shape}")
        logger.info(f"X_vl_prep shape: {X_vl_prep.shape}")
        logger.info(f"col_encoders type: {type(col_encoders)}")

        logger.info("\nPASSED: Prepare data works")
        return X_tr_prep, X_vl_prep, col_encoders

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 27: TRAIN LGBM MODEL
# ============================================================================

def test_train_lgbm_model(loaded_data):
    """Test train_lgbm_model: signature with apply_imbalance, imbalance_strategy."""
    logger.info("=" * 80)
    logger.info("TEST 27: Train LGBM Model")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        X_tr_prep, X_vl_prep, col_encoders = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()
        logger.info(f"n_classes: {n_classes}")

        model, train_columns, col_encs = train_lgbm_model(
            X_tr_prep, y_tr, X_vl_prep, y_vl,
            n_classes=n_classes,
            apply_imbalance=False,
            imbalance_strategy='none'
        )

        logger.info(f"Model type: {type(model)}")
        logger.info(f"Train columns: {len(train_columns)}")
        logger.info(f"Col encoders type: {type(col_encs)}")

        logger.info("\nPASSED: Train LGBM model works")
        return model, train_columns, col_encs

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 28: TRAIN LGBM WITH IMBALANCE
# ============================================================================

def test_train_lgbm_with_imbalance(loaded_data):
    """Test train_lgbm_model with apply_imbalance=True."""
    logger.info("=" * 80)
    logger.info("TEST 28: Train LGBM With Imbalance")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        X_tr_prep, X_vl_prep, col_encoders = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()

        model, train_columns, col_encs = train_lgbm_model(
            X_tr_prep, y_tr, X_vl_prep, y_vl,
            n_classes=n_classes,
            apply_imbalance=True,
            imbalance_strategy='multiclass_moderate'
        )

        logger.info(f"Model type: {type(model)}")
        logger.info(f"Train columns: {len(train_columns)}")

        logger.info("\nPASSED: Train LGBM with imbalance works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 29: EVALUATE ON SET
# ============================================================================

def test_evaluate_on_set(loaded_data, model, train_columns, col_encoders):
    """Test evaluate_on_set: complex metrics dict return."""
    logger.info("=" * 80)
    logger.info("TEST 29: Evaluate On Set")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        X_tr_prep, X_vl_prep, _ = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()

        metrics = evaluate_on_set(
            model, X_vl_prep, y_vl,
            train_columns=train_columns,
            n_classes=n_classes,
            col_encoders=col_encoders
        )

        logger.info(f"Return type: {type(metrics)}")
        logger.info(f"Keys: {list(metrics.keys())}")
        logger.info(f"Sample metrics: accuracy={metrics.get('accuracy')}, roc_auc={metrics.get('roc_auc')}")

        logger.info("\nPASSED: Evaluate on set works")
        return metrics

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 30: PREDICT ON SET
# ============================================================================

def test_predict_on_set(loaded_data, model, train_columns):
    """Test predict_on_set: returns dict with _y_pred, _y_pred_proba."""
    logger.info("=" * 80)
    logger.info("TEST 30: Predict On Set")
    try:
        data = loaded_data['student_dropout']
        X_test = data['X_test']

        X_test_prep = pd.DataFrame(X_test)
        # Align columns with training
        for col in train_columns:
            if col not in X_test_prep.columns:
                X_test_prep[col] = 0

        predictions = predict_on_set(
            model, X_test_prep,
            train_columns=train_columns,
            n_classes=2,
            col_encoders=None
        )

        logger.info(f"Return type: {type(predictions)}")
        logger.info(f"Keys: {list(predictions.keys())}")
        logger.info(f"_y_pred shape: {predictions.get('_y_pred').shape if isinstance(predictions.get('_y_pred'), np.ndarray) else 'N/A'}")
        logger.info(f"_y_pred_proba shape: {predictions.get('_y_pred_proba').shape if isinstance(predictions.get('_y_pred_proba'), np.ndarray) else 'N/A'}")

        logger.info("\nPASSED: Predict on set works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 31: FIT AND APPLY SUGGESTIONS
# ============================================================================

def test_fit_and_apply_suggestions(loaded_data, suggestions):
    """Test fit_and_apply_suggestions: (X_train, y_train, suggestions) -> (X_enhanced, fitted_params)."""
    logger.info("=" * 80)
    logger.info("TEST 31: Fit And Apply Suggestions")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        if not suggestions:
            logger.info("No suggestions to fit")
            return None, None

        X_enhanced, fitted_params = fit_and_apply_suggestions(X_train, y_train, suggestions[:5])

        logger.info(f"X_enhanced type: {type(X_enhanced)}")
        logger.info(f"X_enhanced shape: {X_enhanced.shape}")
        logger.info(f"fitted_params type: {type(fitted_params)}")
        logger.info(f"fitted_params length: {len(fitted_params)}")

        logger.info("\nPASSED: Fit and apply suggestions works")
        return X_enhanced, fitted_params

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 32: APPLY FITTED TO TEST
# ============================================================================

def test_apply_fitted_to_test(loaded_data, fitted_params):
    """Test apply_fitted_to_test: X_test, fitted_params -> X_enhanced."""
    logger.info("=" * 80)
    logger.info("TEST 32: Apply Fitted To Test")
    try:
        if fitted_params is None:
            logger.info("No fitted params")
            return None

        data = loaded_data['student_dropout']
        X_test = data['X_test']

        X_test_enhanced = apply_fitted_to_test(X_test, fitted_params)

        logger.info(f"Return type: {type(X_test_enhanced)}")
        logger.info(f"Shape: {X_test_enhanced.shape}")
        logger.info(f"Columns added: {X_test_enhanced.shape[1] - X_test.shape[1]}")

        logger.info("\nPASSED: Apply fitted to test works")
        return X_test_enhanced

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 33: COMPUTE SUGGESTION VERDICTS
# ============================================================================

def test_compute_suggestion_verdicts(loaded_data, suggestions):
    """Test _compute_suggestion_verdicts."""
    logger.info("=" * 80)
    logger.info("TEST 33: Compute Suggestion Verdicts")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        if not suggestions or len(suggestions) == 0:
            logger.info("No suggestions")
            return

        # Need baseline and enhanced models
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_tr_prep, X_vl_prep, col_encs = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()

        # Baseline model
        baseline_model, baseline_train_cols, _ = train_lgbm_model(
            X_tr_prep, y_tr, X_vl_prep, y_vl, n_classes=n_classes
        )
        baseline_metrics = evaluate_on_set(baseline_model, X_vl_prep, y_vl, train_columns=baseline_train_cols, n_classes=n_classes)

        # Enhanced model (just use same for testing)
        enhanced_model = baseline_model
        enhanced_train_cols = baseline_train_cols
        enhanced_metrics = baseline_metrics

        verdicts, low_imp = _compute_suggestion_verdicts(
            fitted_params=[],
            suggestions=suggestions[:3],
            selected_indices=[0, 1, 2],
            enhanced_model=enhanced_model,
            enhanced_train_cols=enhanced_train_cols,
            baseline_model=baseline_model,
            baseline_train_cols=baseline_train_cols,
            baseline_val_metrics=baseline_metrics,
            enhanced_val_metrics=enhanced_metrics,
            apply_imbalance=False
        )

        logger.info(f"Verdicts type: {type(verdicts)}")
        logger.info(f"Verdicts length: {len(verdicts)}")
        logger.info(f"Low imp type: {type(low_imp)}")

        logger.info("\nPASSED: Compute verdicts works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 34: BUILD REPORT DATA
# ============================================================================

def test_build_report_data(loaded_data):
    """Test build_report_data: needs session_state dict."""
    logger.info("=" * 80)
    logger.info("TEST 34: Build Report Data")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']

        # Build mock session state
        session_state = {
            'X_train': X_train,
            'y_train': y_train,
            'target_col': 'Dropout',
            'n_classes': y_train.nunique(),
            '_col_type_info': get_column_type_info(X_train),
            'suggestions': [],
            'advisories': [],
            'skipped_info': {},
            'baseline_val_metrics': {'roc_auc': 0.75},
            'enhanced_val_metrics': {'roc_auc': 0.78},
            'baseline_model': None,
            'enhanced_model': None,
            'baseline_train_cols': [],
            'enhanced_train_cols': [],
            'fitted_params': [],
        }

        report_data = build_report_data(
            session_state,
            dataset_name='test_dataset',
            report_stage='validation'
        )

        logger.info(f"Return type: {type(report_data)}")
        logger.info(f"Keys: {list(report_data.keys())[:10]}")

        logger.info("\nPASSED: Build report data works")
        return report_data

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 35: GENERATE HTML REPORT
# ============================================================================

def test_generate_html_report(report_data):
    """Test generate_html_report: d -> bytes."""
    logger.info("=" * 80)
    logger.info("TEST 35: Generate HTML Report")
    try:
        if report_data is None:
            logger.info("No report data")
            return

        html_bytes = generate_html_report(report_data)

        logger.info(f"Return type: {type(html_bytes)}")
        logger.info(f"Size: {len(html_bytes)} bytes")
        logger.info(f"Is bytes: {isinstance(html_bytes, bytes)}")

        logger.info("\nPASSED: Generate HTML report works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# TEST 36: GENERATE MARKDOWN REPORT
# ============================================================================

def test_generate_markdown_report(report_data):
    """Test generate_markdown_report: d -> bytes."""
    logger.info("=" * 80)
    logger.info("TEST 36: Generate Markdown Report")
    try:
        if report_data is None:
            logger.info("No report data")
            return

        md_bytes = generate_markdown_report(report_data)

        logger.info(f"Return type: {type(md_bytes)}")
        logger.info(f"Size: {len(md_bytes)} bytes")
        logger.info(f"Is bytes: {isinstance(md_bytes, bytes)}")

        logger.info("\nPASSED: Generate markdown report works")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# E2E TEST 1: STUDENT DROPOUT (BINARY CLASSIFICATION)
# ============================================================================

def test_e2e_student_dropout(loaded_data, meta_models):
    """Complete end-to-end pipeline: student dropout."""
    logger.info("=" * 80)
    logger.info("E2E TEST 1: Student Dropout Pipeline")
    try:
        data = loaded_data['student_dropout']
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data.get('y_test')

        logger.info(f"Train: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Test: {X_test.shape}")

        # Split train/val
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        logger.info(f"After split: train {X_tr.shape}, val {X_vl.shape}")

        # Prepare
        X_tr_prep, X_vl_prep, col_encs = prepare_data_for_model(X_tr, X_vl)
        logger.info(f"After prepare: train {X_tr_prep.shape}, val {X_vl_prep.shape}")

        # Baseline model
        n_classes = y_train.nunique()
        baseline_model, baseline_cols, _ = train_lgbm_model(
            X_tr_prep, y_tr, X_vl_prep, y_vl, n_classes=n_classes
        )
        logger.info(f"Baseline model trained on {len(baseline_cols)} features")

        baseline_metrics = evaluate_on_set(baseline_model, X_vl_prep, y_vl, train_columns=baseline_cols, n_classes=n_classes)
        logger.info(f"Baseline val metrics: roc_auc={baseline_metrics.get('roc_auc'):.3f}")

        # Generate suggestions
        suggestions, skipped, advisories, ds_meta = generate_suggestions(
            X_train, y_train, meta_models, baseline_score=baseline_metrics.get('roc_auc', 0.75), baseline_std=0.05
        )
        logger.info(f"Suggestions: {len(suggestions)}, Advisories: {len(advisories)}")

        # Dedup
        suggestions = deduplicate_suggestions(suggestions)
        logger.info(f"After dedup: {len(suggestions)}")

        # Fit suggestions
        X_enhanced, fitted_params = fit_and_apply_suggestions(X_train, y_train, suggestions[:5])
        logger.info(f"Enhanced data: {X_enhanced.shape}")

        # Split enhanced
        X_tr_enh, X_vl_enh, y_tr_enh, y_vl_enh = train_test_split(X_enhanced, y_train, test_size=0.2, random_state=42)
        X_tr_enh_prep, X_vl_enh_prep, _ = prepare_data_for_model(X_tr_enh, X_vl_enh)

        # Enhanced model
        enhanced_model, enhanced_cols, _ = train_lgbm_model(
            X_tr_enh_prep, y_tr_enh, X_vl_enh_prep, y_vl_enh, n_classes=n_classes
        )
        logger.info(f"Enhanced model trained on {len(enhanced_cols)} features")

        enhanced_metrics = evaluate_on_set(enhanced_model, X_vl_enh_prep, y_vl_enh, train_columns=enhanced_cols, n_classes=n_classes)
        logger.info(f"Enhanced val metrics: roc_auc={enhanced_metrics.get('roc_auc'):.3f}")

        # Apply to test
        if y_test is not None:
            X_test_enh = apply_fitted_to_test(X_test, fitted_params)
            X_test_prep = pd.DataFrame(X_test_enh)
            for col in enhanced_cols:
                if col not in X_test_prep.columns:
                    X_test_prep[col] = 0
            test_metrics = evaluate_on_set(enhanced_model, X_test_prep, y_test, train_columns=enhanced_cols, n_classes=n_classes)
            logger.info(f"Test metrics: roc_auc={test_metrics.get('roc_auc'):.3f}")

        logger.info("\nPASSED: E2E Student Dropout pipeline complete")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# E2E TEST 2: BOOKING CANCELLATION (WITH DATES)
# ============================================================================

def test_e2e_booking(loaded_data, meta_models):
    """Complete pipeline: Booking with date columns."""
    logger.info("=" * 80)
    logger.info("E2E TEST 2: Booking Cancellation Pipeline")
    try:
        if 'booking' not in loaded_data:
            logger.info("Booking dataset not loaded, skipping")
            return

        data = loaded_data['booking']
        X_train = data['X_train']
        y_train = data['y_train']

        logger.info(f"Train: {X_train.shape}, y: {y_train.shape}")

        # Date detection
        date_cols = detect_date_columns(X_train)
        logger.info(f"Date columns: {list(date_cols.keys())}")

        # Split and prepare
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_tr_prep, X_vl_prep, col_encs = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()
        baseline_model, baseline_cols, _ = train_lgbm_model(
            X_tr_prep, y_tr, X_vl_prep, y_vl, n_classes=n_classes
        )

        baseline_metrics = evaluate_on_set(baseline_model, X_vl_prep, y_vl, train_columns=baseline_cols, n_classes=n_classes)
        logger.info(f"Baseline roc_auc: {baseline_metrics.get('roc_auc'):.3f}")

        logger.info("\nPASSED: E2E Booking pipeline complete")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# E2E TEST 3: CUSTOMER CONVERSION (WITH TEXT + DATES)
# ============================================================================

def test_e2e_customer_conversion(loaded_data, meta_models):
    """Complete pipeline: customer conversion with text and dates."""
    logger.info("=" * 80)
    logger.info("E2E TEST 3: Customer Conversion Pipeline")
    try:
        if 'customer_conversion' not in loaded_data:
            logger.info("Customer conversion not loaded, skipping")
            return

        data = loaded_data['customer_conversion']
        X_train = data['X_train']
        y_train = data['y_train']

        logger.info(f"Train: {X_train.shape}, y: {y_train.shape}")

        # Text and date detection
        text_cols = detect_text_columns(X_train)
        date_cols = detect_date_columns(X_train)
        logger.info(f"Text columns: {list(text_cols.keys())}")
        logger.info(f"Date columns: {list(date_cols.keys())}")

        # Check for missing
        missing_pct = X_train.isnull().sum() / len(X_train) * 100
        if (missing_pct > 0).any():
            logger.info(f"Missing: {missing_pct[missing_pct > 0]}")

        # Basic pipeline
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_tr_prep, X_vl_prep, _ = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()
        model, cols, _ = train_lgbm_model(X_tr_prep, y_tr, X_vl_prep, y_vl, n_classes=n_classes)

        metrics = evaluate_on_set(model, X_vl_prep, y_vl, train_columns=cols, n_classes=n_classes)
        logger.info(f"Val roc_auc: {metrics.get('roc_auc'):.3f}")

        logger.info("\nPASSED: E2E Customer Conversion pipeline complete")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# E2E TEST 4: RECIPE REVIEWS (MULTICLASS + TEXT)
# ============================================================================

def test_e2e_recipe_reviews(loaded_data, meta_models):
    """Complete pipeline: Recipe reviews multiclass with text."""
    logger.info("=" * 80)
    logger.info("E2E TEST 4: Recipe Reviews (Multiclass)")
    try:
        if 'recipe_reviews' not in loaded_data:
            logger.info("Recipe reviews not loaded, skipping")
            return

        data = loaded_data['recipe_reviews']
        X_train = data['X_train']
        y_train = data['y_train']

        logger.info(f"Train: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Target classes: {y_train.nunique()}")

        # Text detection
        text_cols = detect_text_columns(X_train)
        logger.info(f"Text columns: {list(text_cols.keys())}")

        # Basic pipeline
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_tr_prep, X_vl_prep, _ = prepare_data_for_model(X_tr, X_vl)

        n_classes = y_train.nunique()
        logger.info(f"Training multiclass model with {n_classes} classes")

        model, cols, _ = train_lgbm_model(X_tr_prep, y_tr, X_vl_prep, y_vl, n_classes=n_classes)

        metrics = evaluate_on_set(model, X_vl_prep, y_vl, train_columns=cols, n_classes=n_classes)
        logger.info(f"Val accuracy: {metrics.get('accuracy'):.3f}")

        logger.info("\nPASSED: E2E Recipe Reviews pipeline complete")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN: RUN ALL TESTS
# ============================================================================

def main():
    """Execute all tests and summarize results."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80)

    test_functions = []
    passed = 0
    failed = 0
    errors = []

    # Fixture: load data
    try:
        loaded_data = test_load_all_datasets()
        test_functions.append(("test_load_all_datasets", True, None))
        passed += 1
    except Exception as e:
        logger.error(f"FIXTURE FAILED: load_all_datasets: {e}")
        return

    # Fixture: load meta-models
    try:
        meta_models = test_load_meta_models()
        test_functions.append(("test_load_meta_models", True, None))
        passed += 1
    except Exception as e:
        logger.error(f"FIXTURE FAILED: load_meta_models: {e}")
        return

    # Unit tests
    unit_tests = [
        ("test_get_column_type_info", lambda: test_get_column_type_info(loaded_data)),
        ("test_detect_problematic_columns", lambda: test_detect_problematic_columns(loaded_data)),
        ("test_detect_date_columns", lambda: test_detect_date_columns(loaded_data)),
        ("test_detect_date_has_hour", lambda: test_detect_date_has_hour(loaded_data)),
        ("test_detect_dow_columns", lambda: test_detect_dow_columns(loaded_data)),
        ("test_detect_text_columns", lambda: test_detect_text_columns(loaded_data)),
        ("test_generate_dataset_advisories", lambda: test_generate_dataset_advisories(loaded_data)),
        ("test_get_dataset_meta", lambda: test_get_dataset_meta(loaded_data)),
        ("test_get_row_dataset_meta", lambda: test_get_row_dataset_meta(loaded_data)),
        ("test_get_baseline_importances", lambda: test_get_baseline_importances(loaded_data)),
        ("test_get_numeric_column_meta", lambda: test_get_numeric_column_meta(loaded_data)),
        ("test_get_categorical_column_meta", lambda: test_get_categorical_column_meta(loaded_data)),
        ("test_get_pair_meta_features", lambda: test_get_pair_meta_features(loaded_data)),
        ("test_should_test_numerical", lambda: test_should_test_numerical(loaded_data)),
        ("test_should_test_categorical", lambda: test_should_test_categorical(loaded_data)),
        ("test_build_feature_vector", lambda: test_build_feature_vector(loaded_data, meta_models)),
        ("test_ensure_numeric_target", lambda: test_ensure_numeric_target(loaded_data)),
        ("test_sanitize_feature_names", lambda: test_sanitize_feature_names(loaded_data)),
        ("test_override_options_for", lambda: test_override_options_for()),
        ("test_validate_col_override", lambda: test_validate_col_override(loaded_data)),
        ("test_constants", lambda: test_constants()),
    ]

    for test_name, test_fn in unit_tests:
        try:
            test_fn()
            test_functions.append((test_name, True, None))
            passed += 1
        except Exception as e:
            test_functions.append((test_name, False, str(e)))
            failed += 1
            errors.append((test_name, e))

    # Integration tests
    try:
        suggestions, advisories = test_generate_suggestions(loaded_data, meta_models)
        test_functions.append(("test_generate_suggestions", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_generate_suggestions", False, str(e)))
        failed += 1
        errors.append(("test_generate_suggestions", e))
        suggestions = []
        advisories = []

    try:
        suggestions = test_deduplicate_suggestions(suggestions)
        test_functions.append(("test_deduplicate_suggestions", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_deduplicate_suggestions", False, str(e)))
        failed += 1
        errors.append(("test_deduplicate_suggestions", e))

    try:
        X_tr_prep, X_vl_prep, col_encs = test_prepare_data_for_model(loaded_data)
        test_functions.append(("test_prepare_data_for_model", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_prepare_data_for_model", False, str(e)))
        failed += 1
        errors.append(("test_prepare_data_for_model", e))

    try:
        model, train_columns, col_encoders = test_train_lgbm_model(loaded_data)
        test_functions.append(("test_train_lgbm_model", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_train_lgbm_model", False, str(e)))
        failed += 1
        errors.append(("test_train_lgbm_model", e))
        model = None
        train_columns = []
        col_encoders = None

    try:
        test_train_lgbm_with_imbalance(loaded_data)
        test_functions.append(("test_train_lgbm_with_imbalance", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_train_lgbm_with_imbalance", False, str(e)))
        failed += 1
        errors.append(("test_train_lgbm_with_imbalance", e))

    try:
        metrics = test_evaluate_on_set(loaded_data, model, train_columns, col_encoders)
        test_functions.append(("test_evaluate_on_set", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_evaluate_on_set", False, str(e)))
        failed += 1
        errors.append(("test_evaluate_on_set", e))

    try:
        test_predict_on_set(loaded_data, model, train_columns)
        test_functions.append(("test_predict_on_set", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_predict_on_set", False, str(e)))
        failed += 1
        errors.append(("test_predict_on_set", e))

    try:
        X_enhanced, fitted_params = test_fit_and_apply_suggestions(loaded_data, suggestions)
        test_functions.append(("test_fit_and_apply_suggestions", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_fit_and_apply_suggestions", False, str(e)))
        failed += 1
        errors.append(("test_fit_and_apply_suggestions", e))
        fitted_params = None

    try:
        X_test_enh = test_apply_fitted_to_test(loaded_data, fitted_params)
        test_functions.append(("test_apply_fitted_to_test", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_apply_fitted_to_test", False, str(e)))
        failed += 1
        errors.append(("test_apply_fitted_to_test", e))

    try:
        test_compute_suggestion_verdicts(loaded_data, suggestions)
        test_functions.append(("test_compute_suggestion_verdicts", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_compute_suggestion_verdicts", False, str(e)))
        failed += 1
        errors.append(("test_compute_suggestion_verdicts", e))

    try:
        report_data = test_build_report_data(loaded_data)
        test_functions.append(("test_build_report_data", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_build_report_data", False, str(e)))
        failed += 1
        errors.append(("test_build_report_data", e))
        report_data = None

    try:
        test_generate_html_report(report_data)
        test_functions.append(("test_generate_html_report", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_generate_html_report", False, str(e)))
        failed += 1
        errors.append(("test_generate_html_report", e))

    try:
        test_generate_markdown_report(report_data)
        test_functions.append(("test_generate_markdown_report", True, None))
        passed += 1
    except Exception as e:
        test_functions.append(("test_generate_markdown_report", False, str(e)))
        failed += 1
        errors.append(("test_generate_markdown_report", e))

    # E2E tests
    e2e_tests = [
        ("test_e2e_student_dropout", lambda: test_e2e_student_dropout(loaded_data, meta_models)),
        ("test_e2e_booking", lambda: test_e2e_booking(loaded_data, meta_models)),
        ("test_e2e_customer_conversion", lambda: test_e2e_customer_conversion(loaded_data, meta_models)),
        ("test_e2e_recipe_reviews", lambda: test_e2e_recipe_reviews(loaded_data, meta_models)),
    ]

    for test_name, test_fn in e2e_tests:
        try:
            test_fn()
            test_functions.append((test_name, True, None))
            passed += 1
        except Exception as e:
            test_functions.append((test_name, False, str(e)))
            failed += 1
            errors.append((test_name, e))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"PASSED: {passed}")
    logger.info(f"FAILED: {failed}")
    logger.info(f"TOTAL:  {passed + failed}")

    if errors:
        logger.info("\n" + "=" * 80)
        logger.info("FAILED TESTS DETAILS")
        logger.info("=" * 80)
        for test_name, exc in errors:
            logger.error(f"\n{test_name}:")
            logger.error(f"  {exc}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Test log written to: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
