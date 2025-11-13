import pandas as pd
from autogluon.tabular import TabularPredictor
from typing import Dict, Any, Optional
import tempfile
import shutil
import logging

# Configure logging to suppress verbose AutoGluon output if desired
logging.basicConfig(level=logging.WARNING)


def _detect_problem_type(df: pd.DataFrame, target_column: str):
    """
    Detect problem type based on the target column and choose metrics:

    - Regression  → problem_type="regression",  eval_metric="root_mean_squared_error" (RMSE)
    - Binary      → problem_type="binary",      eval_metric="roc_auc" (AUC)
    - Multiclass  → problem_type="multiclass",  eval_metric="log_loss"
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the dataset.")

    target = df[target_column]
    n_unique = target.nunique(dropna=True)

    if n_unique <= 1:
        raise ValueError("Target column must have at least 2 distinct values.")

    # Convert to numeric if possible, but handle non-numeric gracefully
    numeric_target = pd.to_numeric(target, errors='coerce')
    is_numeric = pd.api.types.is_numeric_dtype(target) and not numeric_target.isna().all()


    # Regression: More than a few unique numeric values
    # Heuristic: if > 20 unique values, and it's a number, it's likely regression.
    if is_numeric and n_unique > 20:
        return "regression", "root_mean_squared_error"

    # Classification (Binary or Multiclass)
    if n_unique == 2:
        return "binary", "roc_auc"
    else:
        # Also handles numeric data with few unique values (e.g., 1, 2, 3, 4)
        return "multiclass", "log_loss"


def _infer_model_family(model_name: str) -> str:
    """
    Heuristic to map AutoGluon model names to a coarse model family.
    This is updated to be consistent with model_scorer.py.
    """
    name = model_name.lower()

    if "catboost" in name:
        return "catboost"
    if "lightgbm" in name:
        return "lightgbm"
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "randomforest" in name or "extratrees" in name or "rf_" in name or "et_" in name:
        return "ensemble_trees"
    if "knn" in name or "kneighbors" in name:
        return "knn"
    if "nn_" in name or "neuralnet" in name:
        return "neural_network"
    if "linear" in name or "lr_" in name:
        return "linear_model"
    
    return "unknown"


def run_model_suggestions(
    df: pd.DataFrame,
    target_column: str,
    time_limit: Optional[int] = 300,
    presets: str = "medium_quality",
    purpose: str = "full_run" # NEW: Add a purpose flag
) -> Dict[str, Any]:
    """
    Train models with AutoGluon.
    - purpose='full_run': Standard behavior for recommendations.
    - purpose='feature_importance_only': A quick run just to get FI.
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the dataset.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    df = df.copy()
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("All rows have missing values in the target column.")

    problem_type, eval_metric = _detect_problem_type(df, target_column)
    model_path = tempfile.mkdtemp(prefix="ag_models_")
    
    if purpose == 'feature_importance_only':
        print(f"AutoGluon running in 'feature_importance_only' mode in: {model_path}")
    else:
        print(f"AutoGluon models will be saved to temporary directory: {model_path}")

    results = {}

    try:
        predictor = TabularPredictor(
            label=target_column, problem_type=problem_type, eval_metric=eval_metric, path=model_path,
        ).fit(
            train_data=df, time_limit=time_limit, presets=presets, dynamic_stacking=False, verbosity=0
        )

        leaderboard = predictor.leaderboard(silent=True)
        if leaderboard is None or leaderboard.empty:
            raise RuntimeError("No models were trained successfully; leaderboard is empty.")

        # For a full run, we care about the leaderboard and other details.
        # For a quick FI run, we just need the importance dataframe.
        fi_df = predictor.feature_importance(data=df)

        results = {
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "leaderboard": leaderboard,
            "feature_importance": fi_df,
        }

    except Exception as e:
        print(f"An error occurred during AutoGluon training: {e}")
        raise

    finally:
        print(f"Cleaning up temporary directory: {model_path}")
        shutil.rmtree(model_path, ignore_errors=True)

    return results