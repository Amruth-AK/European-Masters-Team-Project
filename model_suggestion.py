from autogluon.tabular.predictor.predictor import os
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.common.features.feature_metadata import FeatureMetadata 



from typing import Dict, Any, Optional
import tempfile
import shutil
import logging
import os
import time
import glob

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
    Heuristic to map AutoGluon model names to a coarse model family,
    which you can later use to decide what to tune with Optuna.
    """
    name = model_name.lower()

    # Common AutoGluon model prefixes
    if "lightgbm" in name:
        return "lightgbm"
    if "catboost" in name:
        return "catboost"
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "randomforest" in name or "rf_" in name:
        return "random_forest"
    if "extratrees" in name or "et_" in name:
        return "extra_trees"
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
    time_limit: Optional[int] = None,
    presets: str = "medium_quality", # Use medium_quality for faster feedback
) -> Dict[str, Any]:
    """
    Train models with AutoGluon and return artifacts that can be used
    downstream (Optuna, feature selection, etc.), WITHOUT any Streamlit UI.

    This version creates and cleans up a temporary directory to avoid
    Windows PermissionError issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (already preprocessed).
    target_column : str
        Target column name in `df`.
    time_limit : int, optional
        Global training time limit in seconds.
    presets : str
        AutoGluon presets.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing analysis results.
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the dataset.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    # Work on a copy so we don't mutate the original dataframe
    df_processed = df.copy()

    # Drop rows where target is missing (AutoGluon cannot train with NaN labels)
    df_processed = df_processed.dropna(subset=[target_column])
    if df_processed.empty:
        raise ValueError("All rows have missing values in the target column.")

    # Detect problem type and metric
    problem_type, eval_metric = _detect_problem_type(df_processed, target_column)

    if time_limit is None:
        time_limit = _calculate_adaptive_time_limit(df_processed, target_column)
        print(f"Auto-calculated time_limit: {time_limit} seconds ({time_limit/60:.1f} minutes) based on dataset size")

    # Optional but recommended: Tell AutoGluon the type of each column, prevent it from re-inferring & triggering extra preprocessing
    feature_metadata = FeatureMetadata.from_df(df_processed)

    temp_dir = tempfile.gettempdir()
    old_ag_dirs = glob.glob(f"{temp_dir}/ag_models_*")
    for old_dir in old_ag_dirs:
        try:
            
            if os.path.getmtime(old_dir) < time.time() - 3600:
                shutil.rmtree(old_dir, ignore_errors=True)
        except:
            pass

    model_path = tempfile.mkdtemp(prefix=f"ag_models_{int(time.time())}_")
    
    
    if os.path.exists(model_path) and os.listdir(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
        model_path = tempfile.mkdtemp(prefix=f"ag_models_{int(time.time())}_")

    print(f"AutoGluon models will be saved to temporary directory: {model_path}")

    try:
        predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=model_path,
        ).fit(
            train_data=df_processed,  # Use processed DataFrame
            feature_generator=IdentityFeatureGenerator(),  # basic AutoGluon built-in preprocessing
            feature_metadata=feature_metadata,  # optional: let AutoGluon not guess dtype
            time_limit=time_limit,
            presets=presets,
            dynamic_stacking=False,
            verbosity=0,
        )

        # Get leaderboard
        leaderboard = predictor.leaderboard(silent=True)  # silent=True to suppress output  
        
        print("\nTop 5 Models by Validation Score:")
        print(leaderboard.head(5)[['model', 'score_val']])

        
        if leaderboard is None or leaderboard.empty:
            raise RuntimeError("No models were trained successfully; leaderboard is empty.")

        # Prefer the best NON-ENSEMBLE model as the base model family
        model_series = leaderboard["model"].astype(str)
        non_ensemble_rows = leaderboard[
            ~model_series.str.contains("weightedensemble", case=False, na=False) &
            ~model_series.str.contains("stack", case=False, na=False)
        ]

        if not non_ensemble_rows.empty:
            # Take the best base model (first non-ensemble row)
            best_row = non_ensemble_rows.iloc[0]
        else:
            # Fallback: if only ensembles were trained, use the top one
            best_row = leaderboard.iloc[0]

        best_model_name = str(best_row["model"])
        best_model_family = _infer_model_family(best_model_name)

        # Compute feature importance for downstream feature selection
        try:
            fi_df = predictor.feature_importance(data=df)
        except Exception as e:
            print(f"Could not compute feature importance: {e}")
            fi_df = None

        # Prepare results before the directory is deleted
        results = {
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "leaderboard": leaderboard,
            "best_model_name": best_model_name,
            "best_model_family": best_model_family,
            "feature_importance": fi_df,
            "predictor_path": model_path, # Return path in case it needs to be accessed
        }

    except Exception as e:
        # If anything goes wrong, still try to clean up
        print(f"An error occurred during AutoGluon training: {e}")
        raise # Re-raise the exception after printing

    finally:
        # --- IMPORTANT ---
        # Reliably clean up the temporary directory and all its contents
        print(f"Cleaning up temporary directory: {model_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        # Note: The 'predictor' object is now invalid as its files are deleted.
        # This function's purpose is to return METADATA, not a usable predictor.
        results["predictor_path"] = None # Path is no longer valid

    return results




def _calculate_adaptive_time_limit(df: pd.DataFrame, target_column: str) -> int:
    """
    Calculate adaptive time limit based on dataset characteristics.
    
    Formula:
    - Base time: 2 minutes per 10k samples
    - Feature penalty: +30 seconds per 100 features
    - Problem type multiplier: multiclass (1.5x), regression (1.2x), binary (1.0x)
    
    Returns:
    - Minimum: 300 seconds (5 minutes)
    - Maximum: 3600 seconds (1 hour)
    """
    n_samples = len(df)
    n_features = len(df.columns) - 1  # Exclude target column
    
    # Base time: 2 minutes per 10k samples
    base_time = max(60, (n_samples / 10000) * 120)
    
    # Feature penalty: +30 seconds per 100 features
    feature_penalty = (n_features / 100) * 30
    
    # Problem type multiplier
    problem_type, _ = _detect_problem_type(df, target_column)
    if problem_type == "multiclass":
        multiplier = 1.5
    elif problem_type == "regression":
        multiplier = 1.2
    else:  # binary
        multiplier = 1.0
    
    recommended = int((base_time + feature_penalty) * multiplier)
    
    # Cap between 5 minutes and 1 hour
    return max(300, min(recommended, 3600))