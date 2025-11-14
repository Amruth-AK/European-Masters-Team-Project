# model_suggestion.py

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
    if is_numeric and n_unique > 20:
        return "regression", "root_mean_squared_error"

    # Classification (Binary or Multiclass)
    if n_unique == 2:
        return "binary", "roc_auc"
    else:
        return "multiclass", "log_loss"


def _infer_model_family(model_name: str) -> str:
    """
    Heuristic to map AutoGluon model names to a coarse model family.
    """
    name = model_name.lower()

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
    # 👉 Let AutoGluon use its strongest tabular preset by default
    presets: str = "best_quality",
) -> Dict[str, Any]:
    """
    Train models with AutoGluon and return artifacts that can be used
    downstream (Optuna, feature selection, etc.), WITHOUT any Streamlit UI.

    This version creates and cleans up a temporary directory to avoid
    Windows PermissionError issues.
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the dataset.")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    # Work on a copy so we don't mutate the original dataframe
    df = df.copy()

    # Drop rows where target is missing (AutoGluon cannot train with NaN labels)
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("All rows have missing values in the target column.")

    # Detect problem type and metric
    problem_type, eval_metric = _detect_problem_type(df, target_column)

    # Default training time_limit if not provided
    if time_limit is None:
        # 👉 Give AutoGluon a more generous time budget
        time_limit = 300  # seconds; adjust if this is too slow for you

    # --- Use a temporary directory for AutoGluon models ---
    model_path = tempfile.mkdtemp(prefix="ag_models_")
    print(f"AutoGluon models will be saved to temporary directory: {model_path}")

    results: Dict[str, Any] = {}
    try:
        # 👉 IMPORTANT: do *not* restrict dynamic stacking or hyperparameters here.
        # Let AutoGluon choose the best models/ensembles it can within the preset/time.
        predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=model_path,
        ).fit(
            train_data=df,
            time_limit=time_limit,
            presets=presets,
            verbosity=0,
        )

        # Get leaderboard
        leaderboard = predictor.leaderboard(silent=True)

        print("\nTop 5 Models by Validation Score:")
        print(leaderboard.head(5)[['model', 'score_val']])

        if leaderboard is None or leaderboard.empty:
            raise RuntimeError("No models were trained successfully; leaderboard is empty.")

        # Prefer the best NON-ENSEMBLE model as the *family* for Optuna,
        # but ensembles and stacks are still trained and can be the overall best.
        model_series = leaderboard["model"].astype(str)
        non_ensemble_rows = leaderboard[
            ~model_series.str.contains("weightedensemble", case=False, na=False)
            & ~model_series.str.contains("stack", case=False, na=False)
        ]

        if not non_ensemble_rows.empty:
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

        results = {
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "leaderboard": leaderboard,
            "best_model_name": best_model_name,      # may be base model or ensemble
            "best_model_family": best_model_family,  # for Optuna
            "feature_importance": fi_df,
            "predictor_path": model_path,
        }

    except Exception as e:
        print(f"An error occurred during AutoGluon training: {e}")
        raise

    finally:
        # Clean up temp dir
        print(f"Cleaning up temporary directory: {model_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        # Path no longer valid after cleanup
        results["predictor_path"] = None

    return results



# -------------------------------------------------------------------
# NEW: train_raw_autogluon_model – this is the one used for train/test
# -------------------------------------------------------------------

def train_raw_autogluon_model(
    df: pd.DataFrame,
    target_column: str,
    time_limit: Optional[int] = 600,      # ⏱ 10 minutes default
    presets: str = "best_quality",        # 💪 strongest built-in preset
) -> Dict[str, Any]:
    """
    Train AutoGluon on the *raw* train dataset and KEEP the model directory
    so we can later use the exact same best model on the test set.

    - Uses df exactly as provided (after your manual dtype tweaks on Home).
    - Does NOT use your preprocessing pipeline or suggestions.
    - Lets AutoGluon choose its full model zoo + ensembles (no restrictions).
    - Picks the top model from the leaderboard and stores its name + path.
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the dataset.")
    if df.empty:
        raise ValueError("The input dataset is empty.")

    df = df.copy()

    # Drop rows where target is missing (AG cannot train with NaN labels)
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("All rows have missing values in the target column.")

    problem_type, eval_metric = _detect_problem_type(df, target_column)

    # Create a dedicated directory for this raw model
    model_path = tempfile.mkdtemp(prefix="ag_raw_model_")
    print(f"[RAW AG] Training raw AutoGluon model at: {model_path}")
    print(f"[RAW AG] problem_type={problem_type}, eval_metric={eval_metric}")
    print(f"[RAW AG] presets={presets}, time_limit={time_limit}")

    # 👉 No dynamic_stacking=False, no manual hyperparameter overrides.
    #    Let AutoGluon use everything it wants within the preset + time_limit.
    predictor = TabularPredictor(
        label=target_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=model_path,
    ).fit(
        train_data=df,
        time_limit=time_limit,
        presets=presets,
        verbosity=2,   # a bit of logging in terminal; Streamlit will still be fine
    )

    # Get leaderboard and pick the absolute best model
    leaderboard = predictor.leaderboard(silent=True)
    if leaderboard is None or leaderboard.empty:
        raise RuntimeError("[RAW AG] No models were trained successfully; leaderboard is empty.")

    # AutoGluon sorts leaderboard so the best model is on top
    best_row = leaderboard.iloc[0]
    best_model_name = str(best_row["model"])

    print(f"[RAW AG] Best model selected for train/test: {best_model_name}")

    return {
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,   # this can be an ensemble/stack
        "predictor_path": model_path,         # directory we will load later for test
    }

