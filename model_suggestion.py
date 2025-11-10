import pandas as pd
from autogluon.tabular import TabularPredictor
from typing import Dict, Any, Optional


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

    is_numeric = pd.api.types.is_numeric_dtype(target)

    # Numeric with > 2 unique values → regression (continuous)
    if is_numeric and n_unique > 2:
        return "regression", "root_mean_squared_error"

    # Otherwise, treat as classification
    if n_unique == 2:
        return "binary", "roc_auc"
    else:
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
    if "knn" in name:
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
    presets: str = "medium_quality_faster_train",
) -> Dict[str, Any]:
    """
    Train models with AutoGluon and return artifacts that can be used
    downstream (Optuna, feature selection, etc.), WITHOUT any Streamlit UI.

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
        {
            "problem_type": str,          # "regression" / "binary" / "multiclass"
            "eval_metric": str,           # "root_mean_squared_error" / "roc_auc" / "log_loss"
            "leaderboard": pd.DataFrame,  # full AutoGluon leaderboard
            "best_model_name": str,       # AutoGluon internal model name (e.g. "LightGBM_BAG_L1")
            "best_model_family": str,     # coarse family label for Optuna logic (e.g. "lightgbm")
            "feature_importance": pd.DataFrame or None,  # importance values (may be None)
            "predictor": TabularPredictor # fitted predictor object
        }

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., missing target column, too few target classes).
    Exception
        Any underlying AutoGluon error will propagate upward.
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
        time_limit = 60  # seconds, adjust if you want

    # Train AutoGluon predictor
    predictor = TabularPredictor(
        label=target_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
    ).fit(
        train_data=df,
        time_limit=time_limit,
        presets=presets,
    )

    # Get leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    if leaderboard is None or leaderboard.empty:
        raise RuntimeError("No models were trained successfully; leaderboard is empty.")

    # Best model is first row in leaderboard
    best_model_name = str(leaderboard.iloc[0]["model"])
    best_model_family = _infer_model_family(best_model_name)

    # Compute feature importance for downstream feature selection
    try:
        fi_df = predictor.feature_importance(data=df)
    except Exception:
        fi_df = None

    return {
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_model_family": best_model_family,
        "feature_importance": fi_df,
        "predictor": predictor,
    }
