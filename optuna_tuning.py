# optuna_tuning.py

from typing import Dict, Any, Optional
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import optuna

# ✅ Always use the native libraries for boosting families
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def _choose_model_class(model_family: str, problem_type: str):
    """
    Map AutoGluon's coarse model family to a concrete implementation.

    Expected families (from your model_suggestion.py):
        - "lightgbm"
        - "xgboost"
        - "catboost"
        - "random_forest"
        - "extra_trees"
        - "knn"
        - "neural_network"
        - "linear_model"
        - "unknown" (fallback)
    """
    mf = (model_family or "").lower()
    pt = (problem_type or "").lower()

    is_reg = pt == "regression"
    is_clf = pt in ("binary", "multiclass")

    if mf == "lightgbm":
        return lgb.LGBMRegressor if is_reg else lgb.LGBMClassifier

    if mf == "xgboost":
        return xgb.XGBRegressor if is_reg else xgb.XGBClassifier

    if mf == "catboost":
        return cb.CatBoostRegressor if is_reg else cb.CatBoostClassifier

    if mf == "random_forest":
        return RandomForestRegressor if is_reg else RandomForestClassifier

    if mf == "extra_trees":
        return ExtraTreesRegressor if is_reg else ExtraTreesClassifier

    if mf == "knn":
        return KNeighborsRegressor if is_reg else KNeighborsClassifier

    if mf == "neural_network":
        return MLPRegressor if is_reg else MLPClassifier

    if mf == "linear_model":
        return LinearRegression if is_reg else LogisticRegression

    # Fallback if family is "unknown" or something unexpected
    return RandomForestRegressor if is_reg else RandomForestClassifier


def _get_search_range( #Add Zhiqi
    trial: optuna.Trial,
    param_name: str,
    default_low: float,
    default_high: float,
    ref_params: Dict[str, Any],
    is_int: bool = False,
    log: bool = False,
    step: Optional[float] = None,
) -> Any:
    """
    Helper to define a search range centered around a reference value if available.
    """
    kwargs = {}
    if step is not None:
        kwargs['step'] = step

    if not ref_params or param_name not in ref_params:
        if is_int:
            return trial.suggest_int(param_name, int(default_low), int(default_high), log=log, **kwargs)
        return trial.suggest_float(param_name, default_low, default_high, log=log, **kwargs)

    ref_val = ref_params[param_name]
    
    # If reference is not numeric (e.g. string/None), fallback to default
    if not isinstance(ref_val, (int, float)):
         if is_int:
            return trial.suggest_int(param_name, int(default_low), int(default_high), log=log, **kwargs)
         return trial.suggest_float(param_name, default_low, default_high, log=log, **kwargs)

    # Define a window around the reference value (e.g., +/- 50% or +/- fixed amount)
    # We want to be conservative but allow exploration
    
    if log:
        # For log scale, use multiplicative window
        low = max(default_low, ref_val * 0.5)
        high = min(default_high, ref_val * 2.0)
    else:
        # For linear scale, use additive or multiplicative window
        # Using a heuristic: +/- 30% of the value, or at least some absolute margin
        margin = abs(ref_val * 0.3)
        if margin == 0: margin = 1.0 # Handle zero case
        
        low = max(default_low, ref_val - margin)
        high = min(default_high, ref_val + margin)
        
    if is_int:
        low = int(low)
        high = int(high)
        # Ensure low <= high
        if low > high: low, high = high, low
        return trial.suggest_int(param_name, low, high, log=log, **kwargs)
    
    if low > high: low, high = high, low
    return trial.suggest_float(param_name, low, high, log=log, **kwargs)


def _build_model_params(
    trial: optuna.Trial,
    ModelClass,
    model_family: str,
    problem_type: str,
    reference_params: Optional[Dict[str, Any]] = None,# Add Zhiqi
) -> Dict[str, Any]:
    """
    Define Optuna search space for the chosen model class.
    If reference_params are provided, the search space is narrowed around them.
    """
    mf = (model_family or "").lower()
    pt = (problem_type or "").lower()
    is_reg = pt == "regression"
    params: Dict[str, Any] = {}
    
    ref = reference_params or {} # Add Zhiqi

    # --- LightGBM ---
    if ModelClass in (lgb.LGBMRegressor, lgb.LGBMClassifier):
        params = {
            "n_estimators": _get_search_range(trial, "n_estimators", 100, 1000, ref, is_int=True),
            "num_leaves": _get_search_range(trial, "num_leaves", 20, 300, ref, is_int=True),
            "learning_rate": _get_search_range(trial, "learning_rate", 1e-3, 0.3, ref, log=True),
            "max_depth": _get_search_range(trial, "max_depth", -1, 20, ref, is_int=True),
            "subsample": _get_search_range(trial, "subsample", 0.5, 1.0, ref),
            "colsample_bytree": _get_search_range(trial, "colsample_bytree", 0.5, 1.0, ref),
            "min_child_samples": _get_search_range(trial, "min_child_samples", 5, 100, ref, is_int=True),
            "reg_lambda": _get_search_range(trial, "reg_lambda", 1e-4, 10.0, ref, log=True),
            "reg_alpha": _get_search_range(trial, "reg_alpha", 1e-4, 10.0, ref, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

    # --- XGBoost ---
    elif ModelClass in (xgb.XGBRegressor, xgb.XGBClassifier):
        params = {
            "n_estimators": _get_search_range(trial, "n_estimators", 100, 1000, ref, is_int=True),
            "max_depth": _get_search_range(trial, "max_depth", 3, 15, ref, is_int=True),
            "learning_rate": _get_search_range(trial, "learning_rate", 1e-3, 0.3, ref, log=True),
            "subsample": _get_search_range(trial, "subsample", 0.5, 1.0, ref),
            "colsample_bytree": _get_search_range(trial, "colsample_bytree", 0.5, 1.0, ref),
            "min_child_weight": _get_search_range(trial, "min_child_weight", 1e-2, 10.0, ref, log=True),
            "gamma": _get_search_range(trial, "gamma", 0.0, 5.0, ref),
            "reg_lambda": _get_search_range(trial, "reg_lambda", 1e-4, 10.0, ref, log=True),
            "reg_alpha": _get_search_range(trial, "reg_alpha", 1e-4, 10.0, ref, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    # --- CatBoost ---
    elif ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
        params = {
            "depth": _get_search_range(trial, "depth", 4, 10, ref, is_int=True),
            "learning_rate": _get_search_range(trial, "learning_rate", 1e-3, 0.3, ref, log=True),
            "l2_leaf_reg": _get_search_range(trial, "l2_leaf_reg", 1.0, 10.0, ref),
            "random_strength": _get_search_range(trial, "random_strength", 0.0, 1.0, ref),
            "bagging_temperature": _get_search_range(trial, "bagging_temperature", 0.0, 5.0, ref),
            "border_count": _get_search_range(trial, "border_count", 32, 255, ref, is_int=True),
            "n_estimators": _get_search_range(trial, "n_estimators", 200, 1000, ref, is_int=True),
            "random_state": 42,
            "verbose": 0,
        }
        if not is_reg:
            if pt == "binary":
                params["loss_function"] = "Logloss"
            else:
                params["loss_function"] = "MultiClass"

    # --- RandomForest / ExtraTrees ---
    elif ModelClass in (
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesRegressor,
        ExtraTreesClassifier,
    ):
        params = {
            "n_estimators": _get_search_range(trial, "n_estimators", 100, 500, ref, is_int=True),
            "max_depth": _get_search_range(trial, "max_depth", 3, 30, ref, is_int=True),
            "min_samples_split": _get_search_range(trial, "min_samples_split", 2, 20, ref, is_int=True),
            "min_samples_leaf": _get_search_range(trial, "min_samples_leaf", 1, 10, ref, is_int=True),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": 42,
        }

    # --- KNN ---
    elif ModelClass in (KNeighborsRegressor, KNeighborsClassifier):
        params = {
            "n_neighbors": _get_search_range(trial, "n_neighbors", 3, 50, ref, is_int=True),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
            "leaf_size": _get_search_range(trial, "leaf_size", 20, 60, ref, is_int=True),
        }

    # --- MLP ---
    elif ModelClass in (MLPRegressor, MLPClassifier):
        hidden_layer_options = [
            (64,),
            (128,),
            (64, 64),
            (128, 64),
        ]
        params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", hidden_layer_options
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": _get_search_range(trial, "alpha", 1e-5, 1e-1, ref, log=True),
            "learning_rate_init": _get_search_range(trial, "learning_rate_init", 1e-4, 1e-2, ref, log=True),
            "max_iter": 200,
            "random_state": 42,
        }

    # --- Linear models ---
    elif ModelClass is LinearRegression:
        params = {}

    elif ModelClass is LogisticRegression:
        params = {
            "C": _get_search_range(trial, "C", 1e-3, 10.0, ref, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000,
            "n_jobs": -1,
        }
        if pt == "multiclass":
            params["multi_class"] = "multinomial"

    else:
        # Generic tiny space for unhandled models
        params = {}

    return params


def tune_model_with_optuna(
    df: pd.DataFrame,
    target_column: str,
    model_family: str,
    problem_type: str,
    eval_metric: str,
    n_trials: int = 30,
    time_limit: Optional[int] = None,
    initial_params: Optional[Dict[str, Any]] = None,#Add Zhiqi
) -> Dict[str, Any]:
    """
    Tune hyperparameters with Optuna, optionally starting from AutoGluon's best parameters.
    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataframe.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    # Work on a safe copy and drop rows with missing target
    df = df.copy()
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("All rows have missing values in the target column.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    pt = (problem_type or "").lower()
    is_reg = pt == "regression"
    is_clf = pt in ("binary", "multiclass")

    stratify = y if is_clf and y.nunique() > 1 else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Keep both the original name and a lower-cased internal version
    metric_name = eval_metric
    metric_internal = (eval_metric or "").lower()

    def objective(trial: optuna.Trial) -> float:
        ModelClass = _choose_model_class(model_family, problem_type)
        # Pass initial_params as reference to define targeted search space # Add Zhiqi
        params = _build_model_params(trial, ModelClass, model_family, problem_type, reference_params=initial_params)

        # Special defaults for CatBoost
        if ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
            params.setdefault("verbose", 0)
            params.setdefault("random_state", 42)

        model = ModelClass(**params)
        model.fit(X_train, y_train)

        if is_reg:
            y_pred = model.predict(X_valid)
            # Old sklearn doesn't support squared=False, so compute RMSE manually
            mse = mean_squared_error(y_valid, y_pred)
            rmse = np.sqrt(mse)
            # For regression we MINIMIZE RMSE directly
            return rmse

        # Classification: need probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_valid)
        else:
            # Very unlikely with the models we use, but safety fallback
            y_pred = model.predict(X_valid)
            classes = np.unique(y_train)
            proba = np.zeros((len(y_pred), len(classes)))
            for i, c in enumerate(classes):
                proba[:, i] = (y_pred == c).astype(float)

        # Binary AUC → we MINIMIZE (1 - AUC)
        if metric_internal == "roc_auc" and pt == "binary":
            auc = roc_auc_score(y_valid, proba[:, 1])
            return 1.0 - auc

        # Default for classification: log_loss (works for binary & multiclass)
        return log_loss(y_valid, proba)

    study = optuna.create_study(direction="minimize")
    
    # Enqueue the initial parameters to ensure they are evaluated first
    if initial_params:
        print(f"Enqueuing initial parameters from AutoGluon: {initial_params}")
        # We need to filter initial_params to only include those that are in the search space
        # But Optuna's enqueue_trial is robust to extra params usually, or we can just pass it.
        # However, we need to make sure the keys match what _build_model_params expects.
        # AutoGluon params might have different names or extra keys. 
        # For now, we pass it as is, Optuna will ignore unknown keys or we might need to map them.
        # Given the complexity of mapping, we'll try passing them directly.
        study.enqueue_trial(initial_params)

    # If time_limit is set (in seconds), stop after that many seconds overall
    if time_limit is not None:
        study.optimize(objective, n_trials=n_trials, timeout=time_limit)
    else:
        study.optimize(objective, n_trials=n_trials)


    # --- Convert Optuna objective back into human-friendly score ---
    best_objective = study.best_value
    if is_reg:
        # objective is RMSE already
        best_eval_score = best_objective
    elif metric_internal == "roc_auc" and pt == "binary":
        # objective = 1 - AUC → AUC = 1 - objective
        best_eval_score = 1.0 - best_objective
    else:
        # log_loss or any other "lower is better" metric
        best_eval_score = best_objective

    # --- Refit best model on FULL data (X, y) ---
    ModelClass = _choose_model_class(model_family, problem_type)
    best_params = study.best_params.copy()

    mf_lower = (model_family or "").lower()
    if mf_lower in ("lightgbm", "xgboost", "random_forest", "extra_trees"):
        best_params.setdefault("random_state", 42)
        best_params.setdefault("n_jobs", -1)

    if ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
        best_params.setdefault("verbose", 0)
        best_params.setdefault("random_state", 42)

    # 👉 This is your "another model" with the exact best hyperparameters
    best_model = ModelClass(**best_params)
    best_model.fit(X, y)

    return {
        "best_model_class": ModelClass.__name__,
        "best_params": best_params,
        "best_model": best_model,
        "best_objective": best_objective,
        "best_eval_score": best_eval_score,
        "metric_name": metric_name,
        "study": study,
    }
