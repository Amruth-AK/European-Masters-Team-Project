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
    UPDATED to handle 'ensemble_trees'.
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
    if mf == "ensemble_trees": # MODIFIED: Handles the new combined family
        return RandomForestRegressor if is_reg else RandomForestClassifier
    if mf == "knn":
        return KNeighborsRegressor if is_reg else KNeighborsClassifier
    if mf == "neural_network":
        return MLPRegressor if is_reg else MLPClassifier
    if mf == "linear_model":
        return LinearRegression if is_reg else LogisticRegression
    # Fallback if family is "unknown" or something unexpected
    return RandomForestRegressor if is_reg else RandomForestClassifier


def _build_model_params(
    trial: optuna.Trial,
    ModelClass,
    model_family: str,
    problem_type: str,
) -> Dict[str, Any]:
    """
    Define Optuna search space for the chosen model class.
    """
    mf = (model_family or "").lower()
    pt = (problem_type or "").lower()
    is_reg = pt == "regression"
    params: Dict[str, Any] = {}

    # --- LightGBM ---
    if ModelClass in (lgb.LGBMRegressor, lgb.LGBMClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

    # --- XGBoost ---
    elif ModelClass in (xgb.XGBRegressor, xgb.XGBClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-2, 10.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    # --- CatBoost ---
    elif ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 5.0
            ),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
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
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
            "leaf_size": trial.suggest_int("leaf_size", 20, 60),
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
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "max_iter": 200,
            "random_state": 42,
        }

    # --- Linear models ---
    elif ModelClass is LinearRegression:
        params = {}

    elif ModelClass is LogisticRegression:
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
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
) -> Dict[str, Any]:
    """
    Tune hyperparameters with Optuna for the SAME model family that AutoGluon chose.
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

    metric_name = eval_metric
    metric_internal = (eval_metric or "").lower()

    def objective(trial: optuna.Trial) -> float:
        ModelClass = _choose_model_class(model_family, problem_type)
        params = _build_model_params(trial, ModelClass, model_family, problem_type)

        if ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
            params.setdefault("verbose", 0)
            params.setdefault("random_state", 42)

        model = ModelClass(**params)
        model.fit(X_train, y_train)

        if is_reg:
            y_pred = model.predict(X_valid)
            mse = mean_squared_error(y_valid, y_pred)
            return np.sqrt(mse)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_valid)
        else:
            y_pred = model.predict(X_valid)
            classes = np.unique(y_train)
            proba = np.zeros((len(y_pred), len(classes)))
            for i, c in enumerate(classes):
                proba[:, i] = (y_pred == c).astype(float)

        if metric_internal == "roc_auc" and pt == "binary":
            auc = roc_auc_score(y_valid, proba[:, 1])
            return 1.0 - auc

        return log_loss(y_valid, proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=time_limit)

    best_objective = study.best_value
    if is_reg:
        best_eval_score = best_objective
    elif metric_internal == "roc_auc" and pt == "binary":
        best_eval_score = 1.0 - best_objective
    else:
        best_eval_score = best_objective

    ModelClass = _choose_model_class(model_family, problem_type)
    best_params = study.best_params.copy()

    mf_lower = (model_family or "").lower()
    if mf_lower in ("lightgbm", "xgboost", "ensemble_trees"):
        best_params.setdefault("random_state", 42)
        best_params.setdefault("n_jobs", -1)

    if ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
        best_params.setdefault("verbose", 0)
        best_params.setdefault("random_state", 42)

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