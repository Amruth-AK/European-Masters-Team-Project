import time

# optuna_tuning.py

from typing import Dict, Any, Optional, Tuple
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, make_scorer


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


def suggest_cv_strategy(n_samples: int) -> Tuple[bool, int]:
    """
    Automatically determine CV strategy based on dataset size.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    
    Returns
    -------
    Tuple[bool, int]
        (use_cv, cv_folds)
        - use_cv: Whether to use cross-validation
        - cv_folds: Number of CV folds
    
    Strategy:
    - < 1000 samples: No CV (use train/test split)
    - 1000-5000 samples: 5-fold CV
    - > 5000 samples: 8-fold CV
    """
    if n_samples < 1000:
        return False, 1  # No CV, use train/test split
    elif n_samples < 5000:
        return True, 5   # 5-fold CV
    else:
        return True, 8   # 8-fold CV


def suggest_optuna_params(
    n_samples: int,
    n_features: int,
    model_family: str,
    use_cv: Optional[bool] = None,
    cv_folds: Optional[int] = None,
    time_budget_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Suggest optimal n_trials and time_limit based on dataset size and model complexity.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    n_features : int
        Number of features
    model_family : str
        Model family name (e.g., "lightgbm", "xgboost", etc.)
    use_cv : Optional[bool]
        Whether cross-validation is used. If None, automatically determined based on n_samples.
    cv_folds : Optional[int]
        Number of CV folds. If None, automatically determined based on n_samples.
    time_budget_seconds : Optional[int]
        Total time budget in seconds. If None, estimates based on data size.
    
    Returns
    -------
    dict with keys: n_trials, time_limit, patience, min_improvement, use_cv, cv_folds
    """
    # Auto-determine CV strategy if not provided
    if use_cv is None or cv_folds is None:
        use_cv, cv_folds = suggest_cv_strategy(n_samples)
    # Model complexity factors (trials per 1000 samples)
    complexity_factors = {
        "lightgbm": 15,
        "xgboost": 15,
        "catboost": 10,
        "random_forest": 8,
        "extra_trees": 8,
        "knn": 5,
        "neural_network": 12,
        "linear_model": 3,
    }
    
    factor = complexity_factors.get(model_family.lower(), 10)
    
    # Base trials calculation based on sample size
    # Minimum 30 trials for meaningful search, maximum 200
    base_trials = max(30, min(200, int(n_samples / 1000 * factor)))
    
    # Adjust for feature count (more features = more exploration needed)
    feature_factor = min(1.3, 1.0 + (n_features / 100) * 0.1)
    suggested_trials = int(base_trials * feature_factor)
    
    # Adjust for CV (CV takes longer, so moderately reduce trials)
    # But ensure we still have enough trials for exploration
    if use_cv:
        suggested_trials = max(20, int(suggested_trials * 0.75))  # Reduce by 25%, but at least 20
    else:
        # Without CV, we can afford more trials
        suggested_trials = max(30, suggested_trials)
    
    # Estimate time per trial (rough estimates in seconds)
    time_per_trial = {
        "lightgbm": 0.5,
        "xgboost": 0.6,
        "catboost": 0.8,
        "random_forest": 0.3,
        "extra_trees": 0.3,
        "knn": 0.1,
        "neural_network": 1.0,
        "linear_model": 0.05,
    }
    
    base_time = time_per_trial.get(model_family.lower(), 0.5)
    
    # Scale by data size and CV
    cv_multiplier = cv_folds if use_cv else 1
    sample_factor = 1.0 + (n_samples / 10000) * 0.5
    feature_time_factor = 1.0 + (n_features / 50) * 0.2
    
    estimated_time_per_trial = base_time * cv_multiplier * sample_factor * feature_time_factor
    
    # Calculate suggested time limit
    if time_budget_seconds is None:
        # Default: allow 2-5 minutes for small datasets, up to 10 minutes for large
        if n_samples < 1000:
            suggested_time = 120  
        elif n_samples < 10000:
            suggested_time = 300  
        else:
            suggested_time = 600  
    else:
        suggested_time = time_budget_seconds
    
    # Adjust trials based on time budget
    max_trials_by_time = int(suggested_time / estimated_time_per_trial * 0.85)  # 85% of budget
    suggested_trials = min(suggested_trials, max_trials_by_time)
    # Ensure minimum trials for meaningful search
    min_trials = 20 if use_cv else 30
    suggested_trials = max(min_trials, min(200, suggested_trials))  # Clamp between min-200
    
    # Adaptive patience (20% of trials, but at least 5, at most 30)
    patience = max(5, min(30, int(suggested_trials * 0.2)))
    
    # Adaptive min_improvement
    min_improvement = 0.001  # Default
    
    return {
        "n_trials": suggested_trials,
        "time_limit": suggested_time,
        "patience": patience,
        "min_improvement": min_improvement,
        "use_cv": use_cv,
        "cv_folds": cv_folds,
    }


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
                "learning_rate", 0.01, 0.3, log=True  
            ),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),  
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),  
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),  
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),  
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
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
    n_trials: int = 100,
    time_limit: Optional[int] = None,
    min_improvement: float = 0.001,  # Minimum improvement to continue
    patience: int = 20,  # Stop if no improvement for N trials
    use_cv: Optional[bool] = None,  # Whether to use cross-validation
    cv_folds: Optional[int] = None,  # Number of cross-validation folds
) -> Dict[str, Any]:
    """
    Tune hyperparameters with Optuna for the SAME model family that AutoGluon chose.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset used for modeling.
    target_column : str
        Target column name.
    model_family : str
        Coarse family from AutoGluon ("lightgbm", "xgboost", "catboost", etc.).
    problem_type : str
        "regression", "binary", or "multiclass".
    eval_metric : str
        "root_mean_squared_error", "roc_auc", or "log_loss".
        We always define the objective so that LOWER is better.
    n_trials : int
        Number of Optuna trials.
    use_cv : Optional[bool]
        Whether to use cross-validation. If None, automatically determined based on dataset size.
    cv_folds : Optional[int]
        Number of folds for cross-validation. If None, automatically determined based on dataset size.

    Returns
    -------
    dict with keys:
        - best_model_class : str
        - best_params      : Dict[str, Any]
        - best_model       : fitted model (trained on ALL data)
        - best_objective   : float (value minimized by Optuna)
        - best_eval_score  : float (human-friendly metric: RMSE / AUC / log_loss)
        - metric_name      : str (original metric name)
        - study            : Optuna study object
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

    # Auto-determine CV strategy if not provided
    n_samples = len(df)
    if use_cv is None or cv_folds is None:
        use_cv, cv_folds = suggest_cv_strategy(n_samples)

    pt = (problem_type or "").lower()
    is_reg = pt == "regression"
    is_clf = pt in ("binary", "multiclass")

    # Keep both the original name and a lower-cased internal version
    metric_name = eval_metric
    metric_internal = (eval_metric or "").lower()

    if use_cv:
        if is_clf and y.nunique() > 1:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        stratify = y if is_clf and y.nunique() > 1 else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

    def objective(trial: optuna.Trial) -> float:
        ModelClass = _choose_model_class(model_family, problem_type)
        params = _build_model_params(trial, ModelClass, model_family, problem_type)

        # Special defaults for CatBoost
        if ModelClass in (cb.CatBoostRegressor, cb.CatBoostClassifier):
            params.setdefault("verbose", 0)
            params.setdefault("random_state", 42)

        if use_cv:
            model = ModelClass(**params)
            
            if is_reg:
                scorer = make_scorer(
                    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                    greater_is_better=False
                )
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                return -scores.mean()
            
            elif metric_internal == "roc_auc" and pt == "binary":
                # Use built-in 'roc_auc' scorer which handles predict_proba automatically
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                return 1.0 - scores.mean()
            
            else:
                # Use built-in 'neg_log_loss' scorer which handles predict_proba automatically
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
                return -scores.mean()
        
        else:
            model = ModelClass(**params)
            model.fit(X_train, y_train)

            if is_reg:
                y_pred = model.predict(X_valid)
                mse = mean_squared_error(y_valid, y_pred)
                rmse = np.sqrt(mse)
                return rmse

            # Classification: need probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_valid)
            else:
                y_pred = model.predict(X_valid)
                classes = np.unique(y_train)
                proba = np.zeros((len(y_pred), len(classes)))
                for i, c in enumerate(classes):
                    proba[:, i] = (y_pred == c).astype(float)

            # Binary AUC → we MINIMIZE (1 - AUC)
            if metric_internal == "roc_auc" and pt == "binary":
                auc = roc_auc_score(y_valid, proba[:, 1])
                return 1.0 - auc

            # Default for classification: log_loss
            return log_loss(y_valid, proba)

    study = optuna.create_study(direction="minimize")

    best_value_history = []
    trial_times = []
    start_time = time.time()

    def callback(study, trial):
        """Callback to track convergence"""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            return  
        
        best_value_history.append(study.best_value)
        trial_times.append(time.time() - start_time)
        
        # Early stopping if no improvement
        if len(best_value_history) > patience:
            recent_best = min(best_value_history[-patience:])
            if abs(study.best_value - recent_best) < min_improvement:
                study.stop()
    
    # Run optimization
    if time_limit is not None:
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=time_limit,
            callbacks=[callback]
        )
    else:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    # Calculate convergence metrics
    actual_time = time.time() - start_time
    num_trials_completed = len(study.trials)
    
    # --- Check if we have any completed trials first ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        raise ValueError("No completed trials found. All trials failed. Check your objective function, data, and model parameters.")
    
    # Check if converged
    if len(best_value_history) >= 10:
        recent_improvement = abs(best_value_history[-1] - best_value_history[-10])
        early_best = min(best_value_history[:10])
        total_improvement = abs(best_value_history[-1] - early_best)
        improvement_rate = recent_improvement / total_improvement if total_improvement > 0 else 0
    else:
        improvement_rate = 1.0  # Still exploring
    
    convergence_metrics = {
        "n_trials_set": n_trials,
        "n_trials_completed": num_trials_completed,
        "time_limit_set": time_limit,
        "actual_time": actual_time,
        "time_utilization": actual_time / time_limit if time_limit else None,
        "improvement_rate": improvement_rate,
        "converged": improvement_rate < 0.05,
        "best_trial_number": study.best_trial.number,
        "best_found_at_pct": study.best_trial.number / num_trials_completed if num_trials_completed > 0 else 0,
        "used_cv": use_cv,
        "cv_folds": cv_folds if use_cv else None,
    }

    # --- Convert Optuna objective back into human-friendly score ---
    
    best_objective = study.best_value
    if is_reg:
        best_eval_score = best_objective
    elif metric_internal == "roc_auc" and pt == "binary":
        best_eval_score = 1.0 - best_objective
    else:
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
        "convergence_metrics": convergence_metrics,
        "optimization_history": {
            "best_values": best_value_history,
            "trial_times": trial_times,
        }
    }