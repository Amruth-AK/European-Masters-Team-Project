# preprocessing_pipeline.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from preprocessing_function import (
    _calculate_intelligent_replace_ratio,
    _select_features_to_replace
)

from preprocessing_registry import FUNC_MAP

# ---------------------------------------------------------------------
#  Train-only vs value-level transformations
# ---------------------------------------------------------------------
# These functions REMOVE ROWS and must NEVER run on test/prediction data.
TRAIN_ONLY_FUNCTIONS = {
    "delete_missing_rows",
    "delete_duplicates",
    "remove_outliers_iqr",
}


@dataclass
class FittedStep:
    """
    A fitted preprocessing step that knows how to transform new data.

    - name:       name of the preprocessing function (string)
    - params:     learned parameters (means, stds, bounds, mappings, etc.)
    - kwargs:     original kwargs that were passed at fit-time
    - train_only: if True, this step is applied only on training data and
                  skipped on test/prediction data (e.g. row deletions)
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    train_only: bool = False


def _coerce_to_list(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    if val is None:
        return []
    return [val]


# =====================================================================
#   NUMERIC IMPUTERS (fit / transform)
# =====================================================================

def _fit_imputer_step(df: pd.DataFrame, name: str, kwargs: Dict[str, Any]):
    cols = _coerce_to_list(kwargs.get("columns"))
    df_proc = df.copy()
    mapping: Dict[str, Any] = {}

    for col in cols:
        if col not in df_proc.columns:
            continue

        if name == "impute_mean":
            val = df_proc[col].mean()
        elif name == "impute_median":
            val = df_proc[col].median()
        else:  # "impute_mode"
            mode_series = df_proc[col].mode(dropna=True)
            val = mode_series.iloc[0] if not mode_series.empty else None

        mapping[col] = val
        if val is not None:
            df_proc[col] = df_proc[col].fillna(val)

    fitted = FittedStep(name=name, params={"values": mapping}, kwargs=kwargs)
    return df_proc, fitted


def _transform_imputer_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    values = step.params.get("values", {})
    for col, val in values.items():
        if col in df_proc.columns and val is not None:
            df_proc[col] = df_proc[col].fillna(val)
    return df_proc


# =====================================================================
#   NUMERIC SCALERS (fit / transform)
# =====================================================================

def _fit_scaler_step(df: pd.DataFrame, name: str, kwargs: Dict[str, Any]):
    cols = _coerce_to_list(kwargs.get("column"))
    df_proc = df.copy()
    params: Dict[str, Any] = {}

    if name == "standard_scaler":
        col_stats: Dict[str, Dict[str, float]] = {}
        for col in cols:
            if col not in df_proc.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df_proc[col]):
                continue
            mean = df_proc[col].mean()
            std = df_proc[col].std()
            if std == 0 or pd.isna(std):
                std = 1.0
            df_proc[col] = (df_proc[col] - mean) / std
            col_stats[col] = {"mean": float(mean), "std": float(std)}

        params["scaler"] = "standard"
        params["stats"] = col_stats

    elif name == "minmax_scaler":
        feature_range = kwargs.get("feature_range", (0, 1))
        min_r, max_r = feature_range
        col_stats: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            if col not in df_proc.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df_proc[col]):
                continue
            col_min = df_proc[col].min()
            col_max = df_proc[col].max()
            if col_max == col_min or pd.isna(col_max) or pd.isna(col_min):
                scale = 1.0
            else:
                scale = (max_r - min_r) / (col_max - col_min)
            df_proc[col] = (df_proc[col] - col_min) * scale + min_r
            col_stats[col] = {
                "min": float(col_min),
                "max": float(col_max),
                "feature_range": (float(min_r), float(max_r)),
            }

        params["scaler"] = "minmax"
        params["stats"] = col_stats

    fitted = FittedStep(name=name, params=params, kwargs=kwargs)
    return df_proc, fitted


def _transform_scaler_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    scaler_type = step.params.get("scaler")
    stats = step.params.get("stats", {})

    if scaler_type == "standard":
        for col, s in stats.items():
            if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
                mean = s["mean"]
                std = s["std"] or 1.0
                df_proc[col] = (df_proc[col] - mean) / std

    elif scaler_type == "minmax":
        for col, s in stats.items():
            if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
                col_min = s["min"]
                col_max = s["max"]
                min_r, max_r = s["feature_range"]
                if col_max == col_min:
                    scale = 1.0
                else:
                    scale = (max_r - min_r) / (col_max - col_min)
                df_proc[col] = (df_proc[col] - col_min) * scale + min_r

    return df_proc

def _fit_robust_scaler_step(df: pd.DataFrame, name: str, kwargs: Dict[str, Any]):
    cols = _coerce_to_list(kwargs.get("column"))
    df_proc = df.copy()
    q_range = kwargs.get("quantile_range", (25.0, 75.0))
    q_min, q_max = q_range
    
    params = {"stats": {}}
    
    for col in cols:
        if col not in df_proc.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_proc[col]):
            continue
            
        median = float(df_proc[col].median())
        q1 = float(df_proc[col].quantile(q_min / 100.0))
        q3 = float(df_proc[col].quantile(q_max / 100.0))
        iqr = q3 - q1
        
        # Apply to train data
        if iqr == 0:
            df_proc[col] = df_proc[col] - median
        else:
            df_proc[col] = (df_proc[col] - median) / iqr
            
        params["stats"][col] = {
            "median": median,
            "iqr": iqr
        }

    fitted = FittedStep(name=name, params=params, kwargs=kwargs)
    return df_proc, fitted


def _transform_robust_scaler_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    stats = step.params.get("stats", {})
    
    for col, stat in stats.items():
        if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
            median = stat["median"]
            iqr = stat["iqr"]
            
            if iqr == 0:
                df_proc[col] = df_proc[col] - median
            else:
                df_proc[col] = (df_proc[col] - median) / iqr
                
    return df_proc


# =====================================================================
#   OUTLIER CLIPPING (fit / transform)
# =====================================================================

def _fit_clip_outliers_step(
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    analysis_results: Optional[Dict[str, Any]],
):
    df_proc = df.copy()
    # Backward-compatible: suggestion currently passes "column"
    cols_arg = kwargs.get("column") or kwargs.get("columns")
    cols = _coerce_to_list(cols_arg)
    if not cols:
        cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info = (analysis_results or {}).get("outlier_info", {})
    bounds: Dict[str, Dict[str, float]] = {}

    for col in cols:
        if col not in df_proc.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_proc[col]):
            continue

        col_info = outlier_info.get(col)
        if col_info:
            lower = col_info.get("lower_bound")
            upper = col_info.get("upper_bound")
        else:
            q1 = df_proc[col].quantile(0.25)
            q3 = df_proc[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        if pd.isna(lower) or pd.isna(upper):
            continue

        bounds[col] = {"lower": float(lower), "upper": float(upper)}
        df_proc[col] = df_proc[col].clip(lower=lower, upper=upper)

    fitted = FittedStep(name="clip_outliers_iqr", params={"bounds": bounds}, kwargs=kwargs)
    return df_proc, fitted


def _transform_clip_outliers_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    bounds = step.params.get("bounds", {})
    for col, b in bounds.items():
        if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
            df_proc[col] = df_proc[col].clip(lower=b["lower"], upper=b["upper"])
    return df_proc


# =====================================================================
#   WINSORIZATION (fit / transform)
# =====================================================================

def _fit_winsorize_step(df: pd.DataFrame, kwargs: Dict[str, Any]):
    df_proc = df.copy()
    col = kwargs.get("column")
    limits = kwargs.get("limits", (0.01, 0.99))
    
    params = {}
    if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
        lower = float(df_proc[col].quantile(limits[0]))
        upper = float(df_proc[col].quantile(limits[1]))
        params = {"lower": lower, "upper": upper}
        df_proc[col] = df_proc[col].clip(lower=lower, upper=upper)
        
    fitted = FittedStep(name="winsorize_column", params=params, kwargs=kwargs)
    return df_proc, fitted

def _transform_winsorize_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    col = step.kwargs.get("column")
    lower = step.params.get("lower")
    upper = step.params.get("upper")
    
    if col in df_proc.columns and lower is not None and upper is not None:
        df_proc[col] = df_proc[col].clip(lower=lower, upper=upper)
        
    return df_proc


# =====================================================================
#   POWER TRANSFORM (fit / transform)
# =====================================================================

def _fit_power_transform_step(df: pd.DataFrame, kwargs: Dict[str, Any]):
    from sklearn.preprocessing import PowerTransformer
    df_proc = df.copy()
    col = kwargs.get("column")
    method = kwargs.get("method", "yeo-johnson")
    
    params = {}
    if col in df_proc.columns and pd.api.types.is_numeric_dtype(df_proc[col]):
        # Fit logic
        pt = PowerTransformer(method=method, standardize=False)
        vals = df_proc[col].dropna().values.reshape(-1, 1)
        
        if len(vals) > 0:
            pt.fit(vals)
            # Apply to current df
            mask = df_proc[col].notna()
            if mask.sum() > 0:
                transformed = pt.transform(df_proc.loc[mask, col].values.reshape(-1, 1)).flatten()
                df_proc.loc[mask, col] = transformed
            params = {"model": pt}

    fitted = FittedStep(name="apply_power_transform", params=params, kwargs=kwargs)
    return df_proc, fitted

def _transform_power_transform_step(df: pd.DataFrame, step: FittedStep):
    df_proc = df.copy()
    col = step.kwargs.get("column")
    pt = step.params.get("model")
    
    if col in df_proc.columns and pt is not None:
        mask = df_proc[col].notna()
        if mask.sum() > 0:
            try:
                vals = df_proc.loc[mask, col].values.reshape(-1, 1)
                df_proc.loc[mask, col] = pt.transform(vals).flatten()
            except Exception:
                pass 
                
    return df_proc

# =====================================================================
#   CORRELATION-BASED FEATURES (fit / transform)
# =====================================================================

def _decode_corr_feature_name(name: str):
    """
    Decode feature name patterns like:
        col1_x_col2, col1_div_col2, col1_minus_col2, col1_plus_col2, col1_interact_col2
    into (op, col1, col2).
    """
    patterns = [
        ("product", "_x_"),
        ("ratio", "_div_"),
        ("difference", "_minus_"),
        ("sum", "_plus_"),
        ("interaction", "_interact_"),
    ]
    for op, token in patterns:
        if token in name:
            col1, col2 = name.split(token, 1)
            return op, col1, col2
    return None, None, None


def _fit_corr_features_step(
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    analysis_results: Optional[Dict[str, Any]],
):
    """
    Fit-time: actually call the existing create_features_from_correlation_analysis
    to decide which interaction features to create, then *record* the formulas
    so we can re-create exactly the same columns on test data.

    We do NOT re-run correlation logic on test; we just apply the formulas.
    """
    df_proc = df.copy()
    base_df = df_proc.copy()  # used for training means in interaction features

    # Ensure analysis_results is passed through if provided
    if analysis_results is not None:
        kwargs = dict(kwargs)  # shallow copy
        kwargs.setdefault("analysis_results", analysis_results)

    func = FUNC_MAP.get("create_features_from_correlation_analysis")
    if func is None:
        # Nothing we can do; return original df and an empty step
        return df_proc, FittedStep(
            name="create_features_from_correlation_analysis",
            params={"features": []},
            kwargs=kwargs,
        )

    old_cols = list(df_proc.columns)
    try:
        df_proc = func(df_proc, **kwargs)
    except Exception as e:
        print(f"[fit_preprocessing_pipeline] Error in create_features_from_correlation_analysis: {e}")
        return df, FittedStep(
            name="create_features_from_correlation_analysis",
            params={"features": []},
            kwargs=kwargs,
        )

    new_cols = [c for c in df_proc.columns if c not in old_cols]
    features: List[Dict[str, Any]] = []

    for new_col in new_cols:
        # Robust decoding: try all patterns and all split positions
        found_match = False
        best_op = None
        best_c1 = None
        best_c2 = None
        
        # Patterns from _decode_corr_feature_name
        patterns = [
            ("product", "_x_"),
            ("ratio", "_div_"),
            ("difference", "_minus_"),
            ("sum", "_plus_"),
            ("interaction", "_interact_"),
        ]
        
        valid_cols = set(df_proc.columns)
        
        for op, token in patterns:
            if token not in new_col:
                continue
                
            # Try splitting at every occurrence of the token
            parts = new_col.split(token)
            # We need to reconstruct c1 and c2 from parts
            # e.g. A_x_B_x_C with token _x_ -> parts [A, B, C]
            # Split could be (A, B_x_C) or (A_x_B, C)
            
            # Optimization: The generator usually puts the complex feature first (A_x_B) and simple second (C)
            # So we iterate splits.
            # However, simple split(token) consumes ALL tokens.
            # We want to find a partition.
            
            # Let's use string find/rfind approach
            start = 0
            while True:
                idx = new_col.find(token, start)
                if idx == -1:
                    break
                
                c1 = new_col[:idx]
                c2 = new_col[idx + len(token):]
                
                if c1 in valid_cols and c2 in valid_cols:
                    best_op = op
                    best_c1 = c1
                    best_c2 = c2
                    found_match = True
                    break
                
                start = idx + len(token)
            
            if found_match:
                break
        
        if not found_match:
            continue

        feat_spec: Dict[str, Any] = {
            "name": new_col,
            "op": best_op,
            "col1": best_c1,
            "col2": best_c2,
        }

        if best_op == "interaction":
            # Store training means so we can use the same centering on test
            # Use df_proc because c1/c2 might be newly created features (not in base_df)
            feat_spec["col1_mean"] = float(df_proc[best_c1].mean())
            feat_spec["col2_mean"] = float(df_proc[best_c2].mean())

        features.append(feat_spec)

    fitted = FittedStep(
        name="create_features_from_correlation_analysis",
        params={"features": features},
        kwargs=kwargs,
    )
    return df_proc, fitted


def _transform_corr_features_step(df: pd.DataFrame, step: FittedStep):
    """
    Test-time: re-create the exact same interaction features using
    the recorded formulas from training.
    """
    df_proc = df.copy()
    features = step.params.get("features", []) or []

    for spec in features:
        name = spec["name"]
        op = spec["op"]
        col1 = spec["col1"]
        col2 = spec["col2"]

        if col1 not in df_proc.columns or col2 not in df_proc.columns:
            # Cannot compute if base columns are missing
            continue

        s1 = df_proc[col1]
        s2 = df_proc[col2]

        if op == "product":
            df_proc[name] = s1 * s2
        elif op == "ratio":
            df_proc[name] = np.where(s2 != 0, s1 / s2, 0)
        elif op == "difference":
            df_proc[name] = s1 - s2
        elif op == "sum":
            df_proc[name] = s1 + s2
        elif op == "interaction":
            m1 = spec.get("col1_mean", 0.0)
            m2 = spec.get("col2_mean", 0.0)
            df_proc[name] = (s1 - m1) * (s2 - m2)

    return df_proc


# =====================================================================
#   FASTICA (fit / transform) - simplified, stateful version
# =====================================================================

def _fit_fastica_step(
    df: pd.DataFrame,
    kwargs: Dict[str, Any],
    analysis_results: Optional[Dict[str, Any]],
):
    """
    Fit-time: learn a FastICA decomposition on numeric features (excluding target),
    and append ICA components as new columns. We store the fitted scaler and ICA
    so we can apply the *same* transform on test data.

    This is a simplified but fully stateful variant of apply_fastica.
    """
    df_proc = df.copy()

    target_column = kwargs.get("target_column")
    exclude_columns = _coerce_to_list(kwargs.get("exclude_columns"))
    random_state = kwargs.get("random_state", 42)
    max_iter = kwargs.get("max_iter", 1000)
    whiten = kwargs.get("whiten", "unit-variance")
    n_components = kwargs.get("n_components", None)

    # 1) Select numeric columns
    numerical_cols = [
        col for col in df_proc.columns
        if pd.api.types.is_numeric_dtype(df_proc[col])
    ]
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    numerical_cols = [c for c in numerical_cols if c not in exclude_columns]

    if not numerical_cols:
        print("[_fit_fastica_step] No numeric columns available for FastICA.")
        fitted = FittedStep(
            name="apply_fastica",
            params={"numeric_cols": [], "ica": None, "scaler": None, "component_names": []},
            kwargs=kwargs,
        )
        return df_proc, fitted

    # 2) Choose n_components if not provided
    n_num = len(numerical_cols)
    if n_components is None:
        n_components = min(5, max(1, n_num // 2))
    n_components = max(1, min(n_components, n_num))

    # 3) Prepare data, fill missing
    X = df_proc[numerical_cols].copy()
    if X.isnull().any().any():
        X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Handle whiten parameter similar to original implementation
    if isinstance(whiten, bool):
        if whiten:
            whiten_param = "unit-variance"
        else:
            whiten_param = False
    else:
        whiten_param = whiten

    try:
        ica = FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            whiten=whiten_param,
        )
        S = ica.fit_transform(X_scaled)
    except Exception as e:
        print(f"❌ FastICA (pipeline) failed: {e}")
        fitted = FittedStep(
            name="apply_fastica",
            params={"numeric_cols": [], "ica": None, "scaler": None, "component_names": []},
            kwargs=kwargs,
        )
        return df_proc, fitted

    component_names = [f"ICA_{i}" for i in range(n_components)]
    for i, cname in enumerate(component_names):
        df_proc[cname] = S[:, i]

    # 5) Handle mode (hybrid/replace/add)
    mode = kwargs.get("mode", "hybrid")
    replace_ratio = kwargs.get("replace_ratio")
    
    cols_to_drop = []
    
    if mode == 'replace':
        cols_to_drop = numerical_cols
        print(f"[_fit_fastica_step] Mode 'replace': Dropping {len(cols_to_drop)} original features.")
        
    elif mode == 'hybrid':
        # Calculate ratio if not provided
        if replace_ratio is None:
            replace_ratio = _calculate_intelligent_replace_ratio(
                df_proc,
                numerical_cols,
                n_components,
                analysis_results=analysis_results
            )
            
        # Identify features to replace
        n_to_replace = int(len(numerical_cols) * replace_ratio)
        cols_to_replace, cols_to_keep = _select_features_to_replace(
            df_proc,
            numerical_cols,
            n_to_replace,
            analysis_results
        )
        cols_to_drop = cols_to_replace
        print(f"[_fit_fastica_step] Mode 'hybrid': Dropping {len(cols_to_drop)} features (ratio={replace_ratio:.2f}).")
        
    # Apply drops
    if cols_to_drop:
        df_proc = df_proc.drop(columns=cols_to_drop)

    fitted = FittedStep(
        name="apply_fastica",
        params={
            "numeric_cols": numerical_cols,
            "scaler": scaler,
            "ica": ica,
            "component_names": component_names,
            "cols_to_drop": cols_to_drop, # Store dropped columns to replicate on test
        },
        kwargs=kwargs,
    )
    return df_proc, fitted


def _transform_fastica_step(df: pd.DataFrame, step: FittedStep):
    """
    Test-time: reuse the stored scaler + ICA model to create the same ICA
    components on new data.
    """
    df_proc = df.copy()
    params = step.params or {}

    numeric_cols = params.get("numeric_cols") or []
    scaler = params.get("scaler")
    ica = params.get("ica")
    component_names = params.get("component_names") or []

    if not numeric_cols or scaler is None or ica is None:
        # Nothing to do
        return df_proc

    # Ensure all required columns are present
    missing = [c for c in numeric_cols if c not in df_proc.columns]
    if missing:
        print(f"[apply_fitted_pipeline] FastICA: missing columns {missing}, skipping transform.")
        return df_proc

    X = df_proc[numeric_cols].copy()
    if X.isnull().any().any():
        X = X.fillna(X.median())

    try:
        X_scaled = scaler.transform(X)
        S = ica.transform(X_scaled)
    except Exception as e:
        print(f"❌ FastICA transform failed: {e}")
        return df_proc

    for i, cname in enumerate(component_names):
        df_proc[cname] = S[:, i]

    # Drop columns if specified (for hybrid/replace mode)
    cols_to_drop = params.get("cols_to_drop", [])
    if cols_to_drop:
        # Only drop columns that exist
        existing_drops = [c for c in cols_to_drop if c in df_proc.columns]
        if existing_drops:
            df_proc = df_proc.drop(columns=existing_drops)

    return df_proc


# =====================================================================
#   INDICATORS (fit / transform)
# =====================================================================

def _fit_indicator_step(df: pd.DataFrame, kwargs: Dict[str, Any]):
    """
    Fit: Record which columns we are adding indicators for.
    Apply: Add the indicators to the training data.
    """
    df_proc = df.copy()
    cols = _coerce_to_list(kwargs.get("columns"))
    
    # Verify columns exist
    valid_cols = [c for c in cols if c in df_proc.columns]
    
    for col in valid_cols:
        df_proc[f"{col}_is_missing"] = df_proc[col].isnull().astype(int)
        
    fitted = FittedStep(name="add_missing_indicator", params={"columns": valid_cols}, kwargs=kwargs)
    return df_proc, fitted

def _transform_indicator_step(df: pd.DataFrame, step: FittedStep):
    """
    Transform: Add the same indicator columns. 
    If source column has no NaNs in test, indicator is all 0s.
    """
    df_proc = df.copy()
    cols = step.params.get("columns", [])
    
    for col in cols:
        if col in df_proc.columns:
            df_proc[f"{col}_is_missing"] = df_proc[col].isnull().astype(int)
        else:
            # Edge case: Source column missing? Fill with 0s or skip?
            # Better to skip or fill 0. Let's fill 0 to maintain shape if possible, 
            # but usually safe to skip if source is gone.
            pass
            
    return df_proc





# =====================================================================
#   FIT / TRANSFORM PIPELINE
# =====================================================================

def fit_preprocessing_pipeline(
    df: pd.DataFrame,
    pipeline: List[Dict[str, Any]],
    analysis_results: Optional[Dict[str, Any]] = None,
) -> (pd.DataFrame, List[FittedStep]):
    """
    Train-time: run pipeline on df while capturing learned parameters
    (means, stds, bounds, etc.) so we can re-use them on test/prediction.

    - Numeric imputation, scaling, and outlier clipping are handled with
      dedicated fit/transform logic.
    - Row-removal steps (delete_missing_rows, delete_duplicates,
      remove_outliers_iqr) are marked as train-only and will be skipped
      on test/prediction data.
    - Correlation-based feature creation and FastICA are treated as
      *stateful* transforms: we record exactly what was done so we can
      replicate it on test data without re-running selection logic.
    - Other steps fall back to calling the original FUNC_MAP functions
      directly and are treated as stateless.
    """
    df_proc = df.copy()
    fitted_steps: List[FittedStep] = []

    for step in pipeline:
        name = step.get("function_to_call")
        kwargs = (step.get("kwargs") or {}).copy()

        # Ensure analysis_results is present where needed
        if name == "create_features_from_correlation_analysis" and analysis_results is not None:
            kwargs.setdefault("analysis_results", analysis_results)
        if name == "clip_outliers_iqr" and analysis_results is not None:
            kwargs.setdefault("analysis_results", analysis_results)
        if name == "apply_fastica" and analysis_results is not None:
            kwargs.setdefault("analysis_results", analysis_results)

        # --- Stateful handlers ---
        if name in ("impute_mean", "impute_median", "impute_mode", "impute_constant"):
            # Special handling for constant to pass fill_value into mapping
            if name == "impute_constant":
                # Reuse _fit_imputer_step logic but inject constant
                cols = _coerce_to_list(kwargs.get("columns"))
                val = kwargs.get("fill_value", -1)
                df_proc = df_proc.copy()
                mapping = {c: val for c in cols if c in df_proc.columns}
                for c, v in mapping.items():
                    df_proc[c] = df_proc[c].fillna(v)
                fitted_steps.append(FittedStep(name=name, params={"values": mapping}, kwargs=kwargs))
            else:
                df_proc, fitted = _fit_imputer_step(df_proc, name, kwargs)
                fitted_steps.append(fitted)
            continue

        # --- ADD INDICATOR BLOCK ---
        if name == "add_missing_indicator": # <--- NEW
            df_proc, fitted = _fit_indicator_step(df_proc, kwargs)
            fitted_steps.append(fitted)
            continue


        if name in ("standard_scaler", "minmax_scaler"):
            df_proc, fitted = _fit_scaler_step(df_proc, name, kwargs)
            fitted_steps.append(fitted)
            continue
        
        if name == "robust_scaler": # <--- NEW
            df_proc, fitted = _fit_robust_scaler_step(df_proc, name, kwargs)
            fitted_steps.append(fitted)
            continue

        if name == "winsorize_column": # <--- NEW
            df_proc, fitted = _fit_winsorize_step(df_proc, kwargs)
            fitted_steps.append(fitted)
            continue

        if name == "apply_power_transform": # <--- NEW
            df_proc, fitted = _fit_power_transform_step(df_proc, kwargs)
            fitted_steps.append(fitted)
            continue

        if name == "clip_outliers_iqr":
            df_proc, fitted = _fit_clip_outliers_step(df_proc, kwargs, analysis_results)
            fitted_steps.append(fitted)
            continue

        if name == "create_features_from_correlation_analysis":
            df_proc, fitted = _fit_corr_features_step(df_proc, kwargs, analysis_results)
            fitted_steps.append(fitted)
            continue

        if name == "apply_fastica":
            df_proc, fitted = _fit_fastica_step(df_proc, kwargs, analysis_results)
            fitted_steps.append(fitted)
            continue

        if name == "combine_categorical_features":
            # This is a stateless transform, but we need to record it to replay it
            # It doesn't learn parameters, but it changes the schema.
            # We treat it as a "stateless" step that is just re-executed.
            try:
                func = FUNC_MAP.get(name)
                df_proc = func(df_proc, **kwargs)
                fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs))
            except Exception as e:
                print(f"[fit_preprocessing_pipeline] Error in {name}: {e}")
                fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs))
            continue

        # --- Train-only row-removal steps ---
        if name in TRAIN_ONLY_FUNCTIONS:
            func = FUNC_MAP.get(name)
            if func is not None:
                try:
                    df_proc = func(df_proc, **kwargs)
                except Exception as e:
                    print(f"[fit_preprocessing_pipeline] Error in train-only step {name}: {e}")
            # Record step as train_only so it will be skipped on test
            fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs, train_only=True))
            continue

        # --- Default: stateless step ---
        func = FUNC_MAP.get(name)
        if func is not None:
            try:
                df_proc = func(df_proc, **kwargs)
                fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs))
            except Exception as e:
                print(f"[fit_preprocessing_pipeline] Error in {name}: {e}")
                fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs))
        else:
            print(f"[fit_preprocessing_pipeline] Unknown step '{name}', skipping.")
            fitted_steps.append(FittedStep(name=name, params={}, kwargs=kwargs))

    return df_proc, fitted_steps


def apply_fitted_pipeline(
    df: pd.DataFrame,
    fitted_steps: Optional[List[FittedStep]],
    analysis_results: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Test/prediction-time: apply previously fitted steps to new data.

    IMPORTANT:
    - Steps marked train_only=True, or whose names are in TRAIN_ONLY_FUNCTIONS,
      are *skipped* here to avoid dropping rows from test data.
    """
    if not fitted_steps:
        return df.copy()

    df_proc = df.copy()

    for step in fitted_steps:
        name = step.name
        kwargs = (step.kwargs or {}).copy()

        # Skip train-only steps on test/prediction
        if step.train_only or name in TRAIN_ONLY_FUNCTIONS:
            continue

        if name == "create_features_from_correlation_analysis" and analysis_results is not None:
            kwargs.setdefault("analysis_results", analysis_results)
        if name == "clip_outliers_iqr" and analysis_results is not None and "bounds" not in step.params:
            kwargs.setdefault("analysis_results", analysis_results)
        if name == "apply_fastica" and analysis_results is not None:
            kwargs.setdefault("analysis_results", analysis_results)

        # --- Stateful handlers ---
        if name in ("impute_mean", "impute_median", "impute_mode", "impute_constant"):
            df_proc = _transform_imputer_step(df_proc, step)
            continue

        # --- ADD INDICATOR BLOCK ---
        if name == "add_missing_indicator": 
            df_proc = _transform_indicator_step(df_proc, step)
            continue


        if name in ("standard_scaler", "minmax_scaler"):
            df_proc = _transform_scaler_step(df_proc, step)
            continue

        if name == "robust_scaler": # <--- NEW
            df_proc = _transform_robust_scaler_step(df_proc, step)
            continue

        if name == "winsorize_column": # <--- NEW
            df_proc = _transform_winsorize_step(df_proc, step)
            continue
            
        if name == "apply_power_transform": # <--- NEW
            df_proc = _transform_power_transform_step(df_proc, step)
            continue

        if name == "clip_outliers_iqr":
            df_proc = _transform_clip_outliers_step(df_proc, step)
            continue

        if name == "create_features_from_correlation_analysis":
            df_proc = _transform_corr_features_step(df_proc, step)
            continue

        if name == "apply_fastica":
            df_proc = _transform_fastica_step(df_proc, step)
            continue

        elif name == "combine_categorical_features":
            func = FUNC_MAP.get(name)
            if func:
                try:
                    df_proc = func(df_proc, **kwargs)
                except Exception as e:
                    print(f"[apply_fitted_pipeline] Error in {name}: {e}")
            continue

        # --- Default: stateless re-apply ---
        # Generic fallback for stateless transforms (if any others exist)
        func = FUNC_MAP.get(name)
        if func is not None:
            try:
                df_proc = func(df_proc, **kwargs)
            except Exception as e:
                print(f"[apply_fitted_pipeline] Error in {name}: {e}")
        else:
            print(f"[apply_fitted_pipeline] Unknown step '{name}', skipping.")

    return df_proc
