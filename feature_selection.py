import pandas as pd
from typing import Tuple, List, Optional


def select_features_by_importance(
    df: pd.DataFrame,
    target_column: str,
    feature_importance: Optional[pd.DataFrame],
    importance_threshold: float = 0.0,
    top_k: Optional[int] = None,
    cumulative_importance_threshold: float = 0.99,
    max_features: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reduce features based on AutoGluon feature importance.

    Strategy:
    1. If no feature_importance is available → keep all features except target.
    2. Optionally drop features with raw importance <= importance_threshold.
    3. Compute relative importance and cumulative importance.
    4. If top_k is provided → keep top_k features by importance.
       Else → keep smallest set of features whose cumulative importance
              >= cumulative_importance_threshold (e.g. 95%).
    5. If max_features is provided → cap the number of features to max_features.
    6. Always keep the target column and ensure at least one feature is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe INCLUDING the target column.
    target_column : str
        Name of the target column.
    feature_importance : pd.DataFrame or None
        Output of predictor.feature_importance() from AutoGluon.
        Expected: index = feature names, column 'importance'.
    importance_threshold : float
        Minimum *raw* importance to keep a feature before applying
        relative/cumulative logic. Default 0.0 (keep all non-negative).
    top_k : int, optional
        If provided, keep only the top_k most important features (after
        applying importance_threshold).
    cumulative_importance_threshold : float
        Cumulative relative importance cutoff (e.g. 0.95 = keep features
        that together explain 95% of total importance).
    max_features : int, optional
        Hard upper bound on number of features to keep.

    Returns
    -------
    reduced_df : pd.DataFrame
        DataFrame containing only the selected features plus target column.
    selected_features : List[str]
        List of selected feature names (WITHOUT the target column).
    """
    # --- No importance → keep everything except target ---
    if feature_importance is None or feature_importance.empty:
        selected = [c for c in df.columns if c != target_column]
        return df.copy(), selected

    if "importance" not in feature_importance.columns:
        raise ValueError("feature_importance must contain an 'importance' column.")

    # Sort by importance (highest first)
    fi_sorted = feature_importance.sort_values("importance", ascending=False).copy()

    # --- 1) Optional raw-importance filter ---
    if importance_threshold is not None:
        filtered = fi_sorted[fi_sorted["importance"] > importance_threshold]
        if not filtered.empty:
            fi_sorted = filtered

    # Safety: if still empty for some reason, revert to original sorted order
    if fi_sorted.empty:
        fi_sorted = feature_importance.sort_values("importance", ascending=False).copy()

    # --- 2) Relative + cumulative importance ---
    total_importance = fi_sorted["importance"].sum()
    if total_importance <= 0:
        # Degenerate case: all zero or negative importance → keep all features except target
        kept_features = [c for c in df.columns if c != target_column]
    else:
        fi_sorted["rel_importance"] = fi_sorted["importance"] / total_importance
        fi_sorted["cum_importance"] = fi_sorted["rel_importance"].cumsum()

        # --- 3) Selection logic ---
        if top_k is not None:
            kept_features = fi_sorted.head(top_k).index.tolist()
        else:
            # Keep smallest set that reaches cumulative_importance_threshold
            kept = fi_sorted[fi_sorted["cum_importance"] <= cumulative_importance_threshold]
            if kept.empty:
                # If threshold is too low/high and nothing is selected, keep all
                kept_features = fi_sorted.index.tolist()
            else:
                kept_features = kept.index.tolist()

        # --- 4) Optional cap on number of features ---
        if max_features is not None and len(kept_features) > max_features:
            kept_features = kept_features[:max_features]

    # --- 5) Intersect with df columns, exclude target ---
    kept_features = [
        f for f in kept_features
        if f in df.columns and f != target_column
    ]

    # Final safety: never end up with no features
    if not kept_features:
        kept_features = [c for c in df.columns if c != target_column]

    selected_columns = kept_features + [target_column]
    reduced_df = df[selected_columns].copy()

    return reduced_df, kept_features
