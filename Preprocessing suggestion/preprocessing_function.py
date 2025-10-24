import pandas as pd
import numpy as np
from typing import Union, List


# ============================================================================
# Numerical Features - Zhiqi
# ============================================================================

def standard_scaler(df: pd.DataFrame, column: Union[str, List[str]]) -> pd.DataFrame:
    """
    Apply standard scaling (z-score normalization) to numerical column(s).
    
    Formula: z = (x - mean) / std
    
    Args:
        df: Input DataFrame
        column: Column name(s) to scale. Can be a single string or list of strings.
    
    Returns:
        DataFrame with scaled column(s)
    
    Example:
        >>> df = standard_scaler(df, 'Age')
        >>> df = standard_scaler(df, ['Age', 'Fare'])
    """
    df = df.copy()
    
    # Convert single column to list for uniform processing
    columns = [column] if isinstance(column, str) else column
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric for standard scaling")
        
        # Calculate mean and std, ignoring NaN values
        mean = df[col].mean()
        std = df[col].std()
        
        # Avoid division by zero
        if std == 0:
            print(f"Warning: Column '{col}' has zero standard deviation. Setting to 0.")
            df[col] = 0
        else:
            df[col] = (df[col] - mean) / std
    
    return df


def minmax_scaler(df: pd.DataFrame, column: Union[str, List[str]], 
                  feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """
    Apply Min-Max scaling to numerical column(s).
    
    Formula: x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range
    
    Args:
        df: Input DataFrame
        column: Column name(s) to scale. Can be a single string or list of strings.
        feature_range: Desired range of transformed data (min, max). Default is (0, 1).
    
    Returns:
        DataFrame with scaled column(s)
    
    Example:
        >>> df = minmax_scaler(df, 'Age')
        >>> df = minmax_scaler(df, ['Age', 'Fare'], feature_range=(0, 1))
    """
    df = df.copy()
    
    # Convert single column to list for uniform processing
    columns = [column] if isinstance(column, str) else column
    
    min_range, max_range = feature_range
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric for min-max scaling")
        
        # Calculate min and max, ignoring NaN values
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Avoid division by zero
        if col_max == col_min:
            print(f"Warning: Column '{col}' has constant values. Setting to min_range.")
            df[col] = min_range
        else:
            # Apply min-max scaling
            df[col] = (df[col] - col_min) / (col_max - col_min) * (max_range - min_range) + min_range
    
    return df


# ============================================================================
# Outlier Handling - Zhiqi
# ============================================================================

def clip_outliers_iqr(df: pd.DataFrame, column: str, analysis_results: dict = None, whisker_width: float = 1.5) -> pd.DataFrame:
    """
    Cap/clip outliers using IQR method - replaces outliers with boundary values.
    
    Can use pre-calculated boundaries from analysis_results or calculate new ones.
    
    Args:
        df: Input DataFrame
        column: Column name to process
        analysis_results: Optional analysis results containing pre-calculated boundaries
        whisker_width: Multiplier for IQR to define outlier boundaries. 
                      Default is 1.5 (standard box plot definition)
    
    Returns:
        DataFrame with outliers clipped to boundary values
    
    Example:
        >>> df = clip_outliers_iqr(df, 'Fare', analysis_results)
        >>> df = clip_outliers_iqr(df, 'Age', whisker_width=2.0)
    """
    df = df.copy()
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for outlier detection")
    
    # Try to use pre-calculated boundaries from analysis_results
    if analysis_results and 'outlier_info' in analysis_results:
        outlier_info = analysis_results['outlier_info'].get(column, {})
        if 'lower_bound' in outlier_info and 'upper_bound' in outlier_info:
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            print(f"Using pre-calculated boundaries from analysis: [{lower_bound:.2f}, {upper_bound:.2f}]")
        else:
            # Fallback to calculation
            lower_bound, upper_bound = _calculate_iqr_bounds(df, column, whisker_width)
    else:
        # Calculate boundaries if not provided
        lower_bound, upper_bound = _calculate_iqr_bounds(df, column, whisker_width)
    
    # Count outliers before clipping
    outliers_lower = (df[column] < lower_bound).sum()
    outliers_upper = (df[column] > upper_bound).sum()
    total_outliers = outliers_lower + outliers_upper
    
    if total_outliers > 0:
        print(f"Column '{column}': Found {total_outliers} outliers "
              f"({outliers_lower} below {lower_bound:.2f}, "
              f"{outliers_upper} above {upper_bound:.2f})")
        print(f"Clipping outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Clip outliers to boundary values
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def _calculate_iqr_bounds(df: pd.DataFrame, column: str, whisker_width: float) -> tuple:
    """Helper function to calculate IQR boundaries."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr
    return lower_bound, upper_bound


def remove_outliers_iqr(df: pd.DataFrame, column: str, analysis_results: dict = None, whisker_width: float = 1.5) -> pd.DataFrame:
    """
    Remove rows containing outliers using IQR method.
    
    Can use pre-calculated boundaries from analysis_results or calculate new ones.
    
    Args:
        df: Input DataFrame
        column: Column name to process
        analysis_results: Optional analysis results containing pre-calculated boundaries
        whisker_width: Multiplier for IQR to define outlier boundaries.
                      Default is 1.5 (standard box plot definition)
    
    Returns:
        DataFrame with outlier rows removed
    
    Example:
        >>> df = remove_outliers_iqr(df, 'Fare', analysis_results)
        >>> df = remove_outliers_iqr(df, 'Age', whisker_width=2.0)
    
    Warning:
        This method permanently removes rows from the dataset. 
        Use with caution, especially if you have limited data.
    """
    df = df.copy()
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for outlier detection")
    
    original_rows = len(df)
    
    # Try to use pre-calculated boundaries from analysis_results
    if analysis_results and 'outlier_info' in analysis_results:
        outlier_info = analysis_results['outlier_info'].get(column, {})
        if 'lower_bound' in outlier_info and 'upper_bound' in outlier_info:
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            print(f"Using pre-calculated boundaries from analysis: [{lower_bound:.2f}, {upper_bound:.2f}]")
        else:
            # Fallback to calculation
            lower_bound, upper_bound = _calculate_iqr_bounds(df, column, whisker_width)
    else:
        # Calculate boundaries if not provided
        lower_bound, upper_bound = _calculate_iqr_bounds(df, column, whisker_width)
    
    # Identify outliers
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    if outliers_count > 0:
        print(f"Column '{column}': Removing {outliers_count} rows with outliers "
              f"(outside range [{lower_bound:.2f}, {upper_bound:.2f}])")
        print(f"Original rows: {original_rows}, Remaining rows: {original_rows - outliers_count}")
    
    # Remove outliers
    df = df[~outliers_mask].reset_index(drop=True)
    
    return df

