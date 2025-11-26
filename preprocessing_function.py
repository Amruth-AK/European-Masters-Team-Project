import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Any
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import optuna
from sklearn.decomposition import FastICA
import itertools
from sklearn.preprocessing import PowerTransformer



# ===================================================================
# Missing values - Tecla
# ===================================================================

#================DELETION=============================

def delete_missing_rows(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Delete rows with missing values exceeding a given threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed fraction of missing values per row.
                   Rows with missing fraction > threshold will be removed.
                   Default is 0.5 (remove rows with >50% missing values).

    Returns:
        DataFrame with rows containing too many missing values removed.

    Example:
        >>> df = delete_missing_rows(df, threshold=0.3)
    """
    df = df.copy()
    missing_fraction = df.isnull().mean(axis=1)
    rows_to_drop = missing_fraction > threshold
    print(f"Removed {rows_to_drop.sum()} rows exceeding missing threshold {threshold}")
    return df.loc[~rows_to_drop].reset_index(drop=True)


def delete_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Delete columns with missing values exceeding a given threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed fraction of missing values per column.
                   Columns with missing fraction > threshold will be removed.
                   Default is 0.5.

    Returns:
        DataFrame with high-missing columns removed.

    Example:
        >>> df = delete_missing_columns(df, threshold=0.4)
    """
    df = df.copy()
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index
    print(f"Removed {len(cols_to_drop)} columns exceeding missing threshold {threshold}: {list(cols_to_drop)}")
    return df.drop(columns=cols_to_drop)






#==================IMPUTATION=============
def impute_mean(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Impute missing values in numeric column(s) using the mean.

    Args:
        df: Input DataFrame
        columns: Column name(s) to impute. Can be string or list of strings.

    Returns:
        DataFrame with missing values imputed.

    Example:
        >>> df = impute_mean(df, 'Age')
        >>> df = impute_mean(df, ['Age', 'Fare'])
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"Imputed missing values in '{col}' with mean={mean_value:.3f}")
    return df


def impute_median(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Impute missing values in numeric column(s) using the median.
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Imputed missing values in '{col}' with median={median_value:.3f}")
    return df


def impute_mode(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Impute missing values using the mode (most frequent value).

    Suitable for categorical or discrete numerical columns.
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)
        print(f"Imputed missing values in '{col}' with mode={mode_value}")
    return df


def impute_constant(df: pd.DataFrame, columns: Union[str, List[str]], fill_value: float = -1) -> pd.DataFrame:
    """
    Impute missing values with a constant (e.g. -1, -999, 0).
    Useful for tree models to isolate missing data in a specific leaf.
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        df[col] = df[col].fillna(fill_value)
        print(f"Imputed missing values in '{col}' with constant={fill_value}")
    return df


def add_missing_indicator(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Create a boolean feature indicating if a value was missing.
    Crucial for financial data where 'missing' often means 'not applicable' or 'zero denominator'.
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            continue
        # Create new column
        new_col = f"{col}_is_missing"
        df[new_col] = df[col].isnull().astype(int)
        print(f"Added indicator column '{new_col}'")
        
    return df

# ===================================================================
# Duplicate Values - Tecla
# ===================================================================

def delete_duplicates(df: pd.DataFrame, subset: Union[str, List[str], None] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Column name(s) to consider when identifying duplicates.
                If None, considers all columns. Default=None.

    Returns:
        DataFrame with duplicate rows removed.

    Example:
        >>> df = delete_duplicates(df)
        >>> df = delete_duplicates(df, subset=['Name', 'Age'])
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows (kept first occurrence).")
    return df



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

def clip_outliers_iqr(df: pd.DataFrame, column: str, 
                     whisker_width: float = 1.5,
                     analysis_results: dict = None) -> pd.DataFrame:
    """
    Cap/clip outliers using IQR method - replaces outliers with boundary values.
    
    Outliers are detected using the IQR (Interquartile Range) method:
    - Lower bound: Q1 - whisker_width * IQR
    - Upper bound: Q3 + whisker_width * IQR
    
    Values outside these bounds are replaced with the boundary values (capping).
    
    Args:
        df: Input DataFrame
        column: Column name to process
        whisker_width: Multiplier for IQR to define outlier boundaries. 
                      Default is 1.5 (standard box plot definition)
        analysis_results: Optional dict from DataAnalyzer with pre-calculated bounds.
                         If provided and contains bounds for this column, uses those.
    
    Returns:
        DataFrame with outliers clipped to boundary values
    
    Example:
        >>> df = clip_outliers_iqr(df, 'Fare')
        >>> df = clip_outliers_iqr(df, 'Age', whisker_width=2.0)
        >>> df = clip_outliers_iqr(df, 'Age', analysis_results=analysis_dict)
    
    Note:
        - whisker_width=1.5: Standard outlier detection (default)
        - whisker_width=3.0: More conservative, detects only extreme outliers
    """
    df = df.copy()
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for outlier detection")
    
    # Try to use pre-calculated boundaries from analysis_results
    if analysis_results and 'outlier_info' in analysis_results:
        col_outlier_info = analysis_results['outlier_info'].get(column, {})
        if 'lower_bound' in col_outlier_info and 'upper_bound' in col_outlier_info:
            lower_bound = col_outlier_info['lower_bound']
            upper_bound = col_outlier_info['upper_bound']
        else:
            # Calculate boundaries ourselves
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - whisker_width * iqr
            upper_bound = q3 + whisker_width * iqr
    else:
        # Calculate boundaries ourselves
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - whisker_width * iqr
        upper_bound = q3 + whisker_width * iqr
    
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

def winsorize_column(df: pd.DataFrame, column: str, 
                     limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """
    Clip values to specified percentiles (e.g., 1st and 99th).
    Preserves the range better than IQR clipping for financial data.
    """
    df = df.copy()
    if column not in df.columns:
        return df
        
    lower_quantile, upper_quantile = limits
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    
    print(f"Winsorizing '{column}': [{lower_bound:.4f}, {upper_bound:.4f}]")
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def apply_power_transform(df: pd.DataFrame, column: str, method: str = 'yeo-johnson') -> pd.DataFrame:
    """
    Apply Power Transformation (Yeo-Johnson) to reduce skewness.
    Handles positive and negative values.
    """
    df = df.copy()
    if column not in df.columns:
        return df
        
    # Reshape for sklearn
    values = df[column].values.reshape(-1, 1)
    
    # Handle NaNs: fit on valid data, transform maintaining NaNs
    mask = ~pd.isna(values).flatten()
    if mask.sum() > 0:
        pt = PowerTransformer(method=method, standardize=False)
        pt.fit(values[mask])
        df.loc[mask, column] = pt.transform(values[mask]).flatten()
        print(f"Applied {method} transform to '{column}'")
        
    return df


def robust_scaler(df: pd.DataFrame, column: Union[str, List[str]], 
                  quantile_range: tuple = (25.0, 75.0)) -> pd.DataFrame:
    """
    Scale features using statistics that are robust to outliers.
    Formula: (x - median) / (q75 - q25)
    """
    df = df.copy()
    columns = [column] if isinstance(column, str) else column
    
    q_min, q_max = quantile_range
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        median = df[col].median()
        q1 = df[col].quantile(q_min / 100.0)
        q3 = df[col].quantile(q_max / 100.0)
        iqr = q3 - q1
        
        if iqr == 0:
            print(f"Warning: IQR is 0 for '{col}', RobustScaling limited to centering.")
            df[col] = df[col] - median
        else:
            df[col] = (df[col] - median) / iqr
            
    return df

def remove_outliers_iqr(df: pd.DataFrame, column: str, 
                       whisker_width: float = 1.5,
                       analysis_results: dict = None) -> pd.DataFrame:
    """
    Remove rows containing outliers using IQR method.
    
    Outliers are detected using the IQR (Interquartile Range) method:
    - Lower bound: Q1 - whisker_width * IQR
    - Upper bound: Q3 + whisker_width * IQR
    
    Rows with values outside these bounds are deleted from the DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to process
        whisker_width: Multiplier for IQR to define outlier boundaries.
                      Default is 1.5 (standard box plot definition)
        analysis_results: Optional dict from DataAnalyzer with pre-calculated bounds.
                         If provided and contains bounds for this column, uses those.
    
    Returns:
        DataFrame with outlier rows removed
    
    Example:
        >>> df = remove_outliers_iqr(df, 'Fare')
        >>> df = remove_outliers_iqr(df, 'Age', whisker_width=2.0)
        >>> df = remove_outliers_iqr(df, 'Age', analysis_results=analysis_dict)
    
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
        col_outlier_info = analysis_results['outlier_info'].get(column, {})
        if 'lower_bound' in col_outlier_info and 'upper_bound' in col_outlier_info:
            lower_bound = col_outlier_info['lower_bound']
            upper_bound = col_outlier_info['upper_bound']
        else:
            # Calculate boundaries ourselves
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - whisker_width * iqr
            upper_bound = q3 + whisker_width * iqr
    else:
        # Calculate boundaries ourselves
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - whisker_width * iqr
        upper_bound = q3 + whisker_width * iqr
    
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


# ============================================================================
# Categorical Encoding - Amruth
# ============================================================================

def one_hot_encode(df: pd.DataFrame, columns: Union[str, List[str]], drop_first: bool = False) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical columns.
    
    Converts categorical variables into multiple binary (0/1) columns.
    
    Args:
        df: Input DataFrame
        columns: Column names to encode. Can be a single string or list of strings.
        drop_first: If True, drops the first category to avoid dummy variable trap.
    
    Returns:
        DataFrame with one-hot encoded columns
    
    Example:
        >>> df = one_hot_encode(df, 'Gender')
        >>> df = one_hot_encode(df, ['Gender', 'City'], drop_first=True)
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is numeric; skipping one-hot encoding.")
            continue

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def label_encode(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Apply Label Encoding to categorical columns.
    
    Converts categories to integer labels (0, 1, 2, ...).
    
    Args:
        df: Input DataFrame
        columns: Column names to encode.
    
    Returns:
        DataFrame with label-encoded columns
    
    Example:
        >>> df = label_encode(df, 'Gender')
        >>> df = label_encode(df, ['City', 'Education'])
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is numeric; skipping label encoding.")
            continue

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def ordinal_encode(df: pd.DataFrame, columns: Union[str, List[str]], category_orders: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Apply Ordinal Encoding to categorical columns.
    
    Maps categories to integers according to specified order.
    If no order is provided, uses alphabetical order by default.
    
    Args:
        df: Input DataFrame
        columns: Column names to encode.
        category_orders: Optional dictionary specifying category order per column.
    
    Returns:
        DataFrame with ordinal-encoded columns
    
    Example:
        >>> orders = {'Education': ['High School', 'Bachelors', 'Masters', 'PhD']}
        >>> df = ordinal_encode(df, ['Education'], category_orders=orders)
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is numeric; skipping ordinal encoding.")
            continue

        if category_orders and col in category_orders:
            categories = [category_orders[col]]
        else:
            categories = [sorted(df[col].dropna().unique().tolist())]

        oe = OrdinalEncoder(categories=categories)
        df[col] = oe.fit_transform(df[[col]])

    return df


def binary_encode(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Apply Binary Encoding to categorical columns.
    
    Converts categories into binary code representations.
    
    Args:
        df: Input DataFrame
        columns: Column names to encode.
    
    Returns:
        DataFrame with binary encoded columns
    
    Example:
        >>> df = binary_encode(df, 'City')
        >>> df = binary_encode(df, ['Country', 'Gender'])
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is numeric; skipping binary encoding.")
            continue

        unique_vals = df[col].astype(str).unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col + '_bin'] = df[col].astype(str).map(mapping)

        max_bits = int(np.ceil(np.log2(len(unique_vals)))) if len(unique_vals) > 1 else 1
        for i in range(max_bits):
            df[f"{col}_bit_{i}"] = df[col + '_bin'].apply(lambda x: (x >> i) & 1)
        
        df = df.drop(columns=[col, col + '_bin'])

    return df


def frequency_encode(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Apply Frequency Encoding to categorical columns.
    
    Replaces each category with its relative frequency in the column.
    
    Args:
        df: Input DataFrame
        columns: Column names to encode.
    
    Returns:
        DataFrame with frequency encoded columns
    
    Example:
        >>> df = frequency_encode(df, 'City')
        >>> df = frequency_encode(df, ['Gender', 'Country'])
    """
    df = df.copy()
    columns = [columns] if isinstance(columns, str) else columns

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)

    return df

# ============================================================================
# Identifier Column Removal - Zhiqi
# ============================================================================

def remove_identifier_columns(df, id_columns=None, **kwargs):
    """
    Remove identifier columns detected by DataAnalyzer.
    """
    if not id_columns:
        return df

    cols_to_drop = [col for col in id_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


# ============================================================================
# Datetime Feature Engineering - Zhiqi
# ============================================================================

def extract_datetime_features(df: pd.DataFrame, 
                              column: str,
                              features: List[str] = None,
                              drop_original: bool = True) -> pd.DataFrame:
    """
    Extract useful features from datetime columns.
    
    Converts datetime column into multiple numerical features that models can use.
    
    Args:
        df: Input DataFrame
        column: Datetime column name to process
        features: List of features to extract. Available options:
                 ['year', 'month', 'day', 'dayofweek', 'quarter', 
                  'hour', 'minute', 'is_weekend', 'is_month_start', 'is_month_end']
                 If None, extracts all available features.
        drop_original: Whether to drop the original datetime column (default True)
    
    Returns:
        DataFrame with datetime features extracted
    
    Example:
        >>> df = extract_datetime_features(df, 'date')
        >>> df = extract_datetime_features(df, 'timestamp', features=['year', 'month', 'hour'])
        >>> df = extract_datetime_features(df, 'date', drop_original=False)
    
    Note:
        The column must be datetime type or convertible to datetime.
        Most ML models cannot handle datetime directly, so this converts them to numbers.
    """
    df = df.copy()
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Try to convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        try:
            df[column] = pd.to_datetime(df[column])
            print(f"Converted column '{column}' to datetime type")
        except Exception as e:
            raise TypeError(f"Column '{column}' cannot be converted to datetime: {e}")
    
    # Default: extract all features
    if features is None:
        features = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                   'hour', 'minute', 'is_weekend', 'is_month_start', 'is_month_end']
    
    # Extract requested features
    extracted_count = 0
    
    if 'year' in features:
        df[f'{column}_year'] = df[column].dt.year
        extracted_count += 1
    
    if 'month' in features:
        df[f'{column}_month'] = df[column].dt.month
        extracted_count += 1
    
    if 'day' in features:
        df[f'{column}_day'] = df[column].dt.day
        extracted_count += 1
    
    if 'dayofweek' in features:
        df[f'{column}_dayofweek'] = df[column].dt.dayofweek  # Monday=0, Sunday=6
        extracted_count += 1
    
    if 'quarter' in features:
        df[f'{column}_quarter'] = df[column].dt.quarter
        extracted_count += 1
    
    if 'hour' in features:
        try:
            df[f'{column}_hour'] = df[column].dt.hour
            extracted_count += 1
        except AttributeError:
            pass  # Skip if datetime doesn't have time component
    
    if 'minute' in features:
        try:
            df[f'{column}_minute'] = df[column].dt.minute
            extracted_count += 1
        except AttributeError:
            pass
    
    if 'is_weekend' in features:
        df[f'{column}_is_weekend'] = (df[column].dt.dayofweek >= 5).astype(int)  # 5=Sat, 6=Sun
        extracted_count += 1
    
    if 'is_month_start' in features:
        df[f'{column}_is_month_start'] = df[column].dt.is_month_start.astype(int)
        extracted_count += 1
    
    if 'is_month_end' in features:
        df[f'{column}_is_month_end'] = df[column].dt.is_month_end.astype(int)
        extracted_count += 1
    
    print(f"Extracted {extracted_count} datetime features from column '{column}'")
    
    # Drop original datetime column if requested
    if drop_original:
        df = df.drop(columns=[column])
        print(f"Dropped original datetime column '{column}'")
    
    return df


def calculate_datetime_diff(df: pd.DataFrame, 
                           col1: str, 
                           col2: str,
                           unit: str = 'days',
                           new_col_name: str = None) -> pd.DataFrame:
    """
    Calculate time difference between two datetime columns.
    
    Useful for features like "days_since_registration", "hours_between_events", etc.
    
    Args:
        df: Input DataFrame
        col1: First datetime column (later/end time)
        col2: Second datetime column (earlier/start time)
        unit: Unit for difference calculation. Options:
              'days', 'hours', 'minutes', 'seconds'
        new_col_name: Name for the new difference column.
                     If None, uses format '{col1}_minus_{col2}_{unit}'
    
    Returns:
        DataFrame with new time difference column
    
    Example:
        >>> df = calculate_datetime_diff(df, 'end_date', 'start_date', unit='days')
        >>> df = calculate_datetime_diff(df, 'current_time', 'signup_time', 
                                        unit='hours', new_col_name='hours_since_signup')
    """
    df = df.copy()
    
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' not found in DataFrame")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[col1]):
        df[col1] = pd.to_datetime(df[col1])
    if not pd.api.types.is_datetime64_any_dtype(df[col2]):
        df[col2] = pd.to_datetime(df[col2])
    
    # Calculate difference
    time_diff = df[col1] - df[col2]
    
    # Convert to requested unit
    if unit == 'days':
        result = time_diff.dt.days
    elif unit == 'hours':
        result = time_diff.dt.total_seconds() / 3600
    elif unit == 'minutes':
        result = time_diff.dt.total_seconds() / 60
    elif unit == 'seconds':
        result = time_diff.dt.total_seconds()
    else:
        raise ValueError(f"Invalid unit '{unit}'. Use 'days', 'hours', 'minutes', or 'seconds'")
    
    # Create new column
    if new_col_name is None:
        new_col_name = f'{col1}_minus_{col2}_{unit}'
    
    df[new_col_name] = result
    print(f"Created time difference column '{new_col_name}' ({unit})")
    
    return df

def create_features_from_high_correlation(
    df: pd.DataFrame, 
    correlation_threshold: float = 0.3,
    target_column: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    feature_types: List[str] = ['product', 'ratio'],
    analysis_results: Optional[Dict] = None, 
    # Filtering parameters
    use_basic_filter: bool = True,
    use_correlation_filter: bool = True,
    min_cardinality: int = 3,
    max_new_features: int = 500,
    corr_filter_threshold: float = 0.9,
    min_variance: float = 1e-6,
    **kwargs
) -> pd.DataFrame:
    """
    Generate new features based on highly correlated feature pairs.
    
    This function acts as the 'Executor'. It takes the threshold suggested by the 
    'Planner' and performs the actual feature engineering.
    
    Optimization:
    It attempts to reuse the pre-calculated correlation matrix from 'analysis_results'
    to avoid redundant computation on large datasets.

    Args:
        df: Input DataFrame.
        correlation_threshold: Minimum correlation to trigger interaction.
        target_column: Name of the target column.
        exclude_columns: List of columns to ignore.
        feature_types: List of operations.
        analysis_results: Analysis dictionary containing pre-computed correlations.
        use_basic_filter: Filter constant/low-variance features.
        use_correlation_filter: Filter highly redundant new features.
        min_cardinality: Minimum unique values.
        max_new_features: Cap on new features.
        corr_filter_threshold: Threshold for redundancy filtering.
        min_variance: Minimum variance threshold.
        **kwargs: Additional arguments for 3rd-order generation.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    df = df.copy()
    
    # 1. Prepare numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if exclude_columns:
        numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
    
    # Ensure target is not used as a raw feature for interaction
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if len(numerical_cols) < 2:
        print("⚠️ [Feature Gen] Need at least 2 numerical columns to generate interactions.")
        return df
    
    # 2. Get Correlation Matrix (Optimized)
    # Strategy: Try to load from analysis_results first, calculate only if needed.
    corr_matrix = None
    
    if analysis_results and 'correlations' in analysis_results:
        try:
            matrix_data = analysis_results['correlations'].get('correlation_matrix')
            if matrix_data:
                # Convert dict/json back to DataFrame
                cached_corr = pd.DataFrame(matrix_data)
                
                # Check if the cached matrix covers our current numerical columns
                # We filter it to match the current df columns exactly
                common_cols = [c for c in numerical_cols if c in cached_corr.index]
                
                if len(common_cols) >= 2:
                    corr_matrix = cached_corr.loc[common_cols, common_cols]
                    # If cached matrix is missing significant columns, we might need to recalc
                    # But usually, analysis_results comes from the same dataset version.
                    if len(common_cols) < len(numerical_cols) * 0.9: 
                        print("ℹ️ [Feature Gen] Cached matrix is missing many columns. Recalculating...")
                        corr_matrix = None
                    else:
                        print("✓ [Feature Gen] Using pre-calculated correlation matrix from analysis_results.")
        except Exception as e:
            print(f"ℹ️ [Feature Gen] Failed to load cached correlation matrix: {e}. Recalculating...")
            corr_matrix = None

    # Fallback: Calculate if not found or invalid
    if corr_matrix is None:
        print("⏳ [Feature Gen] Calculating correlation matrix...")
        corr_matrix = df[numerical_cols].corr()
    
    # 3. Identify High-Correlation Pairs
    high_corr_pairs = _find_high_corr_pairs(
        corr_matrix, numerical_cols, target_column, correlation_threshold
    )
    
    if not high_corr_pairs:
        print(f"ℹ️ [Feature Gen] No pairs found with correlation > {correlation_threshold}. Skipping.")
        return df
    
    print(f"✓ [Feature Gen] Found {len(high_corr_pairs)} high correlation pairs.")
    
    # 4. Generate Candidate 2nd-Order Features
    candidate_df = _generate_candidate_features(df, high_corr_pairs, feature_types)
    
    if candidate_df.empty:
        print("❌ [Feature Gen] No candidate features were generated.")
        return df
    
    print(f"✓ [Feature Gen] Generated {len(candidate_df.columns)} raw candidate features.")
    
    # 5. Apply Quality Filters (Basic & Redundancy)
    if use_basic_filter:
        candidate_df = _apply_basic_filter(candidate_df, min_cardinality, min_variance)
    
    if use_correlation_filter and len(candidate_df.columns) > 1:
        candidate_df = _apply_correlation_filter(candidate_df, corr_filter_threshold)
    
    # 6. Smart Selection: Top N Features based on Target Correlation
    if len(candidate_df.columns) > max_new_features:
        print(f"⚠️ [Feature Gen] Limiting to top {max_new_features} features based on target correlation.")
        
        if target_column and target_column in df.columns:
            target_corrs = {}
            for col in candidate_df.columns:
                try:
                    corr = abs(candidate_df[col].corr(df[target_column]))
                    target_corrs[col] = corr if not pd.isna(corr) else 0.0
                except Exception:
                    target_corrs[col] = 0.0
            
            sorted_features = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
            selected_cols = [f[0] for f in sorted_features[:max_new_features]]
            candidate_df = candidate_df[selected_cols]
            
            avg_corr = np.mean([f[1] for f in sorted_features[:max_new_features]])
            print(f"   -> Average target correlation of selected: {avg_corr:.4f}")
        else:
            print("   -> No target column specified. Using random sampling.")
            candidate_df = candidate_df.sample(n=max_new_features, random_state=42, axis=1)
    
    # 7. Merge 2nd-Order Features
    second_order_count = len(candidate_df.columns)
    df = pd.concat([df, candidate_df], axis=1)
    
    print(f"✅ [Feature Gen] Successfully added {second_order_count} 2nd-order features.")
    
    # 8. Generate 3rd-Order Features (Optional)
    generate_third_order = kwargs.get('generate_third_order', True)
    min_second_for_third = kwargs.get('min_second_for_third', 20)
    
    if generate_third_order and second_order_count >= min_second_for_third:
        print("\n🔬 [Feature Gen] Attempting 3rd-order interaction features...")
        
        max_third_order = kwargs.get('max_third_order_features', 30)
        top_second_for_third = kwargs.get('top_second_for_third', 50)
        top_original_for_third = kwargs.get('top_original_for_third', 30)
        
        # Debugging
        print(f"🔍 [Debug] Before 3rd-order generation:")
        print(f"   candidate_df shape: {candidate_df.shape}")
        print(f"   candidate_df columns: {len(candidate_df.columns)}")
        print(f"   df shape: {df.shape}")
        print(f"   target_column in df: {target_column in df.columns}")
        print(f"   Index match: {candidate_df.index.equals(df.index)}")

        
        third_order_df = _generate_third_order_features(
            df=df,
            second_order_features=candidate_df,
            original_features=numerical_cols,
            target_column=target_column,
            max_third_order=max_third_order,
            top_second_for_third=min(top_second_for_third, second_order_count),
            top_original_for_third=top_original_for_third
        )
        
        if not third_order_df.empty:
            third_order_count = len(third_order_df.columns)
            df = pd.concat([df, third_order_df], axis=1)
            print(f"✅ [Feature Gen] Added {third_order_count} 3rd-order features.")
    
    return df


# Helper functions
def _find_high_corr_pairs(corr_matrix: pd.DataFrame, numerical_cols: List[str], target_column: Optional[str], threshold: float) -> List[tuple]:
    """
    Identify pairs of features with correlation above the threshold.
    Optimized to iterate only the upper triangle, avoiding duplicate checks.
    """
    pairs = []
    
    # Strategy A: Include all pairs (Threshold = 0)
    if threshold == 0.0:
        for i, col1 in enumerate(numerical_cols):
            if col1 not in corr_matrix.index: continue
            for col2 in numerical_cols[i+1:]:
                if col2 not in corr_matrix.columns: continue
                val = corr_matrix.loc[col1, col2]
                pairs.append((col1, col2, val))
        return pairs
    
    # Strategy B: Normal iteration (Threshold > 0)
    # We iterate upper triangle to avoid duplicates (A,B) and (B,A)
    # This loop structure inherently prevents duplicates, so no "if not in pairs" check is needed.
    for i, col1 in enumerate(numerical_cols):
        if col1 not in corr_matrix.index: continue
        
        for col2 in numerical_cols[i+1:]:
            if col2 not in corr_matrix.columns: continue
            
            try:
                val = corr_matrix.loc[col1, col2]
                # Check for NaN and threshold
                if pd.notna(val) and abs(val) > threshold:
                    pairs.append((col1, col2, val))
            except Exception:
                continue
    
    return pairs

def _generate_candidate_features(df, high_corr_pairs, feature_types):
    """Generate candidate features from pairs."""
    candidates = {}
    
    for col1, col2, _ in high_corr_pairs:
        # Skip if too many missing values
        if df[col1].isnull().mean() > 0.5 or df[col2].isnull().mean() > 0.5:
            continue
        
        if 'product' in feature_types:
            candidates[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        if 'ratio' in feature_types:
            candidates[f'{col1}_div_{col2}'] = np.where(
                df[col2] != 0, df[col1] / df[col2], 0
            )
        
        if 'difference' in feature_types:
            candidates[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        if 'sum' in feature_types:
            candidates[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        
        if 'interaction' in feature_types:
            candidates[f'{col1}_interact_{col2}'] = (
                (df[col1] - df[col1].mean()) * (df[col2] - df[col2].mean())
            )
    
    return pd.DataFrame(candidates, index=df.index) if candidates else pd.DataFrame()


def _apply_basic_filter(df: pd.DataFrame, min_cardinality: int, min_variance: float = 1e-6) -> pd.DataFrame:
    """
    Apply basic filtering: removes columns with low cardinality, high missing rate, or low variance.
    
    Includes detailed logging to track why features were dropped.
    
    Args:
        df: Input DataFrame containing candidate features.
        min_cardinality: Minimum number of unique values required (e.g., 3).
        min_variance: Minimum variance threshold (default: 1e-6).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    n_before = len(df.columns)
    to_keep = []
    
    # Counters for logging (Essential for debugging feature engineering)
    dropped_counts = {'low_cardinality': 0, 'high_missing': 0, 'low_variance': 0}
    
    for col in df.columns:
        # 1. Cardinality Check (Are there enough unique values?)
        if df[col].nunique() < min_cardinality:
            dropped_counts['low_cardinality'] += 1
            continue
            
        # 2. Missing Value Check (Is it mostly empty?)
        if df[col].isnull().mean() > 0.5:
            dropped_counts['high_missing'] += 1
            continue
            
        # 3. Variance Check (Does it carry information?)
        # We explicitly handle NaN variance (happens if only 1 value exists)
        var = df[col].var()
        if pd.isna(var) or var < min_variance:
            dropped_counts['low_variance'] += 1
            continue
            
        to_keep.append(col)
        
    df_filtered = df[to_keep]
    n_after = len(df_filtered.columns)
    
    # Log the results clearly
    print(f"  Basic Filter: {n_before} → {n_after} features")
    if n_before != n_after:
        details = [f"{k}: {v}" for k, v in dropped_counts.items() if v > 0]
        print(f"    Dropped: {', '.join(details)}")
    
    return df_filtered


def _apply_correlation_filter(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove redundant features with high correlation using a greedy strategy.
    
    Logic:
    Iterates through features. If Feature A and Feature B are highly correlated,
    Feature A (the first one) is kept, and Feature B is dropped.
    
    Args:
        df: Input DataFrame.
        threshold: Correlation threshold (default 0.9).
        
    Returns:
        pd.DataFrame: DataFrame with redundant features removed.
    """
    # 1. Safety Check
    if df.empty or len(df.columns) < 2:
        return df
        
    n_before = len(df.columns)
    
    # 2. Calculate Correlation Matrix once
    # Use absolute values because -0.99 is just as redundant as 0.99
    corr_matrix = df.corr().abs()
    
    # 3. Greedy Selection Loop
    to_drop = set()
    columns = df.columns.tolist()
    
    for i in range(len(columns)):
        col1 = columns[i]
        
        # If col1 is already marked for removal, it's "dead". 
        # Dead features don't get to decide the fate of others.
        if col1 in to_drop:
            continue
            
        for j in range(i + 1, len(columns)):
            col2 = columns[j]
            
            # Optimization: If col2 is already dead, skip checking it
            if col2 in to_drop:
                continue
            
            # Check correlation
            # Try-except block handles rare edge cases (like NaNs in correlation)
            try:
                if corr_matrix.loc[col1, col2] > threshold:
                    to_drop.add(col2)
            except:
                continue
    
    # 4. Drop Features
    df_filtered = df.drop(columns=list(to_drop))
    n_after = len(df_filtered.columns)
    
    # 5. Logging (Crucial for observability)
    print(f"  Correlation Filter: {n_before} → {n_after} features")
    if to_drop:
        print(f"    Dropped {len(to_drop)} redundant features (> {threshold})")
        
    return df_filtered


from typing import List, Optional
import pandas as pd
import numpy as np

def _generate_third_order_features(
    df: pd.DataFrame,
    second_order_features: pd.DataFrame,
    original_features: List[str],
    target_column: Optional[str],
    max_third_order: int = 30,
    top_second_for_third: int = 50,
    top_original_for_third: int = 30
) -> pd.DataFrame:
    """
    Generate diverse 3rd-order features (Product, Ratio, Sum, Diff).
    
    Logic:
    Selects the "Elite" 2nd-order features and "Elite" original features,
    then combines them using +, -, *, / to capture various interaction types.
    """
    # 1. Validation
    if not target_column or target_column not in df.columns:
        print("  ℹ️ [3rd Order] Target column missing. Skipping.")
        return pd.DataFrame()

    target_series = df[target_column]
    if not pd.api.types.is_numeric_dtype(target_series):
        le = LabelEncoder()
        target_series = pd.Series(
            le.fit_transform(target_series.astype(str)),
            index=target_series.index,
            name=target_column
        )
        print(f"  ℹ️ [3rd Order] Target column is categorical, using label encoding for correlation calculation.")
    else:
        target_series = df[target_column]

    
    # 2. Rank 2nd Order Features 
    second_corrs = {}
    for col in second_order_features.columns:
        try:
            series1 = second_order_features[col]
            common_index = series1.index.intersection(target_series.index)
            if len(common_index) < 2:
                continue
            aligned_s1 = series1.loc[common_index]
            aligned_s2 = target_series.loc[common_index]

            val = abs(aligned_s1.corr(aligned_s2))
            if pd.notna(val):
                second_corrs[col] = val
        except Exception as e:
            print(f"  ⚠️ [3rd Order] Failed to compute correlation for {col}: {e}")
            continue
        
    if not second_corrs:
        print("  ℹ️ [3rd Order] No valid 2nd-order features found.")
        print(f"     Debug: second_order_features has {len(second_order_features.columns)} columns")
        print(f"     Debug: target_column '{target_column}' in df: {target_column in df.columns}")
        return pd.DataFrame()
        
    top_second = sorted(second_corrs, key=second_corrs.get, reverse=True)[:top_second_for_third]
    
    # 3. Rank Original Features
    orig_corrs = {}
    for col in original_features:
        if col in df.columns and col != target_column:
            try:
                common_index = df[col].index.intersection(target_series.index)
                if len(common_index) < 2:
                    continue


                aligned_col = df[col].loc[common_index]
                aligned_target = target_series.loc[common_index]
                val = abs(aligned_col.corr(aligned_target))
                if pd.notna(val): 
                    orig_corrs[col] = val
                
            except Exception as e:
                continue
            
    top_orig = sorted(orig_corrs, key=orig_corrs.get, reverse=True)[:top_original_for_third]
    
    print(f"  🔬 [3rd Order] Combining {len(top_second)} interaction features × {len(top_orig)} original features...")
    
    # 4. Generate Candidates (The 4 Operations)
    candidates = {}
    for f2 in top_second:
        v2 = second_order_features[f2]
        for f1 in top_orig:
            v1 = df[f1]
            
            # Pre-calculate safe denominator for division
            # Replacing 0 with NaN prevents "inf" errors
            v1_safe = v1.replace(0, np.nan)
            
            # --- Operation 1: Product ((A*B)*C) ---
            candidates[f'{f2}_x_{f1}'] = v2 * v1
            
            # --- Operation 2: Ratio ((A*B)/C) ---
            candidates[f'{f2}_div_{f1}'] = v2 / v1_safe
            
            # --- Operation 3: Sum ((A*B)+C) ---
            # Captures additive offsets
            candidates[f'{f2}_plus_{f1}'] = v2 + v1
            
            # --- Operation 4: Difference ((A*B)-C) ---
            candidates[f'{f2}_minus_{f1}'] = v2 - v1
            
    if not candidates:
        return pd.DataFrame()
        
    cand_df = pd.DataFrame(candidates, index=df.index)
    
    # 5. Quality Filter (Variance/Nulls)
    # Using the robust filter we defined earlier
    cand_df = _apply_basic_filter(cand_df, min_cardinality=3, min_variance=1e-5)
    
    if cand_df.empty:
        return cand_df

    # 6. Final Selection by Target Correlation
    final_corrs = {}
    for col in cand_df.columns:
        try:
            series_col = cand_df[col]
            common_index = series_col.index.intersection(target_series.index)
            if len(common_index) < 2:
                continue

            aligned_col = series_col.loc[common_index]
            aligned_target = target_series.loc[common_index]
            val = abs(aligned_col.corr(aligned_target))
            if pd.notna(val): final_corrs[col] = val
        except Exception as e:
            continue
        
    best_features = sorted(final_corrs, key=final_corrs.get, reverse=True)[:max_third_order]
    
    result = cand_df[best_features]
    
    # Stats
    if len(best_features) > 0:
        avg = np.mean([final_corrs[f] for f in best_features])
        print(f"  ✅ [3rd Order] Selected {len(result.columns)} features. Avg Corr: {avg:.4f}")
        print(f"  -> Top feature: {best_features[0]} ({final_corrs[best_features[0]]:.4f})")
        
    return result


def create_features_from_correlation_analysis(
    df: pd.DataFrame,
    analysis_results: Dict = None,
    **kwargs
) -> pd.DataFrame:
    """Wrapper function using pre-computed analysis results."""
    return create_features_from_high_correlation(
        df=df, 
        analysis_results=analysis_results, 
        **kwargs
    )
    
   
def combine_categorical_features(df: pd.DataFrame,
                                 columns_to_combine: List[str],
                                 new_col_name: str,
                                 separator: str = '_',
                                 drop_original: bool = False) -> pd.DataFrame:
    """
    Combines two or more categorical columns into a single new feature.

    This can help models capture interaction effects between categorical variables.

    Args:
         df: Input DataFrame.
        columns_to_combine: List of column names to combine.
        new_col_name: Name for the newly created combined column.
        separator: The string to use for joining the values. Default is '_'.
        drop_original: If True, drops the original columns after combination.

    Returns:
        DataFrame with the new combined feature.
    """
    df = df.copy()

    # Check if columns exist
    for col in columns_to_combine:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in DataFrame.")

    # Combine columns by converting them to string type and joining
    df[new_col_name] = df[columns_to_combine].astype(str).agg(separator.join, axis=1)
    print(f"Created new combined feature: '{new_col_name}'")

    # Drop original columns if requested
    if drop_original:
        df = df.drop(columns=columns_to_combine)
        print(f"Dropped original columns: {', '.join(columns_to_combine)}")

    return df




def apply_fastica(
    df: pd.DataFrame,
    n_components: int = None,
    target_column: str = None,
    exclude_columns: List[str] = None,
    random_state: int = 42,
    max_iter: int = 1000,
    whiten: str = 'unit-variance',
    mode: str = 'hybrid',   # 'hybrid' is the default mode
    replace_ratio: float = None,  # Automatically calculate the replace ratio
    replace_columns: List[str] = None,
    keep_columns: List[str] = None,
    add_interaction_features: bool = True,  # Add ICA interaction features
    analysis_results: dict = None  # Use analysis results to make intelligent decisions
) -> pd.DataFrame:
    """
    Apply FastICA with intelligent hybrid mode.

    Args:
        df: Input dataframe
        n_components: Number of ICA components. If None, auto-selected.
        target_column: Target column to exclude from ICA.
        exclude_columns: Additional columns to exclude from ICA.
        random_state: Random state for FastICA.
        max_iter: Max iterations for FastICA.
        whiten: Whether to whiten in FastICA.
        mode: 'hybrid' or 'selective'.
        replace_ratio: For 'hybrid' mode. If None, auto-calculated.
        replace_columns: For 'selective' mode: columns to replace.
        keep_columns: For 'selective' mode: columns to keep.
        add_interaction_features: If True, add ICA interaction features.
        analysis_results: Optional analysis dict for intelligent feature selection.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    # 1. Select numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 2. Exclude target and specified columns
    if exclude_columns:
        numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)

    if len(numerical_cols) < 2:
        print("⚠️ Need at least 2 numerical columns for FastICA")
        return df

    # 3. Auto-select n_components if not specified
    if n_components is None:
        n_components = min(len(numerical_cols), len(df) // 2, 10)
        n_components = max(2, n_components)

    if n_components >= len(numerical_cols):
        print(f"⚠️ n_components ({n_components}) >= n_features ({len(numerical_cols)}). "
              f"Using {len(numerical_cols) - 1}")
        n_components = len(numerical_cols) - 1

    # 4. Prepare data
    X = df[numerical_cols].copy()
    if X.isnull().any().any():
        print("⚠️ Missing values detected. Filling with median.")
        X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if isinstance(whiten, bool):
        if whiten:  # True → pick a valid string
            whiten_param = "unit-variance"
        else:
            whiten_param = False
    else:
        # assume user passed a valid string or False
        whiten_param = whiten

    # 5. Apply FastICA
    try:
        ica = FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            whiten=whiten_param
        )
        ica_components = ica.fit_transform(X_scaled)
    except Exception as e:
        print(f"❌ FastICA failed: {e}")
        return df

    component_names = [f'ICA_{i}' for i in range(n_components)]
    ica_df = pd.DataFrame(
        ica_components,
        columns=component_names,
        index=df.index
    )

    # Placeholder for final result
    df_result = df.copy()

    # 6. Handle different modes
    if mode == 'hybrid':
        # Intelligent calculation of replace ratio
        if replace_ratio is None:
            replace_ratio = _calculate_intelligent_replace_ratio(
                df, numerical_cols, n_components, analysis_results
            )

        # How many features to replace
        n_to_replace = max(1, int(len(numerical_cols) * replace_ratio))
        n_to_replace = min(n_to_replace, len(numerical_cols) - 1)  # Keep at least 1

        # Intelligent selection of features to replace/keep
        cols_to_replace, cols_to_keep = _select_features_to_replace(
            df, numerical_cols, n_to_replace, analysis_results
        )

        non_numerical_cols = [col for col in df.columns if col not in numerical_cols]

        result_parts = []
        if non_numerical_cols:
            result_parts.append(df[non_numerical_cols])
        if cols_to_keep:
            result_parts.append(df[cols_to_keep])
        result_parts.append(ica_df)

        df_result = pd.concat(result_parts, axis=1)

        print(f"✅ FastICA (Intelligent Hybrid Mode):")
        print(f"   Replace ratio: {replace_ratio:.2%} (auto-calculated)")
        print(f"   Replaced {len(cols_to_replace)} features, kept {len(cols_to_keep)}, "
              f"added {n_components} ICA components")
        print(f"   Total features: {len(df.columns)} → {len(df_result.columns)}")
        if len(cols_to_replace) <= 10:
            print(f"   Replaced: {cols_to_replace}")
        else:
            print(f"   Replaced: {cols_to_replace[:5]} ... ({len(cols_to_replace)} total)")
        if len(cols_to_keep) <= 10:
            print(f"   Kept: {cols_to_keep}")
        else:
            print(f"   Kept: {cols_to_keep[:5]} ... ({len(cols_to_keep)} total)")

    elif mode == 'selective':
        # Selective: User specifies which to replace/keep
        if replace_columns is None:
            replace_columns = []
        if keep_columns is None:
            keep_columns = []

        # Validate
        replace_columns = [col for col in replace_columns if col in numerical_cols]
        keep_columns = [col for col in keep_columns if col in numerical_cols]

        # If both empty, default to replace all numerical
        if not replace_columns and not keep_columns:
            replace_columns = numerical_cols

        # Ensure no overlap
        keep_columns = [col for col in keep_columns if col not in replace_columns]

        # Keep non-numerical columns
        non_numerical_cols = [col for col in df.columns if col not in numerical_cols]

        # Build result
        result_parts = []
        if non_numerical_cols:
            result_parts.append(df[non_numerical_cols])
        if keep_columns:
            result_parts.append(df[keep_columns])
        result_parts.append(ica_df)

        df_result = pd.concat(result_parts, axis=1)

        print(f"✅ FastICA (Selective Mode): Replaced {len(replace_columns)} features, "
              f"kept {len(keep_columns)}, added {n_components} ICA components")
        if replace_columns:
            print(f"   Replaced: {replace_columns}")
        if keep_columns:
            print(f"   Kept: {keep_columns}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'hybrid' or 'selective'")

    # 7. Optional: Add interaction features from ICA components
    if add_interaction_features:
        interaction_features = _create_ica_interactions(
            ica_df,
            n_interactions=min(5, n_components)
        )
        if not interaction_features.empty:
            df_result = pd.concat([df_result, interaction_features], axis=1)
            print(f"   Added {len(interaction_features.columns)} ICA interaction features")

    return df_result

def _calculate_intelligent_replace_ratio(
    df: pd.DataFrame,
    numerical_cols: List[str],
    n_components: int,
    analysis_results: dict = None,
    min_ratio: float = 0.1,
    max_ratio: float = 0.7,
) -> float:
    """
    Calculates replacement ratio based on feature redundancy metrics.
    
    Optimized for stability:
    1. Handles mismatch between current columns and cached analysis results.
    2. Uses both average correlation and high-correlation density.
    """
    # 1. Base Ratio: Proportional to dimensionality reduction needs
    n_features = max(1, len(numerical_cols))
    base_ratio = n_components / n_features + 0.1
    
    # If no analysis results, return safe base ratio immediately
    if not analysis_results:
        return float(max(min_ratio, min(max_ratio, base_ratio)))

    try:
        # 2. Extract Correlation Matrix Safely
        corr_data = analysis_results.get("correlations", {}).get("correlation_matrix")
        if not corr_data:
            return float(max(min_ratio, min(max_ratio, base_ratio)))

        # 3. Robust Indexing (Key Stability Fix)
        # analysis_results might be old (pre-feature-generation), so we must find the intersection
        # of columns to avoid KeyError during .loc
        cached_cols = pd.DataFrame(corr_data).columns
        valid_cols = list(set(numerical_cols).intersection(cached_cols))
        
        if len(valid_cols) < 2:
            # Not enough overlapping columns to judge redundancy
            return float(max(min_ratio, min(max_ratio, base_ratio)))

        corr_df = pd.DataFrame(corr_data).loc[valid_cols, valid_cols]

        # 4. Calculate Statistics (Vectorized)
        # Extract off-diagonal elements
        mask = ~np.eye(len(corr_df), dtype=bool)
        values = corr_df.abs().values[mask]
        
        if len(values) == 0:
            return float(max(min_ratio, min(max_ratio, base_ratio)))

        avg_corr = values.mean()
        # Density of very high correlations (> 0.7)
        # This catches cases where avg is low, but specific clusters are redundant
        high_corr_density = (values > 0.7).mean()

        # 5. Dynamic Adjustment Strategy
        if avg_corr < 0.2:
            base_ratio -= 0.15  # Mostly independent -> keep original
        elif avg_corr < 0.3:
            base_ratio -= 0.08
        elif avg_corr >= 0.6:
            # If extremely redundant OR high density of strong pairs -> replace aggressively
            if high_corr_density > 0.3:
                base_ratio += 0.15
            else:
                base_ratio += 0.08
        elif avg_corr >= 0.5:
             base_ratio += 0.05

    except Exception as e:
        # Fail safe: log if needed, but return base ratio to keep pipeline alive
        # print(f"Warning in replace ratio calc: {e}")
        pass
    # 6. Final Clamp
    return float(max(min_ratio, min(max_ratio, base_ratio)))


def _select_features_to_replace(
    df: pd.DataFrame,
    numerical_cols: List[str],
    n_to_replace: int,
    analysis_results: dict = None
) -> tuple:
    """
    Decides which features to replace based on Importance (Target Corr > Variance).
    
    Optimized for:
    1. Speed: Uses vectorized operations for variance fallback.
    2. Stability: Guarantees at least 1 feature is kept.
    """
    # 1. Boundary Check
    total_feats = len(numerical_cols)
    n_to_replace = max(0, min(total_feats, n_to_replace))
    
    if n_to_replace == 0:
        return [], numerical_cols

    # 2. Initialize Scores
    scores = {col: 0.0 for col in numerical_cols}
    used_target_corr = False

    # 3. Strategy A: Target Correlation (Priority)
    if analysis_results:
        target_corr = analysis_results.get("correlations", {}).get("target_correlation", {})
        
        # FIX: Ensure it is a dictionary before trying to access keys
        if isinstance(target_corr, dict) and target_corr:
            for col in numerical_cols:
                val = target_corr.get(col)
                if isinstance(val, (int, float)) and not pd.isna(val):
                    scores[col] = abs(val)
            
            # Check if we actually got any non-zero signal
            if any(s > 0 for s in scores.values()):
                used_target_corr = True
    # 4. Strategy B: Variance (Fallback) - Vectorized
    # Only run if Target Correlation failed or provided no signal
    if not used_target_corr:
        try:
            # Vectorized calculation is faster than loop
            variances = df[numerical_cols].var().fillna(0)
            scores = variances.to_dict()
        except Exception:
            # Absolute fallback if something goes wrong
            pass
    else:
        try:
            variances = df[numerical_cols].var().fillna(0)
            for col in numerical_cols:
                if scores.get(col, 0.0) == 0.0:  # No correlation info available
                    var_score = variances.get(col, 0.0)
                    # Use a small positive value for zero-variance features to avoid confusion
                    scores[col] = var_score if var_score > 0 else 1e-10
        except Exception:
            pass


    # 5. Sort: Lowest Score -> Replace First
    # Primary sort key: Score (ascending). Secondary key: Name (for deterministic tie-breaking)
    ordered_cols = sorted(numerical_cols, key=lambda c: (scores.get(c, 0.0), c))

    # 6. Slicing with Safety Net
    # If algorithm wants to replace ALL features, forcefully keep the best one
    if n_to_replace == total_feats:
        n_to_replace -= 1

    cols_to_replace = ordered_cols[:n_to_replace]
    cols_to_keep = ordered_cols[n_to_replace:]

    return cols_to_replace, cols_to_keep


def _create_ica_interactions(ica_df: pd.DataFrame, n_interactions: int = 5) -> pd.DataFrame:
    """
    Creates pairwise interactions of ICA components.
    
    Optimized for:
    1. Readability: Uses itertools to replace nested loops.
    2. Performance: Builds dict first to avoid DataFrame fragmentation.
    """
    # 1. Boundary Checks
    if ica_df.empty or n_interactions <= 0 or len(ica_df.columns) < 2:
        return pd.DataFrame(index=ica_df.index)

    interactions = {}
    
    # 2. Generate Combinations (Safe & Pythonic)
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    for col1, col2 in itertools.combinations(ica_df.columns, 2):
        
        # Stop if quota reached
        if len(interactions) >= n_interactions:
            break
            
        # Create Interaction
        interactions[f'{col1}_x_{col2}'] = ica_df[col1] * ica_df[col2]

    return pd.DataFrame(interactions, index=ica_df.index)

def _evaluate_fastica_performance(
    replace_ratio: float,
    df_work: pd.DataFrame,
    target_column: str,
    original_numerical: List[str],
    model,
    cv,
    is_reg: bool,
    eval_metric: str,
    pt: str,
    apply_fastica_fn, 
    fastica_args: dict
) -> Dict[str, Any]:
    """
    Common evaluation logic shared by Grid Search and Optuna.
    Executes the entire FastICA + CV pipeline for a single replace_ratio.
    """
    try:
        # A. Transform Data
        df_trans = apply_fastica_fn(
            df_work,
            target_column=target_column,
            replace_ratio=replace_ratio,
            **fastica_args
        )
        
        X = df_trans.drop(columns=[target_column])
        y = df_trans[target_column]
        
        if X.empty: raise ValueError("Empty features after transformation")

        # B. Metadata Counting
        n_features_after = len(X.columns)
        n_replaced = max(1, int(len(original_numerical) * replace_ratio))
        
        # C. Cross-Validation
        model_clone = clone(model) if hasattr(model, 'fit') else model.__class__(**model.get_params())
        
        # Select Scoring based on problem type
        if is_reg:
            scoring = 'neg_root_mean_squared_error'
            raw_scores = cross_val_score(model_clone, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            scores = -raw_scores # Convert to positive RMSE
        else:
            if eval_metric == "roc_auc" and pt == "binary":
                raw_scores = cross_val_score(model_clone, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                scores = 1.0 - raw_scores # Minimize (1 - AUC)
            else:
                scoring = 'neg_log_loss' if hasattr(model_clone, "predict_proba") else 'accuracy'
                raw_scores = cross_val_score(model_clone, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                scores = -raw_scores if 'neg' in scoring else (1.0 - raw_scores)

        return {
            "score": np.mean(scores),
            "std": np.std(scores),
            "n_features_after": n_features_after,
            "n_replaced": n_replaced,
            "error": None
        }

    except Exception as e:
        return {"score": float('inf'), "error": str(e)}

def grid_search_replace_ratio(
    df: pd.DataFrame,
    target_column: str,
    model,
    problem_type: str,
    replace_ratios: List[float] = None,
    n_components: int = None,
    exclude_columns: List[str] = None,
    analysis_results: dict = None,
    cv_folds: int = 5,
    random_state: int = 42,
    **fastica_kwargs
) -> pd.DataFrame:
    """
    Grid search wrapper reusing the unified evaluation engine.
    Significantly shorter and cleaner.
    """
    # Local import to avoid circular dependency    
    # 1. Setup
    if replace_ratios is None: replace_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    df_work = df.copy()
    if target_column not in df_work.columns: raise ValueError(f"Target column '{target_column}' not found")
    
    pt = problem_type.lower()
    is_reg = pt == "regression"
    eval_metric = "rmse" if is_reg else "roc_auc" if pt == "binary" else "log_loss"
    
    # CV Setup
    cv_cls = KFold if is_reg else StratifiedKFold
    cv = cv_cls(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Identify original numerical columns (for counting stats)
    original_numerical = df_work.select_dtypes(include=np.number).columns.tolist()
    if target_column in original_numerical: original_numerical.remove(target_column)
    
    fastica_args = {
        "n_components": n_components,
        "exclude_columns": exclude_columns,
        "mode": 'hybrid',
        "analysis_results": analysis_results,
        "random_state": random_state,
        **fastica_kwargs
    }
    
    # 2. Loop
    results = []
    print(f"🔎 Starting Grid Search for Replace Ratio ({len(replace_ratios)} values)...")
    
    for ratio in replace_ratios:
        # Call the engine!
        res = _evaluate_fastica_performance(
            ratio, df_work, target_column, original_numerical, 
            model, cv, is_reg, eval_metric, pt, apply_fastica, fastica_args
        )
        
        status = "error" if res["error"] else "success"
        mean_score = res.get("score", np.nan)
        
        print(f"  Ratio {ratio:.2f}: Score={mean_score:.4f} ({status})")
        
        results.append({
            "replace_ratio": ratio,
            "mean_score": mean_score,
            "std_score": res.get("std", np.nan),
            "n_features_after": res.get("n_features_after", np.nan),
            "status": status,
            "error_message": res.get("error")
        })
        
    return pd.DataFrame(results).sort_values("mean_score")


def tune_fastica_replace_ratio(
    df: pd.DataFrame,
    target_column: str,
    model,
    problem_type: str,
    n_trials: int = 20,
    **fastica_kwargs
) -> dict:
    """
    Optuna Tuning Wrapper reusing the unified evaluation engine.
    
    Optimized for:
    1. DRY Principle: Reuses _evaluate_fastica_performance.
    2. Observability: Records metadata (feature counts) into Optuna trials.
    """
    # Local import    
    # 1. Standard Setup
    df_work = df.copy()
    pt = problem_type.lower()
    is_reg = pt == "regression"
    eval_metric = "rmse" if is_reg else "roc_auc" if pt == "binary" else "log_loss"
    
    cv_cls = KFold if is_reg else StratifiedKFold
    cv = cv_cls(n_splits=5, shuffle=True, random_state=42)
    
    original_numerical = df_work.select_dtypes(include=np.number).columns.tolist()
    if target_column in original_numerical: original_numerical.remove(target_column)

    # 2. Define Objective
    def objective(trial: optuna.Trial) -> float:
        # Suggest parameter
        replace_ratio = trial.suggest_float("replace_ratio", 0.1, 0.7, step=0.05)
        
        # Call Unified Engine
        res = _evaluate_fastica_performance(
            replace_ratio, df_work, target_column, original_numerical, 
            model, cv, is_reg, eval_metric, pt, apply_fastica, fastica_kwargs
        )
        
        # Error Handling
        if res["error"]: 
            return float('inf')
            
        # [CRITICAL ADDITION] Record Metadata for Analysis
        # This restores the functionality of the "Long Version" while keeping code short
        trial.set_user_attr("n_features_after", res["n_features_after"])
        trial.set_user_attr("n_replaced", res["n_replaced"])
        trial.set_user_attr("std_score", res["std"])
        
        return res["score"]

    # 3. Run Optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    return {
        "study": study, 
        "best_replace_ratio": study.best_params["replace_ratio"],
        "best_score": study.best_value
    }



def decide_correlation_threshold(all_corrs, method='auto', **kwargs):
    """
    Decide an adaptive correlation threshold using AutoML-style distribution analysis.
    """
    if not all_corrs:
        return 0.0

    # Ensure numpy array and sort
    corrs = np.sort(np.array(all_corrs))
    min_threshold = kwargs.get('min_threshold', 0.1) 
    
    if method == 'auto':
        n_points = len(corrs)
        if n_points < 3:
            return max(np.max(corrs) * 0.9, min_threshold)
            
        x = np.linspace(0, 1, n_points)
        y = (corrs - corrs.min()) / (corrs.max() - corrs.min() + 1e-9)
        
        start_point = np.array([x[0], y[0]])
        end_point = np.array([x[-1], y[-1]])
        vec_line = end_point - start_point
        vec_points = np.stack([x, y], axis=1) - start_point
        
        cross_prod = np.cross(vec_points, vec_line)
        distances = np.abs(cross_prod) / np.linalg.norm(vec_line)
        
        knee_index = np.argmax(distances)
        threshold = corrs[knee_index]
        
        if threshold < 0.15:
            threshold = np.percentile(corrs, 90)
            
    elif method == 'percentile':
        p = kwargs.get('percentile', 75)
        threshold = np.percentile(corrs, p)
        
    elif method == 'mean_std':
        sigma = kwargs.get('sigma', 1.5)
        threshold = np.mean(corrs) + sigma * np.std(corrs)
        threshold = min(threshold, 1.0)
    else:
        threshold = 0.5

    return float(max(threshold, min_threshold))


def assess_ica_necessity(n_samples, n_features, corr_matrix_dict, potential_new_features=0):
    """
    AutoML logic to determine if ICA/Dimensionality Reduction is necessary.
    Returns: (should_apply: bool, reasons: str, suggested_n: int)
    """
    if n_features < 4: return False, None, 0

    reasons = []
    total_estimated_features = n_features + potential_new_features
    np_ratio = n_samples / (total_estimated_features + 1e-9)
    
    is_crowded = np_ratio < 10.0 
    if is_crowded:
        reasons.append(f"Low sample-to-feature ratio ({np_ratio:.1f})")

    redundancy_score = 0.0
    if corr_matrix_dict:
        try:
            df_corr = pd.DataFrame(corr_matrix_dict)
            mask = np.triu(np.ones(df_corr.shape), k=1).astype(bool)
            abs_corr = df_corr.where(mask).abs()
            n_pairs = mask.sum()
            n_high_corr = (abs_corr > 0.5).sum().sum()
            
            if n_pairs > 0: redundancy_score = n_high_corr / n_pairs
            if redundancy_score > 0.1: 
                reasons.append(f"High feature redundancy (density: {redundancy_score:.1%})")
        except Exception: pass

    is_huge = n_features > 100
    if is_huge: reasons.append(f"High absolute dimensionality ({n_features} features)")

    should_apply = is_crowded or (redundancy_score > 0.15) or is_huge
    
    compression_rate = 0.3 if np_ratio < 5 else 0.5
    suggested_n = int(n_features * compression_rate)
    suggested_n = max(2, min(suggested_n, n_samples // 2, 100))

    return should_apply, "; ".join(reasons), suggested_n