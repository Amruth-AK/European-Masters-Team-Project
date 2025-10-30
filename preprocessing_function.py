import pandas as pd
import numpy as np
from typing import Union, List, Dict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer

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

def remove_identifier_columns(df: pd.DataFrame, 
                             pattern: str = 'id',
                             max_unique_ratio: float = 0.95) -> pd.DataFrame:
    """
    Automatically remove identifier/id-like columns (e.g., id, index, uid).
    
    Detection Logic:
    1. Column name contains 'id' (or specified pattern)
    2. OR unique values > 95% of total rows (high cardinality)
    
    Args:
        df: Input DataFrame
        pattern: Keyword to search in column names (case-insensitive)
        max_unique_ratio: Threshold for unique value ratio (0.95 = 95%)
    
    Returns:
        DataFrame with identifier columns removed
    
    Example:
        >>> df = remove_identifier_columns(df)
        >>> df = remove_identifier_columns(df, pattern='key', max_unique_ratio=0.99)
    """
    df = df.copy()
    to_remove = []
    identifier_keywords = ['id', 'index', 'uid', 'key', 'identifier', 'pk']
    
    for col in df.columns:
        
        if any(kw == col.lower() or col.lower().endswith(f"_{kw}") for kw in identifier_keywords):
            to_remove.append(col)
            continue

        
        # Check 2: High unique ratio
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > max_unique_ratio:
            to_remove.append(col)
    
    if to_remove:
        df = df.drop(columns=to_remove)
        print(f"Removed identifier columns: {to_remove}")
    
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