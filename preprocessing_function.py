import pandas as pd
import numpy as np
from typing import Union, List, Dict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, log_loss
import optuna

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
    target_column: str = None,
    exclude_columns: List[str] = None,
    feature_types: List[str] = ['product', 'ratio'],
    analysis_results: Dict = None, 
    # Filtering parameters
    use_basic_filter: bool = True,
    use_correlation_filter: bool = True,
    min_cardinality: int = 3,
    max_new_features: int = 500,
    corr_filter_threshold: float = 0.95,
    min_variance: float = 1e-6  
) -> pd.DataFrame:
    """
    Create new features from highly correlated pairs with multi-layer filtering.
    
    Uses the same correlation calculation method as analyze.py.
    
    Args:
        df: Input DataFrame
        correlation_threshold: Min correlation for feature pairs (default: 0.3)
        target_column: Optional target column
        exclude_columns: Columns to exclude
        feature_types: ['product', 'ratio', 'difference', 'sum', 'interaction']
        analysis_results: Optional (kept for compatibility, but not used)
        use_basic_filter: Remove constant/low-cardinality/low-variance features
        use_correlation_filter: Remove redundant features (corr > threshold)
        min_cardinality: Min unique values (default: 3)
        max_new_features: Max features to create (default: 500)
        corr_filter_threshold: Threshold for redundancy filtering (default: 0.95)
        min_variance: Minimum variance threshold for filtering (default: 1e-6)
    
    Returns:
        DataFrame with new features
    """
    df = df.copy()
    
    # 1. Prepare numerical columns (same as analyze.py line 54)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if exclude_columns:
        numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if len(numerical_cols) < 2:
        print("⚠️ Need at least 2 numerical columns")
        return df
    
    # 2. Get correlation matrix (same logic as analyze.py line 59)
    corr_matrix = df[numerical_cols].corr()
    
    # 3. Find high correlation pairs
    high_corr_pairs = _find_high_corr_pairs(
        corr_matrix, numerical_cols, target_column, correlation_threshold
    )
    
    if not high_corr_pairs:
        print(f"❌ No pairs found with |corr| > {correlation_threshold}")
        return df
    
    print(f"✓ Found {len(high_corr_pairs)} high correlation pairs")
    
    # 4. Generate candidate features
    candidate_df = _generate_candidate_features(df, high_corr_pairs, feature_types)
    
    if candidate_df.empty:
        print("❌ No candidate features generated")
        return df
    
    print(f"✓ Generated {len(candidate_df.columns)} candidate features")
    
    # 5. Apply filters
    if use_basic_filter:
        candidate_df = _apply_basic_filter(candidate_df, min_cardinality, min_variance)
    
    if use_correlation_filter and len(candidate_df.columns) > 1:
        candidate_df = _apply_correlation_filter(candidate_df, corr_filter_threshold)
    
    # 6. Limit features
    if len(candidate_df.columns) > max_new_features:
        print(f"⚠️ Limiting to {max_new_features} features")
        candidate_df = candidate_df.sample(n=max_new_features, random_state=42, axis=1)
    
    # 7. Add to DataFrame
    for col in candidate_df.columns:
        df[col] = candidate_df[col]
    
    print(f"✅ Added {len(candidate_df.columns)} new features")
    return df


# Helper functions
def _get_correlation_matrix(df, numerical_cols, analysis_results):
    """Get correlation matrix using the same logic as analyze.py."""
    # Same calculation as analyze.py line 59: self.df[numerical_cols].corr()
    return df[numerical_cols].corr()


def _find_high_corr_pairs(corr_matrix, numerical_cols, target_column, threshold):
    """Find pairs with high correlation."""
    pairs = []
    
    # Handle threshold = 0.0 (include all pairs)
    if threshold == 0.0:
        for i, col1 in enumerate(numerical_cols):
            if col1 not in corr_matrix.index:
                continue
            for col2 in numerical_cols[i+1:]:
                if col2 not in corr_matrix.columns:
                    continue
                try:
                    corr_value = corr_matrix.loc[col1, col2]
                    if pd.notna(corr_value):
                        pairs.append((col1, col2, corr_value))
                except (KeyError, IndexError):
                    continue
        return pairs
    
    # Normal case: threshold > 0
    # Prioritize target correlations
    if target_column and target_column in corr_matrix.index:
        target_corrs = corr_matrix[target_column].abs()
        high_corr_cols = target_corrs[target_corrs > threshold].index.tolist()
        if target_column in high_corr_cols:
            high_corr_cols.remove(target_column)
        
        for col in high_corr_cols:
            if col in numerical_cols:
                pairs.append((target_column, col, corr_matrix.loc[target_column, col]))
    
    # Find all high correlation pairs
    for i, col1 in enumerate(numerical_cols):
        for col2 in numerical_cols[i+1:]:
            try:
                corr_value = corr_matrix.loc[col1, col2]
                if pd.notna(corr_value) and abs(corr_value) > threshold:
                    if not any((c1, c2) in [(col1, col2), (col2, col1)] for c1, c2, _ in pairs):
                        pairs.append((col1, col2, corr_value))
            except (KeyError, IndexError):
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


def _apply_basic_filter(candidate_df, min_cardinality, min_variance=1e-6):
    """
    Apply basic filtering: constant, low cardinality, high missing, low variance.
    
    Args:
        candidate_df: DataFrame with candidate features
        min_cardinality: Minimum number of unique values required
        min_variance: Minimum variance threshold (default: 1e-6)
    
    Returns:
        Filtered DataFrame
    """
    n_before = len(candidate_df.columns)
    
    # Remove constant, low cardinality, high missing, and low variance features
    to_keep = []
    removed_reasons = {'constant': 0, 'low_cardinality': 0, 'high_missing': 0, 'low_variance': 0}
    
    for col in candidate_df.columns:
        # Check cardinality
        unique_count = candidate_df[col].nunique()
        if unique_count < min_cardinality:
            removed_reasons['low_cardinality'] += 1
            continue
        
        # Check missing values
        missing_pct = candidate_df[col].isnull().mean()
        if missing_pct > 0.5:
            removed_reasons['high_missing'] += 1
            continue
        
        # Check variance (new filtering criterion)
        variance = candidate_df[col].var()
        if pd.isna(variance) or variance < min_variance:
            removed_reasons['low_variance'] += 1
            continue
        
        # Check if constant (variance = 0)
        if variance == 0:
            removed_reasons['constant'] += 1
            continue
        
        to_keep.append(col)
    
    candidate_df = candidate_df[to_keep]
    
    # Print detailed filtering results
    print(f"  Basic filter: {n_before} → {len(candidate_df.columns)} features")
    if any(removed_reasons.values()):
        filter_details = [f"{reason}: {count}" for reason, count in removed_reasons.items() if count > 0]
        print(f"    Removed - {', '.join(filter_details)}")
    
    return candidate_df


def _apply_correlation_filter(candidate_df, threshold):
    """Remove redundant features with high correlation."""
    n_before = len(candidate_df.columns)
    
    corr_matrix = candidate_df.corr().abs()
    to_remove = set()
    
    for i, col1 in enumerate(candidate_df.columns):
        if col1 in to_remove:
            continue
        for col2 in candidate_df.columns[i+1:]:
            if col2 not in to_remove and corr_matrix.loc[col1, col2] > threshold:
                to_remove.add(col2)
    
    candidate_df = candidate_df.drop(columns=list(to_remove))
    print(f"  Correlation filter: {n_before} → {len(candidate_df.columns)} features")
    
    return candidate_df


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
    Grid search to test different replace_ratio values for FastICA.
    
    This function provides a macro-level view by testing multiple replace_ratio values
    and recording detailed information for each trial, including feature counts and
    cross-validation scores.
    
    Args:
        df: Input dataframe
        target_column: Target column name
        model: Model instance or model class (must support fit/predict)
        problem_type: 'regression', 'binary', or 'multiclass'
        replace_ratios: List of replace_ratio values to test. 
                       If None, uses default range [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
        n_components: Number of ICA components
        exclude_columns: Columns to exclude from ICA
        analysis_results: Analysis results dictionary
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed
        **fastica_kwargs: Additional arguments for apply_fastica
        
    Returns:
        DataFrame with columns:
            - replace_ratio: Tested replace_ratio value
            - mean_score: Mean CV score
            - std_score: Standard deviation of CV scores
            - min_score: Minimum CV score
            - max_score: Maximum CV score
            - n_features_before: Number of features before FastICA
            - n_features_after: Number of features after FastICA
            - n_components: Number of ICA components used
            - n_replaced: Number of features replaced
            - n_kept: Number of features kept
            - n_interactions: Number of interaction features added (if enabled)
            - status: 'success' or 'error'
            - error_message: Error message if status is 'error'
    """
    import numpy as np
    
    if replace_ratios is None:
        replace_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    
    df_work = df.copy()
    if target_column not in df_work.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    pt = problem_type.lower()
    is_reg = pt == "regression"
    
    # Setup cross-validation
    if is_reg:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    y = df_work[target_column]
    n_features_before = len(df_work.columns) - 1  # Exclude target
    
    results = []
    
    for replace_ratio in replace_ratios:
        print(f"Testing replace_ratio = {replace_ratio:.2f}...")
        
        try:
            # Apply FastICA
            df_transformed = apply_fastica(
                df_work,
                n_components=n_components,
                target_column=target_column,
                exclude_columns=exclude_columns,
                mode='hybrid',
                replace_ratio=replace_ratio,
                analysis_results=analysis_results,
                random_state=random_state,
                **fastica_kwargs
            )
            
            # Prepare features and target
            X = df_transformed.drop(columns=[target_column])
            y_transformed = df_transformed[target_column]
            
            if X.empty or len(X.columns) == 0:
                print(f"  ⚠️ Empty features, skipping")
                results.append({
                    "replace_ratio": replace_ratio,
                    "mean_score": np.nan,
                    "std_score": np.nan,
                    "min_score": np.nan,
                    "max_score": np.nan,
                    "n_features_before": n_features_before,
                    "n_features_after": 0,
                    "n_components": n_components,
                    "n_replaced": 0,
                    "n_kept": 0,
                    "n_interactions": 0,
                    "status": "error",
                    "error_message": "Empty features after transformation"
                })
                continue
            
            n_features_after = len(X.columns)
            
            # Count ICA components and interaction features
            ica_cols = [col for col in X.columns if col.startswith('ICA_')]
            interaction_cols = [col for col in X.columns if '_x_' in col and any(c.startswith('ICA_') for c in col.split('_x_'))]
            n_ica_components = len(ica_cols)
            n_interactions = len(interaction_cols)
            
            # Estimate n_replaced and n_kept (approximate)
            original_numerical = df_work.select_dtypes(include=np.number).columns.tolist()
            if target_column in original_numerical:
                original_numerical.remove(target_column)
            if exclude_columns:
                original_numerical = [col for col in original_numerical if col not in exclude_columns]
            
            n_replaced = max(1, int(len(original_numerical) * replace_ratio))
            n_kept = len(original_numerical) - n_replaced
            
            # Clone model
            if hasattr(model, 'fit'):
                model_clone = clone(model)
            else:
                model_clone = model.__class__(**model.get_params())
            
            # Cross-validation
            if is_reg:
                scores = -cross_val_score(
                    model_clone, X, y_transformed, cv=cv,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
            else:
                if hasattr(model_clone, "predict_proba"):
                    if pt == "binary":
                        scores = cross_val_score(
                            model_clone, X, y_transformed, cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1
                        )
                        scores = 1.0 - scores  # Convert to minimize (lower is better)
                    else:
                        scores = cross_val_score(
                            model_clone, X, y_transformed, cv=cv,
                            scoring='neg_log_loss',
                            n_jobs=-1
                        )
                        scores = -scores  # Convert to minimize
                else:
                    scores = cross_val_score(
                        model_clone, X, y_transformed, cv=cv,
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    scores = 1.0 - scores  # Convert to minimize
            
            results.append({
                "replace_ratio": replace_ratio,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "n_features_before": n_features_before,
                "n_features_after": n_features_after,
                "n_components": n_ica_components,
                "n_replaced": n_replaced,
                "n_kept": n_kept,
                "n_interactions": n_interactions,
                "status": "success",
                "error_message": None
            })
            print(f"  ✅ Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f} "
                  f"(Features: {n_features_before} → {n_features_after})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "replace_ratio": replace_ratio,
                "mean_score": np.nan,
                "std_score": np.nan,
                "min_score": np.nan,
                "max_score": np.nan,
                "n_features_before": n_features_before,
                "n_features_after": np.nan,
                "n_components": n_components,
                "n_replaced": np.nan,
                "n_kept": np.nan,
                "n_interactions": np.nan,
                "status": "error",
                "error_message": str(e)
            })
            continue
    
    results_df = pd.DataFrame(results)
    # Sort by mean_score (ascending for regression/log_loss, but for AUC we want descending)
    # Since we converted AUC to (1-AUC) for minimization, ascending sort works for all
    results_df = results_df.sort_values("mean_score", ascending=True, na_last=True)
    
    return results_df


def tune_fastica_replace_ratio(
    df: pd.DataFrame,
    target_column: str,
    model,
    problem_type: str,
    eval_metric: str = None,
    n_components: int = None,
    exclude_columns: List[str] = None,
    analysis_results: dict = None,
    n_trials: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    **fastica_kwargs
) -> dict:
    """
    Use Optuna to automatically search for optimal replace_ratio.
    
    This function provides micro-level fine-tuning by using Optuna's optimization
    to find the best replace_ratio value within a continuous range. It uses
    cross-validation to avoid overfitting and records detailed information for
    each trial.
    
    Args:
        df: Input dataframe
        target_column: Target column name
        model: Model instance or model class (must support fit/predict)
        problem_type: 'regression', 'binary', or 'multiclass'
        eval_metric: Evaluation metric name. If None, auto-selected:
                    'rmse' for regression, 'roc_auc' for binary, 'log_loss' for multiclass
        n_components: Number of ICA components
        exclude_columns: Columns to exclude from ICA
        analysis_results: Analysis results dictionary
        n_trials: Number of Optuna trials (default: 20)
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed
        **fastica_kwargs: Additional arguments for apply_fastica
        
    Returns:
        dict with keys:
            - best_replace_ratio: Optimal replace_ratio value
            - best_score: Best CV score achieved
            - study: Optuna study object
            - results_df: DataFrame with all trial results including:
                * replace_ratio: Tested value
                * score: CV score for this trial
                * n_features_before: Features before transformation
                * n_features_after: Features after transformation
                * n_components: ICA components used
                * n_replaced: Features replaced
                * n_kept: Features kept
                * n_interactions: Interaction features added
                * state: Trial state (COMPLETE, PRUNED, etc.)
    """
    import numpy as np
    
    # Prepare data
    df_work = df.copy()
    if target_column not in df_work.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    # Determine evaluation metric
    pt = problem_type.lower()
    is_reg = pt == "regression"
    
    if eval_metric is None:
        eval_metric = "rmse" if is_reg else "roc_auc" if pt == "binary" else "log_loss"
    
    # Setup cross-validation
    if is_reg:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    y = df_work[target_column]
    n_features_before = len(df_work.columns) - 1  # Exclude target
    
    # Get original numerical columns for estimation
    original_numerical = df_work.select_dtypes(include=np.number).columns.tolist()
    if target_column in original_numerical:
        original_numerical.remove(target_column)
    if exclude_columns:
        original_numerical = [col for col in original_numerical if col not in exclude_columns]
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest replace_ratio value
        replace_ratio = trial.suggest_float("replace_ratio", 0.1, 0.7, step=0.05)
        
        # Apply FastICA
        try:
            df_transformed = apply_fastica(
                df_work,
                n_components=n_components,
                target_column=target_column,
                exclude_columns=exclude_columns,
                mode='hybrid',
                replace_ratio=replace_ratio,
                analysis_results=analysis_results,
                random_state=random_state,
                **fastica_kwargs
            )
        except Exception as e:
            print(f"⚠️ FastICA failed with replace_ratio={replace_ratio:.3f}: {e}")
            return float('inf') if is_reg or eval_metric == "log_loss" else 0.0
        
        # Prepare features and target
        X = df_transformed.drop(columns=[target_column])
        y_transformed = df_transformed[target_column]
        
        # Check data validity
        if X.empty or len(X.columns) == 0:
            return float('inf') if is_reg or eval_metric == "log_loss" else 0.0
        
        # Count features
        ica_cols = [col for col in X.columns if col.startswith('ICA_')]
        interaction_cols = [col for col in X.columns if '_x_' in col and any(c.startswith('ICA_') for c in col.split('_x_'))]
        n_ica_components = len(ica_cols)
        n_interactions = len(interaction_cols)
        n_features_after = len(X.columns)
        n_replaced = max(1, int(len(original_numerical) * replace_ratio))
        n_kept = len(original_numerical) - n_replaced
        
        # Store trial metadata
        trial.set_user_attr("n_features_before", n_features_before)
        trial.set_user_attr("n_features_after", n_features_after)
        trial.set_user_attr("n_components", n_ica_components)
        trial.set_user_attr("n_replaced", n_replaced)
        trial.set_user_attr("n_kept", n_kept)
        trial.set_user_attr("n_interactions", n_interactions)
        
        # Cross-validation
        scores = []
        for train_idx, val_idx in cv.split(X, y_transformed):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_transformed.iloc[train_idx], y_transformed.iloc[val_idx]
            
            # Clone model
            if hasattr(model, 'fit'):
                model_clone = clone(model)
            else:
                model_clone = model.__class__(**model.get_params())
            
            model_clone.fit(X_train, y_train)
            
            # Predict
            if is_reg:
                y_pred = model_clone.predict(X_val)
                if eval_metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                else:
                    score = mean_squared_error(y_val, y_pred)
            else:
                if hasattr(model_clone, "predict_proba"):
                    y_proba = model_clone.predict_proba(X_val)
                    if eval_metric == "roc_auc" and pt == "binary":
                        score = roc_auc_score(y_val, y_proba[:, 1])
                        score = 1.0 - score  # Optuna minimizes
                    else:
                        score = log_loss(y_val, y_proba)
                else:
                    y_pred = model_clone.predict(X_val)
                    score = log_loss(y_val, y_pred)
            
            scores.append(score)
        
        return np.mean(scores)
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Extract results
    best_replace_ratio = study.best_params["replace_ratio"]
    best_score = study.best_value
    
    # Create results DataFrame with all trial information
    results_data = []
    for trial in study.trials:
        results_data.append({
            "replace_ratio": trial.params.get("replace_ratio", np.nan),
            "score": trial.value,
            "n_features_before": trial.user_attrs.get("n_features_before", np.nan),
            "n_features_after": trial.user_attrs.get("n_features_after", np.nan),
            "n_components": trial.user_attrs.get("n_components", np.nan),
            "n_replaced": trial.user_attrs.get("n_replaced", np.nan),
            "n_kept": trial.user_attrs.get("n_kept", np.nan),
            "n_interactions": trial.user_attrs.get("n_interactions", np.nan),
            "state": trial.state.name
        })
    results_df = pd.DataFrame(results_data)
    
    return {
        "best_replace_ratio": best_replace_ratio,
        "best_score": best_score,
        "study": study,
        "results_df": results_df
    }
