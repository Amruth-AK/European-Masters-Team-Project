import pandas as pd
import numpy as np
from typing import Union, List, Dict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


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

def clip_outliers_iqr(df: pd.DataFrame, column: str, whisker_width: float = 1.5) -> pd.DataFrame:
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
    
    Returns:
        DataFrame with outliers clipped to boundary values
    
    Example:
        >>> df = clip_outliers_iqr(df, 'Fare')
        >>> df = clip_outliers_iqr(df, 'Age', whisker_width=2.0)
    
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
    
    # Calculate quartiles and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Calculate boundaries
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


def remove_outliers_iqr(df: pd.DataFrame, column: str, whisker_width: float = 1.5) -> pd.DataFrame:
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
    
    Returns:
        DataFrame with outlier rows removed
    
    Example:
        >>> df = remove_outliers_iqr(df, 'Fare')
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
    
    # Calculate quartiles and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    # Calculate boundaries
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
