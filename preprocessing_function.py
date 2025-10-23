import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.impute import KNNImputer

# ===================================================================
# Missing values
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
# Duplicate Values
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

