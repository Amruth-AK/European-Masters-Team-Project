import pandas as pd

# -------------------------------
# Drop constant features
# -------------------------------
def drop_constant_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop columns that have a single unique value (constant)."""
    df = df.copy()
    cols_to_drop = [col for col in columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


# -------------------------------
# Drop low variance numeric features
# -------------------------------
def drop_low_variance_features(df: pd.DataFrame, columns: list, threshold: float = 1e-5) -> pd.DataFrame:
    """Drop numeric columns with variance below threshold."""
    df = df.copy()
    cols_to_drop = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].std() < threshold:
                cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


# -------------------------------
# Drop one of highly correlated features
# -------------------------------
def drop_correlated_features(df: pd.DataFrame, columns: list, threshold: float = 0.95) -> pd.DataFrame:
    """Drop one column from pairs of highly correlated features."""
    df = df.copy()
    if len(columns) < 2:
        return df
    # drop second column of pair by default
    col_to_drop = columns[1]
    if col_to_drop in df.columns:
        df = df.drop(columns=[col_to_drop])
    return df


# -------------------------------
# Reduce high-cardinality categorical features
# -------------------------------
def reduce_categorical_cardinality(df: pd.DataFrame, columns: list, max_cardinality: int = 1000) -> pd.DataFrame:
    """Reduce cardinality of categorical columns by frequency encoding."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals > max_cardinality:
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
    return df


# -------------------------------
# Suggest irrelevant feature removal
# -------------------------------
def suggest_irrelevant_feature_removal(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate suggestions for removing irrelevant features (constant, low variance, highly correlated, or high-cardinality).
    Identifier columns are excluded because they are handled separately.
    """
    suggestions = []

    # Extract needed info
    categorical_info = analysis_results.get('categorical_info', {})
    numeric_summary = analysis_results.get('numerical_summary', {})
    correlation_info = analysis_results.get('correlation_matrix', {})

    # Rule 1: Constant or near-constant features
    for col, info in categorical_info.items():
        if col == target_column:
            continue
        unique_vals = info.get('unique_values', 0)
        if unique_vals <= 1:
            suggestions.append({
                'feature': col,
                'issue': f'Feature "{col}" has a constant value.',
                'suggestion': 'Drop constant features — they provide no variance or information.',
                'function_to_call': 'drop_constant_features',
                'kwargs': {'columns': [col]}
            })

    for col, stats in numeric_summary.items():
        if col == target_column:
            continue
        std_val = stats.get('std', 0)
        if std_val == 0 or std_val < 1e-5:
            suggestions.append({
                'feature': col,
                'issue': f'Feature "{col}" has near-zero variance (std={std_val:.6f}).',
                'suggestion': 'Drop near-constant numeric features.',
                'function_to_call': 'drop_low_variance_features',
                'kwargs': {'columns': [col], 'threshold': 1e-5}
            })

    # Rule 2: Highly correlated features (redundant)
    if correlation_info:
        correlated_pairs = []
        for col1, vals in correlation_info.items():
            for col2, corr in vals.items():
                if col1 != col2 and abs(corr) > 0.95:
                    correlated_pairs.append((col1, col2, corr))

        for col1, col2, corr in correlated_pairs:
            suggestions.append({
                'feature': f'{col1}, {col2}',
                'issue': f'High correlation ({corr:.2f}) between "{col1}" and "{col2}".',
                'suggestion': 'Drop one of the correlated features to avoid multicollinearity.',
                'function_to_call': 'drop_correlated_features',
                'kwargs': {'columns': [col1, col2], 'threshold': 0.95}
            })

    # Rule 3: High-cardinality categorical features (exclude ID-like columns)
    for col, info in categorical_info.items():
        unique_vals = info.get('unique_values', 0)
        if unique_vals > 1000:
            suggestions.append({
                'feature': col,
                'issue': f'High-cardinality categorical feature "{col}" ({unique_vals} unique values).',
                'suggestion': 'Consider frequency encoding or dimensionality reduction.',
                'function_to_call': 'reduce_categorical_cardinality',
                'kwargs': {'columns': [col], 'max_cardinality': 1000}
            })

    return suggestions
