import pandas as pd
import numpy as np


def suggest_missing_value_handling(analysis_results: dict) -> list:
    """
    Generate handling suggestions for missing values based on analysis results.

    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()

    Returns:
        list: Preprocessing suggestions for missing value imputation or deletion.
    """
    suggestions = []

    missing_info = analysis_results.get('missing_values', {})
    data_types = analysis_results.get('general_info', {}).get('data_types', {})

    for col, info in missing_info.items():
        missing_pct = info.get('missing_percentage', 0)
        missing_count = info.get('missing_count', 0)
        dtype = data_types.get(col, '')

        if missing_pct == 0:
            continue

        # --- Rule 1: High missingness (>50%) — delete column
        if missing_pct > 50:
            suggestions.append({
                'feature': col,
                'issue': f'{missing_pct:.2f}% missing values (too high)',
                'suggestion': 'Column has excessive missing values; consider dropping it.',
                'function_to_call': 'delete_missing_columns',
                'kwargs': {'threshold': 0.5}
            })

        # --- Rule 2: Moderate missingness (20–50%) — impute intelligently
        elif 20 < missing_pct <= 50:
            if 'float' in dtype or 'int' in dtype:
                suggestions.append({
                    'feature': col,
                    'issue': f'{missing_pct:.2f}% missing numeric data',
                    'suggestion': 'Use median imputation to handle moderate missing values.',
                    'function_to_call': 'impute_median',
                    'kwargs': {'columns': col}
                })
            else:
                suggestions.append({
                    'feature': col,
                    'issue': f'{missing_pct:.2f}% missing categorical data',
                    'suggestion': 'Use mode imputation to fill missing categorical values.',
                    'function_to_call': 'impute_mode',
                    'kwargs': {'columns': col}
                })

        # --- Rule 3: Low missingness (<=20%) — simple imputation
        else:
            if 'float' in dtype or 'int' in dtype:
                suggestions.append({
                    'feature': col,
                    'issue': f'Low missingness ({missing_pct:.2f}%) detected',
                    'suggestion': 'Use mean imputation for small missing value percentages.',
                    'function_to_call': 'impute_mean',
                    'kwargs': {'columns': col}
                })
            else:
                suggestions.append({
                    'feature': col,
                    'issue': f'Low missingness ({missing_pct:.2f}%) detected in categorical feature',
                    'suggestion': 'Use mode imputation for categorical columns.',
                    'function_to_call': 'impute_mode',
                    'kwargs': {'columns': col}
                })

    # Optional: check row-level missingness across the dataset
    total_missing_rows = sum(1 for row in analysis_results.get('missing_values', {}).values()
                             if row.get('missing_count', 0) > 0)
    if total_missing_rows > 0:
        suggestions.append({
            'feature': 'rows',
            'issue': 'Some rows contain multiple missing values.',
            'suggestion': 'Consider deleting rows with more than 50% missing data.',
            'function_to_call': 'delete_missing_rows',
            'kwargs': {'threshold': 0.5}
        })

    return suggestions


def suggest_duplicate_handling(analysis_results: dict) -> list:
    """
    Generate handling suggestions for duplicate rows based on analysis results.

    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()

    Returns:
        list: Preprocessing suggestions for duplicate removal.
    """
    suggestions = []

    dup_info = analysis_results.get('duplicate_info', {})
    total_dup = dup_info.get('total_duplicates', 0)
    dup_pct = dup_info.get('duplicate_percentage', 0)

    # --- Rule 1: No duplicates
    if total_dup == 0:
        return suggestions

    # --- Rule 2: Few duplicates (<1%)
    if dup_pct <= 1:
        suggestions.append({
            'feature': 'rows',
            'issue': f'Low duplicate ratio ({dup_pct:.2f}%) detected.',
            'suggestion': 'Remove duplicate rows to avoid redundancy.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    # --- Rule 3: Moderate duplicates (1–10%)
    elif 1 < dup_pct <= 10:
        suggestions.append({
            'feature': 'rows',
            'issue': f'{dup_pct:.2f}% duplicate rows detected.',
            'suggestion': 'Remove duplicates; verify if they are genuine or data collection artifacts.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    # --- Rule 4: High duplicates (>10%)
    else:
        suggestions.append({
            'feature': 'rows',
            'issue': f'High duplicate percentage ({dup_pct:.2f}%) detected.',
            'suggestion': 'Investigate source of duplication before deletion; may indicate merging issues.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    return suggestions


def suggest_numerical_scaling(analysis_results: dict) -> list:
    """
    Generate scaling suggestions for numerical features based on analysis results.
    
    Args:
        analysis_results: Dictionary from Adrian's analysis() function
    
    Returns:
        list: Preprocessing suggestions for numerical scaling
    """
    suggestions = []
    
    # Get data types to identify numerical columns
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    numerical_cols = [col for col, dtype in data_types.items() 
                     if 'int' in dtype or 'float' in dtype]
    
    # Get distribution info (skewness)
    distributions = analysis_results.get('distributions', {})
    
    # Get outlier info
    outlier_info = analysis_results.get('outlier_info', {})
    
    for col in numerical_cols:
        # Get skewness for this column
        skewness = distributions.get(col, {}).get('skewness', 0)
        
        # Check if column has outliers
        has_outliers = outlier_info.get(col, {}).get('outlier_count', 0) > 0
        
        # Rule 1: High skewness (>1 or <-1) -> use standard scaler
        if abs(skewness) > 1:
            suggestions.append({
                'feature': col,
                'issue': f'High skewness detected ({skewness:.2f})',
                'suggestion': 'Apply standard scaling (z-score normalization) to normalize distribution.',
                'function_to_call': 'standard_scaler',
                'kwargs': {'column': col}
            })
        
        # Rule 2: Low skewness and no outliers -> use minmax scaler
        elif not has_outliers:
            suggestions.append({
                'feature': col,
                'issue': 'Numerical feature requires scaling for model compatibility',
                'suggestion': 'Apply min-max scaling to transform values to [0,1] range.',
                'function_to_call': 'minmax_scaler',
                'kwargs': {'column': col, 'feature_range': (0, 1)}
            })
        
        # Rule 3: Has outliers but low skewness -> still recommend minmax but warn
        else:
            suggestions.append({
                'feature': col,
                'issue': 'Numerical feature with some outliers detected',
                'suggestion': 'Apply min-max scaling after handling outliers.',
                'function_to_call': 'minmax_scaler',
                'kwargs': {'column': col, 'feature_range': (0, 1)}
            })
    
    return suggestions


def suggest_outlier_handling(analysis_results: dict) -> list:
    """
    Generate outlier handling suggestions based on analysis results.
    
    Args:
        analysis_results: Dictionary from Adrian's analysis() function
    
    Returns:
        list: Preprocessing suggestions for outlier handling
    """
    suggestions = []
    
    # Get outlier information from Adrian's analysis
    outlier_info = analysis_results.get('outlier_info', {})
    
    for col, info in outlier_info.items():
        outlier_count = info.get('outlier_count', 0)
        outlier_percentage = info.get('outlier_percentage', 0)
        
        # Only suggest if outliers exist
        if outlier_count > 0:
            
            # Rule 1: Small percentage of outliers (<5%) -> clip outliers
            if outlier_percentage < 5.0:
                suggestions.append({
                    'feature': col,
                    'issue': f'Contains {outlier_count} outliers ({outlier_percentage:.1f}% of data)',
                    'suggestion': 'Clip outliers to IQR boundaries to preserve all rows while limiting extreme values.',
                    'function_to_call': 'clip_outliers_iqr',
                    'kwargs': {'column': col, 'whisker_width': 1.5, 'analysis_results': analysis_results}
                })
            
            # Rule 2: Large percentage of outliers (>=5%) -> consider removal
            else:
                suggestions.append({
                    'feature': col,
                    'issue': f'High proportion of outliers ({outlier_count} rows, {outlier_percentage:.1f}%)',
                    'suggestion': 'Consider removing outlier rows if dataset is sufficiently large.',
                    'function_to_call': 'remove_outliers_iqr',
                    'kwargs': {'column': col, 'whisker_width': 1.5, 'analysis_results': analysis_results}
                })
    
    return suggestions




def suggest_categorical_encoding(analysis_results: dict) -> list:
    """
    Generate encoding suggestions for categorical features based on analysis results.
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
    
    Returns:
        list: Preprocessing suggestions for categorical encoding
    """
    suggestions = []
    
    # Extract info
    categorical_info = analysis_results.get('categorical_info', {})
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    
    for col, info in categorical_info.items():
        unique_count = info.get('unique_values', 0)
        value_counts = info.get('value_counts', {})
        total_values = sum(value_counts.values()) if value_counts else 0
        
        # Skip numeric columns
        if 'int' in data_types.get(col, '') or 'float' in data_types.get(col, ''):
            continue
        
        # Rule 1: Binary categorical (2 unique)
        if unique_count == 2:
            suggestions.append({
                'feature': col,
                'issue': 'Binary categorical feature detected (2 unique values)',
                'suggestion': 'Apply Binary Encoding to represent the two categories with binary bits.',
                'function_to_call': 'binary_encode',
                'kwargs': {'columns': col}
            })
        
        # Rule 2: Ordinal-like pattern
        elif any(keyword in col.lower() for keyword in ['level', 'grade', 'rank', 'score', 'stage', 'class']):
            suggestions.append({
                'feature': col,
                'issue': 'Possible ordinal feature (name suggests order, e.g., Level, Rank, Grade)',
                'suggestion': 'Apply Ordinal Encoding if the categories follow a natural order.',
                'function_to_call': 'ordinal_encode',
                'kwargs': {'columns': col, 'category_orders': None}
            })
        
        # Rule 3: Low cardinality (3–10)
        elif 3 <= unique_count <= 10:
            suggestions.append({
                'feature': col,
                'issue': f'Low-cardinality feature ({unique_count} unique values)',
                'suggestion': 'Apply One-Hot Encoding to create binary columns for each category.',
                'function_to_call': 'one_hot_encode',
                'kwargs': {'columns': col, 'drop_first': False}
            })
        
        # Rule 4: Medium cardinality (10–50)
        elif 10 < unique_count <= 50:
            suggestions.append({
                'feature': col,
                'issue': f'Medium-cardinality categorical feature ({unique_count} unique values)',
                'suggestion': 'Apply Label Encoding to convert categories to integer labels.',
                'function_to_call': 'label_encode',
                'kwargs': {'columns': col}
            })
        
        # Rule 5: High cardinality (>50)
        elif unique_count > 50:
            suggestions.append({
                'feature': col,
                'issue': f'High-cardinality categorical feature ({unique_count} unique values)',
                'suggestion': 'Apply Frequency Encoding to represent categories by their occurrence frequency.',
                'function_to_call': 'frequency_encode',
                'kwargs': {'columns': col}
            })
    
    return suggestions

def suggest_identifier_removal(analysis_results: dict) -> list:
    """
    Generate suggestions to remove identifier columns.
    
    Detection Rules:
    1. Column name contains 'id', 'index', 'uid', 'key'
    2. OR unique values > 95% of total rows (high cardinality)
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
    
    Returns:
        list: Preprocessing suggestions for identifier removal
    """
    suggestions = []
    
    # Get data info
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    total_rows = analysis_results.get('general_info', {}).get('shape', [0])[0]
    
    # Keywords to detect
    identifier_keywords = ['id', 'index', 'uid', 'key', 'identifier', '_id', 'pk']
    
    # Collect all identifier columns
    identifier_cols = []
    
    for col in data_types.keys():
        # Check 1: Name contains identifier keyword
        is_identifier = any(keyword in col.lower() for keyword in identifier_keywords)
        
        # Check 2: High cardinality (>95% unique values)
        is_high_cardinality = False
        if total_rows > 0:
            # Try to get unique count from feature_duplicate_info
            feat_dup = analysis_results.get('feature_duplicate_info', {}).get(col, {})
            if feat_dup:
                duplicate_count = feat_dup.get('duplicate_count', 0)
                unique_count = total_rows - duplicate_count
                if unique_count / total_rows > 0.95:
                    is_high_cardinality = True
        
        # Add to list if detected as identifier
        if is_identifier or is_high_cardinality:
            identifier_cols.append(col)
    
    # Only create ONE suggestion if any identifier columns found
    # Because remove_identifier_columns processes all columns at once
    if identifier_cols:
        suggestions.append({
            'feature': ', '.join(identifier_cols),
            'issue': f'Identifier column(s) detected: {", ".join(identifier_cols)}',
            'suggestion': 'Remove identifier columns as they provide no predictive value and can harm model performance.',
            'function_to_call': 'remove_identifier_columns',
            'kwargs': {'pattern': 'id', 'max_unique_ratio': 0.95}
        })
    
    return suggestions



def suggest_datetime_features(analysis_results: dict) -> list:
    """
    Generate suggestions for datetime feature engineering.
    
    Detects datetime columns and suggests extracting useful temporal features.
    Most ML models cannot handle datetime objects directly, so we need to
    convert them into numerical features.
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
    
    Returns:
        list: Preprocessing suggestions for datetime feature extraction
    """
    suggestions = []
    
    # Get data types
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    
    # Detect datetime columns
    datetime_cols = []
    potential_datetime_cols = []
    
    for col, dtype in data_types.items():
        # Check if already datetime type
        if 'datetime' in dtype.lower():
            datetime_cols.append(col)
        # Check if might be datetime (object/string with date-like names)
        elif 'object' in dtype:
            if any(keyword in col.lower() for keyword in 
                   ['date', 'time', 'timestamp', 'dt', 'datetime', 'day']):
                potential_datetime_cols.append(col)
    
    # --- Rule 1: Extract features from confirmed datetime columns ---
    for col in datetime_cols:
        suggestions.append({
            'feature': col,
            'issue': f'Datetime column detected: {col}',
            'suggestion': 'Extract temporal features (year, month, day, etc.) as most models cannot process datetime directly.',
            'function_to_call': 'extract_datetime_features',
            'kwargs': {
                'column': col,
                'features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'],
                'drop_original': True
            }
        })
    
    # --- Rule 2: Suggest checking potential datetime columns ---
    if potential_datetime_cols:
        suggestions.append({
            'feature': ', '.join(potential_datetime_cols),
            'issue': f'Potential datetime columns detected (currently string/object type): {", ".join(potential_datetime_cols)}',
            'suggestion': 'Verify if these are datetime columns and convert them. Then extract temporal features.',
            'function_to_call': 'extract_datetime_features',
            'kwargs': {
                'column': potential_datetime_cols[0] if potential_datetime_cols else '',
                'features': ['year', 'month', 'day', 'dayofweek'],
                'drop_original': True
            }
        })
    
    # --- Rule 3: Suggest time difference if multiple datetime columns exist ---
    if len(datetime_cols) >= 2:
        suggestions.append({
            'feature': f'{datetime_cols[0]}, {datetime_cols[1]}',
            'issue': f'Multiple datetime columns found: {", ".join(datetime_cols)}',
            'suggestion': 'Consider calculating time differences between datetime columns to create duration features.',
            'function_to_call': 'calculate_datetime_diff',
            'kwargs': {
                'col1': datetime_cols[0],
                'col2': datetime_cols[1],
                'unit': 'days',
                'new_col_name': None
            }
        })
    
    return suggestions    




# ============================================================================
# Ignore Target Columns
# While use it in main.py or dashboard.py
# suggestions = get_all_suggestions(analysis_results, target_column='accident_risk')
# ============================================================================


def filter_target_from_suggestions(suggestions: list, target_column: str = None) -> list:
    """
    Remove any suggestions that involve the target column.
    
    This ensures the target variable is never modified during preprocessing.
    
    Args:
        suggestions: List of suggestion dictionaries
        target_column: Name of the target column to exclude
    
    Returns:
        Filtered list of suggestions without target column
    
    Example:
        >>> all_suggestions = suggest_missing_value_handling(results)
        >>> filtered = filter_target_from_suggestions(all_suggestions, 'price')
    """
    if not target_column:
        return suggestions
    
    filtered = []
    for suggestion in suggestions:
        feature = suggestion.get('feature', '')
        
        # Check if this suggestion involves the target column
        # Handle both single column and comma-separated multiple columns
        if ',' in feature:
            # Multiple columns case (e.g., "col1, col2")
            columns = [col.strip() for col in feature.split(',')]
            if target_column not in columns:
                filtered.append(suggestion)
        else:
            # Single column case
            if feature != target_column:
                filtered.append(suggestion)
    
    return filtered


def get_all_suggestions(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate all preprocessing suggestions and automatically filter out target column.
    
    This is a convenience function that calls all suggestion functions and filters
    the results in one go.
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
        target_column: Name of target column to exclude from all suggestions
    
    Returns:
        Complete list of filtered preprocessing suggestions
    
    Example:
        >>> suggestions = get_all_suggestions(analysis_results, target_column='accident_risk')
    """
    all_suggestions = []
    
    # Collect all suggestions
    all_suggestions.extend(suggest_missing_value_handling(analysis_results))
    all_suggestions.extend(suggest_duplicate_handling(analysis_results))
    all_suggestions.extend(suggest_numerical_scaling(analysis_results))
    all_suggestions.extend(suggest_outlier_handling(analysis_results))
    all_suggestions.extend(suggest_identifier_removal(analysis_results))
    all_suggestions.extend(suggest_datetime_features(analysis_results))
    all_suggestions.extend(suggest_categorical_encoding(analysis_results))
    
    # Filter out target column
    return filter_target_from_suggestions(all_suggestions, target_column)


# ============================================================================
# Add new features - Create Features from High Correlation Pairs - Zhiqi
# ============================================================================
def suggest_correlation_based_features(analysis_results: dict) -> list:
    """
    Generate suggestions for creating new features from highly correlated feature pairs.
    Adaptively adjusts threshold based on actual correlation distribution.
    """
    suggestions = []
    
    # Get correlation matrix from analysis results
    correlations = analysis_results.get('correlations', {})
    corr_matrix_dict = correlations.get('correlation_matrix', {})
    
    if not corr_matrix_dict:
        return suggestions
    
    try:
        corr_matrix = pd.DataFrame(corr_matrix_dict)
    except Exception:
        return suggestions
    
    # Get data types to identify numerical columns
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    numerical_cols = [col for col, dtype in data_types.items() 
                     if 'int' in dtype or 'float' in dtype]
    
    # Exclude ID columns (they have abnormal correlation values)
    id_keywords = ['id', 'index', 'uid', 'key', 'identifier', '_id', 'pk']
    numerical_cols = [col for col in numerical_cols 
                     if not any(keyword in col.lower() for keyword in id_keywords)]
    
    if len(numerical_cols) < 2:
        return suggestions
    
    # Collect all correlation values (excluding diagonal)
    all_corrs = []
    for i, col1 in enumerate(numerical_cols):
        if col1 not in corr_matrix.index:
            continue
        for col2 in numerical_cols[i+1:]:
            if col2 not in corr_matrix.columns:
                continue
            try:
                corr_value = corr_matrix.loc[col1, col2]
                if pd.notna(corr_value):
                    # Filter out abnormal values (should be between -1 and 1)
                    if -1 <= corr_value <= 1:
                        all_corrs.append(abs(corr_value))
            except (KeyError, IndexError):
                continue
    
    if not all_corrs:
        return suggestions
    
    # Analyze correlation distribution
    max_corr = max(all_corrs)
    median_corr = np.median(all_corrs)
    
    # Adaptive threshold selection based on actual data
    if max_corr < 0.3:
        # Very low correlations - use all pairs
        correlation_threshold = 0.01
        strategy = "all_pairs"
        feature_types = ['product','ratio']
    elif max_corr < 0.5:
        # Low correlations - use very low threshold
        correlation_threshold = 0.2
        strategy = "low_correlation"
        feature_types = ['product', 'ratio']
    elif max_corr < 0.7:
        # Moderate correlations - use moderate threshold
        correlation_threshold = 0.4
        strategy = "moderate_correlation"
        feature_types = ['product', 'ratio', 'difference']
    elif max_corr < 0.8:
        # Good correlations - use standard threshold
        correlation_threshold = 0.7
        strategy = "standard"
        feature_types = ['product', 'ratio', 'difference', 'sum']
    else:
        # High correlations - use high threshold
        correlation_threshold = 0.8
        strategy = "high_correlation"
        feature_types = ['product', 'ratio', 'difference', 'sum']
    
    # Count pairs at selected threshold
    high_corr_pairs = []
    for i, col1 in enumerate(numerical_cols):
        if col1 not in corr_matrix.index:
            continue
        for col2 in numerical_cols[i+1:]:
            if col2 not in corr_matrix.columns:
                continue
            try:
                corr_value = corr_matrix.loc[col1, col2]
                if pd.notna(corr_value) and -1 <= corr_value <= 1:
                    if correlation_threshold == 0.0 or abs(corr_value) > correlation_threshold:
                        high_corr_pairs.append((col1, col2, corr_value))
            except (KeyError, IndexError):
                continue
    
    if len(high_corr_pairs) == 0:
        return suggestions
    
    # Determine max features based on dataset size
    shape = analysis_results.get('general_info', {}).get('shape', [0, 0])
    n_samples, n_features = shape[0], shape[1]
    
    # Adaptive max_new_features based on dataset size and strategy
    if strategy == "all_pairs":
        if n_samples < 1000:
            max_new_features = 10
        elif n_samples < 10000:
            max_new_features = 20
        else:
            max_new_features = 30
    else:
        if n_samples < 1000:
            max_new_features = 50
        elif n_samples < 10000:
            max_new_features = 100
        else:
            max_new_features = 200
    
    # Create suggestion with adaptive message
    if strategy == "all_pairs":
        issue_msg = (
            f"Very low correlations detected (max: {max_corr:.2f}, median: {median_corr:.2f}). "
            f"Suggesting interaction features from all {len(high_corr_pairs)} feature pairs "
            f"to explore non-linear relationships."
        )
        suggestion_msg = (
            f"Even with low correlations, interaction features (A*B, A/B, etc.) can capture "
            f"non-linear patterns that linear correlation misses. This will generate features "
            f"from all {len(high_corr_pairs)} feature pairs."
        )
    elif strategy == "low_correlation":
        issue_msg = (
            f"Low correlations detected (max: {max_corr:.2f}, median: {median_corr:.2f}). "
            f"Found {len(high_corr_pairs)} pairs with |corr| > {correlation_threshold}"
        )
        suggestion_msg = (
            f"Create interaction features from {len(high_corr_pairs)} feature pairs "
            f"with |correlation| > {correlation_threshold} to capture potential non-linear relationships."
        )
    else:
        issue_msg = (
            f"Found {len(high_corr_pairs)} pairs with |corr| > {correlation_threshold} "
            f"(max correlation: {max_corr:.2f}, median: {median_corr:.2f})"
        )
        suggestion_msg = (
            f"Create interaction features from {len(high_corr_pairs)} feature pairs "
            f"with |correlation| > {correlation_threshold} to capture non-linear relationships."
        )
    
    suggestions.append({
        'feature': 'multiple',
        'issue': issue_msg,
        'suggestion': suggestion_msg,
        'function_to_call': 'create_features_from_correlation_analysis',
        'kwargs': {
            'correlation_threshold': correlation_threshold,
            'feature_types': feature_types,
            'use_basic_filter': True,
            'use_correlation_filter': True,
            'min_cardinality': 5,
            'max_new_features': max_new_features,
            'corr_filter_threshold': 0.9,
            'min_variance': 1e-4,

        }
    })
    
    return suggestions
