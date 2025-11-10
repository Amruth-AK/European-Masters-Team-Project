# preprocessing_suggestions.py
import itertools


def suggest_missing_value_handling(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate handling suggestions for missing values based on analysis results.

    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
        target_column: Name of the target column to skip or deal with differently.

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
        
        if col == target_column:
            # Skip target column suggestions for now
            continue

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


def suggest_numerical_scaling(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate scaling suggestions for numerical features based on analysis results.
    
    Args:
        analysis_results: Dictionary from Adrian's analysis() function
        target_column: Name of the target column to skip or deal with differently.
    
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
        
        if col == target_column:
            # Skip target column suggestions for now
            continue
        
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


def suggest_outlier_handling(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate outlier handling suggestions based on analysis results.
    
    Args:
        analysis_results: Dictionary from Adrian's analysis() function
        target_column: Name of the target column to skip or deal with differently.
    
    Returns:
        list: Preprocessing suggestions for outlier handling
    """
    suggestions = []
    
    # Get outlier information from Adrian's analysis
    outlier_info = analysis_results.get('outlier_info', {})
    
    for col, info in outlier_info.items():
        outlier_count = info.get('outlier_count', 0)
        outlier_percentage = info.get('outlier_percentage', 0)
        
        if col == target_column:
            # Skip target column suggestions for now
            continue
        
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




def suggest_categorical_encoding(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate encoding suggestions for categorical features based on analysis results.
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
        target_column: Name of the target column to skip or deal with differently.
    
    Returns:
        list: Preprocessing suggestions for categorical encoding
    """
    suggestions = []
    
    # Extract info
    categorical_info = analysis_results.get('categorical_info', {})
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    
    for col, info in categorical_info.items():
        if col == target_column:
            # Skip target column suggestions for now
            continue
        
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
    Generate suggestions to remove identifier columns using the ID columns
    already detected by DataAnalyzer in analyze.py.
    """
    suggestions = []

    # Get the ID columns that were already detected and ignored in duplicate analysis
    row_dup_info = analysis_results.get('row_duplicate_info', {})
    identifier_cols = row_dup_info.get('ignored_columns', [])

    if not identifier_cols:
        return suggestions

    suggestions.append({
        'feature': ', '.join(identifier_cols),
        'issue': f'Identifier column(s) detected: {', '.join(identifier_cols)}',
        'suggestion': 'Remove identifier columns as they typically have no '
                      'predictive value and can negatively impact model performance.',
        'function_to_call': 'remove_identifier_columns',
        'kwargs': {'id_columns': identifier_cols}
    })

    return suggestions



def suggest_datetime_features(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate suggestions for datetime feature engineering.
    
    Detects datetime columns and suggests extracting useful temporal features.
    Most ML models cannot handle datetime objects directly, so we need to
    convert them into numerical features.
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
        target_column: Name of the target column to skip or deal with differently.
    
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
        if col == target_column:
            # Skip target column suggestions for now
            continue
        
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



def suggest_feature_combination(analysis_results: dict) -> list:
    """
    Generates suggestions for creating new features by combining categorical columns.

    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis().

    Returns:
        list: Preprocessing suggestions for feature combination.
    """
    suggestions = []
    categorical_info = analysis_results.get('categorical_info', {})
    categorical_cols = list(categorical_info.keys())

    # We need at least two categorical columns to create combinations
    if len(categorical_cols) < 2:
        return suggestions

    # Generate all unique pairs of categorical columns
    column_pairs = list(itertools.combinations(categorical_cols, 2))

    for col1, col2 in column_pairs:
        new_col_name = f"{col1}_{col2}_combined"
        suggestions.append({
            'feature': f"'{col1}' and '{col2}'",
            'issue': "Potential to capture interaction effects between features.",
            'suggestion': (f"Combine '{col1}' and '{col2}' into a single feature "
                           "to potentially improve model performance by representing their interaction."),
            'function_to_call': 'combine_categorical_features',
            'kwargs': {
                'columns_to_combine': [col1, col2],
                'new_col_name': new_col_name,
                'drop_original': False
            }
        })

    return suggestions