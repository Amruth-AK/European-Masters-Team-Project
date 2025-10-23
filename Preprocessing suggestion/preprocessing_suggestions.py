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
                    'kwargs': {'column': col, 'whisker_width': 1.5}
                })
            
            # Rule 2: Large percentage of outliers (>=5%) -> consider removal
            else:
                suggestions.append({
                    'feature': col,
                    'issue': f'High proportion of outliers ({outlier_count} rows, {outlier_percentage:.1f}%)',
                    'suggestion': 'Consider removing outlier rows if dataset is sufficiently large.',
                    'function_to_call': 'remove_outliers_iqr',
                    'kwargs': {'column': col, 'whisker_width': 1.5}
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
                'kwargs': {'columns': col, 'drop_first': True}
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
