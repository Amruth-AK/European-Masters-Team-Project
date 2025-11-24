import numpy as np
import pandas as pd
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

    dup_info = (
        analysis_results.get('row_duplicate_info')
        or analysis_results.get('duplicate_info', {})
    )
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
            
            # --- MODIFIED LOGIC ---
            # ALWAYS suggest clipping to avoid accidentally deleting too many rows in wide datasets.
            # The original logic that suggested removal for high outlier percentages is too risky.
            
            suggestions.append({
                'feature': col,
                'issue': f'Contains {outlier_count} outliers ({outlier_percentage:.2f}% of data)',
                'suggestion': 'Clip outliers to the IQR boundaries. This contains extreme values without deleting rows, which is safer for datasets with many columns.',
                'function_to_call': 'clip_outliers_iqr',  # Always use clip
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
        'issue': f"Identifier column(s) detected: {', '.join(identifier_cols)}",
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

def decide_correlation_threshold(all_corrs, method='percentile', percentile=75):
    """
    Decide an adaptive correlation threshold based on the distribution of correlations.

    Parameters:
        all_corrs (list): List of absolute correlation values between features.
        method (str): 'percentile' or 'iqr' for threshold calculation.
        percentile (float): Percentile to use if method='percentile'.

    Returns:
        threshold (float): Suggested threshold.
    """
    if not all_corrs:
        return 0.0

    all_corrs = np.array(all_corrs)

    if method == 'percentile':
        threshold = np.percentile(all_corrs, percentile)
    elif method == 'iqr':
        q1 = np.percentile(all_corrs, 25)
        q3 = np.percentile(all_corrs, 75)
        threshold = min(q3 + 1.5 * (q3 - q1), 1.0)
    else:
        raise ValueError("Unsupported method. Choose 'percentile' or 'iqr'.")

    # Avoid thresholds that are too low
    return max(threshold, 0.1)


def suggest_correlation_based_features(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate suggestions for creating new features from highly correlated feature pairs.
    Adaptively adjusts threshold based on actual correlation distribution.
    """
    suggestions = []

    # Get correlation matrix
    correlations = analysis_results.get('correlations', {})
    corr_matrix_dict = correlations.get('correlation_matrix', {})

    if not corr_matrix_dict:
        return suggestions

    try:
        corr_matrix = pd.DataFrame(corr_matrix_dict)
    except Exception:
        return suggestions

    if target_column is not None:
        corr_matrix = corr_matrix.drop(index=target_column, columns=target_column, errors='ignore')

    # Identify numerical columns (excluding ID-like columns)
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    numerical_cols = [
        col for col, dtype in data_types.items()
        if ('int' in dtype or 'float' in dtype) and col != target_column
    ]
    id_keywords = ['id', 'index', 'uid', 'key', 'identifier', '_id', 'pk']
    numerical_cols = [col for col in numerical_cols if not any(k in col.lower() for k in id_keywords)]

    if len(numerical_cols) < 2:
        return suggestions

    # Collect all absolute correlations (excluding diagonal)
    all_corrs = []
    for i, col1 in enumerate(numerical_cols):
        if col1 not in corr_matrix.index:
            continue
        for col2 in numerical_cols[i + 1:]:
            if col2 not in corr_matrix.columns:
                continue
            corr_value = corr_matrix.loc[col1, col2]
            if pd.notna(corr_value) and -1 <= corr_value <= 1:
                all_corrs.append(abs(corr_value))

    if not all_corrs:
        return suggestions

    # Adaptive threshold
    correlation_threshold = decide_correlation_threshold(all_corrs, method='percentile', percentile=75)

    # Decide strategy and feature types
    if correlation_threshold < 0.2:
        strategy = "all_pairs"
        feature_types = ['product', 'ratio']
    elif correlation_threshold < 0.4:
        strategy = "low_correlation"
        feature_types = ['product', 'ratio']
    elif correlation_threshold < 0.7:
        strategy = "moderate_correlation"
        feature_types = ['product', 'ratio', 'difference']
    else:
        strategy = "high_correlation"
        feature_types = ['product', 'ratio', 'difference', 'sum']

    # Filter high-correlation pairs
    high_corr_pairs = []
    for i, col1 in enumerate(numerical_cols):
        if col1 not in corr_matrix.index:
            continue
        for col2 in numerical_cols[i + 1:]:
            if col2 not in corr_matrix.columns:
                continue
            corr_value = corr_matrix.loc[col1, col2]
            if pd.notna(corr_value) and -1 <= corr_value <= 1:
                if abs(corr_value) >= correlation_threshold:
                    high_corr_pairs.append((col1, col2, corr_value))

    if not high_corr_pairs:
        return suggestions

    # Adaptive max_new_features based on dataset characteristics # Add Zhiqi
    shape = analysis_results.get('general_info', {}).get('shape', [0, 0])
    n_samples = shape[0] if shape else 0
    n_features = shape[1] if shape else 0
    
    # Formula: min(samples/10, original_features*2, 200)
    # This ensures: 1) sample-to-feature ratio stays healthy
    #               2) don't add too many features relative to original
    #               3) absolute cap to prevent explosion
    if strategy == "all_pairs":
        # More conservative for all_pairs strategy
        max_new_features = min(
            n_samples // 15,      # Conservative ratio
            n_features,           # Don't exceed original feature count
            100                   # Lower absolute cap
        )
        max_new_features = max(20, max_new_features)  # At least 20
    else:
        # More generous for targeted high-correlation pairs
        max_new_features = min(
            n_samples // 10,      # Healthy sample-to-feature ratio
            n_features * 2,       # Up to 2x original features
            200                   # Reasonable absolute cap
        )
        max_new_features = max(50, max_new_features)  # At least 50
    
    print(f"📊 Dataset: {n_samples} samples, {n_features} features")
    print(f"📊 Adaptive limit: {max_new_features} new features (strategy: {strategy})")

    # Create suggestion messages
    median_corr = np.median(all_corrs)
    if strategy == "all_pairs":
        issue_msg = (
            f"Very low correlations detected (max: {max(all_corrs):.2f}, median: {median_corr:.2f}). "
            f"Suggesting interaction features from all {len(high_corr_pairs)} feature pairs "
            f"to explore non-linear relationships."
        )
        suggestion_msg = (
            f"Even with low correlations, interaction features (A*B, A/B, etc.) can capture "
            f"non-linear patterns that linear correlation misses. This will generate features "
            f"from all {len(high_corr_pairs)} feature pairs."
        )
    else:
        issue_msg = (
            f"Found {len(high_corr_pairs)} pairs with |corr| >= {correlation_threshold:.2f} "
            f"(max correlation: {max(all_corrs):.2f}, median: {median_corr:.2f})"
        )
        suggestion_msg = (
            f"Create interaction features from {len(high_corr_pairs)} feature pairs "
            f"with |correlation| >= {correlation_threshold:.2f} to capture non-linear relationships."
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
            'min_cardinality': 2,              # Relaxed from 4 to 2
            'max_new_features': max_new_features,
            'corr_filter_threshold': 0.95,     # Relaxed from 0.9 to 0.95
            'min_variance': 1e-6,              # Relaxed from 1e-4 to 1e-6
            'target_column': target_column,
            # Third-order feature generation parameters
            'generate_third_order': True,           # Enable 3rd-order features
            'min_second_for_third': 10,             # Reduced from 20 to 10 (easier to trigger)
            'max_third_order_features': 30,         # Maximum 3rd-order features to keep
            'top_second_for_third': 50,             # Top N 2nd-order features to use
            'top_original_for_third': 30            # Top N original features to use
        }
    })

    return suggestions

def suggest_feature_combination(
    analysis_results: dict,
    target_column: str = None,
    max_pairs: int = 9999,          # not a real cap anymore; budget controls the stop
    memory_growth_cap: float = 0.25 # allow at most +25% more columns (as a proxy for memory)
) -> list:
    """
    Adaptive categorical feature-combination suggestions with a data-driven threshold.

    Budget logic (stop-conditions):
      1) Column budget from samples: total_new_columns <= min(n_samples/5, 2000)
      2) Column budget from memory proxy: total_new_columns <= current_cols * memory_growth_cap
      3) Always respect whichever budget is smaller.
    """
    import itertools
    suggestions = []

    cat_info = analysis_results.get("categorical_info", {})
    dtypes = analysis_results.get("general_info", {}).get("data_types", {})
    shape = analysis_results.get("general_info", {}).get("shape", [0, 0])
    n_rows, n_cols = shape if isinstance(shape, (list, tuple)) else (0, 0)

    # ---- 1) Eligible categorical columns by cardinality & type ----
    def is_cat(col):
        t = dtypes.get(col, "")
        return ("object" in t) or ("category" in t)

    eligible = []
    for col, info in cat_info.items():
        if col == target_column:
            continue
        if not is_cat(col):
            continue
        uv = info.get("unique_values", 0)
        # Keep moderate-cardinality only (avoid too tiny/too huge)
        if 2 <= uv <= 30:
            eligible.append(col)

    if len(eligible) < 2:
        return suggestions

    # ---- 2) Rank pairs by estimated joint cardinality (smaller first) ----
    pairs = list(itertools.combinations(eligible, 2))
    pairs.sort(key=lambda p: cat_info[p[0]]["unique_values"] * cat_info[p[1]]["unique_values"])

    # ---- 3) Build an adaptive budget in "new columns" units ----
    # Budget A: sample-based (avoid too many columns vs. rows)
    budget_from_samples = int(min(max(n_rows // 5, 10), 2000))  # at least 10, at most 2000

    # Budget B: memory proxy — cap added columns to ≤ 25% of current column count
    budget_from_memory = max(10, int(n_cols * memory_growth_cap))

    # Final budget = stricter of the two
    new_cols_budget = max(10, min(budget_from_samples, budget_from_memory))

    # ---- 4) Greedy add pairs while not exceeding budget ----
    used_new_cols = 0

    def encoding_for_joint(unique_est: int):
        # Decide encoding & "new columns" cost
        if unique_est <= 10:
            return "one_hot_encode", "Apply One-Hot Encoding since combined cardinality is low.", max(1, unique_est - 1)
        elif unique_est <= 50:
            return "label_encode", "Apply Label Encoding for moderate cardinality.", 1
        else:
            return "frequency_encode", "Apply Frequency Encoding due to high cardinality.", 1

    for c1, c2 in pairs[:max_pairs]:
        new_col = f"{c1}_{c2}_combined"
        u1 = cat_info[c1]["unique_values"]
        u2 = cat_info[c2]["unique_values"]
        unique_est = u1 * u2

        enc_fn, enc_text, added_cols = encoding_for_joint(unique_est)

        # If adding this pair would exceed budget, try a cheaper encoding before skipping
        if used_new_cols + added_cols > new_cols_budget:
            if enc_fn == "one_hot_encode":
                # fall back to label/frequency to keep within budget
                enc_fn, enc_text, added_cols = ("label_encode", "Budget-aware: using Label Encoding to limit expansion.", 1)
            # re-check after fallback
            if used_new_cols + added_cols > new_cols_budget:
                continue  # still too expensive → skip pair

        # (a) Combine
        suggestions.append({
            "feature": f"'{c1}' and '{c2}'",
            "issue": f"Potential interaction between '{c1}' and '{c2}'.",
            "suggestion": f"Combine into '{new_col}' to capture joint effects.",
            "function_to_call": "combine_categorical_features",
            "kwargs": {
                "columns_to_combine": [c1, c2],
                "new_col_name": new_col,
                "drop_original": False
            }
        })

        # (b) Encode with budget-aware choice
        suggestions.append({
            "feature": new_col,
            "issue": f"Estimated {unique_est} unique combos; budget cost ≈ +{added_cols} column(s).",
            "suggestion": enc_text,
            "function_to_call": enc_fn,
            "kwargs": {"columns": new_col}
        })

        used_new_cols += added_cols

        # Optional soft stop if we’ve used ~95% of budget
        if used_new_cols >= int(0.95 * new_cols_budget):
            break

    # Helpful tail note if we truncated
    if used_new_cols >= int(0.95 * new_cols_budget):
        suggestions.append({
            "feature": "budget",
            "issue": "Adaptive threshold reached",
            "suggestion": f"Limited combined features to keep added columns within ~{new_cols_budget} new columns (≈ {int(memory_growth_cap*100)}% of current width).",
            "function_to_call": None,
            "kwargs": {}
        })

    return suggestions


def suggest_fastica_features(analysis_results: dict, target_column: str = None) -> list:
    """
    Suggest applying FastICA to capture global trends across multiple features.
    
    FastICA is recommended when:
    - Dataset has many numerical features (>10)
    - Features show complex correlations
    - Data is likely non-Gaussian
    - Dimensionality reduction is needed
    
    Args:
        analysis_results: Dictionary from DataAnalyzer().run_full_analysis()
        target_column: Name of target column
    
    Returns:
        list: Preprocessing suggestions for FastICA
    """
    suggestions = []
    
    # Get data info
    general_info = analysis_results.get('general_info', {})
    data_types = general_info.get('data_types', {})
    shape = general_info.get('shape', [0, 0])
    n_samples, n_features = shape[0], shape[1]
    
    # Count numerical features
    numerical_cols = [col for col, dtype in data_types.items() 
                     if 'int' in dtype or 'float' in dtype]
    
    # Exclude target
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    n_numerical = len(numerical_cols)
    
    # Decision rules
    should_suggest = False
    issue_msg = ""
    suggestion_msg = ""
    n_components = None
    
    # Rule 1: Many numerical features (>10)
    if n_numerical > 10:
        should_suggest = True
        issue_msg = f"Dataset has {n_numerical} numerical features. FastICA can extract global trends using intelligent hybrid mode."
        suggestion_msg = (
            f"Apply FastICA in hybrid mode to intelligently replace redundant features and add independent components. "
            f"This captures underlying patterns across {n_numerical} features while optimizing dimensionality."
        )
        n_components = min(5, n_numerical // 2, n_samples // 10)
        n_components = max(2, n_components)
    
    # Rule 2: High dimensionality (>20 features)
    elif n_features > 20:
        should_suggest = True
        issue_msg = f"High-dimensional dataset ({n_features} features). FastICA can help reduce complexity using intelligent hybrid mode."
        suggestion_msg = (
            f"Apply FastICA in hybrid mode to extract independent components and reduce feature space. "
            f"The system will automatically determine optimal replacement ratio based on data characteristics."
        )
        n_components = min(5, n_numerical // 2)
        n_components = max(2, n_components)
    
    # Rule 3: Complex correlations detected
    correlations = analysis_results.get('correlations', {})
    if correlations:
        corr_matrix = correlations.get('correlation_matrix', {})
        if corr_matrix:
            # Check if there are many high correlations (complex structure)
            try:
                corr_df = pd.DataFrame(corr_matrix)
                high_corr_count = (corr_df.abs() > 0.5).sum().sum() - len(corr_df)  # Exclude diagonal
                if high_corr_count > n_numerical * 2:  # Many correlations
                    should_suggest = True
                    issue_msg = f"Complex correlation structure detected ({high_corr_count} high correlations)."
                    suggestion_msg = (
                        f"FastICA in hybrid mode can extract independent signals from correlated features, "
                        f"intelligently replacing redundant ones while capturing global trends."
                    )
                    n_components = min(5, n_numerical // 2)
                    n_components = max(2, n_components)
            except:
                pass
    
    if should_suggest and n_components:
        suggestions.append({
            'feature': 'multiple',
            'issue': issue_msg,
            'suggestion': suggestion_msg,
            'function_to_call': 'apply_fastica',
            'kwargs': {
                'n_components': n_components,
                'target_column': target_column,
                'mode': 'hybrid',  
                'replace_ratio': None,
                'analysis_results': analysis_results,
                'random_state': 42,
                'whiten': 'unit-variance'
            }
        })
    
    return suggestions