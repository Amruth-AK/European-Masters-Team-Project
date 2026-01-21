import numpy as np
import pandas as pd
import itertools
from preprocessing_registry import FUNC_MAP, decide_correlation_threshold, assess_ica_necessity

# ===================================================================
# Missing Values
# ===================================================================

def suggest_missing_value_handling(analysis_results: dict, target_column: str = None) -> list:
    """
    Robust, model-agnostic missing value suggestions.
    Strategy: 
    - Numerics: "Median + Indicator".
    - Categoricals: "Unknown" Category (Preserves missingness info).
    """
    suggestions = []

    missing_info = analysis_results.get('missing_values', {})
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    
    for col, info in missing_info.items():
        missing_pct = info.get('missing_percentage', 0)
        dtype = data_types.get(col, '')
        
        #No missing values
        if missing_pct == 0:
            continue

        #Target Variable -> Drop Rows
        if col == target_column:
            suggestions.append({
                'feature': col,
                'issue': f'Target column contains {missing_pct:.2f}% missing values.',
                'suggestion': 'Drop rows where target is missing (cannot train on unknown targets).',
                'function_to_call': 'delete_missing_rows',
                'kwargs': {'threshold': 0.0} # 0.0 = drop any row with NaN in this column
            })
            continue

        #>95% missing -> Drop
        if missing_pct > 95:
            suggestions.append({
                'feature': col,
                'issue': f'{missing_pct:.2f}% missing (effectively empty).',
                'suggestion': 'Drop column (insufficient signal).',
                'function_to_call': 'delete_missing_columns',
                'kwargs': {'threshold': 0.95}
            })
            continue

        #NUMERICAL: Indicator + Median
        if 'int' in dtype or 'float' in dtype:
            # Add Indicator
            suggestions.append({
                'feature': col,
                'issue': f'{missing_pct:.2f}% missing values detected.',
                'suggestion': 'Add "Is_Missing" indicator to capture if missingness itself is predictive.',
                'function_to_call': 'add_missing_indicator',
                'kwargs': {'columns': col}
            })
            
            # Impute Median
            suggestions.append({
                'feature': col,
                'issue': f'Missing values need replacement for model compatibility.',
                'suggestion': 'Impute with Median. This is robust to outliers.',
                'function_to_call': 'impute_median',
                'kwargs': {'columns': col}
            })

        # CATEGORICAL: Constant "Unknown"
        else:
            suggestions.append({
                'feature': col,
                'issue': f'{missing_pct:.2f}% missing values in categorical feature.',
                'suggestion': 'Impute with new category "Unknown". Preserves missingness information without guessing.',
                'function_to_call': 'impute_constant',
                'kwargs': {'columns': col, 'fill_value': 'Unknown'}
            })

    return suggestions

# ===================================================================
# Duplicate Values
# ===================================================================

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

    # No duplicates
    if total_dup == 0:
        return suggestions

    # Few duplicates (<1%)
    if dup_pct <= 1:
        suggestions.append({
            'feature': 'rows',
            'issue': f'Low duplicate ratio ({dup_pct:.2f}%) detected.',
            'suggestion': 'Remove duplicate rows to avoid redundancy.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    # Moderate duplicates (1–10%)
    elif 1 < dup_pct <= 10:
        suggestions.append({
            'feature': 'rows',
            'issue': f'{dup_pct:.2f}% duplicate rows detected.',
            'suggestion': 'Remove duplicates; verify if they are genuine or data collection artifacts.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    # High duplicates (>10%)
    else:
        suggestions.append({
            'feature': 'rows',
            'issue': f'High duplicate percentage ({dup_pct:.2f}%) detected.',
            'suggestion': 'Investigate source of duplication before deletion; may indicate merging issues.',
            'function_to_call': 'delete_duplicates',
            'kwargs': {'subset': None}
        })

    return suggestions


# ===================================================================
# Numerical Scaling
# ===================================================================

def suggest_numerical_scaling(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate scaling suggestions. Prefer RobustScaler if outliers exist.
    """
    suggestions = []
    
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    outlier_info = analysis_results.get('outlier_info', {})
    
    numerical_cols = [col for col, dtype in data_types.items() 
                     if ('int' in dtype or 'float' in dtype) and col != target_column]
    
    for col in numerical_cols:
        outlier_count = outlier_info.get(col, {}).get('outlier_count', 0)
        
        # Outliers Detected -> Robust Scaler
        if outlier_count > 0:
            suggestions.append({
                'feature': col,
                'issue': f'Contains {outlier_count} outliers. Standard scaling would be distorted.',
                'suggestion': 'Apply Robust Scaling (based on IQR). This scales data effectively while ignoring extreme outliers, crucial for subsequent feature engineering.',
                'function_to_call': 'robust_scaler',
                'kwargs': {'column': col, 'quantile_range': (25.0, 75.0)}
            })
            
        # No Outliers -> Standard Scaler (Center data)
        else:
            suggestions.append({
                'feature': col,
                'issue': 'High skewness but no extreme outliers.',
                'suggestion': 'Apply Standard Scaling (Z-score) to center the data.',
                'function_to_call': 'standard_scaler',
                'kwargs': {'column': col}
            })
    
    return suggestions


def suggest_outlier_handling(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate outlier suggestions.
    High Skew (>2.0) -> Yeo-Johnson
    Moderate Skew/Outliers -> Winsorization
    """
    suggestions = []
    
    outlier_info = analysis_results.get('outlier_info', {})
    distributions = analysis_results.get('distributions', {})
    
    for col, info in outlier_info.items():
        if col == target_column:
            continue
        
        outlier_count = info.get('outlier_count', 0)
        if outlier_count == 0:
            continue
            
        skewness = distributions.get(col, {}).get('skewness', 0)
        
        # Rule 1: Extreme Skewness (> 2.0) -> Power Transform
        if abs(skewness) > 2.0:
            suggestions.append({
                'feature': col,
                'issue': f'High skewness ({skewness:.2f}) and {outlier_count} outliers detected.',
                'suggestion': 'Apply Yeo-Johnson transformation. This compresses tails and handles outliers better than clipping for skewed financial data.',
                'function_to_call': 'apply_power_transform',
                'kwargs': {'column': col, 'method': 'yeo-johnson'}
            })
            
        # Rule 2: Outliers exist but not super skewed -> Winsorization
        else:
            suggestions.append({
                'feature': col,
                'issue': f'Feature contains {outlier_count} outliers.',
                'suggestion': 'Apply Winsorization (capping at 1st and 99th percentiles). This limits extreme values without removing data points.',
                'function_to_call': 'winsorize_column',
                'kwargs': {'column': col, 'limits': (0.01, 0.99)}
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

# Add Zhiqi-feature combination


def suggest_correlation_based_features(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate suggestions for creating new features from highly correlated feature pairs.
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

    # Identify numerical columns
    data_types = analysis_results.get('general_info', {}).get('data_types', {})
    numerical_cols = [
        col for col, dtype in data_types.items()
        if ('int' in dtype or 'float' in dtype) and col != target_column
    ]
    #id_keywords = ['id', 'index', 'uid', 'key', 'identifier', '_id', 'pk']
    #numerical_cols = [col for col in numerical_cols if not any(k in col.lower() for k in id_keywords)]
    numerical_cols = [
        col for col in numerical_cols 
        if not col.lower().endswith("_is_missing") 
       #and not any(k in col.lower() for k in ['id', 'index', 'uid', 'key', 'identifier', '_id', 'pk'])
    ]

    # Filter matrix to only valid numerical columns
    valid_cols = [c for c in numerical_cols if c in corr_matrix.index]
    if len(valid_cols) < 2:
        return suggestions
        
    corr_matrix = corr_matrix.loc[valid_cols, valid_cols]

    # [Optimization] Vectorized collection of absolute correlations
    # Instead of nested loops, we use numpy upper triangle indices
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    all_corrs = corr_matrix.where(mask).stack().abs().tolist()

    if not all_corrs:
        return suggestions

    # Adaptive threshold
    correlation_threshold = decide_correlation_threshold(
        all_corrs, 
        method='auto', 
        min_threshold=0.1
    )

    # Decide strategy
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

    # Filter high-correlation pairs for preview
    # (We only need top 10 for the message, so we don't need to collect ALL of them here manually)
    # Let's just grab the top ones from our vectorized data for the preview
    corr_series = corr_matrix.where(mask).stack()
    high_corr_series = corr_series[corr_series.abs() >= correlation_threshold]
    
    # Sort by absolute value descending
    high_corr_series = high_corr_series.iloc[np.argsort(-high_corr_series.abs().values)]
    
    num_high_pairs = len(high_corr_series)
    if num_high_pairs == 0:
        return suggestions

    # Adaptive max_new_features
    shape = analysis_results.get('general_info', {}).get('shape', [0, 0])
    n_samples = shape[0] if shape else 0
    n_features = shape[1] if shape else 0
    
    if strategy == "all_pairs":
        max_new_features = min(n_samples // 15, n_features, 100)
        max_new_features = max(20, max_new_features)
    else:
        max_new_features = min(n_samples // 10, n_features * 2, 200)
        max_new_features = max(50, max_new_features)
    
    print(f"📊 [Suggestion] Threshold: {correlation_threshold:.3f} | Strategy: {strategy} | Limit: {max_new_features}")

    # Create preview string
    top_pairs_preview = []
    for (col1, col2), val in high_corr_series.head(10).items():
        top_pairs_preview.append(f"- {col1} & {col2} (corr={val:.2f})")
    
    preview_str = "\n".join(top_pairs_preview)
    if num_high_pairs > 10:
        preview_str += f"\n- ... and {num_high_pairs - 10} more pairs"

    if strategy == "all_pairs":
        issue_msg = (
            f"Very low correlations detected (max: {max(all_corrs):.2f}). "
            f"Suggesting interaction features from all {num_high_pairs} feature pairs."
        )
        suggestion_msg = (
            f"Even with low correlations, interaction features can capture non-linear patterns.\n"
            f"**Top feature pairs to be combined:**\n{preview_str}"
        )
    else:
        issue_msg = f"Found {num_high_pairs} pairs with |corr| >= {correlation_threshold:.2f}"
        suggestion_msg = (
            f"Create interaction features from {num_high_pairs} feature pairs.\n"
            f"**Top feature pairs to be combined:**\n{preview_str}"
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
            'min_cardinality': 3,
            'max_new_features': max_new_features,
            'corr_filter_threshold': 0.92,
            'min_variance': 1e-5,
            'target_column': target_column,
            'generate_third_order': True,
            'min_second_for_third': 15,
            'max_third_order_features': 30,
            'top_second_for_third': 50,
            'top_original_for_third': 30
        }
    })

    return suggestions


def suggest_feature_combination(
    analysis_results: dict,
    target_column: str = None,
    max_pairs: int = 9999,
    memory_growth_cap: float = 0.25
) -> list:
    """
    Adaptive categorical feature-combination suggestions.
    """
    suggestions = []

    cat_info = analysis_results.get("categorical_info", {})
    dtypes = analysis_results.get("general_info", {}).get("data_types", {})
    shape = analysis_results.get("general_info", {}).get("shape", [0, 0])
    n_rows, n_cols = shape if isinstance(shape, (list, tuple)) else (0, 0)

    def is_cat(col):
        t = dtypes.get(col, "")
        return ("object" in t) or ("category" in t)

    eligible = []
    for col, info in cat_info.items():
        if col == target_column: continue
        if not is_cat(col): continue
        uv = info.get("unique_values", 0)
        if 2 <= uv <= 30: eligible.append(col)

    if len(eligible) < 2: return suggestions

    pairs = list(itertools.combinations(eligible, 2))
    pairs.sort(key=lambda p: cat_info[p[0]]["unique_values"] * cat_info[p[1]]["unique_values"])

    budget_from_samples = int(min(max(n_rows // 5, 10), 2000))
    budget_from_memory = max(10, int(n_cols * memory_growth_cap))
    new_cols_budget = max(10, min(budget_from_samples, budget_from_memory))

    used_new_cols = 0

    def encoding_for_joint(unique_est: int):
        if unique_est <= 10:
            return "one_hot_encode", "Apply One-Hot Encoding (low cardinality).", max(1, unique_est - 1)
        elif unique_est <= 50:
            return "label_encode", "Apply Label Encoding (moderate cardinality).", 1
        else:
            return "frequency_encode", "Apply Frequency Encoding (high cardinality).", 1

    for c1, c2 in pairs[:max_pairs]:
        new_col = f"{c1}_{c2}_combined"
        u1 = cat_info[c1]["unique_values"]
        u2 = cat_info[c2]["unique_values"]
        unique_est = u1 * u2

        enc_fn, enc_text, added_cols = encoding_for_joint(unique_est)

        if used_new_cols + added_cols > new_cols_budget:
            if enc_fn == "one_hot_encode":
                enc_fn, enc_text, added_cols = ("label_encode", "Budget-aware: using Label Encoding.", 1)
            if used_new_cols + added_cols > new_cols_budget:
                continue

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

        suggestions.append({
            "feature": new_col,
            "issue": f"Estimated {unique_est} unique combos.",
            "suggestion": enc_text,
            "function_to_call": enc_fn,
            "kwargs": {"columns": new_col}
        })

        used_new_cols += added_cols
        if used_new_cols >= int(0.95 * new_cols_budget): break

    if used_new_cols >= int(0.95 * new_cols_budget):
        suggestions.append({
            "feature": "budget",
            "issue": "Adaptive threshold reached",
            "suggestion": f"Limited combined features to keep added columns within budget.",
            "function_to_call": None,
            "kwargs": {}
        })

    return suggestions


def suggest_fastica_features(analysis_results: dict, target_column: str = None) -> list:
    """
    Generate FastICA suggestions using adaptive, data-driven thresholds.
    """
    suggestions = []
    
    general_info = analysis_results.get('general_info', {})
    data_types = general_info.get('data_types', {})
    shape = general_info.get('shape', [0, 0])
    n_samples = shape[0]
    
    numerical_cols = [
        col for col, dtype in data_types.items() 
        if ('int' in dtype or 'float' in dtype) and col != target_column
    ]
    n_numerical = len(numerical_cols)
    
    correlations = analysis_results.get('correlations', {})
    corr_matrix_dict = correlations.get('correlation_matrix', {})

    potential_new_features = 0
    if corr_matrix_dict:
        try:
            corr_df = pd.DataFrame(corr_matrix_dict)
            high_corr_pairs_count = (corr_df.abs() >= 0.4).values.sum() // 2
            potential_new_features = high_corr_pairs_count * 2
        except: pass
            
    should_suggest, reason_msg, recommended_n_components = assess_ica_necessity(
        n_samples=n_samples,
        n_features=n_numerical,
        corr_matrix_dict=corr_matrix_dict,
        potential_new_features=potential_new_features
    )

    if should_suggest:
        suggestions.append({
            'feature': 'multiple',
            'issue': f"Dimensionality reduction recommended. Reasons: {reason_msg}",
            'suggestion': (
                f"Apply FastICA (Hybrid Mode) to extract {recommended_n_components} independent components."
            ),
            'function_to_call': 'apply_fastica',
            'kwargs': {
                'n_components': recommended_n_components,
                'target_column': target_column,
                'mode': 'hybrid',  
                'replace_ratio': None,
                'analysis_results': analysis_results,
                'random_state': 42,
                'whiten': 'unit-variance'
                # Removed 'perform_tuning': True because apply_fastica doesn't accept it.
                # Tuning should be triggered separately if needed.
            }
        })
    
    return suggestions