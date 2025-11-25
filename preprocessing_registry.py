# preprocessing_registry.py

"""
Central registry of preprocessing functions.

This avoids circular imports by keeping FUNC_MAP in its own module,
so that both pre_dashboard.py and preprocessing_pipeline.py can import it.
"""

from preprocessing_function import (
    delete_missing_columns,
    delete_missing_rows,
    impute_mean,
    impute_median,
    impute_mode,
    delete_duplicates,
    standard_scaler,
    minmax_scaler,
    clip_outliers_iqr,
    remove_outliers_iqr,
    binary_encode,
    ordinal_encode,
    one_hot_encode,
    label_encode,
    frequency_encode,
    remove_identifier_columns,
    calculate_datetime_diff,
    extract_datetime_features,
    create_features_from_correlation_analysis,
    combine_categorical_features,
    apply_fastica,
    tune_fastica_replace_ratio,
    grid_search_replace_ratio,
    decide_correlation_threshold,
    assess_ica_necessity,
    winsorize_column,       
    apply_power_transform, 
    robust_scaler,      
    impute_constant,        
    add_missing_indicator,    
)


# Map string function names to actual functions
FUNC_MAP = {
    "delete_missing_columns": delete_missing_columns,
    "delete_missing_rows": delete_missing_rows,
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_mode": impute_mode,
    "impute_constant": impute_constant,             
    "add_missing_indicator": add_missing_indicator, 
    "delete_duplicates": delete_duplicates,
    "standard_scaler": standard_scaler,
    "minmax_scaler": minmax_scaler,
    "winsorize_column": winsorize_column,          
    "apply_power_transform": apply_power_transform,
    "robust_scaler": robust_scaler,  
    "clip_outliers_iqr": clip_outliers_iqr,
    "remove_outliers_iqr": remove_outliers_iqr,
    "binary_encode": binary_encode,
    "ordinal_encode": ordinal_encode,
    "one_hot_encode": one_hot_encode,
    "label_encode": label_encode,
    "frequency_encode": frequency_encode,
    "remove_identifier_columns": remove_identifier_columns,
    "calculate_datetime_diff": calculate_datetime_diff,
    "extract_datetime_features": extract_datetime_features,
    "create_features_from_correlation_analysis": create_features_from_correlation_analysis,
    "combine_categorical_features": combine_categorical_features,
    "apply_fastica": apply_fastica,
    "tune_fastica_replace_ratio": tune_fastica_replace_ratio,
    "grid_search_replace_ratio": grid_search_replace_ratio,
    "decide_correlation_threshold": decide_correlation_threshold,
    "assess_ica_necessity": assess_ica_necessity,
}
