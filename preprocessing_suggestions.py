"""
Preprocessing Suggestions Module
Team: Yang/Zhiqi, Amruth, Tecla

This module acts as a rule engine that converts analysis diagnostics 
into actionable preprocessing suggestions.

Input: analysis_report (dict) from analyze.py
Output: suggestions (list of dict) for main.py
"""

import sys
from pathlib import Path

# Add the preprocessing function directory to path
preprocessing_dir = Path(__file__).parent / "Preprocessing suggestion"
sys.path.insert(0, str(preprocessing_dir))


def generate_suggestions(analysis_report):
    """
    Main entry point - generates all preprocessing suggestions.
    
    Args:
        analysis_report (dict): Diagnostic results from analyze.py
    
    Returns:
        list of dict: Preprocessing suggestions with this structure:
            {
                'feature': str,
                'issue': str,
                'suggestion': str,
                'function_to_call': str,
                'kwargs': dict,
                'priority': int (1=high, 2=medium, 3=low)
            }
    """
    suggestions = []
    
    # Priority 1: Handle missing values first (Tecla)
    suggestions.extend(_suggest_missing_values(analysis_report))
    
    # Priority 1: Handle duplicates (Tecla)
    suggestions.extend(_suggest_duplicates(analysis_report))
    
    # Priority 2: Handle outliers (Yang/Zhiqi)
    suggestions.extend(_suggest_outlier_handling(analysis_report))
    
    # Priority 3: Handle categorical encoding (Amruth)
    suggestions.extend(_suggest_categorical_encoding(analysis_report))
    
    # Priority 4: Handle numerical scaling (Yang/Zhiqi)
    suggestions.extend(_suggest_numerical_scaling(analysis_report))
    
    return suggestions


# ============================================================================
# YANG/ZHIQI'S SECTION - Numerical Features & Outliers
# ============================================================================

def _suggest_numerical_scaling(analysis_report):
    """
    Generate scaling suggestions for numerical features.
    
    Rules:
    - If has outliers → robust_scaler (not implemented yet, skip for now)
    - If high skewness (>1) → standard_scaler
    - Otherwise → minmax_scaler
    """
    suggestions = []
    
    feature_types = analysis_report.get('feature_types', {})
    numerical_stats = analysis_report.get('numerical_stats', {})
    outliers = analysis_report.get('outliers', {})
    
    for feature, dtype in feature_types.items():
        if dtype == 'numerical':
            # Check if feature has outliers
            has_outliers = outliers.get(feature, {}).get('has_outliers', False)
            
            # Get skewness
            skewness = numerical_stats.get(feature, {}).get('skewness', 0)
            
            # Rule 1: If high skewness, use standard scaler
            if abs(skewness) > 1:
                suggestions.append({
                    'feature': feature,
                    'issue': f'High skewness detected (skewness: {skewness:.2f})',
                    'suggestion': 'Apply standard scaling (z-score normalization) to normalize distribution.',
                    'function_to_call': 'standard_scaler',
                    'kwargs': {'column': feature},
                    'priority': 4
                })
            # Rule 2: Otherwise use minmax scaler
            else:
                suggestions.append({
                    'feature': feature,
                    'issue': 'Numerical feature requires scaling for model compatibility',
                    'suggestion': 'Apply min-max scaling to transform values to [0,1] range.',
                    'function_to_call': 'minmax_scaler',
                    'kwargs': {'column': feature, 'feature_range': (0, 1)},
                    'priority': 4
                })
    
    return suggestions


def _suggest_outlier_handling(analysis_report):
    """
    Generate outlier handling suggestions.
    
    Rules:
    - If outlier_percentage < 5% → clip_outliers_iqr
    - If outlier_percentage >= 5% → remove_outliers_iqr
    """
    suggestions = []
    
    outliers = analysis_report.get('outliers', {})
    
    for feature, outlier_info in outliers.items():
        if outlier_info.get('has_outliers', False):
            outlier_pct = outlier_info.get('percentage', 0)
            outlier_count = outlier_info.get('count', 0)
            
            # Rule 1: Small percentage - clip outliers
            if outlier_pct < 0.05:
                suggestions.append({
                    'feature': feature,
                    'issue': f'Contains {outlier_count} outliers ({outlier_pct*100:.1f}% of data)',
                    'suggestion': 'Clip outliers to IQR boundaries to preserve all rows while limiting extreme values.',
                    'function_to_call': 'clip_outliers_iqr',
                    'kwargs': {'column': feature, 'whisker_width': 1.5},
                    'priority': 2
                })
            # Rule 2: Large percentage - consider removal
            else:
                suggestions.append({
                    'feature': feature,
                    'issue': f'High proportion of outliers ({outlier_count} rows, {outlier_pct*100:.1f}%)',
                    'suggestion': 'Consider removing outlier rows if dataset is sufficiently large.',
                    'function_to_call': 'remove_outliers_iqr',
                    'kwargs': {'column': feature, 'whisker_width': 1.5},
                    'priority': 2
                })
    
    return suggestions


# ============================================================================
# AMRUTH'S SECTION - Categorical Features
# ============================================================================

def _suggest_categorical_encoding(analysis_report):
    """
    Generate encoding suggestions for categorical features.
    
    Rules (to be implemented by Amruth):
    - If cardinality == 2 → label encoding
    - If cardinality <= 10 → one-hot encoding
    - If cardinality > 10 → target encoding
    
    TODO: Amruth - implement the logic below
    """
    suggestions = []
    
    feature_types = analysis_report.get('feature_types', {})
    categorical_stats = analysis_report.get('categorical_stats', {})
    
    for feature, dtype in feature_types.items():
        if dtype == 'categorical':
            cardinality = categorical_stats.get(feature, {}).get('cardinality', 0)
            
            # Rule 1: Binary categorical - use label encoding
            if cardinality == 2:
                suggestions.append({
                    'feature': feature,
                    'issue': f'Binary categorical variable ({cardinality} unique values)',
                    'suggestion': 'Apply label encoding (0/1) for binary categories.',
                    'function_to_call': 'encode_label',
                    'kwargs': {'column': feature},
                    'priority': 3
                })
            # Rule 2: Low cardinality - use one-hot encoding
            elif cardinality <= 10:
                suggestions.append({
                    'feature': feature,
                    'issue': f'Low cardinality categorical variable ({cardinality} unique values)',
                    'suggestion': 'Apply one-hot encoding to create binary dummy variables.',
                    'function_to_call': 'encode_one_hot',
                    'kwargs': {'column': feature},
                    'priority': 3
                })
            # Rule 3: High cardinality - use target encoding
            else:
                suggestions.append({
                    'feature': feature,
                    'issue': f'High cardinality categorical variable ({cardinality} unique values)',
                    'suggestion': 'Apply target encoding to avoid dimensionality explosion.',
                    'function_to_call': 'encode_target',
                    'kwargs': {'column': feature, 'target': 'target_variable'},
                    'priority': 3
                })
    
    return suggestions


# ============================================================================
# TECLA'S SECTION - Missing Values & Duplicates
# ============================================================================

def _suggest_missing_values(analysis_report):
    """
    Generate missing value handling suggestions.
    
    Rules (to be implemented by Tecla):
    - If missing_rate < 5% → deletion
    - If numerical & missing_rate >= 5% → median/mean imputation
    - If categorical & missing_rate >= 5% → mode imputation
    
    TODO: Tecla - implement the logic below
    """
    suggestions = []
    
    missing_values = analysis_report.get('missing_values', {})
    feature_types = analysis_report.get('feature_types', {})
    
    for feature, missing_info in missing_values.items():
        if missing_info.get('has_missing', False):
            missing_pct = missing_info.get('percentage', 0)
            missing_count = missing_info.get('count', 0)
            feature_type = feature_types.get(feature, 'unknown')
            
            # Rule 1: Very few missing values - consider deletion
            if missing_pct < 0.05:
                suggestions.append({
                    'feature': feature,
                    'issue': f'Missing values detected ({missing_count} rows, {missing_pct*100:.1f}%)',
                    'suggestion': 'Consider deleting rows with missing values as they represent a small portion.',
                    'function_to_call': 'delete_missing_rows',
                    'kwargs': {'column': feature},
                    'priority': 1
                })
            # Rule 2: Numerical with significant missing - impute with median
            elif feature_type == 'numerical':
                suggestions.append({
                    'feature': feature,
                    'issue': f'Significant missing values ({missing_count} rows, {missing_pct*100:.1f}%)',
                    'suggestion': 'Apply median imputation to fill missing numerical values.',
                    'function_to_call': 'impute_median',
                    'kwargs': {'column': feature},
                    'priority': 1
                })
            # Rule 3: Categorical with significant missing - impute with mode
            elif feature_type == 'categorical':
                suggestions.append({
                    'feature': feature,
                    'issue': f'Significant missing values ({missing_count} rows, {missing_pct*100:.1f}%)',
                    'suggestion': 'Apply mode imputation to fill missing categorical values.',
                    'function_to_call': 'impute_mode',
                    'kwargs': {'column': feature},
                    'priority': 1
                })
    
    return suggestions


def _suggest_duplicates(analysis_report):
    """
    Generate duplicate handling suggestions.
    
    TODO: Tecla - implement the logic below
    """
    suggestions = []
    
    duplicates = analysis_report.get('duplicates', {})
    
    if duplicates.get('has_duplicates', False):
        dup_count = duplicates.get('count', 0)
        dup_pct = duplicates.get('percentage', 0)
        
        suggestions.append({
            'feature': 'All columns',
            'issue': f'Duplicate rows detected ({dup_count} rows, {dup_pct*100:.1f}%)',
            'suggestion': 'Remove duplicate rows to avoid data leakage and bias.',
            'function_to_call': 'remove_duplicates',
            'kwargs': {},
            'priority': 1
        })
    
    return suggestions


# ============================================================================
# Utility Functions
# ============================================================================

def print_suggestions(suggestions):
    """Pretty print suggestions for user review."""
    if not suggestions:
        print("No preprocessing suggestions generated.")
        return
    
    print("\n" + "=" * 80)
    print(f"PREPROCESSING SUGGESTIONS ({len(suggestions)} total)")
    print("=" * 80)
    
    # Group by priority
    priority_labels = {1: 'HIGH', 2: 'MEDIUM', 3: 'MEDIUM', 4: 'LOW'}
    
    for priority in [1, 2, 3, 4]:
        priority_suggestions = [s for s in suggestions if s.get('priority') == priority]
        
        if priority_suggestions:
            print(f"\n[Priority {priority_labels[priority]}]")
            print("-" * 80)
            
            for i, suggestion in enumerate(priority_suggestions, 1):
                print(f"\n{i}. Feature: {suggestion['feature']}")
                print(f"   Issue: {suggestion['issue']}")
                print(f"   Suggestion: {suggestion['suggestion']}")
                print(f"   Function: {suggestion['function_to_call']}({suggestion['kwargs']})")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test with mock data
    from mock_analysis_output import get_mock_analysis_report
    
    print("Testing Preprocessing Suggestions Module")
    print("=" * 80)
    
    # Get mock analysis report
    analysis_report = get_mock_analysis_report()
    
    # Generate suggestions
    suggestions = generate_suggestions(analysis_report)
    
    # Print results
    print_suggestions(suggestions)
    
    print("\n" + "=" * 80)
    print(f"Total suggestions generated: {len(suggestions)}")
    print("=" * 80)

