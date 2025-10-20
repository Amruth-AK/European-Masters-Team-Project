"""
Test and demonstration of numerical feature preprocessing functions.
"""

import pandas as pd
import numpy as np
from preprocessing_function import standard_scaler, minmax_scaler, robust_scaler


def demo_scalers():
    """Demonstrate the usage of different scalers."""
    
    # Create sample data
    data = {
        'Age': [22, 38, 26, 35, np.nan, 54, 2, 27, 14, 4],
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.08, 11.13, 30.07, 16.70],
        'SibSp': [1, 1, 0, 1, 0, 0, 3, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Test 1: Standard Scaler on single column
    print("Test 1: Standard Scaler on 'Fare' column")
    df_standard = standard_scaler(df, 'Fare')
    print(df_standard[['Fare']])
    print(f"Mean: {df_standard['Fare'].mean():.6f}, Std: {df_standard['Fare'].std():.6f}")
    print("\n" + "="*50 + "\n")
    
    # Test 2: MinMax Scaler on single column
    print("Test 2: MinMax Scaler on 'Age' column (0-1 range)")
    df_minmax = minmax_scaler(df, 'Age')
    print(df_minmax[['Age']])
    print(f"Min: {df_minmax['Age'].min():.6f}, Max: {df_minmax['Age'].max():.6f}")
    print("\n" + "="*50 + "\n")
    
    # Test 3: MinMax Scaler with custom range
    print("Test 3: MinMax Scaler on 'Fare' column (-1 to 1 range)")
    df_minmax_custom = minmax_scaler(df, 'Fare', feature_range=(-1, 1))
    print(df_minmax_custom[['Fare']])
    print(f"Min: {df_minmax_custom['Fare'].min():.6f}, Max: {df_minmax_custom['Fare'].max():.6f}")
    print("\n" + "="*50 + "\n")
    
    # Test 4: Scaling multiple columns
    print("Test 4: Standard Scaler on multiple columns")
    df_multi = standard_scaler(df, ['Fare', 'SibSp'])
    print(df_multi[['Fare', 'SibSp']])
    print("\n" + "="*50 + "\n")
    
    # Test 5: Robust Scaler (good for outliers)
    print("Test 5: Robust Scaler on 'Fare' column")
    df_robust = robust_scaler(df, 'Fare')
    print(df_robust[['Fare']])
    print(f"Median: {df_robust['Fare'].median():.6f}")
    print("\n" + "="*50 + "\n")
    
    # Test 6: Comparison of all scalers
    print("Test 6: Comparison of all three scalers on 'Fare'")
    comparison = pd.DataFrame({
        'Original': df['Fare'],
        'Standard': standard_scaler(df, 'Fare')['Fare'],
        'MinMax': minmax_scaler(df, 'Fare')['Fare'],
        'Robust': robust_scaler(df, 'Fare')['Fare']
    })
    print(comparison)
    print("\n" + "="*50 + "\n")


def demo_workflow_integration():
    """Demonstrate how scalers integrate with the suggestion workflow."""
    
    print("Workflow Integration Example:")
    print("="*50)
    
    # Simulated analysis results
    analysis_results = {
        'feature_types': {'Age': 'numerical', 'Fare': 'numerical', 'Sex': 'categorical'},
        'skewness': {'Age': 0.38, 'Fare': 4.72},
        'has_outliers': {'Fare': True}
    }
    
    # Simulated preprocessing suggestions
    preprocessing_suggestions = [
        {
            'feature': 'Age',
            'issue': 'Numerical feature needs scaling',
            'suggestion': 'Apply standard scaling for normally distributed data.',
            'function_to_call': 'standard_scaler',
            'kwargs': {'column': 'Age'}
        },
        {
            'feature': 'Fare',
            'issue': 'High skewness (4.72) and outliers detected',
            'suggestion': 'Apply robust scaling to handle outliers better.',
            'function_to_call': 'robust_scaler',
            'kwargs': {'column': 'Fare'}
        }
    ]
    
    print("\nAnalysis Results:")
    print(analysis_results)
    print("\nGenerated Suggestions:")
    for i, suggestion in enumerate(preprocessing_suggestions, 1):
        print(f"\n{i}. Feature: {suggestion['feature']}")
        print(f"   Issue: {suggestion['issue']}")
        print(f"   Suggestion: {suggestion['suggestion']}")
        print(f"   Function: {suggestion['function_to_call']}")
        print(f"   Arguments: {suggestion['kwargs']}")


if __name__ == "__main__":
    demo_scalers()
    print("\n\n")
    demo_workflow_integration()

