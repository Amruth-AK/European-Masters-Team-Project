"""
Test and demonstration of outlier handling functions.
"""

import pandas as pd
import numpy as np

# Mock the functions for demonstration
def clip_outliers_iqr(df, column, whisker_width=1.5):
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr
    
    outliers_lower = (df[column] < lower_bound).sum()
    outliers_upper = (df[column] > upper_bound).sum()
    total_outliers = outliers_lower + outliers_upper
    
    if total_outliers > 0:
        print(f"Column '{column}': Found {total_outliers} outliers "
              f"({outliers_lower} below {lower_bound:.2f}, "
              f"{outliers_upper} above {upper_bound:.2f})")
        print(f"Clipping outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def remove_outliers_iqr(df, column, whisker_width=1.5):
    df = df.copy()
    original_rows = len(df)
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    if outliers_count > 0:
        print(f"Column '{column}': Removing {outliers_count} rows with outliers "
              f"(outside range [{lower_bound:.2f}, {upper_bound:.2f}])")
        print(f"Original rows: {original_rows}, Remaining rows: {original_rows - outliers_count}")
    
    df = df[~outliers_mask].reset_index(drop=True)
    return df


def demo_outlier_handling():
    """Demonstrate outlier handling methods."""
    
    # Create sample data with outliers
    np.random.seed(42)
    data = {
        'PassengerId': range(1, 21),
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.08, 11.13, 30.07, 16.70,
                 500.00,  # Extreme outlier
                 8.45, 15.50, 7.88, 31.27, 
                 1000.00,  # Extreme outlier
                 10.50, 12.35, 26.00, 14.45],
        'Age': [22, 38, 26, 35, 28, 54, 2, 27, 14, 4,
                120,  # Outlier
                35, 28, 40, 32, 50, 25, 30, 45, 33]
    }
    df = pd.DataFrame(data)
    
    print("="*60)
    print("ORIGINAL DATA WITH OUTLIERS")
    print("="*60)
    print(df)
    print(f"\nFare statistics:")
    print(f"Mean: {df['Fare'].mean():.2f}, Median: {df['Fare'].median():.2f}")
    print(f"Min: {df['Fare'].min():.2f}, Max: {df['Fare'].max():.2f}")
    print(f"Q1: {df['Fare'].quantile(0.25):.2f}, Q3: {df['Fare'].quantile(0.75):.2f}")
    
    # Test 1: Clip outliers in Fare
    print("\n" + "="*60)
    print("TEST 1: CLIP OUTLIERS IN 'Fare' (whisker_width=1.5)")
    print("="*60)
    df_clipped = clip_outliers_iqr(df, 'Fare', whisker_width=1.5)
    print("\nData after clipping:")
    print(df_clipped[['PassengerId', 'Fare']])
    print(f"\nFare statistics after clipping:")
    print(f"Mean: {df_clipped['Fare'].mean():.2f}, Median: {df_clipped['Fare'].median():.2f}")
    print(f"Min: {df_clipped['Fare'].min():.2f}, Max: {df_clipped['Fare'].max():.2f}")
    
    # Test 2: Remove outliers in Fare
    print("\n" + "="*60)
    print("TEST 2: REMOVE OUTLIERS IN 'Fare' (whisker_width=1.5)")
    print("="*60)
    df_removed = remove_outliers_iqr(df, 'Fare', whisker_width=1.5)
    print("\nData after removing outliers:")
    print(df_removed[['PassengerId', 'Fare']])
    print(f"\nFare statistics after removal:")
    print(f"Mean: {df_removed['Fare'].mean():.2f}, Median: {df_removed['Fare'].median():.2f}")
    print(f"Min: {df_removed['Fare'].min():.2f}, Max: {df_removed['Fare'].max():.2f}")
    
    # Test 3: Compare different whisker widths
    print("\n" + "="*60)
    print("TEST 3: EFFECT OF whisker_width PARAMETER")
    print("="*60)
    
    print("\nWith whisker_width=1.5 (standard):")
    df_w15 = clip_outliers_iqr(df, 'Fare', whisker_width=1.5)
    
    print("\nWith whisker_width=3.0 (more conservative):")
    df_w30 = clip_outliers_iqr(df, 'Fare', whisker_width=3.0)
    
    # Test 4: Workflow integration example
    print("\n" + "="*60)
    print("TEST 4: WORKFLOW INTEGRATION")
    print("="*60)
    
    # Simulated preprocessing suggestion
    suggestion = {
        'feature': 'Fare',
        'issue': 'Contains extreme outliers (500.00, 1000.00)',
        'suggestion': 'Recommend clipping outliers using IQR method to preserve data.',
        'function_to_call': 'clip_outliers_iqr',
        'kwargs': {'column': 'Fare', 'whisker_width': 1.5}
    }
    
    print("\nGenerated Suggestion:")
    print(f"Feature: {suggestion['feature']}")
    print(f"Issue: {suggestion['issue']}")
    print(f"Suggestion: {suggestion['suggestion']}")
    print(f"Function: {suggestion['function_to_call']}")
    print(f"Arguments: {suggestion['kwargs']}")
    
    print("\nApplying suggestion...")
    df_final = clip_outliers_iqr(df, **suggestion['kwargs'])
    print("Done!")


def demo_comparison():
    """Compare clip vs remove methods."""
    
    print("\n" + "="*60)
    print("COMPARISON: CLIP vs REMOVE")
    print("="*60)
    
    # Simple data
    data = {
        'Value': [10, 12, 11, 13, 12, 11, 10, 100, 9, 11, 12, 13, 200, 11, 10]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df['Value'].tolist())
    print(f"Shape: {df.shape}, Mean: {df['Value'].mean():.2f}")
    
    print("\nAfter CLIP:")
    df_clip = clip_outliers_iqr(df, 'Value')
    print(df_clip['Value'].tolist())
    print(f"Shape: {df_clip.shape}, Mean: {df_clip['Value'].mean():.2f}")
    
    print("\nAfter REMOVE:")
    df_remove = remove_outliers_iqr(df, 'Value')
    print(df_remove['Value'].tolist())
    print(f"Shape: {df_remove.shape}, Mean: {df_remove['Value'].mean():.2f}")
    
    print("\n" + "="*60)
    print("Key Differences:")
    print("- CLIP: Keeps all rows, replaces outlier values with boundaries")
    print("- REMOVE: Deletes rows with outliers, reduces dataset size")
    print("="*60)


if __name__ == "__main__":
    demo_outlier_handling()
    demo_comparison()

