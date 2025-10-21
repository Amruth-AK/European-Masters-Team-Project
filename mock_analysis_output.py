"""
Mock Analysis Output
This is a MOCK/EXAMPLE output from the Analysis Team (Adrian & Alisa).
Used for development and testing until the actual analyze.py is ready.

Purpose: Define the data contract between Analysis and Preprocessing teams.
"""

import pandas as pd


def get_mock_analysis_report():
    """
    Mock analysis report based on Titanic dataset.
    This represents what we EXPECT from analyze.py
    
    Returns:
        dict: Analysis report with diagnostics
    """
    analysis_report = {
        # Feature type classification
        'feature_types': {
            'Age': 'numerical',
            'Fare': 'numerical',
            'SibSp': 'numerical',
            'Parch': 'numerical',
            'Sex': 'categorical',
            'Cabin': 'categorical',
            'Embarked': 'categorical',
            'Pclass': 'categorical'
        },
        
        # Missing value information
        'missing_values': {
            'Age': {
                'count': 177,
                'percentage': 0.1987,
                'has_missing': True
            },
            'Cabin': {
                'count': 687,
                'percentage': 0.7710,
                'has_missing': True
            },
            'Embarked': {
                'count': 2,
                'percentage': 0.0022,
                'has_missing': True
            }
        },
        
        # Outlier detection results
        'outliers': {
            'Fare': {
                'has_outliers': True,
                'count': 23,
                'percentage': 0.0258,
                'method': 'IQR'
            },
            'Age': {
                'has_outliers': False,
                'count': 0,
                'percentage': 0.0,
                'method': 'IQR'
            }
        },
        
        # Numerical feature statistics
        'numerical_stats': {
            'Age': {
                'mean': 29.70,
                'median': 28.0,
                'std': 14.53,
                'min': 0.42,
                'max': 80.0,
                'skewness': 0.389,
                'kurtosis': 0.178
            },
            'Fare': {
                'mean': 32.20,
                'median': 14.45,
                'std': 49.69,
                'min': 0.0,
                'max': 512.33,
                'skewness': 4.787,
                'kurtosis': 33.398
            },
            'SibSp': {
                'mean': 0.523,
                'median': 0.0,
                'std': 1.103,
                'min': 0,
                'max': 8,
                'skewness': 3.695,
                'kurtosis': 17.880
            }
        },
        
        # Categorical feature statistics
        'categorical_stats': {
            'Sex': {
                'cardinality': 2,
                'unique_values': ['male', 'female'],
                'most_frequent': 'male',
                'most_frequent_count': 577
            },
            'Cabin': {
                'cardinality': 147,
                'unique_values': None,  # Too many to list
                'most_frequent': 'C23 C25 C27',
                'most_frequent_count': 4
            },
            'Embarked': {
                'cardinality': 3,
                'unique_values': ['S', 'C', 'Q'],
                'most_frequent': 'S',
                'most_frequent_count': 644
            },
            'Pclass': {
                'cardinality': 3,
                'unique_values': [1, 2, 3],
                'most_frequent': 3,
                'most_frequent_count': 491
            }
        },
        
        # Duplicate detection
        'duplicates': {
            'has_duplicates': False,
            'count': 0,
            'percentage': 0.0
        },
        
        # Dataset metadata
        'metadata': {
            'total_rows': 891,
            'total_columns': 12,
            'memory_usage': '84.1 KB'
        }
    }
    
    return analysis_report


def get_simplified_mock():
    """
    Simplified version for quick testing.
    """
    return {
        'feature_types': {
            'Age': 'numerical',
            'Fare': 'numerical',
            'Sex': 'categorical'
        },
        'missing_values': {
            'Age': {'count': 177, 'percentage': 0.19, 'has_missing': True}
        },
        'outliers': {
            'Fare': {'has_outliers': True, 'count': 23, 'percentage': 0.026}
        },
        'numerical_stats': {
            'Age': {'skewness': 0.38},
            'Fare': {'skewness': 4.72}
        },
        'categorical_stats': {
            'Sex': {'cardinality': 2}
        },
        'duplicates': {
            'has_duplicates': False,
            'count': 0
        }
    }


if __name__ == "__main__":
    # Demo: Show what the mock data looks like
    import json
    
    print("=" * 60)
    print("MOCK ANALYSIS REPORT (Full Version)")
    print("=" * 60)
    report = get_mock_analysis_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 60)
    print("MOCK ANALYSIS REPORT (Simplified Version)")
    print("=" * 60)
    simple = get_simplified_mock()
    print(json.dumps(simple, indent=2))

