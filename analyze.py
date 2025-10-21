import pandas as pd
import numpy as np
from scipy.stats import skew

def analysis(df: pd.DataFrame, target_column: str) -> dict:
    """
    Performs a efficient and comprehensive analysis of a DataFrame,
    optimized for large datasets.

    Args:
        df: The input Pandas DataFrame.
        target_column: The name of the target column.

    Returns:
        A dictionary containing the analysis results.
    """

    # --- 1. Optimize Data Types to Reduce Memory ---
    # Downcast numeric columns to the smallest possible type
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        df[col] = pd.to_numeric(df[col], downcast='float')


    analysis_results = {
        'general_info': {},
        'missing_values': {},
        'descriptive_statistics': {},
        'distributions': {},
        'correlations': {},
        'categorical_info': {},
        'outlier_info': {},
        'histogram_data': {}
    }

    # --- 2. General Information ---
    analysis_results['general_info']['shape'] = df.shape
    analysis_results['general_info']['memory_usage'] = df.memory_usage(deep=True).sum()
    analysis_results['general_info']['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}


    # --- 3. Missing Values ---
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.concat([missing_values, missing_percentage], axis=1, keys=['missing_count', 'missing_percentage'])
    analysis_results['missing_values'] = missing_info.to_dict('index')

    # --- 4. Consolidated Analysis of Numerical Columns ---
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_column in numerical_cols:
        # Calculate correlation matrix once
        correlation_matrix = df[numerical_cols].corr()
        analysis_results['correlations']['correlation_matrix'] = correlation_matrix.to_dict()
        analysis_results['correlations']['target_correlation'] = correlation_matrix[target_column].to_dict()
    else:
        # Handle cases where the target is not numeric
        correlation_matrix = df[numerical_cols].corr()
        analysis_results['correlations']['correlation_matrix'] = correlation_matrix.to_dict()
        analysis_results['correlations']['target_correlation'] = "Target column is not numeric."


    desc_stats = df[numerical_cols].describe().to_dict()
    analysis_results['descriptive_statistics'] = desc_stats

    for col in numerical_cols:
        # Skewness
        analysis_results['distributions'][col] = {'skewness': skew(df[col].dropna())}

        # Histogram Data
        counts, bin_edges = np.histogram(df[col].dropna(), bins='auto')
        analysis_results['histogram_data'][col] = {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist()
        }

        # Outlier Information using IQR
        Q1 = desc_stats[col]['25%']
        Q3 = desc_stats[col]['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        analysis_results['outlier_info'][col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100
        }


    # --- 5. Consolidated Analysis of Categorical Columns ---
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    for col in categorical_cols:
        analysis_results['categorical_info'][col] = {
            'unique_values': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict()
        }

    return analysis_results

def pretty_print_dict(d):
    # (Your existing pretty_print_dict function)
    pretty_dict = ''
    for k, v in d.items():
        pretty_dict += f'{k}: \n'
        if isinstance(v, dict):
            for value in v:
                pretty_dict += f'    {value}: {v[value]}\n'
        else:
            pretty_dict += f'    {v}\n'
    return pretty_dict

if __name__ == '__main__':

    try:
        sample_df = pd.read_csv(r'C:\Users\adria\PycharmProjects\pythonProject3\train.csv')
        target = 'accident_risk'
        analysis_output = analysis(sample_df, target)
        print(pretty_print_dict(analysis_output))
    except FileNotFoundError:
        print("The specified file was not found.")
