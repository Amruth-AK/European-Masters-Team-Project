import pandas as pd
import numpy as np
from scipy.stats import skew


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.results = {
            'general_info': {},
            'missing_values': {},
            'descriptive_statistics': {},
            'distributions': {},
            'correlations': {},
            'categorical_info': {},
            'outlier_info': {},
            'histogram_data': {},
            'row_duplicate_info': {},
            'feature_duplicate_info': {}
        }

    # --- 1. Memory Optimization ---
    def optimize_dtypes(self):
        for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        return self

    # --- 2. General Info ---
    def analyze_general_info(self):
        self.results['general_info'] = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        return self

    # --- 3. Missing Values ---
    def analyze_missing_values(self):
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_info = pd.concat(
            [missing_values, missing_percentage],
            axis=1,
            keys=['missing_count', 'missing_percentage']
        )
        self.results['missing_values'] = missing_info.to_dict('index')
        return self

    # --- 4. Numerical Analysis ---
    def analyze_numerical(self):
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        desc_stats = self.df[numerical_cols].describe().to_dict()
        self.results['descriptive_statistics'] = desc_stats

        # Correlations
        correlation_matrix = self.df[numerical_cols].corr()
        self.results['correlations'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'target_correlation': correlation_matrix[
                self.target_column].to_dict() if self.target_column in numerical_cols else "Target column is not numeric."
        }

        # Distributions and Outliers
        for col in numerical_cols:
            col_data = self.df[col].dropna()
            self.results['distributions'][col] = {'skewness': skew(col_data)}

            # Histogram
            counts, bin_edges = np.histogram(col_data, bins='auto')
            self.results['histogram_data'][col] = {
                'counts': counts.tolist(),
                'bin_edges': bin_edges.tolist()
            }

            # Outliers
            Q1 = desc_stats[col]['25%']
            Q3 = desc_stats[col]['75%']
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            self.results['outlier_info'][col] = {
                'lower_bound': lower,
                'upper_bound': upper,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100
            }
        return self

    # --- 5. Categorical Analysis ---
    def analyze_categorical(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            self.results['categorical_info'][col] = {
                'unique_values': self.df[col].nunique(),
                'value_counts': self.df[col].value_counts().to_dict()
            }
        return self

    # --- 6. Duplicate Analysis ---
    def analyze_row_duplicates(self):
        duplicate_rows = self.df[self.df.duplicated()]
        self.results['duplicate_info'] = {
            'total_duplicates': len(duplicate_rows),
            'duplicate_percentage': (len(duplicate_rows) / len(self.df)) * 100,
            'duplicate_rows': duplicate_rows.to_dict('records')
        }
        return self

    def analyze_feature_duplicates(self):
        """Analyzes each column for duplicate values and their frequency."""
        total_rows = len(self.df)
        for col in self.df.columns:
            if total_rows > 0:
                # Calculate count of values that are not unique
                duplicate_count = total_rows - self.df[col].nunique()

                # Get the most frequent value and its count
                value_counts = self.df[col].value_counts()
                if not value_counts.empty:
                    most_frequent_value = value_counts.index[0]
                    most_frequent_count = int(value_counts.iloc[0])
                else:
                    most_frequent_value = None
                    most_frequent_count = 0

                self.results['feature_duplicate_info'][col] = {
                    'duplicate_count': duplicate_count,
                    'duplicate_percentage': (duplicate_count / total_rows) * 100 if total_rows > 0 else 0,
                    'most_frequent_value': most_frequent_value,
                    'most_frequent_count': most_frequent_count
                }
        return self

    # --- 7. Run Full Analysis ---
    def run_full_analysis(self):
        (self.optimize_dtypes()
         .analyze_general_info()
         .analyze_missing_values()
         .analyze_numerical()
         .analyze_categorical()
         .analyze_row_duplicates()  # Renamed method call
         .analyze_feature_duplicates())  # New method call
        return self.results


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



