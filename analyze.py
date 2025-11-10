import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import List

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.id_columns_to_ignore = self._detect_id_columns()
        self.results = {
            'general_info': {},
            'missing_values': {},
            'descriptive_statistics': {},
            'numerical_summary': {},  # Added this key for compatibility
            'distributions': {},
            'correlations': {},
            'categorical_info': {},
            'outlier_info': {},
            'histogram_data': {},
            'row_duplicate_info': {},
            'feature_duplicate_info': {}
        }

    def _detect_id_columns(self) -> List[str]:
        id_candidates = []
        columns_to_check = [col for col in self.df.columns if col != self.target_column]
        if len(self.df) == 0:
            return []

        for col in columns_to_check:
            uniqueness_score = self.df[col].nunique() / len(self.df)
            dtype_score = 0
            if pd.api.types.is_integer_dtype(self.df[col]):
                dtype_score = 0.2
            elif pd.api.types.is_string_dtype(self.df[col]):
                dtype_score = 0.1

            name_score = 0
            if any(keyword in col.lower() for keyword in ['id', 'key', 'identifier', 'uuid', 'pk']):
                name_score = 0.3

            null_penalty = self.df[col].isnull().sum() / len(self.df)
            total_score = uniqueness_score + dtype_score + name_score - null_penalty

            if total_score > 0.8:
                id_candidates.append(col)

        print(f"Automatically detected ID columns: {id_candidates}")
        return id_candidates

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
        numerical_cols = [col for col in numerical_cols if col not in self.id_columns_to_ignore]

        desc_stats = {}
        numerical_summary = {}
        for col in numerical_cols:
            col_data = self.df[col].dropna()
            stats = col_data.describe().to_dict()
            stats['std'] = float(col_data.std() if not col_data.empty else 0.0)
            desc_stats[col] = stats
            numerical_summary[col] = stats  # populate numerical_summary for the suggestion function

            # Distributions
            self.results['distributions'][col] = {'skewness': float(skew(col_data)) if len(col_data) > 1 else 0.0}

            # Histogram
            counts, bin_edges = np.histogram(col_data, bins='auto')
            self.results['histogram_data'][col] = {
                'counts': counts.tolist(),
                'bin_edges': bin_edges.tolist()
            }

            # Outliers (IQR method)
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower_outliers = self.df[self.df[col] < lower_bound]
            upper_outliers = self.df[self.df[col] > upper_bound]
            lower_count = len(lower_outliers)
            upper_count = len(upper_outliers)
            total_count = lower_count + upper_count
            total_rows = len(self.df)

            self.results['outlier_info'][col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': total_count,
                'outlier_percentage': (total_count / total_rows) * 100 if total_rows > 0 else 0,
                'lower_outlier_count': lower_count,
                'lower_outlier_percentage': (lower_count / total_rows) * 100 if total_rows > 0 else 0,
                'upper_outlier_count': upper_count,
                'upper_outlier_percentage': (upper_count / total_rows) * 100 if total_rows > 0 else 0,
            }

        self.results['descriptive_statistics'] = desc_stats
        self.results['numerical_summary'] = numerical_summary  # <-- important for constant detection

        # Correlations
        if numerical_cols:
            correlation_matrix = self.df[numerical_cols].corr()
            self.results['correlations'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'target_correlation': correlation_matrix[
                    self.target_column].to_dict() if self.target_column in numerical_cols else "Target column is not numeric."
            }
        else:
            self.results['correlations'] = {}

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
        cols_to_check = self.df.columns.drop(self.id_columns_to_ignore).tolist()
        df_for_duplicates_check = self.df if not cols_to_check else self.df[cols_to_check]
        duplicate_mask = df_for_duplicates_check.duplicated(keep=False)
        duplicate_rows_df = self.df[duplicate_mask]
        total_rows = len(self.df)
        self.results['row_duplicate_info'] = {
            'total_duplicates': len(duplicate_rows_df),
            'duplicate_percentage': (len(duplicate_rows_df) / total_rows) * 100 if total_rows > 0 else 0,
            'duplicate_rows': duplicate_rows_df.sort_values(by=list(df_for_duplicates_check.columns)).to_dict('records'),
            'ignored_columns': self.id_columns_to_ignore
        }
        return self

    def analyze_feature_duplicates(self):
        total_rows = len(self.df)
        for col in self.df.columns:
            duplicate_count = total_rows - self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            most_frequent_value = value_counts.index[0] if not value_counts.empty else None
            most_frequent_count = int(value_counts.iloc[0]) if not value_counts.empty else 0
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
         .analyze_row_duplicates()
         .analyze_feature_duplicates())
        return self.results
