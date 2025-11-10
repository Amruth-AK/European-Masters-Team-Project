import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import List, Optional

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        # --- Automatic ID Column Detection ---
        self.id_columns_to_ignore = self._detect_id_columns()
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

    def _detect_id_columns(self) -> List[str]:
        """
        Automatically detects potential ID columns based on a scoring system.
        The target column is always excluded from this check.
        """
        id_candidates = []
        # Exclude the target column from the search for identifiers
        columns_to_check = [col for col in self.df.columns if col != self.target_column]

        if len(self.df) == 0:
            return []

        for col in columns_to_check:
            # Heuristic 1: Uniqueness
            uniqueness_score = self.df[col].nunique() / len(self.df)

            # Heuristic 2: Data Type
            dtype_score = 0
            if pd.api.types.is_integer_dtype(self.df[col]):
                dtype_score = 0.2
            elif pd.api.types.is_string_dtype(self.df[col]):
                dtype_score = 0.1

            # Heuristic 3: Column Name (using the expanded keyword list from the old logic)
            name_score = 0
            if any(keyword in col.lower() for keyword in ['id', 'key', 'identifier', 'uuid', 'pk']):
                name_score = 0.3

            # Heuristic 4: Low Nulls (penalize for nulls)
            null_penalty = self.df[col].isnull().sum() / len(self.df)

            # Combine scores
            total_score = uniqueness_score + dtype_score + name_score - null_penalty

            # A threshold is used to identify suitable ID columns
            if total_score > 0.8:  # This threshold can be adjusted
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
        # Exclude automatically detected ID columns from numerical analysis
        numerical_cols = [col for col in numerical_cols if col not in self.id_columns_to_ignore]

        desc_stats = self.df[numerical_cols].describe().to_dict()
        self.results['descriptive_statistics'] = desc_stats

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

        # Distributions and Outliers
        total_rows = len(self.df)
        for col in numerical_cols:
            col_data = self.df[col].dropna()
            self.results['distributions'][col] = {'skewness': skew(col_data)}

            # Histogram
            counts, bin_edges = np.histogram(col_data, bins='auto')
            self.results['histogram_data'][col] = {
                'counts': counts.tolist(),
                'bin_edges': bin_edges.tolist()
            }

            Q1 = desc_stats[col]['25%']
            Q3 = desc_stats[col]['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Find lower and upper outliers separately
            lower_outliers = self.df[self.df[col] < lower_bound]
            upper_outliers = self.df[self.df[col] > upper_bound]

            # Get the counts
            lower_outlier_count = len(lower_outliers)
            upper_outlier_count = len(upper_outliers)
            total_outlier_count = lower_outlier_count + upper_outlier_count

            # Store all information in the results dictionary
            self.results['outlier_info'][col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': total_outlier_count,
                'outlier_percentage': (total_outlier_count / total_rows) * 100 if total_rows > 0 else 0,
                'lower_outlier_count': lower_outlier_count,
                'lower_outlier_percentage': (lower_outlier_count / total_rows) * 100 if total_rows > 0 else 0,
                'upper_outlier_count': upper_outlier_count,
                'upper_outlier_percentage': (upper_outlier_count / total_rows) * 100 if total_rows > 0 else 0,
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
        # Determine the columns to check for duplicates, excluding ID columns
        cols_to_check = self.df.columns.drop(self.id_columns_to_ignore).tolist()
        
        if not cols_to_check:
            # Handle case where all columns are considered IDs
            df_for_duplicates_check = self.df
        else:
            df_for_duplicates_check = self.df[cols_to_check]

        duplicate_mask = df_for_duplicates_check.duplicated(keep=False)
        duplicate_rows_df = self.df[duplicate_mask]
        num_duplicates = len(duplicate_rows_df)
        total_rows = len(self.df)

        self.results['row_duplicate_info'] = {
            'total_duplicates': num_duplicates,
            'duplicate_percentage': (num_duplicates / total_rows) * 100 if total_rows > 0 else 0,
            'duplicate_rows': duplicate_rows_df.sort_values(
                by=list(df_for_duplicates_check.columns)
            ).to_dict('records'),
            'ignored_columns': self.id_columns_to_ignore
        }
        return self

    def analyze_feature_duplicates(self):
        """Analyzes each column for duplicate values and their frequency."""
        total_rows = len(self.df)
        for col in self.df.columns:
            if total_rows > 0:
                duplicate_count = total_rows - self.df[col].nunique()
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
         .analyze_row_duplicates()
         .analyze_feature_duplicates())
        return self.results