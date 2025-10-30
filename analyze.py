import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import List, Optional 

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str, id_columns_to_ignore: Optional[List[str]] = None):
        self.df = df.copy()
        self.target_column = target_column
        self.id_columns_to_ignore = id_columns_to_ignore if id_columns_to_ignore is not None else []
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
                # Original Info
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': total_outlier_count,
                'outlier_percentage': (total_outlier_count / total_rows) * 100 if total_rows > 0 else 0,
                # New Detailed Info
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
        # Determine the columns to check for duplicates
        if self.id_columns_to_ignore:
            # Create a temporary DataFrame without the ignored columns
            df_for_duplicates_check = self.df.drop(columns=self.id_columns_to_ignore)
            print(f"Ignoring columns for duplicate check: {self.id_columns_to_ignore}")
        else:
            # Use the full DataFrame if no columns are specified to be ignored
            df_for_duplicates_check = self.df
            print("Checking for duplicates across all columns.")
        print(self.id_columns_to_ignore)
        print("we")
        # --- LOGIC FIX STARTS HERE ---

        # 1. Use keep=False to mark ALL occurrences of duplicates as True.
        # This makes it easier to identify and group them.
        duplicate_mask = df_for_duplicates_check.duplicated(keep=False)

        # 2. Get the actual duplicated rows from the original DataFrame using the mask.
        # This correctly handles any index misalignments.
        duplicate_rows_df = self.df[duplicate_mask]
        
        # 3. The total number of duplicate rows is simply the length of this new DataFrame.
        num_duplicates = len(duplicate_rows_df)
        total_rows = len(self.df)
        
        # --- LOGIC FIX ENDS HERE ---

        self.results['row_duplicate_info'] = {
            'total_duplicates': num_duplicates,
            'duplicate_percentage': (num_duplicates / total_rows) * 100 if total_rows > 0 else 0,
            # We sort by the columns used for checking to group duplicates together visually
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


