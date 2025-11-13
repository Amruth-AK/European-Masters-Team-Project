import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import mode as sci_mode
from typing import List, Optional
import re
from math import isfinite

# --- New Helper Functions for ID Detection ---

def _is_uuid_like(series: pd.Series, sample: int = 500) -> float:
    """Return fraction of non-null values matching UUID v1-5 pattern."""
    pat = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
    )
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    s = s.sample(min(sample, len(s)), random_state=42) if len(s) > sample else s
    return (s.str.match(pat)).mean()

def _is_hex_hash_like(series: pd.Series, sample: int = 500) -> float:
    """
    Return fraction matching hex-only tokens of typical hash length (>= 16).
    e.g., 32 (MD5), 40 (SHA1), 56/64 (SHA-2), etc.
    """
    pat = re.compile(r"^[0-9a-fA-F]{16,}$")
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    s = s.sample(min(sample, len(s)), random_state=42) if len(s) > sample else s
    # Also require low length variance (IDs/hashes tend to be constant width)
    lengths = s.str.len()
    length_var = lengths.var() if len(lengths) > 1 else 0.0
    return (s.str.match(pat)).mean() * (1.0 if (length_var is not None and length_var < 4.0) else 0.6)

def _is_constant_width_numeric_string(series: pd.Series, sample: int = 500) -> float:
    """
    Return fraction of values that are digit-only strings with narrow length variance (e.g., 8-12 digits).
    Good at catching zero-padded IDs like '00012345'.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    s = s.sample(min(sample, len(s)), random_state=42) if len(s) > sample else s
    digit_frac = s.str.match(r"^\d+$").mean()
    if digit_frac == 0.0:
        return 0.0
    lengths = s.str.len()
    if lengths.empty:
        return 0.0
    length_var = lengths.var() if len(lengths) > 1 else 0.0
    # Prefer “longish & tight width”
    longish = (lengths.median() >= 8)
    return digit_frac * (1.0 if (length_var < 2.0 and longish) else 0.4)

def _is_datetime_like(series: pd.Series, sample: int = 200) -> float:
    """
    Return fraction parseable as datetime; helps NOT labeling date-like strings as IDs.
    """
    s = series.dropna()
    if s.empty:
        return 0.0
    s = s.sample(min(sample, len(s)), random_state=42) if len(s) > sample else s
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        return parsed.notna().mean()
    except Exception:
        return 0.0

def _is_mono_increasing_integer(series: pd.Series) -> float:
    """
    Score for monotonic/incrementing integer sequence IDs.
    Returns 0..1 score (1 = very likely an autoincrement id).
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype("int64", errors="ignore")
    if s.empty:
        return 0.0
    # require many uniques
    nunique = s.nunique()
    if nunique < max(10, int(0.5 * len(s))):
        return 0.0
    # correlation with rank ~ 1 for sequences
    try:
        order = np.arange(len(s))
        corr = np.corrcoef(order, np.sort(s))[0, 1]
        if not isfinite(corr):
            corr = 0.0
        # also check average step near constant
        diffs = np.diff(np.sort(s))
        step_const = (np.std(diffs) / (np.mean(diffs) + 1e-9)) < 0.25 if len(diffs) > 5 else False
        score = 0.0
        if corr > 0.99:
            score += 0.6
        elif corr > 0.97:
            score += 0.45
        if step_const:
            score += 0.25
        return min(1.0, score)
    except Exception:
        return 0.0

def _name_token_score(colname: str) -> float:
    """
    Strict, token-aware name score. Avoids mid-word hits like 'accident_risk'.
    Matches:
    'id', 'idx', 'index', 'key', 'uuid', 'guid', 'pk', '*_id', 'id_*'
    """
    name = colname.strip().lower()
    tokens = re.split(r"[^0-9a-zA-Z]+", name)
    token_set = set(t for t in tokens if t)
    strong = {"id", "idx", "index", "key", "uuid", "guid", "pk"}
    if name in strong or (name.endswith("_id")) or (name.startswith("id_")):
        return 0.5
    if token_set & strong:
        return 0.4
    common_suffix = name.endswith(("id", "_key", "_uuid", "_guid", "_pk"))
    common_prefix = name.startswith(("id_", "uuid_", "guid_", "pk_"))
    if common_suffix or common_prefix:
        return 0.4
    return 0.0


# --- Main Detection Function with Integrated Logic ---

def auto_detect_id_columns(df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
    """
    Automatically detects potential ID columns using a comprehensive scoring system.
    """
    id_candidates = {}
    columns_to_check = [col for col in df.columns if col != target_column]

    if len(df) == 0:
        return []

    for col in columns_to_check:
        score = 0.0
        col_data = df[col]

        # 1. Uniqueness Score (High weight, based on your successful adjustment)
        uniqueness = col_data.nunique() / len(df) if len(df) > 0 else 0
        if uniqueness > 0.95:
            score += 0.60 * uniqueness
        else:
            score -= (1 - uniqueness) # Penalize non-unique columns

        # 2. Name Score (Using the new, more robust token-based function)
        score += _name_token_score(col)

        # 3. Content-Based Scoring (The core of the new logic)
        is_integer_like = False
        if pd.api.types.is_integer_dtype(col_data.dtype):
            is_integer_like = True
        elif pd.api.types.is_float_dtype(col_data.dtype):
            if col_data.dropna().eq(col_data.dropna().round()).all():
                is_integer_like = True

        if is_integer_like:
            # Use the new monotonic check for integers
            score += _is_mono_increasing_integer(col_data) * 0.4 # Scale the 0-1 score
        
        elif pd.api.types.is_string_dtype(col_data.dtype) or pd.api.types.is_object_dtype(col_data.dtype):
            # Give strong bonuses for clear ID patterns in strings
            if _is_uuid_like(col_data) > 0.9:
                score += 1.0
            if _is_hex_hash_like(col_data) > 0.9:
                score += 1.0
            if _is_constant_width_numeric_string(col_data) > 0.9:
                score += 0.4
            
            # Apply a heavy penalty if the string looks like a datetime
            if _is_datetime_like(col_data) > 0.8:
                score -= 1.0
        
        elif pd.api.types.is_float_dtype(col_data.dtype):
            # Heavy penalty for true floats (those that aren't integer-like)
            score -= 0.5
            
        # 4. Null Penalty
        null_penalty = col_data.isnull().sum() / len(df) if len(df) > 0 else 0
        score -= null_penalty * 0.7

        # Use a new, higher threshold to account for the stronger scoring signals
        if score > 1.0:
            id_candidates[col] = score
        
    sorted_candidates = sorted(id_candidates.keys(), key=lambda k: id_candidates[k], reverse=True)
    print(f"Automatically detected ID columns (and scores): { {k: f'{v:.2f}' for k, v in id_candidates.items()} }")
    return sorted_candidates


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str, id_columns_to_ignore: Optional[List[str]] = None):
        self.df = df.copy()
        self.target_column = target_column
        
        if id_columns_to_ignore is not None:
            self.id_columns_to_ignore = id_columns_to_ignore
            print(f"Using user-defined ID columns: {self.id_columns_to_ignore}")
        else:
            self.id_columns_to_ignore = auto_detect_id_columns(self.df, self.target_column)

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

    # --- (The rest of the DataAnalyzer class remains unchanged) ---

    def optimize_dtypes(self):
        for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        return self

    def analyze_general_info(self):
        self.results['general_info'] = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        return self

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

    def analyze_numerical(self):
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in self.id_columns_to_ignore]
        desc_stats = self.df[numerical_cols].describe().to_dict()
        self.results['descriptive_statistics'] = desc_stats
        if numerical_cols:
            correlation_matrix = self.df[numerical_cols].corr()
            self.results['correlations'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'target_correlation': correlation_matrix[
                    self.target_column].to_dict() if self.target_column in numerical_cols else "Target column is not numeric."
            }
        else:
            self.results['correlations'] = {}
        total_rows = len(self.df)
        for col in numerical_cols:
            col_data = self.df[col].dropna()
            self.results['distributions'][col] = {'skewness': skew(col_data)}
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
            lower_outliers = self.df[self.df[col] < lower_bound]
            upper_outliers = self.df[self.df[col] > upper_bound]
            lower_outlier_count = len(lower_outliers)
            upper_outlier_count = len(upper_outliers)
            total_outlier_count = lower_outlier_count + upper_outlier_count
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

    def analyze_categorical(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            self.results['categorical_info'][col] = {
                'unique_values': self.df[col].nunique(),
                'value_counts': self.df[col].value_counts().to_dict()
            }
        return self

    def analyze_row_duplicates(self):
        cols_to_check = [col for col in self.df.columns if col not in self.id_columns_to_ignore]
        if not cols_to_check:
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

    def run_full_analysis(self):
        (self.optimize_dtypes()
         .analyze_general_info()
         .analyze_missing_values()
         .analyze_numerical()
         .analyze_categorical()
         .analyze_row_duplicates()
         .analyze_feature_duplicates())
        return self.results