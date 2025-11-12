import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import List, Optional
import re
from math import isfinite

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
    # Split on non-alnum to get tokens; also check exact start/end patterns
    tokens = re.split(r"[^0-9a-zA-Z]+", name)
    token_set = set(t for t in tokens if t)

    strong = {"id", "idx", "index", "key", "uuid", "guid", "pk"}
    if name in strong or (name.endswith("_id")) or (name.startswith("id_")):
        return 0.5
    if token_set & strong:
        return 0.4
    # common explicit variants
    common_suffix = name.endswith(("id", "_key", "_uuid", "_guid", "_pk"))
    common_prefix = name.startswith(("id_", "uuid_", "guid_", "pk_"))
    if common_suffix or common_prefix:
        return 0.4
    return 0.0

def _target_predictiveness_penalty(df: pd.DataFrame, col: str, target: str) -> float:
    """
    Down-weight 'ID-looking' columns that actually correlate with the target.
    - For numeric target: absolute Pearson corr > 0.2 → penalty.
    - For categorical target: use ANOVA F-ish proxy via group means variance.
    Returns penalty in [0, 0.25].
    """
    if target is None or target not in df.columns or col == target:
        return 0.0

    try:
        y = df[target]
        x = df[col]
        # if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            if pd.api.types.is_numeric_dtype(x):
                corr = df[[col, target]].corr().iloc[0, 1]
                corr = 0.0 if pd.isna(corr) else abs(corr)
                return 0.25 if corr >= 0.3 else (0.15 if corr >= 0.2 else 0.0)
            else:
                # categorical x → use group mean spread
                means = df.groupby(col, dropna=False)[target].mean()
                spread = means.std()
                return 0.2 if spread and spread > 0.3 * df[target].std() else 0.0
        else:
            # categorical target → mutual info proxy: target rate variance across x buckets
            # (cheap, rough): compute p(target==mode | x)
            mode = y.mode().iloc[0]
            rates = df.groupby(col, dropna=False)[target].apply(lambda s: (s == mode).mean())
            spread = rates.std()
            return 0.25 if spread and spread > 0.1 else 0.0
    except Exception:
        return 0.0


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
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
        Best-effort identifier detection with multiple signals and adaptive thresholds.
        Excludes the target column automatically.
        """
        id_cols = []
        if len(self.df) == 0:
            return id_cols

        n = len(self.df)
        base_thresh = 0.88
        # adapt threshold: a bit stricter on tiny data; a bit looser on very large
        if n < 200:
            base_thresh += 0.04
        elif n > 10_000:
            base_thresh -= 0.06

        cols_to_check = [c for c in self.df.columns if c != self.target_column]

        for col in cols_to_check:
            s = self.df[col]
            # quick skip: all-null or single unique
            nunique = s.nunique(dropna=True)
            if nunique <= 1:
                continue

            null_ratio = s.isnull().mean()
            uniq_ratio = nunique / max(1, n)

            # data type hints
            is_int = pd.api.types.is_integer_dtype(s)
            is_str = pd.api.types.is_string_dtype(s)
            is_num = pd.api.types.is_numeric_dtype(s)

            # name score (strict token logic)
            name_score = _name_token_score(col)

            # pattern scores (strings only)
            uuid_frac = _is_uuid_like(s) if is_str else 0.0
            hash_frac = _is_hex_hash_like(s) if is_str else 0.0
            const_width_digit_frac = _is_constant_width_numeric_string(s) if is_str else 0.0
            datetime_frac = _is_datetime_like(s) if is_str else 0.0  # used as negative signal

            # sequence score (integers only)
            seq_score = _is_mono_increasing_integer(s) if is_int else 0.0

            # type bonus
            dtype_bonus = 0.12 if is_int else (0.06 if is_str else 0.0)

            # build score (cap certain parts so score stays calibrated)
            score = 0.0
            score += min(0.55, 0.55 * uniq_ratio)              # uniqueness
            score += dtype_bonus                               # type hint
            score += min(0.5, name_score)                      # name token
            score += min(0.35, 0.35 * uuid_frac)               # uuid pattern
            score += min(0.30, 0.30 * hash_frac)               # hex/hash pattern
            score += min(0.25, 0.25 * const_width_digit_frac)  # fixed-width digits
            score += min(0.35, seq_score)                      # sequential ints
            score -= min(0.40, 0.40 * null_ratio)              # null penalty
            # date-like strings should NOT be treated as IDs
            score -= min(0.30, 0.30 * datetime_frac)

            # if it actually looks predictive of target, penalize (avoid dropping useful features)
            score -= _target_predictiveness_penalty(self.df, col, self.target_column)

            # final guardrails: require either (very unique) OR (name/pattern evidence)
            has_strong_evidence = (
                name_score >= 0.4
                or uuid_frac >= 0.5
                or hash_frac >= 0.5
                or seq_score >= 0.45
                or const_width_digit_frac >= 0.6
            )
            is_very_unique = uniq_ratio >= 0.98

            if (score >= base_thresh) and (is_very_unique or has_strong_evidence):
                id_cols.append(col)

        print(f"✅ Improved ID detection → {id_cols}")
        return id_cols


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