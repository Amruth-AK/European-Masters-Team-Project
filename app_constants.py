"""
app_constants.py — Shared constants for the Feature Engineering Recommender
============================================================================
All feature-list schemas, method catalogues, suggestion group definitions,
and model hyper-parameters that are shared across recommend_app.py,
transforms.py, and ui_components.py.
"""

# =============================================================================
# CONSTANTS
# =============================================================================

SENTINEL_NC = -10.0   # num+cat  (n_numerical == 1): used for features requiring num+num or cat+cat
SENTINEL_CC = -20.0   # cat+cat  (n_numerical == 0): used for features requiring num+num or num+cat

DATASET_FEATURES = [
    'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
    'cat_ratio', 'missing_ratio', 'row_col_ratio',
    'n_classes', 'class_imbalance_ratio',
    'avg_feature_corr', 'max_feature_corr',
    'avg_target_corr', 'max_target_corr',
    'landmarking_score',
    'baseline_score', 'baseline_std', 'relative_headroom',
]

NUMERICAL_COLUMN_FEATURES = [
    'null_pct', 'unique_ratio', 'outlier_ratio',
    'skewness', 'kurtosis_val', 'coeff_variation',
    'zeros_ratio', 'entropy',
    'is_binary', 'range_iqr_ratio',
    'baseline_feature_importance', 'importance_rank_pct',
    'spearman_corr_target', 'mutual_info_score',
    'shapiro_p_value',
    'bimodality_coefficient',
    'pct_negative', 'pct_in_0_1_range',
]

CATEGORICAL_COLUMN_FEATURES = [
    'null_pct', 'n_unique', 'unique_ratio',
    'entropy', 'normalized_entropy',
    'is_binary', 'is_low_cardinality', 'is_high_cardinality',
    'top_category_dominance', 'top3_category_concentration',
    'rare_category_pct',
    'conditional_entropy',
    'baseline_feature_importance', 'importance_rank_pct',
    'mutual_info_score', 'pps_score',
]

INTERACTION_PAIR_FEATURES = [
    # Type indicator
    'n_numerical_cols',
    # Shared pairwise features (all types)
    'pearson_corr', 'spearman_corr',
    'mutual_info_pair', 'mic_score', 'scale_ratio',
    # Combined order-invariant individual stats (all types)
    'sum_importance', 'max_importance', 'min_importance',
    'sum_null_pct', 'max_null_pct',
    'sum_unique_ratio', 'abs_diff_unique_ratio',
    'sum_entropy', 'abs_diff_entropy',
    'sum_target_corr', 'abs_diff_target_corr',
    'sum_mi_target', 'abs_diff_mi_target',
    'both_binary',
    # num+num specific (SENTINEL_NC=-10 for num+cat, SENTINEL_CC=-20 for cat+cat)
    'product_of_means',
    'abs_mean_ratio',
    'sum_cv', 'abs_diff_cv',
    'sum_skewness', 'abs_diff_skewness',
    'sign_concordance',
    'n_positive_means',
    # num+cat specific (-10 for num+num, -20 for cat+cat)
    'eta_squared',
    'anova_f_stat',
    'n_groups',
    # cat+cat specific (-10 for num+cat, -20 for num+num)
    'cramers_v',
    'joint_cardinality',
    'cardinality_ratio',
    'joint_sparsity',
]

# Row collector: dataset-level features only (no per-column features).
ROW_DATASET_FEATURES = [
    'n_numeric_cols_used',
    'avg_numeric_mean',
    'avg_numeric_std',
    'avg_missing_pct',
    'max_missing_pct',
    'avg_row_variance',
    'pct_rows_with_any_missing',
    'pct_cells_zero',
    'pct_rows_with_any_zero',
    'numeric_col_corr_mean',
    'numeric_col_corr_max',
    'avg_row_entropy',
    'numeric_range_ratio',
]

# Family names exactly as used in collect_row_features.py `method` field.
ROW_FAMILIES = [
    'row_numeric_stats',   # row_mean, row_median, row_sum, row_std, row_min, row_max, row_range
    'row_zero_stats',      # row_zero_count, row_zero_percentage
    'row_missing_stats',   # row_missing_count, row_missing_percentage
]

NUMERICAL_METHODS = [
    'log_transform', 'sqrt_transform', 'polynomial_square',
    'polynomial_cube', 'reciprocal_transform', 'quantile_binning',
    'impute_median', 'missing_indicator',
]

CATEGORICAL_METHODS = [
    'frequency_encoding', 'target_encoding', 'onehot_encoding',
    'hashing_encoding', 'missing_indicator',
]

INTERACTION_METHODS_NUM_NUM = [
    'product_interaction', 'division_interaction',
    'addition_interaction', 'abs_diff_interaction',
]

INTERACTION_METHODS_CAT_NUM = ['group_mean', 'group_std']

INTERACTION_METHODS_CAT_CAT = ['cat_concat']

BASE_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1,
}

METHOD_DESCRIPTIONS = {
    'log_transform': 'Log transform — reduces right skew',
    'sqrt_transform': 'Square root transform — mild skew reduction',
    'polynomial_square': 'Add squared feature — captures U-shaped effects',
    'polynomial_cube': 'Add cubed feature — captures asymmetric nonlinearity',
    'reciprocal_transform': 'Add reciprocal (1/x) — captures diminishing returns',
    'quantile_binning': 'Quantile binning — discretizes into 5 bins',
    'impute_median': 'Impute missing with median',
    'missing_indicator': 'Add binary missing indicator column',
    'frequency_encoding': 'Replace categories with their frequency',
    'target_encoding': 'Replace categories with smoothed target mean',
    'onehot_encoding': 'One-hot encode (drop first)',
    'hashing_encoding': 'Hash encode into 32 buckets',
    'product_interaction': 'Multiply two columns (A × B)',
    'division_interaction': 'Divide columns (A / |B|)',
    'addition_interaction': 'Add columns (A + B)',
    'abs_diff_interaction': 'Absolute difference |A − B|',
    'group_mean': 'Group-by mean (numeric grouped by category)',
    'group_std': 'Group-by std (numeric grouped by category)',
    'cat_concat': 'Concatenate two categories → new feature',
    'row_numeric_stats': 'Row statistics — adds row mean, median, sum, std, min, max, range across numeric cols',
    'row_zero_stats': 'Row zero counts — adds count and % of zeros per row across numeric cols',
    'row_missing_stats': 'Row missing counts — adds count and % of NaNs per row across all cols',
    # Date / time
    'date_features': 'Date extraction — year, month, day, day-of-week, quarter, is_weekend + days-since-min',
    'date_cyclical': 'Cyclic encoding — sin/cos of month & day-of-week (legacy, all-in-one)',
    'date_cyclical_month':  'Cyclic month — sin/cos of month (1–12); keeps Dec ↔ Jan adjacent',
    'date_cyclical_dow':    'Cyclic day-of-week — sin/cos of weekday (0–6); keeps Sun ↔ Mon adjacent',
    'date_cyclical_dom':    'Cyclic day-of-month — sin/cos of day (1–31); keeps month-end ↔ month-start adjacent',
    'date_cyclical_hour':   'Cyclic hour — sin/cos of hour (0–23); keeps 23:00 ↔ 00:00 adjacent',
    # Day-of-week categorical columns
    'dow_ordinal':  'Day-of-week ordinal — maps Mon→0, Tue→1 … Sun→6; replaces text column with numeric weekday',
    'dow_cyclical': 'Day-of-week cyclical — sin/cos of weekday (0–6); keeps Sunday and Monday adjacent',
    # Free-form text
    'text_stats': 'Text statistics — word count, char count, avg word length, uppercase %, digit %, punctuation %',
    'text_tfidf': 'TF-IDF features — top-20 unigram TF-IDF scores as new numeric columns',
}

# =============================================================================
# SUGGESTION PROBLEM GROUPS
# =============================================================================

_SUGGESTION_GROUPS = [
    {
        "id":      "imbalance",
        "icon":    "⚖️",
        "title":   "Class Imbalance",
        "methods": {"class_weight_balance"},
        "color":   "#1a1009",
        "border":  "#f0883e",
        "explain": (
            "The target classes are not equally represented. Without correction, "
            "the model will optimise for the majority class and likely underperform "
            "on the minority. Enabling **class reweighting** tells LightGBM to "
            "penalise mistakes on rare classes more heavily.\n\n"
            "This is applied **only to the enhanced model**, so you can directly "
            "compare its effect against the uncorrected baseline."
        ),
    },
    {
        "id":      "missing_values",
        "icon":    "🕳️",
        "title":   "Missing Values",
        "methods": {"impute_median", "missing_indicator"},
        "color":   "#0d2033",
        "border":  "#58a6ff",
        "explain": (
            "Some columns contain missing values. These transforms fill gaps "
            "with the column median and add a binary **'was missing'** flag, "
            "which can itself be predictive when missingness is not random."
        ),
    },
    {
        "id":      "skewed",
        "icon":    "📉",
        "title":   "Skewed Distributions",
        "methods": {"log_transform", "sqrt_transform", "reciprocal_transform", "quantile_binning"},
        "color":   "#0d1f14",
        "border":  "#3fb950",
        "explain": (
            "One or more numeric columns have long right tails or heavy skew. "
            "Log, square-root, and reciprocal transforms compress the tail and "
            "make distributions more symmetric, helping tree models find cleaner "
            "splits. Quantile binning fully discretises a column when the "
            "distribution is very irregular."
        ),
    },
    {
        "id":      "nonlinear",
        "icon":    "📈",
        "title":   "Nonlinear Effects",
        "methods": {"polynomial_square", "polynomial_cube"},
        "color":   "#150d2a",
        "border":  "#a371f7",
        "explain": (
            "Some numeric columns may have a **U-shaped or asymmetric** "
            "relationship with the target. Adding squared and cubed versions "
            "exposes these patterns without requiring the model to discover "
            "them through deep multiplicative interactions."
        ),
    },
    {
        "id":      "categorical",
        "icon":    "🏷️",
        "title":   "Categorical Encoding",
        "methods": {"frequency_encoding", "target_encoding", "onehot_encoding", "hashing_encoding"},
        "color":   "#0d1f14",
        "border":  "#3fb950",
        "explain": (
            "Categorical columns need numeric representation. "
            "**Target encoding** replaces each category with its smoothed mean "
            "target value (powerful, use with care). **Frequency encoding** uses "
            "how often a category appears. **One-hot** works best for low-cardinality "
            "columns; **hashing** handles high-cardinality ones efficiently."
        ),
    },
    {
        "id":      "interactions",
        "icon":    "🔗",
        "title":   "Feature Interactions",
        "methods": {
            "product_interaction", "division_interaction",
            "addition_interaction", "abs_diff_interaction",
            "group_mean", "group_std", "cat_concat",
        },
        "color":   "#1a0d2a",
        "border":  "#a371f7",
        "explain": (
            "Some column pairs show high mutual information or correlation with "
            "the target when **combined**. Multiplying, dividing, or differencing "
            "two features can reveal signal that neither column captures alone. "
            "Group aggregations (mean/std of a numeric column within each "
            "category) are especially powerful."
        ),
    },
    {
        "id":      "row_patterns",
        "icon":    "📊",
        "title":   "Row-Level Patterns",
        "methods": {"row_numeric_stats", "row_zero_stats", "row_missing_stats"},
        "color":   "#1f1a0d",
        "border":  "#d29922",
        "explain": (
            "These transforms summarise each **row** rather than individual columns, "
            "capturing cross-column patterns the model would otherwise miss.\n\n"
            "- **Numeric stats** (mean, median, sum, std, min, max, range) — a global "
            "'intensity' score useful when many features measure the same kind of thing "
            "(scores, counts, frequencies).\n"
            "- **Zero counts** (count and % of zeros per row) — distinguishes sparse rows "
            "from dense ones; a strong signal in count or indicator matrices.\n"
            "- **Missing counts** (count and % of NaNs per row) — useful when missingness "
            "is not random and its extent per row carries predictive information."
        ),
    },
    {
        "id":      "date_features",
        "icon":    "📅",
        "title":   "Date / Time Features",
        "methods": {
            "date_features",
            "date_cyclical",
            "date_cyclical_month", "date_cyclical_dow",
            "date_cyclical_dom",   "date_cyclical_hour",
            "dow_ordinal",         "dow_cyclical",
        },
        "color":   "#0d1f2a",
        "border":  "#58a6ff",
        "explain": (
            "One or more columns contain dates or timestamps. Raw date strings are "
            "meaningless to LightGBM — the tool splits each column into structured "
            "numeric components and, optionally, wraps periodic ones in **cyclic encoding**.\n\n"
            "**Base extraction** *(📌 checked by default)*: year, month, day, "
            "day-of-week, quarter, is_weekend, days-since-earliest-date. "
            "The original string column is dropped.\n\n"
            "**Cyclic encoding** *(optional — unchecked by default)*: plain integers "
            "treat Jan (1) and Dec (12) as far apart, or Mon (0) and Sun (6) as opposites. "
            "Cyclic encoding wraps each component into a sin/cos pair so the model sees "
            "period boundaries as adjacent. Four independent options are shown — "
            "tick only the ones where periodicity genuinely matters for your problem:\n\n"
            "- 🔵 **Month** — retail seasonality, energy demand, tax deadlines\n"
            "- 🟢 **Day of week** — weekly footfall, fraud, support-ticket volume\n"
            "- 🟡 **Day of month** — billing cycles, salary dates, month-end reporting\n"
            "- 🟠 **Hour** *(shown only for timestamps)* — rush-hour traffic, "
            "overnight fraud, shift patterns\n\n"
            "**When to skip cyclical encoding:** if your model only needs to *identify* "
            "which month or weekday something happened (ordinal identity is sufficient), "
            "or if you have very few rows and want to avoid extra dimensions."
        ),
    },
    {
        "id":      "text_features",
        "icon":    "📝",
        "title":   "Free-Text Features",
        "methods": {"text_stats", "text_tfidf"},
        "color":   "#1a1020",
        "border":  "#bc8cff",
        "explain": (
            "One or more columns contain long free-form text (reviews, descriptions, "
            "addresses, etc.). These cannot be used as-is. Two complementary "
            "approaches are suggested:\n\n"
            "- **Text statistics** extract surface-level signals: word count, "
            "character count, average word length, uppercase ratio, digit ratio, "
            "and punctuation ratio — fast and always applicable.\n"
            "- **TF-IDF features** decompose the text into the top-20 most "
            "discriminative unigrams and add each as a numeric column — more "
            "powerful but adds dimensionality."
        ),
    },
]

_GROUPS_BY_ID    = {g["id"]: g for g in _SUGGESTION_GROUPS}
_METHOD_TO_GROUP = {
    m: g["id"]
    for g in _SUGGESTION_GROUPS
    for m in g["methods"]
}

# Methods available for custom steps, by type
_CUSTOM_METHODS = {
    "numerical":   ["log_transform", "sqrt_transform", "polynomial_square",
                    "polynomial_cube", "reciprocal_transform", "quantile_binning",
                    "impute_median", "missing_indicator"],
    "categorical": ["frequency_encoding", "target_encoding",
                    "onehot_encoding", "hashing_encoding", "missing_indicator"],
    "interaction": ["product_interaction", "division_interaction",
                    "addition_interaction", "abs_diff_interaction",
                    "group_mean", "group_std", "cat_concat"],
    "row":         ["row_numeric_stats", "row_zero_stats", "row_missing_stats"],
}


# =============================================================================
# IMBALANCE THRESHOLDS
# =============================================================================

# Imbalance thresholds
_IMBALANCE_MODERATE = 5.0    # 5:1  → recommend is_unbalance / class_weight
_IMBALANCE_SEVERE   = 20.0   # 20:1 → also recommend scale_pos_weight tuning

# Multiclass-specific: when imbalance is severe, class_weight='balanced' over-corrects
# (e.g. 90%/2% case → 45× weight on minority → model ignores dominant class entirely).
# Above these thresholds, skip reweighting — LightGBM's defaults are better.
_IMBALANCE_MULTICLASS_RATIO_CAP  = 15.0   # max:min ratio above which balanced hurts
_IMBALANCE_MULTICLASS_DOMINANT   = 0.65   # dominant class fraction above which balanced hurts