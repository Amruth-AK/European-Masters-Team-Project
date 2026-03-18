"""
mlcompass.analysis.advisories — Dataset-level advisory generation
================================================================
"""


def generate_dataset_advisories(X, y):
    """
    Inspect dataset-level properties and return a list of advisory dicts.
    Each advisory has: {'category', 'severity', 'title', 'detail', 'suggested_params'}
    """
    advisories = []
    n_rows, n_cols = X.shape
    missing_rate = float(X.isnull().mean().mean())

    # 1. Very small dataset
    if n_rows < 500:
        advisories.append({
            'category': 'Small Dataset',
            'severity': 'medium',
            'title': f"Small dataset ({n_rows} rows) — consider reducing model complexity",
            'detail': (
                f"With only {n_rows} rows, default LightGBM settings risk overfitting. "
                f"The suggested parameters below constrain tree growth and add regularisation."
            ),
            'suggested_params': {
                'num_leaves': 15,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
            },
        })

    # 2. Very wide dataset (n_features >> n_rows)
    feature_cols = X.shape[1]
    if feature_cols > n_rows * 0.5 and feature_cols > 50:
        advisories.append({
            'category': 'High Dimensionality',
            'severity': 'medium',
            'title': f"Wide dataset ({feature_cols} features, {n_rows} rows) — risk of overfitting",
            'detail': (
                f"The dataset has {feature_cols} features vs {n_rows} rows. "
                f"Reduce `colsample_bytree` (0.5–0.7) and add L1 regularisation "
                f"(`reg_alpha`) to encourage sparsity. Consider feature selection "
                f"before adding interaction or polynomial features."
            ),
            'suggested_params': {
                'colsample_bytree': 0.5,
                'reg_alpha': 0.5,
                'min_child_samples': 20,
            },
        })

    # 3. High global missing rate
    if missing_rate > 0.20:
        advisories.append({
            'category': 'High Missingness',
            'severity': 'low',
            'title': f"High global missing rate ({missing_rate*100:.0f}%) — prioritise imputation transforms",
            'detail': (
                f"{missing_rate*100:.0f}% of all cells are missing. "
                f"Prioritise `impute_median` and `missing_indicator` suggestions. "
                f"LightGBM handles NaN natively, but explicit imputation can still help "
                f"downstream feature engineering (e.g. interactions involving NaN columns)."
            ),
            'suggested_params': None,
        })

    return advisories
