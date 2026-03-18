"""
mlcompass.recommendation.meta_models — Meta-model loading & feature vector construction
======================================================================================
"""

import os
import json
import importlib.resources
import lightgbm as lgb
import pandas as pd


def load_meta_models(model_dir=None):
    """Load all available meta-models from disk.

    Parameters
    ----------
    model_dir : str or path-like, optional
        Directory containing subdirectories (numerical/, categorical/,
        interaction/, row/) each with a *_config.json and *_regressor.txt.
        If None, attempts to load bundled models from mlcompass.data.meta_models.
    """
    if model_dir is None:
        try:
            model_dir = str(
                importlib.resources.files("mlcompass") / "data" / "meta_models"
            )
        except Exception:
            raise FileNotFoundError(
                "No model_dir specified and no bundled meta-models found. "
                "Pass model_dir= explicitly."
            )

    models = {}
    for ctype in ['numerical', 'categorical', 'interaction', 'row']:
        type_dir = os.path.join(model_dir, ctype)
        config_path = os.path.join(type_dir, f'{ctype}_config.json')
        reg_path = os.path.join(type_dir, f'{ctype}_regressor.txt')

        if not os.path.exists(config_path) or not os.path.exists(reg_path):
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        booster = lgb.Booster(model_file=reg_path)

        models[ctype] = {
            'booster': booster,
            'config': config,
            'feature_names': config['feature_names'],
            'method_vocab': config['method_vocab'],
        }
    return models


def build_feature_vector(meta_dict, method, config):
    """Build a single feature vector matching training schema."""
    feature_names = config['feature_names']
    method_vocab = config['method_vocab']

    row = {}
    for f in feature_names:
        if f.startswith('method_'):
            row[f] = 0
        else:
            row[f] = meta_dict.get(f, -999)

    method_col = f'method_{method}'
    if method_col in row:
        row[method_col] = 1

    return pd.DataFrame([row])[feature_names].fillna(-999)


def _is_near_bijection(col_a, col_b, threshold=0.95):
    """
    Return True if col_a and col_b encode the same entity under different schemes
    (i.e. they have a near-1:1 mapping in both directions).
    """
    try:
        n_unique_a = col_a.nunique(dropna=True)
        n_unique_b = col_b.nunique(dropna=True)

        if n_unique_a < 3 or n_unique_b < 3:
            return False

        ratio = min(n_unique_a, n_unique_b) / max(n_unique_a, n_unique_b)
        if ratio < 0.80:
            return False

        tmp = pd.DataFrame({'a': col_a.values, 'b': col_b.values}).dropna()
        if len(tmp) < 10:
            return False

        b_per_a = tmp.groupby('a')['b'].nunique()
        pct_a_to_one_b = float((b_per_a == 1).mean())

        a_per_b = tmp.groupby('b')['a'].nunique()
        pct_b_to_one_a = float((a_per_b == 1).mean())

        return pct_a_to_one_b >= threshold and pct_b_to_one_a >= threshold
    except Exception:
        return False
