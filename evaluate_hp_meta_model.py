"""
Offline evaluation of the trained HP meta-model.

For each dataset in the HP tuning DB:
  1. Get dataset meta from one row.
  2. Run the ensemble to recommend an HP config.
  3. In the DB, find the evaluated config that is *closest* (in HP space) to the
     recommended config, and use its actual primary_score as "recommended score".
  4. Compare to default config score and to best config score in that dataset.

Metrics reported:
  - Mean primary_score of the recommended config (vs default, vs best).
  - % of datasets where recommended >= default.
  - Median rank of the chosen config (1 = best in dataset).
  - Mean regret: (best - recommended) / scale per dataset.

Usage:
  python evaluate_hp_meta_model.py --hp-db hp_tuning_db_refined.csv --model-dir hp_meta_model
  python evaluate_hp_meta_model.py --hp-db hp_tuning_db_refined.csv --model-dir hp_meta_model --max-datasets 200
"""

import argparse
import os
import numpy as np
import pandas as pd


# HP bounds for normalizing distance (same as HP_SPACE in HPCollector)
HP_BOUNDS = {
    'hp_num_leaves': (4, 256),
    'hp_max_depth': (3, 15),
    'hp_learning_rate': (0.005, 0.3),
    'hp_n_estimators': (50, 3000),
    'hp_min_child_samples': (5, 100),
    'hp_subsample': (0.5, 1.0),
    'hp_colsample_bytree': (0.3, 1.0),
    'hp_reg_alpha': (1e-8, 10.0),
    'hp_reg_lambda': (1e-8, 10.0),
    'hp_max_bin': (63, 511),
}


def _config_to_hp_vector(config, keys):
    """config: dict with keys num_leaves, learning_rate, ... (no hp_ prefix)."""
    vec = []
    for k in keys:
        param = k.replace('hp_', '')
        val = config.get(param)
        if val is None:
            return None
        low, high = HP_BOUNDS[k]
        if param in ['learning_rate', 'n_estimators', 'reg_alpha', 'reg_lambda']:
            val = np.clip(val, low, high)
            if high > low:
                # log-scale normalize
                log_low, log_high = np.log1p(low), np.log1p(high)
                n = (np.log1p(val) - log_low) / (log_high - log_low)
            else:
                n = 0.0
        else:
            n = (np.clip(val, low, high) - low) / (high - low) if high > low else 0.0
        vec.append(n)
    return np.array(vec, dtype=float)


def _row_to_hp_vector(row, keys):
    """row: Series/dict with hp_* keys."""
    vec = []
    for k in keys:
        val = row.get(k)
        if pd.isna(val):
            return None
        low, high = HP_BOUNDS[k]
        if k in ['hp_learning_rate', 'hp_n_estimators', 'hp_reg_alpha', 'hp_reg_lambda']:
            val = np.clip(float(val), low, high)
            log_low, log_high = np.log1p(low), np.log1p(high)
            n = (np.log1p(val) - log_low) / (log_high - log_low)
        else:
            val = np.clip(float(val), low, high)
            n = (val - low) / (high - low) if high > low else 0.0
        vec.append(n)
    return np.array(vec, dtype=float)


def _distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation of HP meta-model")
    parser.add_argument("--hp-db", required=True, help="Path to hp_tuning_db (with refinement columns)")
    parser.add_argument("--model-dir", default="./hp_meta_model", help="Directory with hp_preparator.pkl, hp_*.pkl")
    parser.add_argument("--max-datasets", type=int, default=0, help="Max datasets to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.hp_db):
        raise FileNotFoundError(f"HP DB not found: {args.hp_db}")

    import json
    from train_hp_meta_model import HPEnsemblePredictor, HP_PARAM_NAMES
    try:
        from train_hp_meta_model import DATASET_FEATURES
    except ImportError:
        DATASET_FEATURES = [
            'n_rows', 'n_cols', 'n_numeric_cols', 'n_cat_cols',
            'cat_ratio', 'missing_ratio', 'row_col_ratio',
            'class_imbalance_ratio', 'n_classes',
            'target_std', 'target_skew', 'target_kurtosis', 'target_nunique_ratio',
            'landmarking_score', 'landmarking_score_norm',
            'avg_feature_corr', 'max_feature_corr', 'avg_target_corr', 'max_target_corr',
            'avg_numeric_sparsity', 'linearity_gap',
            'corr_graph_components', 'corr_graph_clustering', 'corr_graph_density',
            'matrix_rank_ratio',
            'std_feature_importance', 'max_minus_min_importance',
            'pct_features_above_median_importance', 'avg_skewness', 'avg_kurtosis',
        ]

    print("Loading HP DB and ensemble(s)...")
    df = pd.read_csv(args.hp_db)
    if "primary_score" not in df.columns or "dataset_name" not in df.columns:
        raise ValueError("CSV must have primary_score and dataset_name")

    ensemble = None
    ensembles = {}
    manifest_path = os.path.join(args.model_dir, "hp_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("per_task") and manifest.get("task_types"):
            for tt in manifest["task_types"]:
                subdir = os.path.join(args.model_dir, tt)
                if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "hp_scorer.pkl")):
                    ensembles[tt] = HPEnsemblePredictor.load(args.model_dir, task_type=tt)
            print(f"  Loaded per-task: {list(ensembles.keys())}")
    if not ensembles:
        for name in ["hp_preparator.pkl", "hp_scorer.pkl", "hp_predictor.pkl"]:
            if not os.path.isfile(os.path.join(args.model_dir, name)):
                raise FileNotFoundError(f"Model not found: {args.model_dir}/{name}")
        ensemble = HPEnsemblePredictor.load(args.model_dir)
        print("  Loaded single model")

    preparator = ensemble.preparator if ensemble is not None else list(ensembles.values())[0].preparator
    hp_keys = [c for c in HP_PARAM_NAMES if c in HP_BOUNDS]

    # One row per dataset (for ds_meta)
    dataset_rows = df.drop_duplicates(subset="dataset_name", keep="first")
    datasets = dataset_rows["dataset_name"].tolist()
    if args.max_datasets > 0:
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(datasets))[: args.max_datasets]
        datasets = [datasets[i] for i in idx]
    print(f"Evaluating on {len(datasets)} datasets...")

    required_cols = list(DATASET_FEATURES) + ["task_type", "dataset_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns {missing}; evaluation may fail or use NaN.")

    results = []
    for i, dname in enumerate(datasets):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i+1}/{len(datasets)} {dname[:50]}...")
        grp = df[df["dataset_name"] == dname]
        if grp.empty:
            continue
        row0 = grp.iloc[0]
        task_type = str(row0.get("task_type", "classification")).strip().lower()
        if task_type not in ("classification", "regression"):
            task_type = "classification"

        if ensembles:
            ensemble_cur = ensembles.get(task_type)
            if ensemble_cur is None:
                continue
            preparator_cur = ensemble_cur.preparator
        else:
            ensemble_cur = ensemble
            preparator_cur = preparator

        meta_row = row0.to_dict()
        try:
            tt_enc = preparator_cur.task_type_encoder.transform([task_type])[0]
        except Exception:
            tt_enc = 0
        meta_row["task_type_encoded"] = tt_enc
        ref = preparator_cur.encode_refinement_dimensions(task_type, meta_row)
        for k, v in ref.items():
            meta_row[k] = v
        pred_df = pd.DataFrame([meta_row])
        pred_cols = [c for c in preparator_cur.feature_columns_predictor if c in pred_df.columns]
        if len(pred_cols) < len(preparator_cur.feature_columns_predictor):
            pred_df = pred_df.reindex(columns=preparator_cur.feature_columns_predictor)
            pred_df = pred_df.fillna(preparator_cur.fill_values)

        try:
            out = ensemble_cur.predict(pred_df, task_type=task_type, n_perturbations=20, seed=args.seed)
        except Exception as e:
            print(f"    Predict failed for {dname}: {e}")
            continue
        rec_config = out["best_config"]

        rec_vec = _config_to_hp_vector(rec_config, hp_keys)
        if rec_vec is None:
            continue

        primary_scores = grp["primary_score"].values
        best_score = float(np.nanmax(primary_scores))
        worst_score = float(np.nanmin(primary_scores))
        default_row = grp[grp.get("hp_is_default", pd.Series(0)) == 1]
        if default_row.empty:
            default_row = grp[grp.get("hp_config_name", pd.Series("")) == "default"]
        default_score = float(default_row["primary_score"].iloc[0]) if len(default_row) > 0 else primary_scores[0]

        min_dist = np.inf
        chosen_primary = None
        chosen_idx = None
        for idx, (_, r) in enumerate(grp.iterrows()):
            rvec = _row_to_hp_vector(r, hp_keys)
            if rvec is None:
                continue
            d = _distance(rec_vec, rvec)
            if d < min_dist:
                min_dist = d
                chosen_primary = float(r["primary_score"])
                chosen_idx = idx
        if chosen_primary is None:
            continue
        # rank: 1 = best (descending primary_score)
        rank_series = grp["primary_score"].rank(ascending=False, method="min").astype(int)
        chosen_rank = int(rank_series.iloc[chosen_idx]) if chosen_idx is not None else np.nan
        score_range = best_score - worst_score
        regret = (best_score - chosen_primary) / score_range if score_range and not np.isnan(score_range) else 0
        results.append({
            "dataset": dname,
            "task_type": task_type,
            "recommended_score": chosen_primary,
            "default_score": default_score,
            "best_score": best_score,
            "beat_default": chosen_primary >= default_score,
            "rank": chosen_rank,
            "regret": regret,
        })

    if not results:
        print("No valid results.")
        return

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("HP META-MODEL OFFLINE EVALUATION")
    print("=" * 60)
    print(f"  Datasets evaluated: {len(res_df)}")
    print(f"  Mean primary_score (recommended): {res_df['recommended_score'].mean():.6f}")
    print(f"  Mean primary_score (default):     {res_df['default_score'].mean():.6f}")
    print(f"  Mean primary_score (best):        {res_df['best_score'].mean():.6f}")
    print(f"  Mean improvement vs default:      {(res_df['recommended_score'] - res_df['default_score']).mean():.6f}")
    print(f"  % datasets where recommended >= default: {100 * res_df['beat_default'].mean():1f}%")
    print(f"  Median rank of chosen config (1=best):   {res_df['rank'].median():.0f}")
    print(f"  Mean normalized regret (0=best):        {res_df['regret'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
