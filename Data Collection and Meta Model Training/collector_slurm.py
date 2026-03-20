"""
collector_slurm.py — Unified SLURM Worker for All Collectors
=============================================================

Each SLURM array worker:
  1. Reads task_list.json, filters to classification only
  2. Finds its assigned task by SLURM_ARRAY_TASK_ID
  3. Runs the specified collector (numerical, categorical, interaction, or row)
  4. Writes to per-worker CSV and checkpoint files (no clashes)

Usage:
  python collector_slurm.py --slurm_id 42 --collector numerical
  python collector_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID --collector categorical
  python collector_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID --collector interaction
  python collector_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID --collector row

Prerequisites:
  1. task_list.json must exist (from generate_task_list.py)
  2. collector_utils.py + collect_*.py must be importable

Merge all worker outputs afterwards:
  python merge_collector_results.py --output_dir ./output_row --collector row
"""
import argparse
import json
import os
import sys
import time
import traceback
import gc

import pandas as pd
import numpy as np
import openml

from sklearn.preprocessing import LabelEncoder

# Import collectors
from collect_numerical_transforms import (
    collect_numerical_transforms, SCHEMA_NUMERICAL
)
from collect_categorical_transforms import (
    collect_categorical_transforms, SCHEMA_CATEGORICAL
)
from collect_interaction_features import (
    collect_interaction_features, SCHEMA_INTERACTION
)
from collect_row_features import (
    collect_row_features, SCHEMA_ROW
)
from collector_utils import write_csv, sanitize_and_deduplicate_columns, densify_sparse_columns, filter_rare_classes


# =============================================================================
# SCHEMAS PER COLLECTOR
# =============================================================================
COLLECTOR_CONFIG = {
    'numerical': {
        'fn': collect_numerical_transforms,
        'schema': SCHEMA_NUMERICAL,
        'csv_prefix': 'numerical_transforms',
    },
    'categorical': {
        'fn': collect_categorical_transforms,
        'schema': SCHEMA_CATEGORICAL,
        'csv_prefix': 'categorical_transforms',
    },
    'interaction': {
        'fn': collect_interaction_features,
        'schema': SCHEMA_INTERACTION,
        'csv_prefix': 'interaction_features',
    },
    'row': {
        'fn': collect_row_features,
        'schema': SCHEMA_ROW,
        'csv_prefix': 'row_features',
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def write_checkpoint(filepath, task_id, dataset_id, worker_id, status, detail=''):
    with open(filepath, 'w') as f:
        json.dump({
            'task_id': task_id,
            'dataset_id': dataset_id,
            'worker_id': worker_id,
            'status': status,
            'detail': detail,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)


# =============================================================================
# MAIN WORKER
# =============================================================================

def run_single_task(task_entry, collector_name, output_dir,
                    n_folds=5, n_repeats=3, time_budget=7200, worker_id=None):
    """
    Process a single OpenML classification task with the specified collector.
    """
    config = COLLECTOR_CONFIG[collector_name]
    collect_fn = config['fn']
    schema = config['schema']
    csv_prefix = config['csv_prefix']

    task_id = task_entry['task_id']
    dataset_id = task_entry.get('dataset_id')

    # Per-worker output files
    worker_tag = f"worker_{worker_id:05d}" if worker_id is not None else f"task_{task_id}"
    csv_file = os.path.join(output_dir, 'worker_csvs', f'{csv_prefix}_{worker_tag}.csv')
    checkpoint_file = os.path.join(output_dir, 'worker_checkpoints', f'checkpoint_{worker_tag}.json')

    os.makedirs(os.path.join(output_dir, 'worker_csvs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'worker_checkpoints'), exist_ok=True)

    # Check if already done
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        if ckpt.get('status') == 'done':
            print(f"Worker {worker_id}: Task {task_id} already completed. Skipping.")
            return True

    print(f"{'='*80}")
    print(f"Worker {worker_id} [{collector_name}]: task_id={task_id}, dataset_id={dataset_id}")
    print(f"{'='*80}")

    # Mark in-progress
    write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'in_progress')

    try:
        # --- Load data ---
        dataset = None
        target_col = None

        if task_id and task_id > 0:
            print(f"Loading OpenML task {task_id}...")
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            target_col = dataset.default_target_attribute
        elif dataset_id:
            print(f"Loading OpenML dataset {dataset_id}...")
            dataset = openml.datasets.get_dataset(dataset_id)
            target_col = dataset.default_target_attribute
        else:
            raise ValueError(f"No valid task_id or dataset_id: {task_entry}")

        if not target_col:
            print(f"No target attribute — skipping")
            write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped', 'no_target')
            return False

        X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
        X = sanitize_and_deduplicate_columns(X)
        X = densify_sparse_columns(X)
        X, y = filter_rare_classes(X, y, n_folds=n_folds, n_repeats=n_repeats)

        print(f"Dataset: {dataset.name} | Shape: {X.shape}")

        # --- Safety filters ---
        if X.shape[0] < 500:
            print(f"  Too few rows ({X.shape[0]}) — skipping")
            write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped',
                           f'too_few_rows_after_rare_class_filter')
            return False

        if X.shape[1] < 2:
            print(f"  Too few columns ({X.shape[1]}) — skipping")
            write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped',
                           f'too_few_cols_{X.shape[1]}')
            return False

        # --- Size reduction for very large datasets ---
        raw_cells = X.shape[0] * X.shape[1]
        if raw_cells > 100_000_000:
            print(f"  Large dataset ({raw_cells:,} cells) — sampling rows...")
            max_rows = 100_000_000 // max(X.shape[1], 1)
            if X.shape[0] > max_rows:
                X = X.sample(n=max_rows, random_state=42)
                y = y.loc[X.index]
                print(f"  Sampled to {X.shape}")

        # --- Encode target ---
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

        # --- Run collector ---
        t0 = time.time()

        df_result = collect_fn(
            X, y,
            dataset_name=dataset.name,
            n_folds=n_folds,
            n_repeats=n_repeats,
            time_budget=time_budget,
        )

        elapsed = time.time() - t0

        # --- Save ---
        if df_result.empty:
            print(f"Empty result after {elapsed:.1f}s — skipping")
            write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped', 'empty_result')
            return False

        df_result['openml_task_id'] = task_id if (task_id and task_id > 0) else f"ds_{dataset_id}"
        df_result['dataset_name'] = dataset.name
        df_result['dataset_id'] = dataset.dataset_id

        write_csv(df_result, csv_file, schema)
        print(f"Done: {len(df_result)} rows in {elapsed:.1f}s -> {csv_file}")

        write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'done',
                        f'{len(df_result)} rows in {elapsed:.1f}s')

        del df_result, X, y
        gc.collect()
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'failed', str(e)[:200])
        return False


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SLURM worker for meta-learning data collection')
    parser.add_argument('--slurm_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID — index into classification tasks')
    parser.add_argument('--collector', type=str, required=True,
                        choices=['numerical', 'categorical', 'interaction', 'row'],
                        help='Which collector to run')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Base output directory')
    parser.add_argument('--task_list', type=str,
                        default=os.environ.get('TASK_LIST', './task_list.json'),
                        help='Path to task_list.json (env: TASK_LIST)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--time_budget', type=int, default=7200,
                        help='Max seconds per dataset (default: 7200 = 2h)')
    args = parser.parse_args()

    # Set up OpenML cache now (not at import time)
    cache_dir = os.environ.get('OPENML_CACHE_DIR', os.path.join(os.getcwd(), '.openml_cache'))
    os.makedirs(cache_dir, exist_ok=True)
    openml.config.cache_directory = cache_dir
    print(f"OpenML cache: {openml.config.cache_directory}")

    # --- Load task list & filter to classification ---
    if not os.path.exists(args.task_list):
        print(f"ERROR: Task list not found: {args.task_list}")
        sys.exit(1)

    with open(args.task_list, 'r') as f:
        task_data = json.load(f)

    # Handle both formats: list of dicts or {'tasks': [...]}
    if isinstance(task_data, dict):
        all_tasks = task_data.get('tasks', [])
    else:
        all_tasks = task_data

    # Filter to classification only
    classification_tasks = [
        t for t in all_tasks
        if t.get('task_type', '').lower() == 'classification'
    ]

    n_tasks = len(classification_tasks)
    print(f"Task list: {len(all_tasks)} total, {n_tasks} classification")

    if args.slurm_id >= n_tasks:
        print(f"Worker {args.slurm_id}: index >= {n_tasks} classification tasks. Nothing to do.")
        sys.exit(0)

    task_entry = classification_tasks[args.slurm_id]

    print(f"Worker {args.slurm_id}/{n_tasks} [{args.collector}]: "
          f"task_id={task_entry['task_id']}, dataset={task_entry.get('dataset_name', '?')}")

    success = run_single_task(
        task_entry=task_entry,
        collector_name=args.collector,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        time_budget=args.time_budget,
        worker_id=args.slurm_id,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()