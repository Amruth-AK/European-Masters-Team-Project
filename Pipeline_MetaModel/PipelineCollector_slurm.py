"""
SLURM Worker for Pipeline Meta-Learning Data Collection

Each SLURM array worker:
  1. Reads task_list.json to find its assigned task (by SLURM_ARRAY_TASK_ID)
  2. Processes that single dataset with PipelineDataCollector
  3. Writes to per-worker CSV and checkpoint files (no clashes)

Usage:
  python PipelineCollector_slurm.py --slurm_id 42
  python PipelineCollector_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID

Prerequisites:
  1. Run generate_task_list.py first to create task_list.json
  2. DataCollector_3.py and PipelineDataCollector.py must be importable

Merge all worker outputs afterwards:
  python merge_pipeline_results.py --output_dir ./pipeline_meta_output
  (NOTE: merge_results.py is for DataCollector_3 only -- use the pipeline-specific merger.)
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import gc

import pandas as pd
import numpy as np
import openml

# Force OpenML to use /work immediately upon loading
WORK_CACHE_DIR = '/work/inestp05/lightgbm_project/openml_cache'
os.makedirs(WORK_CACHE_DIR, exist_ok=True)
openml.config.cache_directory = WORK_CACHE_DIR
print(f"GLOBAL: OpenML cache directory set to: {openml.config.cache_directory}")

from sklearn.preprocessing import LabelEncoder

# Define the project root (the '000' folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from FE_PP_MetaModel.DataCollector_3 import (
        MetaDataCollector,
        _detect_openml_task_type,
        _infer_task_type,
        _smart_size_reduction,
    )
except ImportError as e:
    print(f"Import failed. Current sys.path: {sys.path}")
    raise e

    
from PipelineDataCollector import PipelineDataCollector


def _sanitize_feature_names(df):
    """Clean column names to avoid LightGBM issues."""
    df.columns = [re.sub(r'[\[\]\{\}\":,]', '_', str(col)) for col in df.columns]
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for d in cols[cols.duplicated()].unique():
            mask = cols == d
            cols[mask] = [f"{d}_{i}" if i != 0 else d for i in range(mask.sum())]
        df.columns = cols
    return df


def _write_checkpoint(filepath, task_id, dataset_id, worker_id, status, detail=''):
    with open(filepath, 'w') as f:
        json.dump({
            'task_id': task_id,
            'dataset_id': dataset_id,
            'worker_id': worker_id,
            'status': status,
            'detail': detail,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)


def run_single_task(task_entry, output_dir, n_folds=5, n_repeats=3,
                    time_budget=7200, worker_id=None):
    """
    Process a single OpenML dataset with PipelineDataCollector.
    Writes results to a worker-specific CSV (no file clashes).
    """
    task_id = task_entry['task_id']
    dataset_id = task_entry.get('dataset_id')
    forced_type = task_entry.get('task_type')

    # Per-worker output files
    worker_tag = f"worker_{worker_id:05d}" if worker_id is not None else f"task_{task_id}"
    csv_file = os.path.join(output_dir, 'worker_csvs', f'pipeline_db_{worker_tag}.csv')
    checkpoint_file = os.path.join(output_dir, 'worker_checkpoints', f'checkpoint_{worker_tag}.json')
    worker_log_dir = os.path.join(output_dir, 'worker_logs')

    os.makedirs(os.path.join(output_dir, 'worker_csvs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'worker_checkpoints'), exist_ok=True)
    os.makedirs(worker_log_dir, exist_ok=True)

    # Check if already done
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        if ckpt.get('status') == 'done':
            print(f"Worker {worker_id}: Task {task_id} already completed. Skipping.")
            return True

    print(f"{'=' * 70}")
    print(f"Worker {worker_id}: Processing task_id={task_id}, dataset_id={dataset_id}")
    print(f"{'=' * 70}")

    # Write "in_progress" checkpoint
    _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'in_progress')

    try:
        # --- Load data ---
        dataset = None
        target_col = None
        task_type = forced_type

        if task_id and task_id > 0:
            print(f"Loading OpenML task {task_id}...")
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            target_col = dataset.default_target_attribute
            if task_type is None:
                task_type = _detect_openml_task_type(task)
        elif dataset_id:
            print(f"Loading OpenML dataset {dataset_id} directly...")
            dataset = openml.datasets.get_dataset(dataset_id)
            target_col = dataset.default_target_attribute
        else:
            raise ValueError(f"No valid task_id or dataset_id in entry: {task_entry}")

        if not target_col:
            print("WARNING: No default target attribute. Skipping.")
            _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped', 'no_target')
            return False

        print(f"Loading data... (Target: {target_col})")
        X, y, _, _ = dataset.get_data(target=target_col, dataset_format='dataframe')
        X = _sanitize_feature_names(X)

        if task_type is None:
            task_type = _infer_task_type(y)

        print(f"Dataset: {dataset.name} | Shape: {X.shape} | Type: {task_type}")

        # --- Runtime size filter ---
        if X.shape[0] < 500:
            print(f"  Skipping: too few rows ({X.shape[0]} < 500)")
            _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped',
                              f'too_few_rows_{X.shape[0]}')
            return False
        if X.shape[1] < 3:
            print(f"  Skipping: too few columns ({X.shape[1]} < 3)")
            _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'skipped',
                              f'too_few_cols_{X.shape[1]}')
            return False

        # --- Smart size reduction ---
        raw_cells = X.shape[0] * X.shape[1]
        if raw_cells > 100_000_000:
            print(f"  Large dataset ({X.shape}, {raw_cells:,} cells) -- reducing...")
            X, y, msg = _smart_size_reduction(X, y, task_type, cell_limit=100_000_000)
            print(f"  -> {msg}")
            if X.shape[1] == 0:
                _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id,
                                  'skipped', 'no_columns_after_pruning')
                return False

        # --- Encode target for classification ---
        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)

        # --- Run pipeline collector ---
        dataset_start = time.time()
        slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

        collector = PipelineDataCollector(
            task_type=task_type, n_folds=n_folds, n_repeats=n_repeats,
            output_dir=worker_log_dir, time_budget_seconds=time_budget,
            n_jobs=slurm_cpus)

        # Worker-specific log file
        collector.log_file = os.path.join(worker_log_dir, f'pipeline_log_{worker_tag}.txt')

        full_df = pd.concat([X, y], axis=1)
        df_result = collector.collect(full_df, y.name, dataset_name=dataset.name)
        elapsed = time.time() - dataset_start

        # --- Save results ---
        if df_result.empty:
            print(f"Skipped (empty result) after {elapsed:.1f}s")
            _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id,
                              'skipped', 'empty_result')
            return False

        df_result['openml_task_id'] = task_id if task_id and task_id > 0 else f"ds_{dataset_id}"
        df_result['dataset_name'] = dataset.name
        df_result['dataset_id'] = dataset.dataset_id if dataset_id else dataset.dataset_id
        df_result['task_type'] = task_type

        PipelineDataCollector.write_csv(df_result, csv_file)
        print(f"Done: {len(df_result)} pipelines in {elapsed:.1f}s -> {csv_file}")

        _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'done',
                          f'{len(df_result)} pipelines in {elapsed:.1f}s')

        del collector, full_df, df_result, X, y
        gc.collect()
        return True

    except Exception as e:
        print(f"FAILED: {str(e)}")
        traceback.print_exc()
        _write_checkpoint(checkpoint_file, task_id, dataset_id, worker_id, 'failed', str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description='SLURM worker for pipeline meta-learning collection')
    parser.add_argument('--slurm_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID - index into task_list.json')
    parser.add_argument('--output_dir', type=str, default='./pipeline_meta_output',
                        help='Base output directory')
    parser.add_argument('--task_list', type=str, default=None,
                        help='Path to task_list.json (default: <output_dir>/task_list.json)')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--time_budget', type=int, default=7200,
                        help='Max seconds per dataset (default: 7200 = 2h)')
    args = parser.parse_args()

    # Load task list
    task_list_path = args.task_list or os.path.join(args.output_dir, 'task_list.json')
    if not os.path.exists(task_list_path):
        print(f"ERROR: Task list not found at {task_list_path}")
        print("Run generate_task_list.py first!")
        sys.exit(1)

    with open(task_list_path, 'r') as f:
        task_data = json.load(f)

    tasks = task_data['tasks']
    n_tasks = len(tasks)

    if args.slurm_id >= n_tasks:
        print(f"Worker {args.slurm_id}: Index {args.slurm_id} >= {n_tasks} tasks. Nothing to do.")
        sys.exit(0)

    task_entry = tasks[args.slurm_id]

    print(f"Worker {args.slurm_id}/{n_tasks}: task_id={task_entry['task_id']}, "
          f"dataset={task_entry.get('dataset_name', '?')}")

    success = run_single_task(
        task_entry=task_entry,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        time_budget=args.time_budget,
        worker_id=args.slurm_id,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()