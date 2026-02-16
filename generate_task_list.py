"""
Generate a shared task list from OpenML for SLURM array jobs.

Run this ONCE before submitting the SLURM array:
    python generate_task_list.py --output_dir ./meta_learning_output --max_tasks 5000

This creates:
    ./meta_learning_output/task_list.json
    
Each SLURM worker reads this file and picks its task by array index.
"""
import argparse
import json
import os
import openml
import pandas as pd
import numpy as np


def _infer_task_type(y):
    """Robustly infer whether a target variable is classification or regression."""
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    y_clean = y.dropna()
    if len(y_clean) == 0:
        return 'classification'
    nunique = int(y_clean.nunique())
    n = len(y_clean)
    if nunique <= 2:
        return 'classification'
    try:
        is_integer_like = (y_clean % 1 == 0).all()
    except (TypeError, ValueError):
        is_integer_like = False
    if not is_integer_like and nunique > 20:
        return 'regression'
    unique_ratio = nunique / n
    if unique_ratio < 0.05:
        return 'classification'
    if nunique <= 200 and unique_ratio < 0.10:
        return 'classification'
    return 'regression'


def _detect_openml_task_type(task_row):
    """Detect task type from a task list dataframe row."""
    CLASSIFICATION_SIGNALS = {'1', 'supervised classification', 'classification'}
    REGRESSION_SIGNALS = {'2', 'supervised regression', 'regression'}

    for col_name in ('task_type_id', 'task_type'):
        if col_name in task_row.index:
            val = task_row[col_name]
            if hasattr(val, 'value'):
                val = val.value
            val_str = str(val).strip().lower()
            if val_str in CLASSIFICATION_SIGNALS:
                return 'classification'
            if val_str in REGRESSION_SIGNALS:
                return 'regression'
    return None


def generate_task_list(output_dir, max_tasks=20000, min_rows=500, max_rows=5000000,
                       min_cols=4, max_cols=6000, max_cells=100_000_000):
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching ALL supervised tasks from OpenML...")

    # Fetch classification and regression tasks
    all_tasks = []
    for task_type_id in [1, 2]:  # 1=classification, 2=regression
        try:
            tasks_df = openml.tasks.list_tasks(
                task_type=openml.tasks.TaskType(task_type_id),
                output_format='dataframe'
            )
            if 'status' in tasks_df.columns:
                before_status = len(tasks_df)
                tasks_df = tasks_df[tasks_df['status'] == 'active']
                print(f"  Task type {task_type_id}: {len(tasks_df)} active tasks (dropped {before_status - len(tasks_df)} inactive)")
            all_tasks.append(tasks_df)
        except Exception as e:
            print(f"  Failed to list task type {task_type_id}: {e}")
            # Fallback: try integer
            try:
                tasks_df = openml.tasks.list_tasks(
                    task_type=task_type_id,
                    output_format='dataframe'
                )
                print(f"  Task type {task_type_id} (fallback): {len(tasks_df)} tasks")
                all_tasks.append(tasks_df)
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")

    if not all_tasks:
        print("ERROR: Could not fetch any tasks from OpenML!")
        return

    tasks_df = pd.concat(all_tasks, ignore_index=False)

    # Deduplicate: keep one task per dataset (prefer 10-fold CV, most runs)
    print(f"Total raw tasks: {len(tasks_df)}")

    # Standardize task_type_id if needed
    if 'task_type_id' not in tasks_df.columns and 'task_type' in tasks_df.columns:
        type_map = {
            'Supervised Classification': 1,
            'Supervised Regression': 2,
        }
        tasks_df['task_type_id'] = tasks_df['task_type'].map(type_map).fillna(0).astype(int)

    # Filter to supervised only
    tasks_df = tasks_df[tasks_df['task_type_id'].isin([1, 2])]
    print(f"After filtering to supervised: {len(tasks_df)}")

    # Apply size filters using available columns
    size_cols = {
        'NumberOfInstances': (min_rows, max_rows),
        'NumberOfFeatures': (min_cols, max_cols),
    }
    for col, (lo, hi) in size_cols.items():
        if col in tasks_df.columns:
            before = len(tasks_df)
            tasks_df = tasks_df[
                (tasks_df[col] >= lo) & (tasks_df[col] <= hi)
            ]
            print(f"After {col} filter [{lo}, {hi}]: {len(tasks_df)} (dropped {before - len(tasks_df)})")

    # Filter by cell count if both columns available
    if 'NumberOfInstances' in tasks_df.columns and 'NumberOfFeatures' in tasks_df.columns:
        before = len(tasks_df)
        tasks_df = tasks_df[
            tasks_df['NumberOfInstances'] * tasks_df['NumberOfFeatures'] <= max_cells * 2
            # Use 2x buffer since _smart_size_reduction can handle some oversize
        ]
        print(f"After cell count filter: {len(tasks_df)} (dropped {before - len(tasks_df)})")

    # Deduplicate: one task per dataset_id (keep the best task variant)
    if 'did' in tasks_df.columns:
        did_col = 'did'
    elif 'data_id' in tasks_df.columns:
        did_col = 'data_id'
    else:
        did_col = None
        print("WARNING: No dataset_id column found — cannot deduplicate by dataset!")
        print("         This may cause the same dataset to be processed multiple times.")

    if did_col:
        before = len(tasks_df)

        # Build a priority score for sorting (higher = keep)
        # 1) Prefer regression (task_type_id=2) for datasets with many unique target values,
        #    classification (task_type_id=1) otherwise — approximated by preferring
        #    classification for low NumberOfClasses, regression for high
        # 2) Prefer 10-fold CV estimation procedure
        # 3) Prefer tasks with more runs (better community-vetted)
        sort_cols = []
        sort_asc = []

        if 'estimation_procedure' in tasks_df.columns:
            tasks_df['_is_10fold'] = tasks_df['estimation_procedure'].str.contains(
                '10-fold', case=False, na=False).astype(int)
            sort_cols.append('_is_10fold')
            sort_asc.append(False)

        if 'NumberOfRuns' in tasks_df.columns:
            sort_cols.append('NumberOfRuns')
            sort_asc.append(False)

        if sort_cols:
            tasks_df = tasks_df.sort_values(sort_cols, ascending=sort_asc)

        # For datasets that appear as BOTH classification AND regression tasks,
        # keep the task type that better matches the data characteristics
        dupes = tasks_df[tasks_df.duplicated(subset=[did_col], keep=False)]
        if len(dupes) > 0:
            n_duped_datasets = dupes[did_col].nunique()
            both_types = dupes.groupby(did_col)['task_type_id'].nunique()
            n_both = (both_types > 1).sum()
            print(f"  {n_duped_datasets} datasets have multiple tasks, "
                  f"{n_both} appear as both classification AND regression")

        tasks_df = tasks_df.drop_duplicates(subset=[did_col], keep='first')
        if '_is_10fold' in tasks_df.columns:
            tasks_df = tasks_df.drop(columns=['_is_10fold'])
        print(f"After dedup by {did_col}: {len(tasks_df)} (dropped {before - len(tasks_df)})")

    # Cap at max_tasks
    if len(tasks_df) > max_tasks:
        # Shuffle deterministically so we get a diverse sample
        tasks_df = tasks_df.sample(n=max_tasks, random_state=42)
        print(f"Capped at {max_tasks} tasks")

    # Build task list
    task_list = []
    for idx, row in tasks_df.iterrows():
        tid = int(row['tid']) if 'tid' in row.index else int(idx)
        task_type = _detect_openml_task_type(row)
        dataset_id = int(row[did_col]) if did_col and did_col in row.index else None
        n_rows = int(row['NumberOfInstances']) if 'NumberOfInstances' in row.index else None
        n_cols = int(row['NumberOfFeatures']) if 'NumberOfFeatures' in row.index else None
        name = str(row.get('name', row.get('source_data', f'dataset_{dataset_id}')))

        task_list.append({
            'task_id': tid,
            'dataset_id': dataset_id,
            'task_type': task_type,
            'dataset_name': name,
            'n_rows': n_rows,
            'n_cols': n_cols,
        })

    # Sort by task_id for deterministic ordering
    task_list.sort(key=lambda x: x['task_id'])

    output_file = os.path.join(output_dir, 'task_list.json')
    with open(output_file, 'w') as f:
        json.dump({
            'n_tasks': len(task_list),
            'generation_params': {
                'max_tasks': max_tasks,
                'min_rows': min_rows,
                'max_rows': max_rows,
                'min_cols': min_cols,
                'max_cols': max_cols,
                'max_cells': max_cells,
            },
            'tasks': task_list,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task list saved to {output_file}")
    print(f"Total tasks: {len(task_list)}")
    n_clf = sum(1 for t in task_list if t['task_type'] == 'classification')
    n_reg = sum(1 for t in task_list if t['task_type'] == 'regression')
    n_unk = sum(1 for t in task_list if t['task_type'] is None)
    print(f"  Classification: {n_clf} | Regression: {n_reg} | Unknown: {n_unk}")
    print(f"\nSubmit SLURM array with: --array=0-{len(task_list)-1}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./meta_learning_output')
    parser.add_argument('--max_tasks', type=int, default=20000)
    parser.add_argument('--min_rows', type=int, default=500)
    parser.add_argument('--max_rows', type=int, default=5000000)
    parser.add_argument('--min_cols', type=int, default=4)
    parser.add_argument('--max_cols', type=int, default=6000)
    args = parser.parse_args()
    generate_task_list(args.output_dir, args.max_tasks, args.min_rows, args.max_rows,
                       args.min_cols, args.max_cols)