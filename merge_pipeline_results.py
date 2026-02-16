"""
Merge all per-worker Pipeline CSV files into a single pipeline_meta_learning_db.csv

Usage:
  python merge_pipeline_results.py --output_dir ./pipeline_meta_output

Also prints a summary of worker statuses from checkpoint files and
pipeline-level statistics (improvement rates, archetype breakdown, etc.).

NOTE: This is the merger for PipelineDataCollector / PipelineCollector_slurm.
      For the individual-transform DataCollector_3 results, use merge_results.py.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Add current directory to path to find PipelineDataCollector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PipelineDataCollector import PIPELINE_CSV_SCHEMA


def _summarize_checkpoints(checkpoint_dir):
    """Print checkpoint status summary and return list of failed tasks."""
    print("=" * 70)
    print("WORKER STATUS SUMMARY")
    print("=" * 70)

    status_counts = {'done': 0, 'failed': 0, 'skipped': 0, 'in_progress': 0}
    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint_worker_*.json')))

    failed_tasks = []
    in_progress_tasks = []

    for cp_file in checkpoint_files:
        try:
            with open(cp_file, 'r') as f:
                ckpt = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read {cp_file}: {e}")
            continue

        status = ckpt.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == 'failed':
            failed_tasks.append((
                ckpt.get('worker_id'),
                ckpt.get('task_id'),
                ckpt.get('detail', '')[:100]))
        elif status == 'in_progress':
            in_progress_tasks.append((
                ckpt.get('worker_id'),
                ckpt.get('task_id'),
                ckpt.get('timestamp', '?')))

    for status, count in sorted(status_counts.items()):
        print(f"  {status:15s}: {count}")
    print(f"  {'Total':15s}: {len(checkpoint_files)}")

    if in_progress_tasks:
        print(f"\nStill in progress ({len(in_progress_tasks)}):")
        for wid, tid, ts in in_progress_tasks[:10]:
            print(f"  Worker {wid}, task {tid} (started {ts})")
        if len(in_progress_tasks) > 10:
            print(f"  ... and {len(in_progress_tasks) - 10} more")
        print("  (These may be stale if the jobs crashed without writing a final checkpoint.)")

    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for wid, tid, detail in failed_tasks[:20]:
            print(f"  Worker {wid}, task {tid}: {detail}")
        if len(failed_tasks) > 20:
            print(f"  ... and {len(failed_tasks) - 20} more")

    return failed_tasks


def _print_pipeline_stats(merged):
    """Print detailed pipeline-level statistics."""
    print(f"\n{'=' * 70}")
    print("PIPELINE STATISTICS")
    print("=" * 70)

    n_total = len(merged)
    n_datasets = merged['dataset_name'].nunique()
    print(f"  Total pipeline rows : {n_total}")
    print(f"  Unique datasets     : {n_datasets}")
    print(f"  Avg pipelines/dataset: {n_total / max(n_datasets, 1):.1f}")

    # Task type breakdown
    print(f"\n  Task types:")
    for tt, count in merged['task_type'].value_counts().items():
        ds_count = merged.loc[merged['task_type'] == tt, 'dataset_name'].nunique()
        print(f"    {tt:20s}: {count:5d} pipelines across {ds_count} datasets")

    # Improvement rates
    try:
        improved_col = merged['pipeline_improved'].astype(int)
        n_improved = improved_col.sum()
        print(f"\n  Pipelines improved (significant): {n_improved}/{n_total} "
              f"({100 * n_improved / max(n_total, 1):.1f}%)")
    except Exception:
        pass

    # Significance stats
    try:
        sig_col = merged['pipeline_is_significant']
        # Handle mixed bool/string from CSV round-tripping
        n_sig = ((sig_col == True) | (sig_col == 'True')).sum()
        print(f"  Statistically significant (p<0.05): {n_sig}/{n_total} "
              f"({100 * n_sig / max(n_total, 1):.1f}%)")

        p_values = pd.to_numeric(merged['pipeline_p_value'], errors='coerce')
        valid_p = p_values.dropna()
        if len(valid_p) > 0:
            print(f"  p-value median: {valid_p.median():.4f}, "
                  f"mean: {valid_p.mean():.4f}")
    except Exception:
        pass

    # Delta stats
    try:
        deltas = pd.to_numeric(merged['pipeline_delta'], errors='coerce').dropna()
        if len(deltas) > 0:
            print(f"\n  Delta (score change):")
            print(f"    mean:   {deltas.mean():+.5f}")
            print(f"    median: {deltas.median():+.5f}")
            print(f"    std:    {deltas.std():.5f}")
            print(f"    min:    {deltas.min():+.5f}")
            print(f"    max:    {deltas.max():+.5f}")
    except Exception:
        pass

    # Archetype breakdown
    try:
        print(f"\n  Archetype breakdown:")
        arch_stats = merged.groupby('pipeline_archetype').agg(
            count=('pipeline_improved', 'size'),
            improved=('pipeline_improved', lambda x: x.astype(int).sum()),
            avg_delta=('pipeline_delta', lambda x: pd.to_numeric(x, errors='coerce').mean()),
        ).sort_values('count', ascending=False)

        for arch, row in arch_stats.iterrows():
            rate = 100 * row['improved'] / max(row['count'], 1)
            print(f"    {arch:22s}: {int(row['count']):4d} pipelines, "
                  f"{int(row['improved']):4d} improved ({rate:5.1f}%), "
                  f"avg delta {row['avg_delta']:+.5f}")
    except Exception:
        pass

    # Dataset-level summary: how many datasets had at least one improving pipeline?
    try:
        ds_any_improved = merged.groupby('dataset_name')['pipeline_improved'].apply(
            lambda x: x.astype(int).max()).sum()
        print(f"\n  Datasets with >= 1 significant improvement: "
              f"{int(ds_any_improved)}/{n_datasets} "
              f"({100 * ds_any_improved / max(n_datasets, 1):.1f}%)")
    except Exception:
        pass


def merge_pipeline_results(output_dir):
    """Main merge function: find worker CSVs, merge, validate, write output."""
    worker_csv_dir = os.path.join(output_dir, 'worker_csvs')
    checkpoint_dir = os.path.join(output_dir, 'worker_checkpoints')
    merged_csv = os.path.join(output_dir, 'pipeline_meta_learning_db.csv')

    # --- Summarize checkpoints ---
    _summarize_checkpoints(checkpoint_dir)

    # --- Discover worker CSVs ---
    print(f"\n{'=' * 70}")
    print("MERGING WORKER CSVs")
    print("=" * 70)

    csv_files = sorted(
        glob.glob(os.path.join(worker_csv_dir, 'pipeline_db_worker_*.csv')))
    print(f"Found {len(csv_files)} worker CSV files in {worker_csv_dir}")

    if not csv_files:
        print("No CSV files to merge!")
        return

    # --- Read and validate each CSV ---
    dfs = []
    skipped = 0
    total_rows = 0
    schema_warnings = set()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                skipped += 1
                continue

            # Check for unexpected columns not in schema
            extra_cols = set(df.columns) - set(PIPELINE_CSV_SCHEMA)
            if extra_cols:
                schema_warnings.add(
                    f"Extra columns in {os.path.basename(csv_file)}: "
                    f"{', '.join(sorted(extra_cols))}")

            total_rows += len(df)
            dfs.append(df)

        except Exception as e:
            print(f"  WARNING: Failed to read {os.path.basename(csv_file)}: {e}")
            skipped += 1

    if schema_warnings:
        print(f"\n  Schema warnings ({len(schema_warnings)}):")
        for w in sorted(schema_warnings):
            print(f"    {w}")

    if not dfs:
        print("No valid data found in any CSV!")
        return

    print(f"  Read {total_rows} rows from {len(dfs)} files ({skipped} empty/failed)")

    # --- Merge ---
    merged = pd.concat(dfs, ignore_index=True)

    # Ensure schema alignment: add missing columns, reorder to canonical schema
    missing_cols = [c for c in PIPELINE_CSV_SCHEMA if c not in merged.columns]
    if missing_cols:
        print(f"  Adding {len(missing_cols)} missing schema columns: "
              f"{', '.join(missing_cols)}")
        for col in missing_cols:
            merged[col] = np.nan

    # Drop any extra columns not in schema, reorder to canonical order
    merged = merged[[c for c in PIPELINE_CSV_SCHEMA if c in merged.columns]]

    # --- Deduplicate: same (dataset_name, pipeline_archetype, pipeline_transforms_json) ---
    pre_dedup = len(merged)
    dedup_cols = ['dataset_name', 'pipeline_archetype', 'pipeline_transforms_json']
    if all(c in merged.columns for c in dedup_cols):
        merged = merged.drop_duplicates(subset=dedup_cols, keep='first')
    n_dupes = pre_dedup - len(merged)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate rows")

    # --- Write merged CSV ---
    merged.to_csv(merged_csv, index=False)
    print(f"\n  Output: {merged_csv}")
    print(f"  Final rows: {len(merged)}")

    # --- Print stats ---
    _print_pipeline_stats(merged)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge per-worker Pipeline meta-learning CSV files')
    parser.add_argument('--output_dir', default='./pipeline_meta_output',
                        help='Base output directory (default: ./pipeline_meta_output)')
    args = parser.parse_args()

    merge_pipeline_results(args.output_dir)
