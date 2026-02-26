"""
merge_collector_results.py — Merge per-worker CSVs into a single file
======================================================================

Usage:
    python merge_collector_results.py --output_dir ./output_numerical --prefix numerical_transforms
    python merge_collector_results.py --output_dir ./output_categorical --prefix categorical_transforms
    python merge_collector_results.py --output_dir ./output_interactions --prefix interaction_features
"""
import argparse
import glob
import json
import os

import pandas as pd


def merge_results(output_dir, prefix):
    worker_csv_dir = os.path.join(output_dir, 'worker_csvs')
    checkpoint_dir = os.path.join(output_dir, 'worker_checkpoints')
    merged_csv = os.path.join(output_dir, f'{prefix}_merged.csv')

    # --- Summarize checkpoints ---
    print("=" * 60)
    print("WORKER STATUS SUMMARY")
    print("=" * 60)

    status_counts = {}
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_worker_*.json'))

    failed_tasks = []
    for cp_file in sorted(checkpoint_files):
        try:
            with open(cp_file, 'r') as f:
                ckpt = json.load(f)
            status = ckpt.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == 'failed':
                failed_tasks.append((
                    ckpt.get('worker_id'),
                    ckpt.get('task_id'),
                    ckpt.get('detail', '')[:80]
                ))
        except Exception:
            status_counts['corrupt_checkpoint'] = status_counts.get('corrupt_checkpoint', 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"  Total checkpoints: {len(checkpoint_files)}")

    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for wid, tid, detail in failed_tasks[:20]:
            print(f"  Worker {wid}, task {tid}: {detail}")
        if len(failed_tasks) > 20:
            print(f"  ... and {len(failed_tasks) - 20} more")

    # --- Merge CSVs ---
    print(f"\n{'='*60}")
    print(f"MERGING WORKER CSVs (prefix: {prefix})")
    print("=" * 60)

    pattern = os.path.join(worker_csv_dir, f'{prefix}_worker_*.csv')
    csv_files = sorted(glob.glob(pattern))
    print(f"Found {len(csv_files)} worker CSV files matching: {pattern}")

    if not csv_files:
        print("No CSV files to merge!")
        return

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Failed to read {csv_file}: {e}")

    if not dfs:
        print("No valid data found!")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (same dataset + column + method)
    id_cols = ['dataset_name', 'method']
    if 'column_name' in merged.columns:
        id_cols.append('column_name')
    if 'interaction_col_a' in merged.columns:
        id_cols.extend(['interaction_col_a', 'interaction_col_b'])

    before = len(merged)
    merged = merged.drop_duplicates(subset=id_cols, keep='first')
    after = len(merged)
    if before != after:
        print(f"  Removed {before - after} duplicate rows")

    merged.to_csv(merged_csv, index=False)

    print(f"\nMerged: {len(merged)} rows from {len(dfs)} workers")
    print(f"Output: {merged_csv}")
    print(f"Unique datasets: {merged['dataset_name'].nunique()}")

    # Quick stats
    try:
        if 'is_significant' in merged.columns:
            sig = (merged['is_significant'] == True).sum() + \
                  (merged['is_significant'] == 'True').sum()
            print(f"Significant (p<0.05): {sig} / {len(merged)} "
                  f"({sig/len(merged)*100:.1f}%)")
        if 'method' in merged.columns:
            print(f"\nMethods tested:")
            for method, count in merged['method'].value_counts().items():
                print(f"  {method}: {count}")
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True,
                        help='Directory containing worker_csvs/')
    parser.add_argument('--prefix', required=True,
                        help='CSV file prefix (numerical_transforms, categorical_transforms, interaction_features)')
    args = parser.parse_args()
    merge_results(args.output_dir, args.prefix)
