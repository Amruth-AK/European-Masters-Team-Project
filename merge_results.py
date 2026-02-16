"""
Merge all per-worker CSV files into a single meta_learning_db.csv

Usage:
  python merge_results.py --output_dir ./meta_learning_output

Also prints a summary of worker statuses from checkpoint files.
"""
import argparse
import glob
import json
import os

import pandas as pd

from DataCollector_3 import MetaDataCollector


def merge_results(output_dir):
    worker_csv_dir = os.path.join(output_dir, 'worker_csvs')
    checkpoint_dir = os.path.join(output_dir, 'worker_checkpoints')
    merged_csv = os.path.join(output_dir, 'meta_learning_db.csv')
    
    # --- Summarize checkpoints ---
    print("=" * 60)
    print("WORKER STATUS SUMMARY")
    print("=" * 60)
    
    status_counts = {'done': 0, 'failed': 0, 'skipped': 0, 'in_progress': 0}
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_worker_*.json'))
    
    failed_tasks = []
    for cp_file in sorted(checkpoint_files):
        with open(cp_file, 'r') as f:
            ckpt = json.load(f)
        status = ckpt.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 'failed':
            failed_tasks.append((ckpt.get('worker_id'), ckpt.get('task_id'), ckpt.get('detail', '')[:80]))
    
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
    print("MERGING WORKER CSVs")
    print("=" * 60)
    
    csv_files = sorted(glob.glob(os.path.join(worker_csv_dir, 'meta_learning_db_worker_*.csv')))
    print(f"Found {len(csv_files)} worker CSV files")
    
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
        print("No valid data found in any CSV!")
        return
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # Ensure schema alignment
    for col in MetaDataCollector.CSV_SCHEMA:
        if col not in merged.columns:
            merged[col] = ''
    merged = merged[MetaDataCollector.CSV_SCHEMA]
    
    # Write merged file
    merged.to_csv(merged_csv, index=False)
    
    print(f"\nMerged: {len(merged)} rows from {len(dfs)} workers")
    print(f"Output: {merged_csv}")
    print(f"Unique datasets: {merged['dataset_name'].nunique()}")
    
    # Stats
    try:
        sig_full = (merged['is_significant'] == True).sum() + (merged['is_significant'] == 'True').sum()
        sig_indiv = (merged['individual_is_significant'] == True).sum() + (merged['individual_is_significant'] == 'True').sum()
        print(f"Significant full-model (p<0.05): {sig_full}")
        print(f"Significant individual (p<0.05): {sig_indiv}")
    except:
        pass
    
    print(f"\nTask types: {merged['task_type'].value_counts().to_dict()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./meta_learning_output')
    args = parser.parse_args()
    merge_results(args.output_dir)