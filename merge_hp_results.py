"""
Merge HP Tuning Results from SLURM Workers
============================================

Combines per-worker CSVs from hp_tuning_output/worker_csvs/ into a single
hp_tuning_db.csv, deduplicates, and prints summary statistics.

Usage:
    python merge_hp_results.py --output_dir ./hp_tuning_output
"""

import os
import json
import glob
import argparse
import pandas as pd
import numpy as np


def merge_hp_results(output_dir='./hp_tuning_output'):
    worker_csv_dir = os.path.join(output_dir, 'worker_csvs')
    checkpoint_dir = os.path.join(output_dir, 'worker_checkpoints')
    merged_file = os.path.join(output_dir, 'hp_tuning_db.csv')
    
    # --- Scan checkpoints for status ---
    done = 0
    failed = 0
    skipped = 0
    in_progress = 0
    
    for ckpt_file in sorted(glob.glob(os.path.join(checkpoint_dir, 'hp_checkpoint_*.json'))):
        try:
            with open(ckpt_file, 'r') as f:
                ckpt = json.load(f)
            status = ckpt.get('status', 'unknown')
            if status == 'done':
                done += 1
            elif status == 'failed':
                failed += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'in_progress':
                in_progress += 1
        except:
            pass
    
    print(f"Checkpoint summary: {done} done, {failed} failed, "
          f"{skipped} skipped, {in_progress} in-progress")
    
    # --- Merge CSV files ---
    csv_files = sorted(glob.glob(os.path.join(worker_csv_dir, 'hp_tuning_db_*.csv')))
    print(f"Found {len(csv_files)} worker CSV files")
    
    if not csv_files:
        print("No CSV files to merge.")
        return
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")
    
    if not dfs:
        print("No valid data to merge.")
        return
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Raw merged: {len(merged)} rows")
    
    # --- Deduplicate (same dataset + config should be unique) ---
    dedup_cols = ['dataset_name', 'hp_config_name']
    if all(c in merged.columns for c in dedup_cols):
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_cols, keep='last')
        if before > len(merged):
            print(f"Deduplicated: {before} → {len(merged)} rows")
    
    # --- Re-rank within each dataset after merging ---
    # (in case some configs were from reprocessed datasets)
    if 'primary_score' in merged.columns and 'dataset_name' in merged.columns:
        
        # [FIX] Drop rows where primary_score is NaN and report it
        before_len = len(merged)
        merged = merged.dropna(subset=['primary_score'])
        dropped = before_len - len(merged)
        
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing 'primary_score'.")

        merged['rank_in_dataset'] = merged.groupby('dataset_name')['primary_score'].rank(
            ascending=False, method='dense').astype(int)
        
        # Recompute normalized_score and pct_of_best per dataset
        for ds_name, group in merged.groupby('dataset_name'):
            idx = group.index
            best = group['primary_score'].max()
            worst = group['primary_score'].min()
            score_range = best - worst if best != worst else 1e-8
            
            merged.loc[idx, 'normalized_score'] = (
                (group['primary_score'] - worst) / score_range)
            merged.loc[idx, 'pct_of_best'] = (
                group['primary_score'] / best if best != 0 else 1.0)
            
            # Delta vs default
            default_mask = group['hp_config_name'] == 'default'
            if default_mask.any():
                default_score = group.loc[default_mask, 'primary_score'].iloc[0]
                merged.loc[idx, 'delta_vs_default'] = group['primary_score'] - default_score
    
    # --- Write merged file ---
    merged.to_csv(merged_file, index=False)
    print(f"\nMerged file: {merged_file}")
    print(f"Total rows: {len(merged)}")
    
    # --- Summary statistics ---
    n_datasets = merged['dataset_name'].nunique()
    print(f"Unique datasets: {n_datasets}")
    print(f"Avg configs/dataset: {len(merged) / max(n_datasets, 1):.1f}")
    
    if 'task_type' in merged.columns:
        print(f"\nTask type distribution:")
        print(merged['task_type'].value_counts().to_string())
    
    if 'primary_score' in merged.columns:
        print(f"\nPrimary score statistics:")
        print(f"  Mean: {merged['primary_score'].mean():.5f}")
        print(f"  Std:  {merged['primary_score'].std():.5f}")
        print(f"  Min:  {merged['primary_score'].min():.5f}")
        print(f"  Max:  {merged['primary_score'].max():.5f}")
    
    if 'rank_in_dataset' in merged.columns:
        best_configs = merged[merged['rank_in_dataset'] == 1]
        if len(best_configs) > 0:
            print(f"\nBest config analysis (rank=1 across {len(best_configs)} datasets):")
            for hp_name in ['hp_num_leaves', 'hp_max_depth', 'hp_learning_rate',
                            'hp_n_estimators', 'hp_subsample', 'hp_colsample_bytree',
                            'hp_reg_alpha', 'hp_reg_lambda']:
                if hp_name in best_configs.columns:
                    vals = best_configs[hp_name].dropna()
                    if len(vals) > 0:
                        print(f"  {hp_name}: mean={vals.mean():.4f}, "
                              f"median={vals.median():.4f}, "
                              f"std={vals.std():.4f}")
    
    if 'delta_vs_default' in merged.columns:
        best_only = merged[merged['rank_in_dataset'] == 1]
        if len(best_only) > 0:
            avg_improvement = best_only['delta_vs_default'].mean()
            pct_improved = (best_only['delta_vs_default'] > 0).mean() * 100
            print(f"\nImprovement over default:")
            print(f"  Avg delta (best vs default): {avg_improvement:.5f}")
            print(f"  % datasets where best > default: {pct_improved:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HP tuning results from SLURM workers")
    parser.add_argument('--output_dir', default='./hp_tuning_output')
    args = parser.parse_args()
    merge_hp_results(args.output_dir)
