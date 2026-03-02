"""
Cleanup script for failed/NaN meta-learning workers.

Scans worker error logs, checkpoints, CSVs, and pipeline logs to find
problematic workers, then deletes their output files so they can be rerun.

Detects four types of problems:
  1. FAILED:    Checkpoint status = 'failed' or 'in_progress' (hard crash / SLURM killed)
  2. ERRORS:    Error log exists (some interventions failed — may have partial data)
  3. NAN:       CSV has NaN baseline_score (silent failure — data is garbage)
  4. TIMED_OUT: Pipeline log shows "Time budget exceeded" (valid but incomplete data)

Usage:
  # Dry run (default) — shows what would be deleted:
  python cleanup_failed_workers.py --output_dir /work/inestp05/lightgbm_project/FE_PP_MetaModel/meta_learning_output

  # Also detect timed-out workers:
  python cleanup_failed_workers.py --output_dir /work/inestp05/lightgbm_project/FE_PP_MetaModel/meta_learning_output --include-timeouts

  # Actually delete files:
  python cleanup_failed_workers.py --output_dir /work/inestp05/lightgbm_project/FE_PP_MetaModel/meta_learning_output --execute

  # Only clean up workers with NaN baselines or hard failures (keep partial-error and timeout workers):
  python cleanup_failed_workers.py --output_dir /work/inestp05/lightgbm_project/FE_PP_MetaModel/meta_learning_output --execute --skip-partial-errors
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
import numpy as np


def find_problematic_workers(output_dir, skip_partial_errors=False, include_timeouts=False):
    """
    Scan all worker artifacts and return a dict of worker_tag -> list of problems.
    """
    log_dir = os.path.join(output_dir, 'worker_logs')
    csv_dir = os.path.join(output_dir, 'worker_csvs')
    ckpt_dir = os.path.join(output_dir, 'worker_checkpoints')

    # Collect all known worker tags from any source
    all_worker_tags = set()

    # --- 1. Workers with error logs ---
    error_workers = {}
    error_log_pattern = os.path.join(log_dir, 'error_log_worker_*.txt')
    for path in glob.glob(error_log_pattern):
        fname = os.path.basename(path)
        # error_log_worker_00004.txt -> worker_00004
        tag = fname.replace('error_log_', '').replace('.txt', '')
        # Only count non-empty error logs
        if os.path.getsize(path) > 0:
            # Count number of errors in the file
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            n_errors = content.count('ERROR:')
            error_workers[tag] = n_errors
            all_worker_tags.add(tag)

    # --- 2. Workers with failed checkpoints ---
    failed_workers = {}
    ckpt_pattern = os.path.join(ckpt_dir, 'checkpoint_worker_*.json')
    for path in glob.glob(ckpt_pattern):
        fname = os.path.basename(path)
        tag = fname.replace('checkpoint_', '').replace('.json', '')
        try:
            with open(path, 'r') as f:
                ckpt = json.load(f)
            status = ckpt.get('status', 'unknown')
            if status in ('failed', 'in_progress'):
                failed_workers[tag] = {
                    'status': status,
                    'detail': ckpt.get('detail', '')[:120],
                    'task_id': ckpt.get('task_id'),
                    'dataset_id': ckpt.get('dataset_id'),
                }
                all_worker_tags.add(tag)
        except (json.JSONDecodeError, IOError):
            failed_workers[tag] = {'status': 'corrupt_checkpoint', 'detail': 'Could not read JSON'}
            all_worker_tags.add(tag)

    # --- 3. Workers with NaN baseline in CSV ---
    nan_workers = {}
    csv_pattern = os.path.join(csv_dir, 'meta_learning_db_worker_*.csv')
    for path in glob.glob(csv_pattern):
        fname = os.path.basename(path)
        tag = fname.replace('meta_learning_db_', '').replace('.csv', '')
        try:
            df = pd.read_csv(path, low_memory=False)
            if 'baseline_score' in df.columns:
                n_nan = df['baseline_score'].isna().sum()
                n_total = len(df)
                if n_nan > 0:
                    nan_workers[tag] = {
                        'n_nan_rows': int(n_nan),
                        'n_total_rows': n_total,
                        'pct_nan': f"{n_nan / n_total * 100:.1f}%",
                    }
                    all_worker_tags.add(tag)
            elif len(df) == 0:
                nan_workers[tag] = {
                    'n_nan_rows': 0, 'n_total_rows': 0, 'pct_nan': 'empty CSV',
                }
                all_worker_tags.add(tag)
        except Exception as e:
            nan_workers[tag] = {
                'n_nan_rows': -1, 'n_total_rows': -1, 'pct_nan': f'read error: {e}',
            }
            all_worker_tags.add(tag)

    # --- 4. Workers that hit the internal time budget ---
    # These have checkpoint "done" but the pipeline log shows "Time budget exceeded",
    # meaning they only tested a fraction of the columns/interactions.
    timeout_workers = {}
    if include_timeouts:
        pipeline_log_pattern = os.path.join(log_dir, 'pipeline_log_worker_*.txt')
        for path in glob.glob(pipeline_log_pattern):
            fname = os.path.basename(path)
            tag = fname.replace('pipeline_log_', '').replace('.txt', '')
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                if 'Time budget exceeded' in content:
                    # Extract the progress info: "after X/Y columns"
                    match = re.search(r'Time budget exceeded after (\d+)/(\d+) columns', content)
                    if match:
                        done_cols = int(match.group(1))
                        total_cols = int(match.group(2))
                        pct_done = done_cols / total_cols * 100 if total_cols > 0 else 0
                    else:
                        done_cols, total_cols, pct_done = -1, -1, -1

                    timeout_workers[tag] = {
                        'cols_done': done_cols,
                        'cols_total': total_cols,
                        'pct_completed': f"{pct_done:.0f}%",
                    }
                    all_worker_tags.add(tag)
            except IOError:
                pass

    # --- Combine into a single report ---
    problems = {}
    for tag in sorted(all_worker_tags):
        issues = []

        if tag in failed_workers:
            issues.append(('FAILED/STALE', failed_workers[tag]))

        if tag in nan_workers:
            issues.append(('NAN_BASELINE', nan_workers[tag]))

        if tag in error_workers:
            issues.append(('HAS_ERRORS', {'n_errors': error_workers[tag]}))

        if tag in timeout_workers:
            issues.append(('TIMED_OUT', timeout_workers[tag]))

        # If --skip-partial-errors: only include workers that have FAILED or NAN issues,
        # not workers that merely have some error-log entries but otherwise succeeded
        if skip_partial_errors:
            issue_types = {t for t, _ in issues}
            if issue_types == {'HAS_ERRORS'}:
                # Only has errors, no NaN or failure — skip this worker
                continue
            if issue_types == {'TIMED_OUT'}:
                # Only timed out, no NaN or failure — skip this worker
                continue

        if issues:
            problems[tag] = issues

    return problems


def get_worker_files(output_dir, worker_tag):
    """Return all files associated with a worker tag."""
    files = []
    candidates = [
        os.path.join(output_dir, 'worker_csvs', f'meta_learning_db_{worker_tag}.csv'),
        os.path.join(output_dir, 'worker_checkpoints', f'checkpoint_{worker_tag}.json'),
        os.path.join(output_dir, 'worker_logs', f'error_log_{worker_tag}.txt'),
        os.path.join(output_dir, 'worker_logs', f'pipeline_log_{worker_tag}.txt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Find and clean up failed/NaN meta-learning workers for rerun.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--output_dir', required=True,
                        help='Base output directory (contains worker_csvs/, worker_checkpoints/, worker_logs/)')
    parser.add_argument('--execute', action='store_true', default=False,
                        help='Actually delete files. Without this flag, only a dry-run report is printed.')
    parser.add_argument('--skip-partial-errors', action='store_true', default=False,
                        help='Keep workers that only have error-log entries but valid baselines. '
                             'Only clean up hard failures and NaN baselines.')
    parser.add_argument('--include-timeouts', action='store_true', default=False,
                        help='Also detect and clean up workers that hit the internal time budget. '
                             'These have valid but incomplete data (only a fraction of columns tested).')
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 80)
    print("META-LEARNING WORKER CLEANUP")
    print("=" * 80)
    print(f"  Output dir: {output_dir}")
    print(f"  Mode:       {'EXECUTE (will delete files!)' if args.execute else 'DRY RUN (no changes)'}")
    print(f"  Policy:     {'Skip partial-error workers' if args.skip_partial_errors else 'Clean ALL problematic workers'}")
    print(f"  Timeouts:   {'Include timed-out workers' if args.include_timeouts else 'Ignore timed-out workers (use --include-timeouts to add)'}")
    print()

    # --- Scan ---
    problems = find_problematic_workers(output_dir, skip_partial_errors=args.skip_partial_errors,
                                        include_timeouts=args.include_timeouts)

    if not problems:
        print("No problematic workers found. Everything looks clean!")
        return

    # --- Report ---
    n_failed = sum(1 for issues in problems.values() if any(t == 'FAILED/STALE' for t, _ in issues))
    n_nan = sum(1 for issues in problems.values() if any(t == 'NAN_BASELINE' for t, _ in issues))
    n_errors = sum(1 for issues in problems.values() if any(t == 'HAS_ERRORS' for t, _ in issues))
    n_timeouts = sum(1 for issues in problems.values() if any(t == 'TIMED_OUT' for t, _ in issues))

    print(f"Found {len(problems)} problematic workers:")
    print(f"  - {n_failed} with FAILED/STALE checkpoints")
    print(f"  - {n_nan} with NaN baseline scores in CSV")
    print(f"  - {n_errors} with error log entries")
    print(f"  - {n_timeouts} that hit the time budget (incomplete data)")
    print()

    total_files_to_delete = 0

    for tag, issues in sorted(problems.items()):
        issue_summary = ' + '.join(t for t, _ in issues)
        print(f"  {tag}: [{issue_summary}]")
        for issue_type, detail in issues:
            if issue_type == 'FAILED/STALE':
                print(f"      Status: {detail.get('status')} — {detail.get('detail', '')}")
            elif issue_type == 'NAN_BASELINE':
                print(f"      NaN rows: {detail['n_nan_rows']}/{detail['n_total_rows']} ({detail['pct_nan']})")
            elif issue_type == 'HAS_ERRORS':
                print(f"      Errors in log: {detail['n_errors']}")
            elif issue_type == 'TIMED_OUT':
                print(f"      Completed: {detail['cols_done']}/{detail['cols_total']} columns ({detail['pct_completed']})")

        files = get_worker_files(output_dir, tag)
        for f in files:
            print(f"      -> {'DELETE' if args.execute else 'would delete'}: {os.path.basename(f)}")
        total_files_to_delete += len(files)
        print()

    # --- Execute ---
    if args.execute:
        print("-" * 80)
        print(f"DELETING {total_files_to_delete} files for {len(problems)} workers...")
        deleted = 0
        for tag in sorted(problems.keys()):
            files = get_worker_files(output_dir, tag)
            for f in files:
                try:
                    os.remove(f)
                    deleted += 1
                except OSError as e:
                    print(f"  WARNING: Could not delete {f}: {e}")

        print(f"Done. Deleted {deleted}/{total_files_to_delete} files.")
        print()
        print("Next steps:")
        print("  1. Apply the bug fixes to DataCollector_3.py")
        print("  2. Resubmit the SLURM array job — workers with deleted checkpoints will rerun automatically")
    else:
        print("-" * 80)
        print(f"DRY RUN complete. Would delete {total_files_to_delete} files for {len(problems)} workers.")
        print(f"Rerun with --execute to actually delete them.")


if __name__ == '__main__':
    main()