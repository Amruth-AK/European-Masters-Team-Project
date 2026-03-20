"""
ablate_slurm.py — Parallel Ablation of Composite Target Formulas
=================================================================

Each SLURM array job handles one (collector_type, formula) cell:
  1. Loads the merged CSV for that collector type
  2. Runs GroupKFold CV (same logic as ablate_formulas in train_meta_models.py)
  3. Writes results to a per-worker JSON in <output_dir>/ablation_workers/

A separate --merge pass reads all worker JSONs and prints the sorted
comparison table, identical to the serial ablate_formulas() output.

Usage
-----
# Step 1: submit the array (see run_ablation.sh)
#   sbatch --array=0-<N_CELLS-1>%20 run_ablation.sh

# Step 2: after all jobs finish, merge:
#   python ablate_slurm.py --merge --output_dir ./ablation_results

Grid size = len(COLLECTOR_TYPES) × len(COMPOSITE_FORMULAS)
Currently: 4 types × 20 formulas = 80 cells  →  --array=0-79

The grid index mapping is printed at startup so you can inspect it:
  python ablate_slurm.py --print_grid
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Import shared definitions from train_meta_models
# ---------------------------------------------------------------------------
# We import directly so the formulas, feature lists, and CV logic stay in one
# place — no duplication.
try:
    from train_meta_models import (
        COMPOSITE_FORMULAS,
        CLASSIFICATION_STRATEGIES,
        load_and_prepare,
        evaluate_grouped_cv,
        DEFAULT_CLS_STRATEGY,
    )
except ImportError as e:
    print(f"ERROR: Could not import from train_meta_models.py: {e}")
    print("Make sure this script lives in the same directory as train_meta_models.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

COLLECTOR_TYPES = ['numerical', 'categorical', 'interaction', 'row']

# Build the flat ordered grid: [(ctype, formula_name), ...]
# Order: types outer loop, formulas inner loop — easy to inspect per-type.
ABLATION_GRID = [
    (ctype, fname)
    for ctype in COLLECTOR_TYPES
    for fname in COMPOSITE_FORMULAS
]

N_CELLS = len(ABLATION_GRID)


def print_grid():
    print(f"\nAblation grid ({N_CELLS} cells):")
    print(f"  {'ID':>4}  {'Type':>15}  Formula")
    print(f"  {'----':>4}  {'-'*15}  {'-'*35}")
    for i, (ctype, fname) in enumerate(ABLATION_GRID):
        print(f"  {i:>4}  {ctype:>15}  {fname}")


# ---------------------------------------------------------------------------
# Single-cell worker
# ---------------------------------------------------------------------------

def run_cell(cell_id: int, collector_csvs: dict, output_dir: str,
             n_cv_splits: int = 5, cls_strategy: str = DEFAULT_CLS_STRATEGY):
    """
    Run GroupKFold CV for one (ctype, formula) cell and write results to JSON.

    Args:
        cell_id:        Index into ABLATION_GRID
        collector_csvs: Dict mapping ctype → path to merged CSV
        output_dir:     Base output directory; JSONs go in <output_dir>/ablation_workers/
        n_cv_splits:    Number of GroupKFold splits
        cls_strategy:   Which classification target strategy to use alongside regression
    """
    if cell_id < 0 or cell_id >= N_CELLS:
        print(f"ERROR: cell_id {cell_id} out of range [0, {N_CELLS - 1}]")
        sys.exit(1)

    ctype, formula_name = ABLATION_GRID[cell_id]

    workers_dir = os.path.join(output_dir, 'ablation_workers')
    os.makedirs(workers_dir, exist_ok=True)

    result_path = os.path.join(workers_dir, f'cell_{cell_id:04d}_{ctype}_{formula_name}.json')

    print(f"\n{'='*70}")
    print(f"Ablation cell {cell_id}/{N_CELLS}: type={ctype}  formula={formula_name}")
    print(f"{'='*70}")

    # Idempotency: skip if already done
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        if existing.get('status') == 'done':
            print(f"Already done — skipping. Results at {result_path}")
            return

    # Write in-progress marker
    _write_result(result_path, {
        'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
        'status': 'in_progress', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    csv_path = collector_csvs.get(ctype)
    if not csv_path or not os.path.exists(csv_path):
        msg = f"CSV not found for type={ctype}: {csv_path}"
        print(f"SKIPPED: {msg}")
        _write_result(result_path, {
            'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
            'status': 'skipped', 'reason': msg,
        })
        return

    t0 = time.time()
    try:
        X, y_reg, y_cls, groups, _, _ = load_and_prepare(
            csv_path, ctype,
            formula=formula_name,
            cls_strategy=cls_strategy,
        )
    except Exception as e:
        msg = f"load_and_prepare failed: {e}"
        print(f"ERROR: {msg}")
        traceback.print_exc()
        _write_result(result_path, {
            'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
            'status': 'failed', 'reason': msg,
        })
        return

    # Guard: constant target means a required column was missing from this CSV
    # (e.g. individual_delta doesn't exist in the row collector).
    # sklearn's r2_score returns 1.0 for a constant target — a silent false positive.
    if y_reg.std() < 1e-6:
        msg = (
            f"Target '{formula_name}' is constant (std={y_reg.std():.2e}) — "
            f"formula likely references a column absent from the '{ctype}' CSV "
            f"(e.g. individual_delta / individual_cohens_d). Skipping."
        )
        print(f"SKIPPED: {msg}")
        _write_result(result_path, {
            'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
            'status': 'skipped', 'reason': msg,
        })
        return

    n_datasets = groups.nunique()
    actual_splits = min(n_cv_splits, n_datasets)

    if actual_splits < 2:
        msg = f"Only {n_datasets} dataset(s) — need ≥2 for CV"
        print(f"SKIPPED: {msg}")
        _write_result(result_path, {
            'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
            'status': 'skipped', 'reason': msg,
        })
        return

    print(f"Running {actual_splits}-fold GroupKFold CV "
          f"({n_datasets} datasets, {len(X)} rows)...")
    try:
        cv_results, _ = evaluate_grouped_cv(
            X, y_reg, y_cls, groups,
            n_splits=actual_splits,
        )
    except Exception as e:
        msg = f"evaluate_grouped_cv failed: {e}"
        print(f"ERROR: {msg}")
        traceback.print_exc()
        _write_result(result_path, {
            'cell_id': cell_id, 'ctype': ctype, 'formula': formula_name,
            'status': 'failed', 'reason': msg,
        })
        return

    elapsed = time.time() - t0

    # Extract key metrics
    reg = cv_results['regression']
    cls_ = cv_results['classification']
    metrics = {
        'r2':       reg['r2']['mean'],
        'r2_std':   reg['r2']['std'],
        'mae':      reg['mae']['mean'],
        'rmse':     reg['rmse']['mean'],
        'cls_auc':  cls_.get('auc', {}).get('mean', float('nan')),
        'cls_f1':   cls_.get('f1', {}).get('mean', float('nan')),
        'cls_pct_pos': float(y_cls.mean()) * 100,
    }

    print(f"\nResult  R²={metrics['r2']:+.3f}±{metrics['r2_std']:.3f}  "
          f"MAE={metrics['mae']:.4f}  "
          f"CLS-AUC={metrics['cls_auc']:.3f}  F1={metrics['cls_f1']:.3f}  "
          f"pos={metrics['cls_pct_pos']:.1f}%  elapsed={elapsed:.1f}s")

    _write_result(result_path, {
        'cell_id':       cell_id,
        'ctype':         ctype,
        'formula':       formula_name,
        'cls_strategy':  cls_strategy,
        'status':        'done',
        'n_rows':        len(X),
        'n_datasets':    int(n_datasets),
        'n_cv_splits':   actual_splits,
        'metrics':       metrics,
        'cv_results':    cv_results,
        'elapsed_sec':   elapsed,
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
    })


def _write_result(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Merge + print table
# ---------------------------------------------------------------------------

def merge_results(output_dir: str):
    """
    Read all per-worker JSONs and print the ablation summary table,
    sorted by R² descending — same format as the serial ablate_formulas().
    Also saves a combined JSON for downstream use.
    """
    workers_dir = os.path.join(output_dir, 'ablation_workers')
    if not os.path.isdir(workers_dir):
        print(f"ERROR: workers dir not found: {workers_dir}")
        sys.exit(1)

    json_files = sorted(
        f for f in os.listdir(workers_dir)
        if f.startswith('cell_') and f.endswith('.json')
    )
    if not json_files:
        print("No worker result files found.")
        return

    records = []
    n_done = n_skipped = n_failed = n_inprogress = 0

    for fname in json_files:
        with open(os.path.join(workers_dir, fname)) as f:
            data = json.load(f)
        status = data.get('status', 'unknown')
        if status == 'done':
            n_done += 1
            m = data.get('metrics', {})
            records.append({
                'ctype':        data['ctype'],
                'formula':      data['formula'],
                'r2':           m.get('r2', float('nan')),
                'r2_std':       m.get('r2_std', float('nan')),
                'mae':          m.get('mae', float('nan')),
                'rmse':         m.get('rmse', float('nan')),
                'cls_auc':      m.get('cls_auc', float('nan')),
                'cls_f1':       m.get('cls_f1', float('nan')),
                'cls_pct_pos':  m.get('cls_pct_pos', float('nan')),
                'n_rows':       data.get('n_rows', 0),
                'n_datasets':   data.get('n_datasets', 0),
                'elapsed_sec':  data.get('elapsed_sec', 0),
            })
        elif status == 'skipped':
            n_skipped += 1
            print(f"  SKIPPED  {data.get('ctype'):>15}  {data.get('formula'):>35}  "
                  f"reason: {data.get('reason', '?')}")
        elif status == 'failed':
            n_failed += 1
            print(f"  FAILED   {data.get('ctype'):>15}  {data.get('formula'):>35}  "
                  f"reason: {data.get('reason', '?')}")
        elif status == 'in_progress':
            n_inprogress += 1
            print(f"  IN_PROG  {data.get('ctype'):>15}  {data.get('formula'):>35}")

    print(f"\n{'='*80}")
    print(f"Status: {n_done} done, {n_skipped} skipped, {n_failed} failed, "
          f"{n_inprogress} still running")
    print(f"{'='*80}")

    if not records:
        print("No completed cells to display.")
        return

    # ---- Per-type tables ----
    for ctype in COLLECTOR_TYPES:
        type_records = sorted(
            [r for r in records if r['ctype'] == ctype],
            key=lambda r: r['r2'], reverse=True,
        )
        if not type_records:
            continue
        print(f"\n  Collector: {ctype}")
        print(f"  {'Formula':>35}  {'R²':>7}±{'std':>5}  {'MAE':>7}  "
              f"{'AUC':>6}  {'F1':>6}  {'pos%':>5}")
        print(f"  {'-'*35}  {'-'*7} {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*5}")
        for r in type_records:
            marker = " ◀ best" if r == type_records[0] else ""
            print(f"  {r['formula']:>35}  {r['r2']:+.3f}±{r['r2_std']:.3f}  "
                  f"{r['mae']:.4f}  {r['cls_auc']:.3f}  {r['cls_f1']:.3f}  "
                  f"{r['cls_pct_pos']:>4.1f}%{marker}")

    # ---- Overall ranking across all types ----
    all_sorted = sorted(records, key=lambda r: r['r2'], reverse=True)
    print(f"\n\n{'='*80}")
    print("OVERALL RANKING (all types, sorted by R²)")
    print(f"{'='*80}")
    print(f"  {'Type':>15}  {'Formula':>35}  {'R²':>7}  {'MAE':>7}  {'AUC':>6}  {'F1':>6}")
    print(f"  {'-'*15}  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
    for r in all_sorted:
        print(f"  {r['ctype']:>15}  {r['formula']:>35}  "
              f"{r['r2']:+.3f}  {r['mae']:.4f}  {r['cls_auc']:.3f}  {r['cls_f1']:.3f}")

    # ---- Best formula per type summary ----
    print(f"\n\n{'='*80}")
    print("RECOMMENDED FORMULA PER TYPE")
    print(f"{'='*80}")
    for ctype in COLLECTOR_TYPES:
        type_records = [r for r in records if r['ctype'] == ctype]
        if not type_records:
            print(f"  {ctype:>15}: no data")
            continue
        best = max(type_records, key=lambda r: r['r2'])
        print(f"  {ctype:>15}:  {best['formula']}  "
              f"(R²={best['r2']:+.3f}, AUC={best['cls_auc']:.3f})")

    # ---- Save combined JSON ----
    combined_path = os.path.join(output_dir, 'ablation_results.json')
    with open(combined_path, 'w') as f:
        json.dump({'records': all_sorted, 'grid': ABLATION_GRID}, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parallel ablation of composite target formulas across collector types'
    )

    # --- Mode flags ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--cell_id', type=int,
        help='SLURM_ARRAY_TASK_ID: index into the (type × formula) grid'
    )
    mode_group.add_argument(
        '--merge', action='store_true',
        help='Merge all worker JSONs and print the comparison table'
    )
    mode_group.add_argument(
        '--print_grid', action='store_true',
        help='Print the full (cell_id → type, formula) mapping and exit'
    )

    # --- Data paths (required for worker mode) ---
    parser.add_argument('--numerical_csv',   type=str, default=None)
    parser.add_argument('--categorical_csv', type=str, default=None)
    parser.add_argument('--interaction_csv', type=str, default=None)
    parser.add_argument('--row_csv',         type=str, default=None)

    # --- Common options ---
    parser.add_argument(
        '--output_dir', type=str, default='./ablation_results',
        help='Directory for worker JSONs and merged output'
    )
    parser.add_argument('--n_cv_splits', type=int, default=5)
    parser.add_argument(
        '--cls_strategy', type=str, default=DEFAULT_CLS_STRATEGY,
        choices=list(CLASSIFICATION_STRATEGIES.keys()),
        help='Classification target strategy used alongside each regression formula'
    )

    args = parser.parse_args()

    # ---- Print grid ----
    if args.print_grid:
        print_grid()
        print(f"\nSubmit with:  --array=0-{N_CELLS - 1}")
        return

    # ---- Merge mode ----
    if args.merge:
        merge_results(args.output_dir)
        return

    # ---- Worker mode ----
    os.makedirs(args.output_dir, exist_ok=True)

    collector_csvs = {
        'numerical':   args.numerical_csv,
        'categorical': args.categorical_csv,
        'interaction': args.interaction_csv,
        'row':         args.row_csv,
    }

    run_cell(
        cell_id=args.cell_id,
        collector_csvs=collector_csvs,
        output_dir=args.output_dir,
        n_cv_splits=args.n_cv_splits,
        cls_strategy=args.cls_strategy,
    )


if __name__ == '__main__':
    main()