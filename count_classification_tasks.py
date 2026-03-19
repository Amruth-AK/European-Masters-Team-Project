"""
count_classification_tasks.py — Count classification tasks and print sbatch commands
=====================================================================================

Usage:
    python count_classification_tasks.py --task_list ./task_list.json
"""
import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_list',
                        default=os.environ.get('TASK_LIST', './task_list.json'))
    args = parser.parse_args()

    with open(args.task_list, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        tasks = data.get('tasks', [])
    else:
        tasks = data

    total = len(tasks)
    classification = [t for t in tasks if t.get('task_type', '').lower() == 'classification']
    regression = [t for t in tasks if t.get('task_type', '').lower() == 'regression']
    other = total - len(classification) - len(regression)

    n = len(classification)
    max_idx = n - 1

    print(f"Task list: {args.task_list}")
    print(f"  Total tasks:          {total}")
    print(f"  Classification:       {n}")
    print(f"  Regression (skipped): {len(regression)}")
    if other > 0:
        print(f"  Other (skipped):      {other}")
    print()
    print(f"Array range: --array=0-{max_idx}")
    print()
    print("=" * 70)
    print("SUBMIT COMMANDS (copy & paste):")
    print("=" * 70)
    print()
    print("mkdir -p logs")

    if max_idx < 1000:
        # Simple case: fits in one array
        print()
        print(f"sbatch --array=0-{max_idx}%50 --job-name=num_collect \\")
        print(f"  --export=COLLECTOR=numerical,OUTPUT_DIR=/work/$USER/Final/output_numerical \\")
        print(f"  run_collector.sh")
        print()
        print(f"sbatch --array=0-{max_idx}%50 --job-name=cat_collect \\")
        print(f"  --export=COLLECTOR=categorical,OUTPUT_DIR=/work/$USER/Final/output_categorical \\")
        print(f"  run_collector.sh")
        print()
        print(f"sbatch --array=0-{max_idx}%50 --job-name=int_collect \\")
        print(f"  --export=COLLECTOR=interaction,OUTPUT_DIR=/work/$USER/Final/output_interactions \\")
        print(f"  run_collector.sh")
        print()
        print(f"sbatch --array=0-{max_idx}%50 --job-name=row_collect \\")
        print(f"  --export=COLLECTOR=row,OUTPUT_DIR=/work/$USER/Final/output_row \\")
        print(f"  run_collector.sh")
    else:
        # Need OFFSET batches (MaxArraySize is typically 1000)
        batch_size = 999
        for collector, short, outdir in [
            ('numerical', 'num', 'output_numerical'),
            ('categorical', 'cat', 'output_categorical'),
            ('interaction', 'int', 'output_interactions'),
            ('row', 'row', 'output_row'),
        ]:
            print(f"\n# --- {collector} ---")
            offset = 0
            batch_num = 1
            while offset <= max_idx:
                end = min(batch_size, max_idx - offset)
                print(f"sbatch --array=0-{end}%50 --job-name={short}_b{batch_num} \\")
                print(f"  --export=COLLECTOR={collector},OUTPUT_DIR=/work/$USER/Final/{outdir},OFFSET={offset} \\")
                print(f"  run_collector.sh")
                offset += batch_size + 1
                batch_num += 1
    print()
    print("=" * 70)
    print("AFTER ALL JOBS FINISH — MERGE:")
    print("=" * 70)
    print()
    print("python merge_collector_results.py --output_dir /work/$USER/Final/output_numerical --prefix numerical_transforms")
    print("python merge_collector_results.py --output_dir /work/$USER/Final/output_categorical --prefix categorical_transforms")
    print("python merge_collector_results.py --output_dir /work/$USER/Final/output_interactions --prefix interaction_features")
    print("python merge_collector_results.py --output_dir /work/$USER/Final/output_row --prefix row_features")


if __name__ == '__main__':
    main()