import openml
import json
import os

# 1. Configuration
CACHE_DIR = '/work/inestp05/lightgbm_project/openml_cache'
TASK_LIST = os.path.join(OUTPUT_DIR, 'task_list.json')
# Set a shared cache directory so all nodes look in the same place
CACHE_DIR = os.path.expanduser('~/.openml/cache') 
openml.config.cache_directory = CACHE_DIR

def prefetch():
    if not os.path.exists(TASK_LIST):
        print("Error: Run your task list generator first!")
        return

    with open(TASK_LIST, 'r') as f:
        tasks = json.load(f)['tasks']

    print(f"Starting pre-cache for {len(tasks)} datasets...")
    
    for i, entry in enumerate(tasks):
        try:
            d_id = entry.get('dataset_id')
            t_id = entry.get('task_id')
            
            if t_id and t_id > 0:
                print(f"[{i+1}/{len(tasks)}] Pre-fetching Task {t_id}...")
                openml.tasks.get_task(t_id) # This triggers the download
            elif d_id:
                print(f"[{i+1}/{len(tasks)}] Pre-fetching Dataset {d_id}...")
                openml.datasets.get_dataset(d_id)
                
        except Exception as e:
            print(f"  Failed to download {entry}: {e}")

if __name__ == "__main__":
    prefetch()