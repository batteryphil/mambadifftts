import subprocess
import time
import os
import sys
from pathlib import Path

def run_command(cmd_list, description):
    print(f"\n>>> Starting: {description}")
    print(f"Executing: {' '.join(cmd_list)}")
    process = subprocess.Popen(cmd_list, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        print(f"\n!!! Error: {description} failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f">>> Completed: {description}")

def main():
    # 1. Wait for the existing data_downloader process to finish
    # Since we can't easily wait on a background process by ID from a new script easily without OS-specifics,
    # we'll just run it again. It will skip if the data exists or wait if it's already locked.
    # However, to be cleaner, we'll just run the stages sequentially in this script.
    
    print("--- Mamba TTS Automation Pipeline ---")
    
    # Stage 1: Procurement (already started by user, but this ensures it completes)
    run_command(["python", "data_downloader.py", "--dataset", "vctk", "--cleanup"], "Data Procurement")
    
    # Stage 2: Indexing & Preprocessing
    run_command(["python", "dataset_indexer.py", "--dataset", "vctk"], "Dataset Indexing & Mel Caching")
    
    # Stage 3: Training
    run_command(["python", "tts_trainer.py"], "Mamba TTS Training (Fine-tuning)")

if __name__ == "__main__":
    main()
