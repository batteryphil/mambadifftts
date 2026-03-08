"""
run_optimized_pipeline.py — Storage-Efficient Pipeline for Mamba TTS

This script processes datasets one-by-one to stay under 20GB disk limit.
It downloads, extracts, curates, and DELETES raw data before moving to the next.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = ["ljspeech", "vctk", "libritts_r", "crema_d"]

def run_cmd(args):
    try:
        subprocess.check_call([sys.executable] + args)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(args)}: {e}")
        return False

def main():
    print(">>> Starting Optimized Sequential Pipeline (Target: <20GB Disk)")
    
    for ds in DATASETS:
        print(f"\n--- Processing {ds} ---")
        
        # 1. Download and Extract (with archive cleanup)
        if not run_cmd(["data_downloader.py", "--dataset", ds, "--cleanup"]):
            continue
            
        # 2. Curation & Preprocessing (Appends to master index)
        if not run_cmd(["dataset_indexer.py", "--dataset", ds]):
            print(f"Warning: Indexing failed for {ds}")
            
        # 3. Cleanup Source Folder (The big raw files)
        source_dir = Path(f"data/{ds}")
        if source_dir.exists():
            print(f"Cleaning up source directory: {source_dir}")
            shutil.rmtree(source_dir) 

    print("\n" + "="*40)
    print("Optimized Pipeline Finished!")
    print(f"Final curated dataset: {Path('curated_dataset').absolute()}")
    print("="*40)

if __name__ == "__main__":
    main()
