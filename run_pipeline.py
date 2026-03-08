"""
run_pipeline.py — Master Script for Mamba TTS Dataset Preparation

This script orchestrates:
1. Downloading datasets (LJ Speech, VCTK, LibriTTS-R, CREMA-D)
2. Curation and filtering (Targeting Female British accents)
3. Preprocessing (Resampling, Normalization, Trim, G2P)
4. Indexing (Master JSONL with Style Tags)

Target: ~50 hours of high-quality training data.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_step(cmd, desc):
    print(f"\n>>> Starting Step: {desc}")
    try:
        subprocess.check_call([sys.executable] + cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error in {desc}: {e}")
        return False
    return True

def main():
    # 1. Download
    # Note: Downloading all can take a long time! You might want to run them individually.
    if not run_step(["data_downloader.py", "--dataset", "all"], "Downloading Datasets"):
        print("Download step failed or was partially successful.")

    # 2. Index / Preprocess
    if not run_step(["dataset_indexer.py"], "Curating and Preprocessing"):
        print("Indexing step failed.")
        return

    print("\n" + "="*40)
    print("Pipeline Complete!")
    print(f"Curated dataset is ready in: {Path('curated_dataset').absolute()}")
    print(f"Master Index: {Path('curated_dataset/master_index.jsonl').absolute()}")
    print("="*40)

if __name__ == "__main__":
    main()
