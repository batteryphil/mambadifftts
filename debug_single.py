import json
import torch
from pathlib import Path
import numpy as np
import time
from tts_data_builder import wav_to_mel

def test_single_file():
    index_path = "data/audiobook_british_libritts_train.jsonl"
    with open(index_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        record = json.loads(first_line)
    
    audio_path = record["audio"]
    print(f"Testing audio path: {audio_path}")
    if not Path(audio_path).exists():
        print(f"ERROR: File does not exist at {audio_path}")
        return
        
    try:
        start = time.time()
        mel = wav_to_mel(audio_path)
        print(f"Success! Mel shape: {mel.shape} | Time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_single_file()
