import os
import wave
import numpy as np
import json
from pathlib import Path

def create_dummy_wav(path, duration=2.0, sr=22050):
    """Creates a dummy silent wav file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration * sr)
    # Use random noise to avoid aggressive silence trimming in tests
    data = (np.random.randn(n_samples) * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())

def create_dummy_dataset():
    root = Path("data/ljspeech/LJSpeech-1.1")
    root.mkdir(parents=True, exist_ok=True)
    
    wav_dir = root / "wavs"
    wav_dir.mkdir(exist_ok=True)
    
    metadata = []
    for i in range(5):
        fileid = f"LJ001-000{i}"
        wav_path = wav_dir / f"{fileid}.wav"
        create_dummy_wav(wav_path)
        metadata.append(f"{fileid}|dummy text|This is dummy sentence {i}.")
        
    with open(root / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
    
    print("Created dummy LJSpeech dataset for testing.")

if __name__ == "__main__":
    create_dummy_dataset()
