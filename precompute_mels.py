import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa

# Audio config (MUST match tts_data_builder.py)
SAMPLE_RATE = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
N_MELS      = 80
FMIN        = 0.0
FMAX        = 8000.0

def wav_to_mel(wav_path: str) -> np.ndarray:
    y, sr = sf.read(wav_path, dtype='float32')
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float16) # Cache in float16 to save space

def precompute_mels(index_path: str, cache_dir: str):
    print(f"Pre-computing mels for {index_path}...")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    for r in tqdm(records):
        wav_path = Path(r["audio"])
        mel_path = cache_path / (wav_path.stem + ".npy")
        
        if not mel_path.exists():
            try:
                mel = wav_to_mel(str(wav_path))
                np.save(mel_path, mel)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    precompute_mels("data/audiobook_british_libritts_train.jsonl", "data/mel_cache")
    precompute_mels("data/audiobook_british_libritts_val.jsonl", "data/mel_cache")
