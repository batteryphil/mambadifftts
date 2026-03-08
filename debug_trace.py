import json
import torch
from pathlib import Path
import numpy as np
import time
import soundfile as sf
import librosa

SAMPLE_RATE = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
N_MELS      = 80
FMIN        = 0.0
FMAX        = 8000.0

def wav_to_mel_debug(wav_path: str) -> np.ndarray:
    print(f"    [TRACING] Entering wav_to_mel_debug for {wav_path}")
    print(f"    [TRACING] Calling sf.read...")
    t = time.time()
    y, sr = sf.read(wav_path)
    print(f"    [TRACING] sf.read done in {time.time() - t:.2f}s (sr={sr})")
    
    if sr != SAMPLE_RATE:
        print(f"    [TRACING] Resampling from {sr} to {SAMPLE_RATE}...")
        t = time.time()
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"    [TRACING] Resampling done in {time.time() - t:.2f}s")
        
    print(f"    [TRACING] Calling melspectrogram...")
    t = time.time()
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0
    )
    print(f"    [TRACING] melspectrogram done in {time.time() - t:.2f}s")
    
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float32)

def test_single_file():
    index_path = "data/audiobook_british_libritts_train.jsonl"
    print(f"Opening index {index_path}...")
    with open(index_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Read {len(lines)} lines.")
        record = json.loads(lines[0])
    
    audio_path = record["audio"]
    print(f"Target audio: {audio_path}")
    
    try:
        print("[TRACING] Calling wav_to_mel_debug...")
        mel = wav_to_mel_debug(audio_path)
        print(f"[TRACING] Success! Mel shape: {mel.shape}")
        
        print("[TRACING] Simulating text processing...")
        phonemes = record["phoneme_sequence"].split()
        print(f"[TRACING] Phonemes: {len(phonemes)}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_single_file()
