"""
tts_data_builder.py — LJSpeech Dataset Builder for Mamba TTS

Downloads LJSpeech-1.1, preprocesses text/audio pairs into mel spectrograms
and phoneme IDs, then saves train_tts.pkl / val_tts.pkl.

Run:
    python tts_data_builder.py
"""

import os
import re
import csv
import math
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import librosa
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LJSPEECH_URL = (
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
)
DATA_DIR = Path("./data/ljspeech")
ARCHIVE   = Path("data/ljspeech.tar.bz2")
TRAIN_OUT = Path("train_tts.pkl")
VAL_OUT   = Path("val_tts.pkl")

JENNY_META_URL = "https://huggingface.co/datasets/reach-vb/jenny_tts_dataset/resolve/main/metadata.csv"
JENNY_DATA_DIR = Path("./data/jenny_tts")
JENNY_TRAIN_OUT = Path("train_jenny.pkl")
JENNY_VAL_OUT   = Path("val_jenny.pkl")

# Audio config
SAMPLE_RATE = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
N_MELS      = 80
FMIN        = 0.0
FMAX        = 8000.0

# Split config
VAL_FRACTION = 0.05
MAX_SAMPLES  = 13100   # full LJSpeech; reduce for quick tests


# ---------------------------------------------------------------------------
# Text → character IDs
# ---------------------------------------------------------------------------

# Simple character-level vocabulary (printable ASCII subset)
PAD_ID    = 0
CHAR_MAP: Dict[str, int] = {c: i + 1 for i, c in enumerate(
    " !\"'()*+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)}
VOCAB_SIZE = len(CHAR_MAP) + 1   # +1 for PAD


def text_to_ids(text: str) -> List[int]:
    """Convert a string to a list of integer character IDs."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z !\"'()*+,-./:;?]", "", text)
    return [CHAR_MAP.get(c, PAD_ID) for c in text]


# ---------------------------------------------------------------------------
# Mel extraction
# ---------------------------------------------------------------------------

def wav_to_mel(wav_path: str) -> np.ndarray:
    """
    Load a wav file and compute a log-mel spectrogram.

    Returns shape: (n_mels, time_frames)
    """
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0,
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _progress_hook(t: tqdm) -> callable:
    """Create a urllib download progress hook for tqdm."""
    last_b = [0]

    def hook(b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return hook


def download_ljspeech() -> None:
    """Download and extract LJSpeech-1.1 if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATA_DIR / "LJSpeech-1.1").exists():
        if not ARCHIVE.exists():
            print(f"Downloading LJSpeech-1.1 (~2.6 GB) from {LJSPEECH_URL} ...")
            with tqdm(unit="B", unit_scale=True, miniters=1,
                      desc="ljspeech.tar.bz2") as t:
                urllib.request.urlretrieve(
                    LJSPEECH_URL, ARCHIVE, reporthook=_progress_hook(t)
                )
        print("Extracting archive ...")
        with tarfile.open(ARCHIVE, "r:bz2") as tar:
            tar.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print(f"LJSpeech already present at {DATA_DIR / 'LJSpeech-1.1'}.")


# ---------------------------------------------------------------------------
# Compute durations from text length and mel length
# ---------------------------------------------------------------------------

def compute_durations(n_text: int, n_mel: int) -> np.ndarray:
    """
    Build a simple uniform duration array: distribute mel frames evenly
    across text tokens. Returns int array of shape (n_text,) summing to n_mel.
    """
    base = n_mel // n_text
    durations = np.full(n_text, base, dtype=np.int32)
    remainder = n_mel - base * n_text
    durations[:remainder] += 1
    return durations


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset() -> None:
    """
    Read LJSpeech metadata, compute mel spectrograms, tokenize text,
    compute durations, and save train/val pickle files.
    """
    download_ljspeech()

    lj_root = DATA_DIR / "LJSpeech-1.1"
    wav_dir  = lj_root / "wavs"
    meta_csv = lj_root / "metadata.csv"

    # Read metadata
    samples: List[Tuple[str, str]] = []
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            fileid, _, normalized = row[0], row[1], row[2]
            wav_path = str(wav_dir / f"{fileid}.wav")
            if os.path.exists(wav_path):
                samples.append((wav_path, normalized.strip()))

    samples = samples[:MAX_SAMPLES]
    print(f"Processing {len(samples)} samples ...")

    records = []
    n_skip = 0
    for wav_path, text in tqdm(samples, desc="Extracting mels"):
        text_ids = text_to_ids(text)
        if len(text_ids) < 2:
            n_skip += 1
            continue
        try:
            mel = wav_to_mel(wav_path)   # (80, T)
        except Exception as e:
            print(f"  [WARN] Skipping {wav_path}: {e}")
            n_skip += 1
            continue

        n_text = len(text_ids)
        n_mel  = mel.shape[1]
        if n_mel < n_text:
            n_skip += 1
            continue

        durations = compute_durations(n_text, n_mel)
        records.append({
            "text_ids":  np.array(text_ids, dtype=np.int32),
            "mel":       mel,
            "durations": durations,
        })

    print(f"Built {len(records)} records | {n_skip} skipped.")

    # Split
    val_cut = max(1, int(len(records) * VAL_FRACTION))
    import random
    random.shuffle(records)
    val_records   = records[:val_cut]
    train_records = records[val_cut:]

    with open(TRAIN_OUT, "wb") as f:
        pickle.dump(train_records, f)
    with open(VAL_OUT, "wb") as f:
        pickle.dump(val_records, f)

    print(
        f"Saved {len(train_records)} train → {TRAIN_OUT} | "
        f"{len(val_records)} val → {VAL_OUT}"
    )
    print(f"Vocab size (char-level): {VOCAB_SIZE}")


def build_jenny_dataset() -> None:
    """
    Read Jenny metadata, compute mel spectrograms, tokenize text,
    compute durations, and save train/val pickle files specifically for Jenny.
    """
    JENNY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_csv = JENNY_DATA_DIR / "metadata.csv"

    if not meta_csv.exists():
        print(f"Downloading Jenny metadata from {JENNY_META_URL} ...")
        urllib.request.urlretrieve(JENNY_META_URL, meta_csv)
    
    wav_dir = JENNY_DATA_DIR / "wavs"
    if not wav_dir.exists():
        print("[WARN] Jenny wavs directory not found! You must manually download and extract the dataset audio to `data/jenny_tts/wavs/`")
        print("Dataset available at: https://www.languagereactor.com/dataset.tar.zst")
        # Proceed anyway so it can fail gracefully locally if no wavs exist

    # Read metadata
    samples: List[Tuple[str, str]] = []
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            fileid, _, normalized = row[0], row[1], row[2]
            # Ensure fileid has .wav extension if omitted
            if not fileid.endswith(".wav"):
                fileid = f"{fileid}.wav"
            wav_path = str(wav_dir / fileid)
            if os.path.exists(wav_path):
                samples.append((wav_path, normalized.strip()))

    samples = samples[:MAX_SAMPLES]
    print(f"Processing {len(samples)} Jenny samples ...")

    records = []
    n_skip = 0
    for wav_path, text in tqdm(samples, desc="Extracting Jenny mels"):
        text_ids = text_to_ids(text)
        if len(text_ids) < 2:
            n_skip += 1
            continue
        try:
            mel = wav_to_mel(wav_path)   # (80, T)
        except Exception as e:
            n_skip += 1
            continue

        n_text = len(text_ids)
        n_mel  = mel.shape[1]
        if n_mel < n_text:
            n_skip += 1
            continue

        durations = compute_durations(n_text, n_mel)
        records.append({
            "text_ids":  np.array(text_ids, dtype=np.int32),
            "mel":       mel,
            "durations": durations,
        })

    print(f"Built {len(records)} records | {n_skip} skipped.")
    if len(records) == 0:
        print("ERROR: No valid records built. Check if the audio exists in data/jenny_tts/wavs/")
        return

    # Split
    val_cut = max(1, int(len(records) * VAL_FRACTION))
    import random
    random.shuffle(records)
    val_records   = records[:val_cut]
    train_records = records[val_cut:]

    with open(JENNY_TRAIN_OUT, "wb") as f:
        pickle.dump(train_records, f)
    with open(JENNY_VAL_OUT, "wb") as f:
        pickle.dump(val_records, f)

    print(
        f"Saved {len(train_records)} train → {JENNY_TRAIN_OUT} | "
        f"{len(val_records)} val → {JENNY_VAL_OUT}"
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LJSpeechDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset over preprocessed LJSpeech pickle files.

    Each item yields:
        text_ids  : LongTensor (L_text,)
        mel       : FloatTensor (n_mels, L_mel)
        durations : LongTensor  (L_text,)
    """

    def __init__(self, pkl_path: str) -> None:
        """Load preprocessed records from pickle file."""
        with open(pkl_path, "rb") as f:
            self.records = pickle.load(f)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        """Return text_ids, mel, and durations for sample at idx."""
        r = self.records[idx]
        return {
            "text_ids":  torch.tensor(r["text_ids"],  dtype=torch.long),
            "mel":       torch.tensor(r["mel"],        dtype=torch.float32),
            "durations": torch.tensor(r["durations"], dtype=torch.long),
        }


def collate_tts(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of variable-length samples to equal length.

    Returns dict with keys: text_ids, mel, durations, mel_mask.
    """
    max_text = max(b["text_ids"].size(0) for b in batch)
    max_mel  = max(b["mel"].size(1)      for b in batch)
    B        = len(batch)
    n_mel    = batch[0]["mel"].size(0)

    text_ids_p  = torch.zeros(B, max_text, dtype=torch.long)
    durations_p = torch.zeros(B, max_text, dtype=torch.long)
    mel_p       = torch.zeros(B, n_mel,    max_mel)
    mel_mask    = torch.zeros(B, max_mel,  dtype=torch.bool)

    for i, b in enumerate(batch):
        lt = b["text_ids"].size(0)
        lm = b["mel"].size(1)
        text_ids_p[i,  :lt]    = b["text_ids"]
        durations_p[i, :lt]    = b["durations"]
        mel_p[i,       :, :lm] = b["mel"]
        mel_mask[i,    :lm]    = True

    return {
        "text_ids":  text_ids_p,
        "mel":       mel_p,
        "durations": durations_p,
        "mel_mask":  mel_mask,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mamba TTS Dataset Builder")
    parser.add_argument("--dataset", type=str, choices=["LJSpeech", "Jenny"], default="LJSpeech",
                        help="Which dataset to build: LJSpeech or Jenny")
    args = parser.parse_args()

    if args.dataset == "Jenny":
        build_jenny_dataset()
    else:
        build_dataset()
