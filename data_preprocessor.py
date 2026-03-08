import os
import re
import librosa
import soundfile as sf
import pyloudnorm as logn
import numpy as np
from pathlib import Path
from num2words import num2words
from g2p_en import G2p
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Preprocessing Config
# ---------------------------------------------------------------------------

TARGET_SR = 22050
TARGET_LUFS = -23.0
MAX_SILENCE_MS = 200
MIN_DURATION = 2.0
MAX_DURATION = 15.0

class Preprocessor:
    def __init__(self):
        self.g2p = G2p()
        self.meter = logn.Meter(TARGET_SR)

    def normalize_text(self, text: str) -> str:
        """Cleans and normalizes text for TTS with punctuation tokens."""
        text = text.lower().strip()
        # Normalize numbers
        text = re.sub(r"(\d+)", lambda m: num2words(int(m.group(0))), text)
        
        # Punctuation to tokens
        text = text.replace("...", " <pause_long> ")
        text = text.replace("?", " <question> ")
        text = text.replace("!", " <exclamation> ")
        # Add short pause for commas
        text = text.replace(",", " <pause_short> ")
        
        # Remove unusual punctuation, keep basic ones
        text = re.sub(r"[^a-z0-9 <_>']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def to_phonemes(self, text: str) -> str:
        """Converts text to ARPAbet phonemes, preserving special tokens."""
        # G2p usually handles <... tokens as individual characters if not careful.
        # We'll split by spaces and only g2p word-like tokens.
        tokens = text.split()
        result = []
        for t in tokens:
            if t.startswith("<") and t.endswith(">"):
                result.append(t.upper()) # Keep <PAUSE_LONG> etc.
            else:
                ph = self.g2p(t)
                result.extend([p for p in ph if p.strip()])
        return " ".join(result)

    def process_audio(self, input_path: Path, output_path: Path) -> float:
        """Resamples, normalizes loudness, trims silence, and saves as FLAC."""
        y, sr = librosa.load(input_path, sr=TARGET_SR)
        
        # 1. Trim leading/trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        # 2. Re-pad with fixed silence
        pad_samples = int(TARGET_SR * (MAX_SILENCE_MS / 1000.0))
        y_padded = np.concatenate([np.zeros(pad_samples), y_trimmed, np.zeros(pad_samples)])
        
        duration = len(y_padded) / TARGET_SR
        if duration < MIN_DURATION or duration > MAX_DURATION:
            return -1.0
            
        # 3. Normalize loudness
        loudness = self.meter.integrated_loudness(y_padded)
        y_norm = logn.normalize.loudness(y_padded, loudness, TARGET_LUFS)
        
        # 4. Check for NaNs or clipping
        if np.isnan(y_norm).any() or np.isinf(y_norm).any():
            return -1.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use soundfile to save as FLAC directly.
        # This is more efficient than calling ffmpeg via subprocess.
        try:
            sf.write(output_path, y_norm, TARGET_SR, format='FLAC', subtype='PCM_16')
        except Exception as e:
            print(f"Warning: soundfile FLAC write failed: {e}. Falling back to default.")
            sf.write(output_path, y_norm, TARGET_SR)
        
        return duration

preprocessor = Preprocessor()
