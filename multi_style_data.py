import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tts_data_builder import wav_to_mel, compute_durations

# These should match dataset_indexer.py
EMOTIONS = ["neutral", "soft", "whisper", "seductive", "happy", "sad", "angry", "dramatic"]
STYLES = ["narration", "dialogue", "expressive"]

# Standard ARPAbet + Punctuation Tokens
PHONEMES = [
    "<pad>", " ", 
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
    "<PAUSE_SHORT>", "<PAUSE_LONG>", "<QUESTION>", "<EXCLAMATION>"
]
PHONE_TO_ID = {p: i for i, p in enumerate(PHONEMES)}
VOCAB_SIZE = len(PHONEMES)

class MultiStyleDataset(Dataset):
    def __init__(self, index_path: str, mel_cache_dir: str, augment: bool = False):
        self.records = []
        path = Path(index_path)
        if not path.exists():
            print(f"Dataset index not found: {path.absolute()}")
            return
            
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        self.mel_cache_dir = Path(mel_cache_dir)
        self.augment = augment
        print(f"MultiStyleDataset: Loaded {len(self.records)} records from {index_path}")

    def __len__(self):
        return len(self.records)

    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Subtle melspectrogram-level augmentations."""
        # mel: (80, T)
        aug_mel = mel.copy()
        
        # 1. Random noise
        if np.random.random() < 0.3:
            aug_mel += np.random.normal(0, 0.01, aug_mel.shape).astype(np.float32)
            
        # 2. Time masking (simple SpecAugment style)
        if np.random.random() < 0.3:
            t = aug_mel.shape[1]
            if t > 5:
                t0 = np.random.randint(0, t - 5)
                aug_mel[:, t0:t0+5] = -10.0 # mel floor
            
        return aug_mel

    def __getitem__(self, idx):
        r = self.records[idx]
        wav_path_str = r.get("audio") or r.get("audio_path")
        if not wav_path_str:
            raise KeyError(f"Record {idx} missing 'audio' or 'audio_path'")
        wav_path = Path(wav_path_str)
        
        # Load float16 cached mel
        mel_cache_path = self.mel_cache_dir / (wav_path.stem + ".npy")
        if mel_cache_path.exists():
            mel = np.load(mel_cache_path).astype(np.float32)
        else:
            mel = wav_to_mel(str(wav_path))
            
        if self.augment:
            mel = self._augment_mel(mel)
            
        # Use phoneme_sequence if available (highly preferred)
        if "phoneme_sequence" in r:
            tokens = r["phoneme_sequence"].split()
            text_ids = torch.tensor([PHONE_TO_ID.get(t, 0) for t in tokens], dtype=torch.long)
        else:
            # Fallback for character-level or dummy data
            text = r.get("text") or r.get("transcript") or ""
            text_ids = torch.tensor([ord(c) % VOCAB_SIZE for c in text], dtype=torch.long)
        
        # Emotion & Style IDs
        emo_id = EMOTIONS.index(r.get("emotion", "neutral"))
        sty_id = STYLES.index(r.get("style", "narration"))
        
        return {
            "mel": torch.from_numpy(mel),
            "text_ids": text_ids,
            "emotion_id": torch.tensor(emo_id, dtype=torch.long),
            "style_id": torch.tensor(sty_id, dtype=torch.long)
        }

def collate_multi(batch):
    max_text = max(b["text_ids"].size(0) for b in batch)
    max_mel = max(b["mel"].size(1) for b in batch)
    B = len(batch)
    n_mel = batch[0]["mel"].size(0)

    text_ids_p = torch.zeros(B, max_text, dtype=torch.long)
    durations_p = torch.zeros(B, max_text, dtype=torch.long)
    mel_p = torch.zeros(B, n_mel, max_mel)
    mel_mask = torch.zeros(B, max_mel, dtype=torch.bool)
    emotion_ids = torch.zeros(B, dtype=torch.long)
    style_ids = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        lt = b["text_ids"].size(0)
        lm = b["mel"].size(1)
        text_ids_p[i, :lt] = b["text_ids"]
        
        # Compute durations for padding
        durs = compute_durations(lt, lm)
        durations_p[i, :lt] = torch.tensor(durs, dtype=torch.long)
        
        mel_p[i, :, :lm] = b["mel"]
        mel_mask[i, :lm] = True
        emotion_ids[i] = b["emotion_id"]
        style_ids[i] = b["style_id"]

    return {
        "text_ids": text_ids_p,
        "mel": mel_p,
        "durations": durations_p,
        "mel_mask": mel_mask,
        "emotion_ids": emotion_ids,
        "style_ids": style_ids
    }
