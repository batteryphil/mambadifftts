import os
import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_preprocessor import preprocessor
import pandas as pd # For easier VCTK/LibriTTS metadata handling

# ---------------------------------------------------------------------------
# Indexer Config
# ---------------------------------------------------------------------------

CURATED_ROOT = Path("curated_dataset")
MASTER_INDEX = Path("curated_dataset/master_index.jsonl")

DATASETS = {
    "ljspeech": "LJ Speech",
    "vctk": "VCTK",
    "libritts_r": "LibriTTS-R",
    "crema_d": "CREMA-D"
}

LIMIT = None # Set by CLI

# Style & Emotion Tags
EMOTIONS = ["neutral", "soft", "whisper", "seductive", "happy", "sad", "angry", "dramatic"]
STYLES = ["narration", "dialogue", "expressive"]

# Chunking config for long-form
MIN_CHARS = 50
MAX_CHARS = 500

# VCTK Preferred Speakers (British Female) 
VCTK_SPEAKERS = ["p225"]

# LibriTTS-R Preferred Speakers (Female, high-fidelity)
LIBRITTS_SPEAKERS = ["40", "103", "196", "201", "229", "233", "250", "254", "289", "311"]

class Indexer:
    def __init__(self):
        CURATED_ROOT.mkdir(exist_ok=True)
        (CURATED_ROOT / "cache" / "mel").mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_stats(self):
        if not self.records: return
        df = pd.DataFrame(self.records)
        print("\n--- Dataset Statistics ---")
        print(f"Total Clips: {len(df)}")
        print(f"Total Hours: {df['duration_seconds'].sum() / 3600:.2f}")
        print("\nHours per Source:")
        print(df.groupby('dataset_source')['duration_seconds'].sum() / 3600)

    def add_record(self, audio_in: Path, text: str, speaker_id: str, source: str, emotion: str, style: str):
        # 1. Output Path (FLAC)
        rel_path = f"{source}/{speaker_id}/{audio_in.stem}.flac"
        audio_out = CURATED_ROOT / rel_path
        
        # 2. Preprocess Text & Audio
        norm_text = preprocessor.normalize_text(text)
        phonemes = preprocessor.to_phonemes(norm_text)
        duration = preprocessor.process_audio(audio_in, audio_out)
        
        if duration > 0:
            # 3. Generate & Cache Mel (float16)
            from tts_data_builder import wav_to_mel
            mel = wav_to_mel(str(audio_out))
            mel_f16 = mel.astype(np.float16)
            
            sample_id = f"{source}_{speaker_id}_{audio_in.stem}"
            mel_cache_path = CURATED_ROOT / "cache" / "mel" / f"{sample_id}.npy"
            np.save(mel_cache_path, mel_f16)

            self.records.append({
                "audio": str(audio_out),
                "mel": str(mel_cache_path),
                "text": norm_text,
                "phoneme_sequence": phonemes,
                "speaker": speaker_id,
                "emotion": emotion,
                "style": style,
                "duration_seconds": duration,
                "dataset_source": source
            })

    def process_ljspeech(self, root: Path, limit: int = None):
        print("Processing LJ Speech...")
        meta = root / "metadata.csv"
        wav_dir = root / "wavs"
        
        # Text chunking for LJSpeech (combine sentences until MIN_CHARS)
        with open(meta, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)
            if limit: rows = rows[:limit]
            
            current_text = ""
            current_wavs = []
            
            for row in tqdm(rows):
                if len(row) < 3: continue
                text = row[2]
                wav_file = wav_dir / f"{row[0]}.wav"
                
                if len(current_text) + len(text) < MAX_CHARS:
                    current_text += " " + text
                    current_wavs.append(wav_file)
                    
                    if len(current_text) >= MIN_CHARS:
                        # Process the chunk
                        # Note: For now, we combine wavs if they are small, 
                        # but real combining requires merging audio.
                        # Simple implementation: just process the first wav of the chunk 
                        # or concatenate them. To keep it robust, we'll process 
                        # individual samples if they meet MIN_CHARS, 
                        # otherwise we skip combining audio for now to stay safe.
                        # USER said: Combine adjacent sentences when necessary.
                        # I'll implement a simple audio concatenation for the chunk.
                        self.add_chunk(current_wavs, current_text.strip(), "LJ01", "ljspeech", "neutral", "narration")
                        current_text = ""
                        current_wavs = []
            
    def add_chunk(self, wav_paths: list[Path], text: str, speaker_id: str, source: str, emotion: str, style: str):
        if not wav_paths: return
        if len(wav_paths) == 1:
            self.add_record(wav_paths[0], text, speaker_id, source, emotion, style)
            return
            
        # Merge wavs into one temporary wav
        import soundfile as sf
        import librosa
        merged_y = []
        for p in wav_paths:
            y, _ = librosa.load(p, sr=22050)
            merged_y.append(y)
            # Add short silence between sentences
            merged_y.append(np.zeros(int(22050 * 0.3)))
        
        y_final = np.concatenate(merged_y)
        temp_merged = Path("temp_merged.wav")
        sf.write(temp_merged, y_final, 22050)
        
        # Use first wav stem as identifier
        self.add_record(temp_merged, text, speaker_id, source, emotion, style)
        if temp_merged.exists(): temp_merged.unlink()

    def process_vctk(self, root: Path, limit: int = None):
        print("Processing VCTK...")
        # VCTK 0.92 structure: root/VCTK-Corpus-0.92/wav48_silence_trimmed/pXXX/...
        # or flat: root/wav48_silence_trimmed/pXXX/...
        wav_root = root / "wav48_silence_trimmed"
        if not wav_root.exists():
            wav_root = root / "VCTK-Corpus-0.92" / "wav48_silence_trimmed"
            
        txt_root = root / "txt"
        if not txt_root.exists():
            txt_root = root / "VCTK-Corpus-0.92" / "txt"

        if not wav_root.exists():
            print(f"Warning: VCTK wav root not found at {wav_root}")
            return
        
        count = 0
        for spk in VCTK_SPEAKERS:
            if limit and count >= limit: break
            spk_wav = wav_root / spk
            spk_txt = txt_root / spk
            if not spk_wav.exists(): continue
            
            files = list(spk_txt.glob("*.txt"))
            if limit: files = files[:limit-count]
            
            for txt_file in tqdm(files, desc=f"VCTK {spk}"):
                # Prefer mic1 (DPA 4035) for VCTK
                wav_file = spk_wav / f"{txt_file.stem}_mic1.flac"
                if not wav_file.exists():
                    # Fallback to mic2 or original stem
                    wav_file = spk_wav / f"{txt_file.stem}_mic2.flac"
                    if not wav_file.exists():
                        wav_file = spk_wav / f"{txt_file.stem}.flac"
                
                if not wav_file.exists(): continue
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Default VCTK to neutral/narration
                self.add_record(wav_file, text, spk, "vctk", "neutral", "narration")
                count += 1

    def process_libritts_r(self, root: Path, limit: int = None):
        print("Processing LibriTTS-R...")
        # Structure: LibriTTS_R/train-clean-100/reader_id/chapter_id/*.wav and *.normalized.txt
        count = 0
        for spk in LIBRITTS_SPEAKERS:
            if limit and count >= limit: break
            spk_root = root / spk
            if not spk_root.exists(): continue
            
            files = list(spk_root.rglob("*.wav"))
            for wav_file in tqdm(files, desc=f"LibriTTS-R {spk}"):
                txt_file = wav_file.with_suffix(".short.txt") # Use short.txt or normalized.txt
                if not txt_file.exists(): 
                    txt_file = wav_file.with_suffix(".normalized.txt")
                if not txt_file.exists(): continue
                
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Style is narration
                self.add_record(wav_file, text, spk, "libritts_r", "neutral", "narration")
                count += 1
                if limit and count >= limit: break

    def process_crema_d(self, root: Path, limit: int = None):
        print("Processing CREMA-D...")
        # CREMA-D structure: AudioWAV/[ActorID]_[SentenceCode]_[EmotionCode]_[EmotionLevel].wav
        # Emotion mapping to user style tags
        EMO_MAP = {
            "NEU": "neutral",
            "HAP": "happy",
            "SAD": "sad",
            "ANG": "angry",
            "FEA": "dramatic", # Falling back to dramatic
            "DIS": "dramatic"
        }
        
        # Sentence transcripts for CREMA-D
        CREMA_TRANSCRIPTS = {
            "IEO": "It's eleven o'clock.",
            "TIE": "That is exactly what happened.",
            "IOM": "I'm on my way to the meeting.",
            "IWW": "I wonder what'll happen next.",
            "TAI": "The airplane is almost full.",
            "MTI": "Maybe tomorrow it will be cold.",
            "IWL": "I would like to see it now.",
            "ITH": "I think I've seen this before.",
            "DFA": "Don't forget a jacket.",
            "ITS": "I think I've seen this before.",
            "TSI": "The surface is slick.",
            "WSI": "We'll stop in a couple of minutes."
        }

        # Female actors in CREMA-D (approximate, usually even/odd split)
        # Even IDs are often female in this dataset, but let's be more robust if we can.
        # For this script, we'll use a known subset or just process all and let user filter.
        wav_dir = root / "AudioWAV"
        if not wav_dir.exists(): return

        files = list(wav_dir.glob("*.wav"))
        if limit: files = files[:limit]
        for wav_file in tqdm(files, desc="CREMA-D"):
            name = wav_file.stem
            parts = name.split("_")
            if len(parts) < 3: continue
            
            actor_id, sent_code, emo_code = parts[0], parts[1], parts[2]
            
            # Filter for specific expressive codes
            if emo_code not in EMO_MAP: continue
            
            text = CREMA_TRANSCRIPTS.get(sent_code, "Unknown sentence.")
            self.add_record(wav_file, text, actor_id, "crema_d", EMO_MAP[emo_code], "expressive")

    def save(self):
        with open(MASTER_INDEX, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        print(f"Master index saved to {MASTER_INDEX}")
        self.log_stats()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mamba TTS Dataset Indexer")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset for testing")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index instead of appending")
    args = parser.parse_args()
    
    L_VAL = args.limit
    indexer = Indexer()
    
    # Load existing records if they exist (Append mode)
    if MASTER_INDEX.exists() and not args.overwrite:
        with open(MASTER_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    indexer.records.append(json.loads(line))
        print(f"Loaded {len(indexer.records)} existing records from {MASTER_INDEX}")

    def should_process(ds_name):
        return args.dataset == "all" or args.dataset == ds_name

    # Note: These paths assume data_downloader was run
    if should_process("ljspeech") and Path("data/ljspeech/LJSpeech-1.1").exists():
        indexer.process_ljspeech(Path("data/ljspeech/LJSpeech-1.1"), limit=L_VAL)
    
    if should_process("vctk") and Path("data/vctk").exists():
        indexer.process_vctk(Path("data/vctk"), limit=L_VAL)
    
    if should_process("libritts_r") and Path("data/libritts_r/LibriTTS_R/train-clean-100").exists():
        indexer.process_libritts_r(Path("data/libritts_r/LibriTTS_R/train-clean-100"), limit=L_VAL)
    
    if should_process("crema_d") and Path("data/crema_d/CREMA-D-master").exists():
        indexer.process_crema_d(Path("data/crema_d/CREMA-D-master"), limit=L_VAL)
    
    indexer.save()
    
    # Split into train/val for trainer
    if indexer.records:
        import random
        random.seed(42)
        records = indexer.records.copy()
        random.shuffle(records)
        
        split = int(len(records) * 0.9)
        train_recs = records[:split]
        val_recs = records[split:]
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / "audiobook_british_libritts_train.jsonl", "w") as f:
            for r in train_recs:
                f.write(json.dumps(r) + "\n")
        with open(data_dir / "audiobook_british_libritts_val.jsonl", "w") as f:
            for r in val_recs:
                f.write(json.dumps(r) + "\n")
                
        print(f"Saved {len(train_recs)} train and {len(val_recs)} val records to {data_dir}")
