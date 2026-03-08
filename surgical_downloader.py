import os
import shutil
from pathlib import Path
from datasets import load_dataset
import soundfile as sf

# Config
SPEAKERS = ["p225", "p228", "p229", "p231", "p240"]
TARGET_ROOT = Path("data/vctk")

def download_speaker_data():
    print(f"--- Surgical VCTK Downloader ---")
    print(f"Targeting speakers: {', '.join(SPEAKERS)}")
    
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    
    # We use streaming to avoid downloading the whole dataset
    # Note: 'vctk' dataset on HF often refers to 'vctk' or 'vctk-corpus'
    try:
        print("Loading VCTK dataset from Hugging Face (streaming mode, trust_remote_code=True)...")
        dataset = load_dataset("vctk", streaming=True, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading 'vctk'. Trying 'vctk-corpus' fallback... {e}")
        dataset = load_dataset("vctk-corpus", streaming=True, trust_remote_code=True)

    counts = {spk: 0 for spk in SPEAKERS}
    
    # Structure needed by indexer:
    # root/wav48_silence_trimmed/pXXX/pXXX_YYY_mic1.flac
    # root/txt/pXXX/pXXX_YYY.txt
    
    wav_dir = TARGET_ROOT / "wav48_silence_trimmed"
    txt_dir = TARGET_ROOT / "txt"
    
    for spk in SPEAKERS:
        (wav_dir / spk).mkdir(parents=True, exist_ok=True)
        (txt_dir / spk).mkdir(parents=True, exist_ok=True)

    print("Fetching samples...")
    for item in dataset:
        spk_id = item.get("speaker_id")
        if not spk_id:
            # Fallback for different field names
            spk_id = item.get("speaker")
            
        if spk_id in SPEAKERS:
            # Save audio
            audio = item["audio"]
            # audio['path'] is usually pXXX_YYY_mic1.flac
            # Or item['file_id']
            file_id = item.get("file_id") or Path(audio["path"]).stem
            
            # Ensure we use mic1 if possible
            if not file_id.endswith("_mic1"):
                # some datasets just have the stem
                pass
            
            wav_path = wav_dir / spk_id / f"{file_id}.flac"
            txt_path = txt_dir / spk_id / f"{file_id.replace('_mic1', '')}.txt"
            
            if not wav_path.exists():
                sf.write(wav_path, audio["array"], audio["sampling_rate"])
                
            if not txt_path.exists() and "text" in item:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(item["text"])
            
            counts[spk_id] += 1
            if sum(counts.values()) % 50 == 0:
                print(f"Progress: {counts}")
        
    print(f"Finished! Total samples: {counts}")

if __name__ == "__main__":
    download_speaker_data()
