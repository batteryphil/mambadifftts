import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Config
REPO_ID = "vctk/vctk" # Commonly used VCTK repo on HF
SPEAKERS = ["p225", "p228", "p229", "p231", "p240"]
TARGET_ROOT = Path("data/vctk")

def download_speaker_data():
    print(f"--- Surgical VCTK Downloader v2 ---")
    print(f"Targeting speakers: {', '.join(SPEAKERS)}")
    
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    
    # We want to download:
    # wav48_silence_trimmed/pXXX/*.flac
    # txt/pXXX/*.txt
    # speaker-info.txt
    
    allow_patterns = [
        "speaker-info.txt",
    ]
    for spk in SPEAKERS:
        allow_patterns.append(f"wav48_silence_trimmed/{spk}/*.flac")
        allow_patterns.append(f"txt/{spk}/*.txt")

    print(f"Starting snapshot download from {REPO_ID}...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=TARGET_ROOT,
            allow_patterns=allow_patterns,
            token=False # Changed from True to allow anonymous download
        )
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading from {REPO_ID}: {e}")
        # Try fallback repo if needed
        FALLBACK_REPO = "vctk/vctk-corpus" 
        print(f"Trying fallback: {FALLBACK_REPO}...")
        snapshot_download(
            repo_id=FALLBACK_REPO,
            repo_type="dataset",
            local_dir=TARGET_ROOT,
            allow_patterns=allow_patterns,
            token=False
        )

if __name__ == "__main__":
    download_speaker_data()
