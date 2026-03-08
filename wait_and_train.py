import time
import os
import subprocess
import sys
from pathlib import Path

def is_process_running(name_fragment):
    try:
        output = subprocess.check_output(['tasklist'], text=True)
        return name_fragment.lower() in output.lower()
    except:
        return False

def main():
    print("--- Mamba TTS Background Automation Waiter ---")
    print("Waiting for background 'data_downloader.py' to finish...")
    
    # Simple polling for the background process
    # We look for 'python' processes since the user started it via cmd/powershell
    # This is a bit coarse but works for this specific setup.
    
    speakers = ["p225", "p228", "p229", "p231", "p240"]
    archive_path = Path("data/vctk.zip")
    target_dir = Path("data/vctk")
    
    while True:
        # Check for all speakers
        all_present = True
        for spk in speakers:
            spk_path = target_dir / "wav48_silence_trimmed" / spk
            if not spk_path.exists():
                # Fallback check for alternate VCTK-Corpus-0.92 structure
                spk_path_alt = target_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed" / spk
                if not spk_path_alt.exists():
                    all_present = False
                    break
        
        if all_present:
            print(f"Detection: All {len(speakers)} speakers found. Proceeding to Indexing.")
            break
            
        time.sleep(60) # Poll every 60s
        
    # Stage 2: Indexing & Preprocessing
    # Use --overwrite to ensure we rebuild with the full speaker set
    print("Starting Indexing...")
    subprocess.run(["python", "dataset_indexer.py", "--dataset", "vctk", "--overwrite"], check=True)
    
    # Stage 3: Training
    print("Starting Training...")
    subprocess.run(["python", "tts_trainer.py"], check=True)

if __name__ == "__main__":
    main()
