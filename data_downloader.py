import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset URLs & Config
# ---------------------------------------------------------------------------

DATASETS = {
    "ljspeech": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "dir": "data/ljspeech",
        "archive": "data/ljspeech.tar.bz2",
        "type": "tar.bz2"
    },
    "vctk": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        "dir": "data/vctk",
        "archive": "data/vctk.zip",
        "type": "zip",
        "speakers": ["p225", "p228", "p229", "p231", "p240"]
    },
    # Note: LibriTTS-R is distributed in parts. We use the 'train-clean-100' subset as requested.
    "libritts_r": {
        "url": "https://www.openslr.org/resources/141/train_clean_100.tar.gz",
        "dir": "data/libritts_r",
        "archive": "data/libritts_r_clean_100.tar.gz",
        "type": "tar.gz"
    },
    "crema_d": {
        "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip",
        "dir": "data/crema_d",
        "archive": "data/crema_d.zip",
        "type": "zip"
    }
}

DATA_ROOT = Path("data")

def _progress_hook(t: tqdm) -> callable:
    last_b = [0]
    def hook(b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return hook

def download_and_extract(name: str, cleanup: bool = False):
    info = DATASETS[name]
    target_dir = Path(info["dir"])
    archive_path = Path(info["archive"])

    DATA_ROOT.mkdir(exist_ok=True)
    
    any_missing_speaker = False
    if name == "vctk" and "speakers" in info:
        for spk in info["speakers"]:
            if not (target_dir / "wav48" / spk).exists() and not (target_dir / spk).exists():
                any_missing_speaker = True
                break
    
    if not target_dir.exists() or any_missing_speaker:
        if not archive_path.exists():
            print(f"Downloading {name} from {info['url']}...")
            with tqdm(unit="B", unit_scale=True, miniters=1, desc=name) as t:
                urllib.request.urlretrieve(info["url"], archive_path, reporthook=_progress_hook(t))
        else:
            # Check if file is still being downloaded or corrupted
            try:
                import zipfile
                if info["type"] == "zip":
                    with zipfile.ZipFile(archive_path, "r") as f:
                        f.testzip()
                print(f"Archive {archive_path} exists and is valid.")
            except:
                print(f"Archive {archive_path} is invalid or incomplete. Deleting and re-downloading...")
                archive_path.unlink()
                print(f"Downloading {name} from {info['url']}...")
                with tqdm(unit="B", unit_scale=True, miniters=1, desc=name) as t:
                    urllib.request.urlretrieve(info["url"], archive_path, reporthook=_progress_hook(t))
        
        print(f"Extracting {name} archive...")
        if info["type"] == "tar.bz2":
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(target_dir)
        elif info["type"] == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(target_dir)
        elif info["type"] == "zip":
            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                if name == "vctk" and "speakers" in info:
                    speakers = info["speakers"]
                    print(f"Surgically extracting {name} (speakers {', '.join(speakers)})...")
                    # Extract only specified speakers' wav and txt
                    members = [m for m in zip_ref.namelist() if any(f"/{spk}/" in m for spk in speakers) or "speaker-info.txt" in m]
                    zip_ref.extractall(target_dir, members=members)
                else:
                    zip_ref.extractall(target_dir)
        print(f"{name} ready at {target_dir}")
        
        if cleanup and archive_path.exists():
            print(f"Cleaning up archive {archive_path}...")
            archive_path.unlink()
    else:
        print(f"{name} already exists at {target_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mamba TTS Data Downloader")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--cleanup", action="store_true", help="Delete archive after extraction")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS:
            download_and_extract(ds, cleanup=args.cleanup)
    else:
        download_and_extract(args.dataset, cleanup=args.cleanup)
