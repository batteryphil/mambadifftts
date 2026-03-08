import torch
from torch.utils.data import DataLoader
from multi_style_data import MultiStyleDataset, collate_multi
import time

TRAIN_INDEX = "data/audiobook_british_libritts_train.jsonl"
MEL_CACHE_DIR = "data/mel_cache"

def test_loader():
    print("Testing DataLoader...")
    dataset = MultiStyleDataset(TRAIN_INDEX, MEL_CACHE_DIR, augment=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_multi)
    
    start = time.time()
    it = iter(loader)
    print("Fetching first batch...")
    try:
        batch = next(it)
        print(f"Success! Batch keys: {batch.keys()}")
        print(f"Batch time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_loader()
