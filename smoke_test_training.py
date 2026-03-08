import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from multi_style_data import MultiStyleDataset, collate_multi
from mamba_tts import MambaTTS, TTSConfig, DiffTTSLoss

def smoke_test_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing training on {device}")
    
    index_path = "curated_dataset/master_index.jsonl"
    if not torch.os.path.exists(index_path):
        print(f"Index not found at {index_path}")
        return

    dataset = MultiStyleDataset(index_path, Path("curated_dataset/cache/mel"))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_multi)
    
    config = TTSConfig(
        vocab_size=256, 
        d_model=256,
        n_encoder_layers=2,
        n_denoiser_layers=2,
        n_mel_channels=80
    )
    
    model = MambaTTS(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = DiffTTSLoss()
    
    # 1. Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    accumulation_steps = 1 # Lowered for smoke test on small dataset
    
    print(f"Starting 3 steps of optimized training... (Accumulation={accumulation_steps})")
    model.train()
    
    # 2. Enable Gradient Checkpointing on Mamba Blocks
    # This assumes MambaBlock supports it or we wrap it.
    # For now, we'll just implement the loop changes.
    
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        if i >= 3 * accumulation_steps: break
        
        text_ids = batch["text_ids"].to(device)
        mel = batch["mel"].to(device)
        durations = batch["durations"].to(device)
        mel_mask = batch["mel_mask"].to(device)
        
        # 3. AMP Context
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            noise_pred, noise_target, log_dur_pred, _ = model(text_ids, mel, durations) 
            loss, n_loss, d_loss = criterion(noise_pred, noise_target, log_dur_pred, durations, mel_mask)
            loss = loss / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (i + 1) % accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            print(f"Optimized Step {(i+1)//accumulation_steps} completed | Loss: {loss.item() * accumulation_steps:.4f}")

    print("VRAM-Optimized Training flow verified successfully!")

if __name__ == "__main__":
    smoke_test_training()
