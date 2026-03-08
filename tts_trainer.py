"""
tts_trainer.py — Training Script for Mamba TTS (Jarvisv5 conventions)

Usage:
    python tts_trainer.py [--max_steps N] [--resume PATH]

Follows Jarvis-v5 training conventions:
  - AdamW + cosine LR decay
  - clip_grad_norm_ (max_norm=1.0)
  - Checkpoint every 200 steps
  - Log every 50 steps
"""

import argparse
import json
import os
import time
import glob
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mamba_tts import MambaTTS, TTSConfig, DiffTTSLoss, GaussianDiffusion
import numpy as np
from multi_style_data import MultiStyleDataset, collate_multi, EMOTIONS, STYLES, VOCAB_SIZE
# from tts_data_builder import VOCAB_SIZE # Switched to phoneme vocab


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = TTSConfig(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    n_encoder_layers=4,
    n_denoiser_layers=6, # Updated to match config
    n_mel_channels=80,
    dropout=0.1,
    n_emotions=8,
    n_styles=3
)

LR            = 1e-3       # base LR per user rules
LR_MIN        = 1e-5
WEIGHT_DECAY  = 0.01       # AdamW WD per user rules
BATCH_SIZE    = 16
MAX_EPOCHS    = 1500
CKPT_EVERY    = 200        # steps per user rules
LOG_EVERY     = 50         # steps per user rules
KEEP_CKPTS    = 3          # rolling window of saved checkpoints
TRAIN_INDEX = "data/audiobook_british_libritts_train.jsonl"
VAL_INDEX = "data/audiobook_british_libritts_val.jsonl"
MEL_CACHE_DIR = "data/mel_cache"
CKPT_PREFIX   = "mamba_tts_p225"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: MambaTTS, optimizer: optim.Optimizer, scheduler, step: int, tag: str = "") -> str:
    """Save model weights, optimizer, and scheduler state."""
    name = f"checkpoints/step_{step:07d}{tag}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step
    }
    torch.save(state, name)
    
    # Prune old checkpoints
    ckpts = sorted(glob.glob("checkpoints/step_*.pt"))
    for old in ckpts[:-KEEP_CKPTS]:
        try:
            os.remove(old)
        except OSError:
            pass
    return name


def load_latest_checkpoint(model: MambaTTS, optimizer: optim.Optimizer = None, scheduler = None, resume_path: Optional[str] = None) -> int:
    """Load full state from checkpoint."""
    path = resume_path
    if path is None:
        candidates = sorted(glob.glob("checkpoints/step_*.pt"))
        path = candidates[-1] if candidates else None

    if path and os.path.exists(path):
        print(f"  -> Resuming from {path}")
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        return state.get("step", 0)
    
    print("  -> No checkpoint found. Starting fresh.")
    return 0


def compute_val_loss(
    model: MambaTTS,
    criterion: DiffTTSLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Run one pass over validation set and return mean loss."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            text_ids  = batch["text_ids"].to(device)
            mel       = batch["mel"].to(device)
            durations = batch["durations"].to(device)
            mel_mask  = batch["mel_mask"].to(device)

            mel_pred, log_dur = model(text_ids, mel, durations)
            loss, _, _ = criterion(mel_pred, mel, log_dur, durations, mel_mask)
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(max_steps: Optional[int] = None, resume_path: Optional[str] = None) -> None:
    """Main training function for Mamba TTS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Mamba TTS Trainer] Device: {device}")

    # --- Data ---
    if not os.path.exists(TRAIN_INDEX):
        raise FileNotFoundError(
            f"{TRAIN_INDEX} not found. Run python dataset_indexer.py first."
        )
    train_ds = MultiStyleDataset(TRAIN_INDEX, MEL_CACHE_DIR, augment=True)
    val_ds   = MultiStyleDataset(VAL_INDEX, MEL_CACHE_DIR, augment=False) if os.path.exists(VAL_INDEX) else None

    # Opt: Increase num_workers for fast data preloading
    num_workers = 4 if os.name == "nt" else min(8, os.cpu_count() or 4)

    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_multi,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_multi,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        if val_ds else None
    )
    print(f"  -> {len(train_ds)} train | {len(val_ds) if val_ds else 0} val samples")

    # --- Model ---
    model = MambaTTS(DEFAULT_CONFIG).to(device)
    model.gradient_checkpointing_enable()
    # --- Optimizer + Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = MAX_EPOCHS * len(train_loader)

    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=total_steps, eta_min=LR_MIN
    # )
    # Simpler scheduler for now
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    start_step = load_latest_checkpoint(model, optimizer, scheduler, resume_path)
    print(f"  -> {model.count_params():,} trainable parameters")

    criterion = DiffTTSLoss(dur_weight=1.0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    if os.path.exists("tts_training_stats.json"):
        try:
            with open("tts_training_stats.json", "r") as f:
                stats = json.load(f)
            stats["step"] = start_step # Ensure step is correctly synced
            if "vram_usage" not in stats: stats["vram_usage"] = []
        except Exception:
            stats = {"train_loss": [], "val_loss": [], "vram_usage": [], "step": start_step}
    else:
        stats = {"train_loss": [], "val_loss": [], "vram_usage": [], "step": start_step}
    _save_stats(stats) # Initialize file
    accumulation_steps = 8 # Optimized for 12GB VRAM

    # --- Training ---
    model.train()
    global_step = start_step
    t0          = time.time()

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(MAX_EPOCHS):
        for i, batch in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                break

            text_ids    = batch["text_ids"].to(device)
            mel_target  = batch["mel"].to(device)
            durations   = batch["durations"].to(device)
            emotion_ids = batch["emotion_ids"].to(device)
            style_ids   = batch["style_ids"].to(device)
            mel_mask    = batch.get("mel_mask")
            if mel_mask is not None: mel_mask = mel_mask.to(device)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                noise_pred, noise, log_dur, _ = model(
                    text_ids, mel_target, durations, emotion_ids, style_ids
                )
                loss, n_loss, d_loss = criterion(
                    noise_pred, noise, log_dur, durations, mel_mask
                )
                loss = loss / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step() # Move here to fix warning
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                # Log more frequently at the start of a run/restart (every 5 steps for 20 steps, then LOG_EVERY)
                curr_log_every = 5 if (global_step - start_step) <= 20 or global_step <= 50 else LOG_EVERY
                if global_step % curr_log_every == 0:
                    train_loss_val = loss.item() * accumulation_steps
                    stats["train_loss"].append(train_loss_val)
                    
                    if device.type == "cuda":
                        vram = torch.cuda.memory_reserved(device) / 1024**3 # GB
                        stats["vram_usage"].append(vram)
                        
                    stats["step"] = global_step
                    _save_stats(stats)
                    
                    dt = time.time() - t0
                    print(f"Step {global_step} | loss={train_loss_val:.4f} | dt={dt:.2f}s")
                    t0 = time.time()

                # Checkpointing & Validation
                if global_step % CKPT_EVERY == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step)
                    
                    if val_loader:
                        val_loss = compute_val_loss(model, criterion, val_loader, device)
                        print(f"  -> Validation Loss: {val_loss:.4f}")
                        
                        stats["val_loss"].append({"step": global_step, "loss": val_loss})
                        _save_stats(stats)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(model, optimizer, scheduler, global_step, tag="_best")
                            print(f"  *** New Best Model Saved ***")
                        
                        # Save a mel preview for the monitor
                        try:
                            with torch.no_grad():
                                # Get one sample from val_loader for preview
                                sample = next(iter(val_loader))
                                m_ids = sample["text_ids"][:1].to(device)
                                e_ids = sample["emotion_ids"][:1].to(device)
                                s_ids = sample["style_ids"][:1].to(device)
                                mel_pred = model.synthesize(m_ids, e_ids, s_ids)
                                preview_np = mel_pred[0].cpu().numpy()
                                np.save(f"tts_mel_preview_step{global_step}.npy", preview_np)
                        except Exception as e:
                            print(f"  -> Preview failed: {e}")
                            
                        model.train()
        
        # scheduler.step() moved to inside the accumulation loop to fix PyTorch warning
    
    print(f"  -> Training complete. Final step: {global_step}")
    save_checkpoint(model, optimizer, scheduler, global_step, tag="_final")

def _save_stats(stats: dict) -> None:
    """Write training statistics to JSON."""
    with open("tts_training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mamba TTS")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N steps (for testing)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(max_steps=args.max_steps, resume_path=args.resume)
