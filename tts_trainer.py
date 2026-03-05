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

from mamba_tts import MambaTTS, TTSConfig, TTSLoss
from tts_data_builder import LJSpeechDataset, collate_tts, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = TTSConfig(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_mel_channels=80,
    dropout=0.1,
)

LR            = 1e-3       # base LR per user rules
LR_MIN        = 1e-5
WEIGHT_DECAY  = 0.01       # AdamW WD per user rules
BATCH_SIZE    = 16
MAX_EPOCHS    = 200
CKPT_EVERY    = 200        # steps per user rules
LOG_EVERY     = 50         # steps per user rules
KEEP_CKPTS    = 3          # rolling window of saved checkpoints
TRAIN_PKL     = "train_tts.pkl"
VAL_PKL       = "val_tts.pkl"
CKPT_PREFIX   = "mamba_tts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: MambaTTS, step: int, tag: str = "") -> str:
    """Save model weights to a checkpoint file and prune old ones."""
    name = f"{CKPT_PREFIX}_step{step:07d}{tag}.pth"
    torch.save(model.state_dict(), name)
    # Prune old checkpoints
    ckpts = sorted(glob.glob(f"{CKPT_PREFIX}_step*.pth"))
    for old in ckpts[:-KEEP_CKPTS]:
        try:
            os.remove(old)
        except OSError:
            pass
    return name


def load_latest_checkpoint(model: MambaTTS, resume_path: Optional[str] = None) -> int:
    """
    Load weights from resume_path, or the latest checkpoint if found.

    Returns the step number extracted from the filename (0 if fresh start).
    """
    path = resume_path
    if path is None:
        candidates = sorted(glob.glob(f"{CKPT_PREFIX}_step*.pth"))
        path = candidates[-1] if candidates else None

    if path and os.path.exists(path):
        print(f"  -> Resuming from {path}")
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        # Extract step from filename
        try:
            return int(Path(path).stem.split("step")[1][:7])
        except (IndexError, ValueError):
            return 0
    print("  -> No checkpoint found. Starting fresh.")
    return 0


def compute_val_loss(
    model: MambaTTS,
    criterion: TTSLoss,
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
    if not os.path.exists(TRAIN_PKL):
        raise FileNotFoundError(
            f"{TRAIN_PKL} not found. Run python tts_data_builder.py first."
        )
    train_ds = LJSpeechDataset(TRAIN_PKL)
    val_ds   = LJSpeechDataset(VAL_PKL) if os.path.exists(VAL_PKL) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_tts,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_tts,
        )
        if val_ds else None
    )
    print(f"  -> {len(train_ds)} train | {len(val_ds) if val_ds else 0} val samples")

    # --- Model ---
    model = MambaTTS(DEFAULT_CONFIG).to(device)
    start_step = load_latest_checkpoint(model, resume_path)
    print(f"  -> {model.count_params():,} trainable parameters")

    # --- Optimizer + Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = MAX_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR_MIN
    )
    # Fast-forward scheduler to resume step
    for _ in range(start_step):
        scheduler.step()

    criterion = TTSLoss(dur_weight=1.0)
    stats = {"train_loss": [], "val_loss": [], "step": start_step}

    # --- Training ---
    model.train()
    global_step = start_step
    best_val    = float("inf")
    t0          = time.time()

    for epoch in range(MAX_EPOCHS):
        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                print(f"  -> max_steps={max_steps} reached. Stopping.")
                _save_stats(stats)
                return

            text_ids  = batch["text_ids"].to(device)
            mel       = batch["mel"].to(device)
            durations = batch["durations"].to(device)
            mel_mask  = batch["mel_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            mel_pred, log_dur = model(text_ids, mel, durations)
            loss, mel_l, dur_l = criterion(mel_pred, mel, log_dur, durations, mel_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            stats["step"] = global_step
            stats["train_loss"].append(loss.item())

            # --- Logging ---
            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]
                print(
                    f"Step {global_step:7d} | "
                    f"loss={loss.item():.4f} mel={mel_l.item():.4f} "
                    f"dur={dur_l.item():.4f} | "
                    f"lr={lr_now:.2e} | "
                    f"{elapsed:.1f}s"
                )
                t0 = time.time()

            # --- Checkpoint ---
            if global_step % CKPT_EVERY == 0:
                ckpt_name = save_checkpoint(model, global_step)
                print(f"  ** Checkpoint saved: {ckpt_name}")

                if val_loader:
                    val_loss = compute_val_loss(model, criterion, val_loader, device)
                    stats["val_loss"].append({"step": global_step, "loss": val_loss})
                    print(f"  ** Val loss: {val_loss:.4f}")

                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(model.state_dict(), f"{CKPT_PREFIX}_best.pth")
                        print(f"  ** New best val={best_val:.4f} -> {CKPT_PREFIX}_best.pth")

                _save_stats(stats)

        epoch_avg = (
            sum(stats["train_loss"][-len(train_loader):]) / len(train_loader)
            if stats["train_loss"] else 0.0
        )
        print(f"=== Epoch {epoch + 1}/{MAX_EPOCHS} | avg_loss={epoch_avg:.4f} ===")

    # Final save
    save_checkpoint(model, global_step, tag="_final")
    print("Training complete.")


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
