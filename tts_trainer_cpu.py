"""
tts_trainer_cpu.py — CPU-Optimized Training Script for Mamba TTS

CPU optimizations applied:
  1. torch.set_num_threads → use all physical cores
  2. torch.autocast(cpu, bfloat16) → ~2x throughput on modern CPUs with AVX2
  3. torch.compile(model, backend='inductor') → kernel fusion + loop tiling
  4. Compiled mamba_scan C++ extension (OpenMP + SIMD via setup.py)
  5. GradScaler-free training (bf16 has enough dynamic range for L1+MSE loss)
  6. Gradient accumulation to simulate larger batches without memory pressure
  7. Pin-memory=False, persistent_workers=True for CPU DataLoader
  8. torch.backends.mkldnn enabled (oneDNN/MKL-DNN for matmuls)

Usage:
    python tts_trainer_cpu.py [--max_steps N] [--resume PATH]
"""

import argparse
import json
import os
import time
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# CPU backend tweaks — set BEFORE any model work
os.environ.setdefault("OMP_NUM_THREADS",      str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS",      str(os.cpu_count() or 4))
os.environ.setdefault("KMP_BLOCKTIME",        "1")      # minimize thread sleep
os.environ.setdefault("KMP_AFFINITY",         "granularity=fine,compact,1,0")
os.environ.setdefault("TORCH_COMPILE_DEBUG",  "0")

import torch
torch.set_flush_denormal(True) # Fix infinite CPU hangs caused by denormal float hardware traps in C++ extension loops
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = True

from mamba_tts import MambaDiffTTS, TTSConfig, DiffTTSLoss
from tts_data_builder import LJSpeechDataset, collate_tts, VOCAB_SIZE


# ---------------------------------------------------------------------------
# EMA Utility
# ---------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, beta=0.9999):
        self.beta = beta
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Keep shadow on CPU to avoid device mismatch
                self.shadow[name] = param.data.clone().to('cpu')

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    # theta_ema = beta * theta_ema + (1 - beta) * theta_current
                    self.shadow[name].copy_(
                        self.beta * self.shadow[name] + (1.0 - self.beta) * param.data.to('cpu')
                    )

# ---------------------------------------------------------------------------
# Hyperparameters (CPU-tuned)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = TTSConfig(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    n_encoder_layers=4,
    n_denoiser_layers=6,
    n_mel_channels=80,
    dropout=0.1,
    n_diff_steps=1000,
    ddim_steps=50,
)

LR              = 2e-5       # Dropped for Transfer Learning (Jenny)
LR_MIN          = 2e-6
WEIGHT_DECAY    = 0.01
BATCH_SIZE      = 1            # CPU: small batches, compensate with grad accum
GRAD_ACCUM      = 1            # effective batch = 4 * 8 = 32
MAX_EPOCHS      = 200
CKPT_EVERY      = 200
LOG_EVERY       = 1
KEEP_CKPTS      = 3
TRAIN_PKL       = "train_jenny.pkl"
VAL_PKL         = "val_jenny.pkl"
CKPT_PREFIX     = "mamba_tts"
USE_BF16        = False        # Disable bfloat16 autocast on CPU due to known hangs
USE_COMPILE     = False        # torch.compile (disabled to avoid CPU hang)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: MambaTTS, step: int, tag: str = "", ema_dict: dict = None) -> str:
    """Save model checkpoint and prune old ones."""
    name = f"{CKPT_PREFIX}_step{step:07d}{tag}.pth"
    # Save in float32 always (unwrap from compile if needed)
    
    if ema_dict is not None:
        torch.save(ema_dict, name)
    else:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(raw.state_dict(), name)
    ckpts = sorted(glob.glob(f"{CKPT_PREFIX}_step*.pth"))
    for old in ckpts[:-KEEP_CKPTS]:
        try:
            os.remove(old)
        except OSError:
            pass
    return name


def load_latest_checkpoint(model: MambaTTS, resume_path: Optional[str] = None) -> int:
    """Load weights from checkpoint. Returns step number."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    path = resume_path
    if path is None:
        candidates = sorted(glob.glob(f"{CKPT_PREFIX}_step*.pth"))
        path = candidates[-1] if candidates else None

    if path and os.path.exists(path):
        print(f"  -> Resuming from {path}")
        state = torch.load(path, map_location="cpu")
        raw.load_state_dict(state, strict=False)
        try:
            return int(Path(path).stem.split("step")[1][:7])
        except (IndexError, ValueError):
            return 0
    print("  -> No checkpoint found. Starting fresh.")
    return 0


def compute_val_loss(
    model: MambaDiffTTS,
    criterion: DiffTTSLoss,
    val_loader: DataLoader,
    ema: Optional[EMA] = None,
) -> float:
    """Run validation pass and return mean total loss. Evaluates using EMA weights if provided."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # Temporarily swap to EMA weights for validation if available
    train_weights = {}
    if ema is not None:
        for name, param in raw.named_parameters():
            if param.requires_grad and name in ema.shadow:
                train_weights[name] = param.data.clone()
                param.data.copy_(ema.shadow[name])

    raw.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            text_ids  = batch["text_ids"]
            mel       = batch["mel"]
            durations = batch["durations"]
            mel_mask  = batch["mel_mask"]

            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=USE_BF16):
                noise_pred, noise_target, log_dur, _ = raw(text_ids, mel, durations)
                loss, _, _ = criterion(noise_pred, noise_target, log_dur, durations, mel_mask)
            total += loss.item()
            count += 1

    # Restore training weights
    if ema is not None:
        for name, param in raw.named_parameters():
            if param.requires_grad and name in train_weights:
                param.data.copy_(train_weights[name])

    raw.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(max_steps: Optional[int] = None, resume_path: Optional[str] = None) -> None:
    """Main CPU-optimized training loop."""
    print(f"[Mamba TTS CPU Trainer]")
    print(f"  Threads   : {torch.get_num_threads()} (MKL={torch.backends.mkl.is_available()}, "
          f"oneDNN={torch.backends.mkldnn.enabled})")
    print(f"  bf16      : {USE_BF16}")
    print(f"  compile   : {USE_COMPILE}")

    # --- Data ---
    if not os.path.exists(TRAIN_PKL):
        raise FileNotFoundError(f"{TRAIN_PKL} not found. Run: python tts_data_builder.py")

    train_ds = LJSpeechDataset(TRAIN_PKL)
    val_ds   = LJSpeechDataset(VAL_PKL) if os.path.exists(VAL_PKL) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        # persistent_workers=True, # Not compatible with num_workers=0
        collate_fn=collate_tts,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            # persistent_workers=True,
            collate_fn=collate_tts,
        )
        if val_ds else None
    )
    print(f"  Data  : {len(train_ds)} train | {len(val_ds) if val_ds else 0} val")
    print(f"  Batch : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")

    # --- Model ---
    model = MambaDiffTTS(DEFAULT_CONFIG)

    # torch.compile for kernel fusion on CPU — uses 'inductor' backend
    # which tiles and fuses the MambaBlock matmuls via C++ codegen
    if USE_COMPILE and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile(backend='inductor') ...")
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
        print("  Compile graph ready.")

    start_step = load_latest_checkpoint(model, resume_path)
    raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
    print(f"  Params: {raw_model.count_params():,}")

    # --- EMA Initialization ---
    ema = EMA(beta=0.9999)
    ema.register(raw_model)
    print("  EMA: initialized and synced to starting weights.")

    # --- Optimizer ---
    optimizer = optim.AdamW(
        raw_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    
    # 🎯 Reset Optimizer Momentum for Transfer Learning
    # We loaded the weights from LJSpeech, but we DO NOT want its old Adam momentum
    # dragging us away from the new Jenny acoustic targets.
    optimizer.state.clear()
    
    total_steps = MAX_EPOCHS * (max(1, len(train_loader)) // GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=LR_MIN
    )
    for _ in range(start_step):
        scheduler.step()

    criterion = DiffTTSLoss(dur_weight=0.1)
    stats = {"train_loss": [], "val_loss": [], "step": start_step}

    # --- Train ---
    model.train()
    global_step = start_step
    best_val    = float("inf")
    t0          = time.time()
    accum_loss  = 0.0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(MAX_EPOCHS):
        for micro_step, batch in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                _save_stats(stats)
                print(f"max_steps={max_steps} reached.")
                return

            text_ids  = batch["text_ids"]
            mel       = batch["mel"]
            durations = batch["durations"]
            mel_mask  = batch["mel_mask"]

            # bf16 autocast — ~2× speedup on modern Intel/AMD with AVX2
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=USE_BF16):
                noise_pred, noise_target, log_dur, _ = model(text_ids, mel, durations)
                loss, noise_l, dur_l = criterion(
                    noise_pred, noise_target, log_dur, durations, mel_mask
                )
                loss = loss / GRAD_ACCUM

            loss.backward()
            # Guard: skip bad micro-steps instead of poisoning the accumulator
            if not torch.isfinite(loss):
                print(f"  [WARN] micro {micro_step}: NaN/inf loss={loss.item():.4f} — skipping")
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                continue
            accum_loss += loss.item()

            # Optimizer step every GRAD_ACCUM micro-batches
            if (micro_step + 1) % GRAD_ACCUM == 0:
                # Skip optimizer step if any gradient is non-finite
                bad_grads = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in raw_model.parameters()
                )
                if bad_grads:
                    print(f"  [WARN] Step {global_step}: non-finite grads — skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    continue
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=0.5)
                optimizer.step()
                ema.update(raw_model)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                stats["step"] = global_step
                norm_loss = accum_loss * GRAD_ACCUM   # un-normalize for logging
                stats["train_loss"].append(norm_loss)
                accum_loss = 0.0

                # --- Log ---
                if global_step % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    lr_now  = scheduler.get_last_lr()[0]
                    samples_sec = (LOG_EVERY * BATCH_SIZE * GRAD_ACCUM) / max(elapsed, 1e-6)
                    print(
                        f"Step {global_step:7d} | "
                        f"loss={norm_loss:.4f} noise={noise_l.item():.4f} dur={dur_l.item():.4f} | "
                        f"lr={lr_now:.2e} | "
                        f"{samples_sec:.1f} samples/s | "
                        f"{elapsed:.1f}s"
                    )
                    t0 = time.time()

                # --- Save Stats for Monitor UI (every LOG_EVERY steps) ---
                if global_step % LOG_EVERY == 0:
                    _save_stats(stats)

                # --- Checkpoint ---
                if global_step % CKPT_EVERY == 0:
                    ckpt = save_checkpoint(model, global_step, ema_dict=ema.shadow)
                    print(f"  ** Checkpoint: {ckpt}")

                    if val_loader:
                        vl = compute_val_loss(model, criterion, val_loader, ema=ema)
                        stats["val_loss"].append({"step": global_step, "loss": vl})
                        print(f"  ** Val loss : {vl:.4f}")
                        if vl < best_val:
                            best_val = vl
                            torch.save(ema.shadow, f"{CKPT_PREFIX}_best.pth")
                            print(f"  ** Best val  -> {CKPT_PREFIX}_best.pth")
                            
                    _save_stats(stats)

                # --- Mel preview for monitor (every 500 steps) ---
                if global_step % 500 == 0:
                    try:
                        with torch.no_grad():
                            # Save noise_pred as proxy for mel preview (80, T)
                            _mel = noise_pred[0].float().detach().cpu().numpy()
                        npy_name = f"tts_mel_preview_step{global_step:07d}.npy"
                        np.save(npy_name, _mel)
                        # Keep only the 2 most recent previews
                        for old_npy in sorted(glob.glob("tts_mel_preview_step*.npy"))[:-2]:
                            os.remove(old_npy)
                    except Exception:
                        pass

        avg = (
            sum(stats["train_loss"][-max(1, len(train_loader) // GRAD_ACCUM):])
            / max(1, len(train_loader) // GRAD_ACCUM)
        )
        print(f"=== Epoch {epoch + 1}/{MAX_EPOCHS} | avg_loss={avg:.4f} ===")

    save_checkpoint(model, global_step, tag="_final", ema_dict=ema.shadow)
    print("Training complete.")


def _save_stats(stats: dict) -> None:
    """Persist training stats to JSON."""
    with open("tts_training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-optimized Mamba TTS training")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N optimizer steps (for testing)")
    parser.add_argument("--resume",    type=str, default=None,
                        help="Checkpoint path to resume from")
    args = parser.parse_args()
    train(max_steps=args.max_steps, resume_path=args.resume)
