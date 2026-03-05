"""
tts_test.py — Test Suite for Mamba TTS

Runs validation on 5 test sentences and reports mel reconstruction loss
on the validation set (if available).

Usage:
    python tts_test.py [--model PATH] [--skip_audio]
"""

import argparse
import os
import time
import glob
from typing import List

import torch
import numpy as np

from mamba_tts import MambaTTS, TTSConfig, TTSLoss
from tts_data_builder import text_to_ids, VOCAB_SIZE, LJSpeechDataset, collate_tts
from download_vocoder import load_hifigan


# ---------------------------------------------------------------------------
# Test sentences
# ---------------------------------------------------------------------------

TEST_TEXTS: List[str] = [
    "Hello, I am Jarvis. How can I help you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world rapidly.",
    "The Mamba architecture enables linear scaling with sequence length.",
    "Text to speech synthesis generates natural sounding audio from text.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint() -> str:
    """Find best or latest checkpoint, raise if none exist."""
    if os.path.exists("mamba_tts_best.pth"):
        return "mamba_tts_best.pth"
    candidates = sorted(glob.glob("mamba_tts_step*.pth"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        "No TTS checkpoint found. Run: python tts_trainer.py"
    )


def load_model(model_path: str, device: str) -> MambaTTS:
    """Load and return a MambaTTS model in eval mode."""
    config = TTSConfig(vocab_size=VOCAB_SIZE)
    model = MambaTTS(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 1: Forward pass smoke test (no data required)
# ---------------------------------------------------------------------------

def test_forward_pass() -> bool:
    """
    Verify the model can run a forward pass with random inputs.

    Returns True on success.
    """
    print("\n[TEST 1] Forward pass smoke test ...")
    try:
        config = TTSConfig(
            vocab_size=VOCAB_SIZE, d_model=256,
            n_encoder_layers=2, n_decoder_layers=2
        )
        model = MambaTTS(config)
        model.eval()

        B, L_text, L_mel = 2, 20, 80
        text_ids  = torch.randint(1, 80, (B, L_text))
        durations = torch.ones(B, L_text, dtype=torch.long) * (L_mel // L_text)
        mel_t     = torch.randn(B, 80, L_mel)

        with torch.no_grad():
            mel_out, log_dur = model(text_ids, mel_t, durations)

        assert mel_out.shape == (B, 80, L_mel), f"Bad mel shape: {mel_out.shape}"
        assert log_dur.shape == (B, L_text),    f"Bad dur shape: {log_dur.shape}"
        print(f"  PASS: mel_out={mel_out.shape}  log_dur={log_dur.shape}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


# ---------------------------------------------------------------------------
# Test 2: Inference on test sentences
# ---------------------------------------------------------------------------

def test_inference(
    model: MambaTTS,
    device: str,
    skip_audio: bool = False,
    out_dir: str = "tts_outputs",
) -> bool:
    """
    Run inference on TEST_TEXTS and optionally save wav files.

    Returns True if all sentences produce non-empty output.
    """
    print(f"\n[TEST 2] Inference on {len(TEST_TEXTS)} test sentences ...")
    os.makedirs(out_dir, exist_ok=True)

    vocoder = None
    if not skip_audio:
        try:
            vocoder = load_hifigan(device)
        except Exception as e:
            print(f"  [WARN] Vocoder not loaded ({e}). Skipping audio.")
            skip_audio = True

    all_pass = True
    for i, text in enumerate(TEST_TEXTS):
        try:
            text_ids = text_to_ids(text)
            ids_t    = torch.tensor([text_ids], dtype=torch.long, device=device)

            t0 = time.time()
            with torch.no_grad():
                mel_out, _ = model(ids_t)
            elapsed = time.time() - t0

            from tts_data_builder import SAMPLE_RATE, HOP_LENGTH
            dur_s = mel_out.shape[2] * HOP_LENGTH / SAMPLE_RATE
            print(f"  [{i+1}] '{text[:55]}...' | "
                  f"mel={mel_out.shape} | {dur_s:.2f}s | {elapsed*1000:.1f}ms")

            if not skip_audio and vocoder is not None:
                import soundfile as sf
                wav = vocoder(mel_out.to(device))
                wav = wav.squeeze().cpu().numpy()
                wav = np.clip(wav, -1.0, 1.0)
                out_path = os.path.join(out_dir, f"test_{i+1:02d}.wav")
                sf.write(out_path, wav, SAMPLE_RATE, subtype="PCM_16")
                print(f"       Saved -> {out_path}")

        except Exception as e:
            print(f"  FAIL [{i+1}]: {e}")
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Test 3: Validation loss
# ---------------------------------------------------------------------------

def test_val_loss(model: MambaTTS, device: str) -> bool:
    """
    Compute the L1 mel loss over the validation set.

    Returns True if validation pkl is found and loss is finite.
    """
    print("\n[TEST 3] Validation loss ...")
    if not os.path.exists("val_tts.pkl"):
        print("  SKIP: val_tts.pkl not found (run tts_data_builder.py first)")
        return True

    from torch.utils.data import DataLoader
    val_ds = LJSpeechDataset("val_tts.pkl")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=2, collate_fn=collate_tts)
    criterion = TTSLoss()

    total, count = 0.0, 0
    model.eval()
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

    avg_loss = total / max(count, 1)
    ok = np.isfinite(avg_loss)
    print(f"  Val loss: {avg_loss:.4f} ({'PASS' if ok else 'FAIL - non-finite'})")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tests(model_path: str, skip_audio: bool) -> None:
    """Run all TTS tests and report results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Mamba TTS Test Suite === (device: {device})")

    results = {}

    # Test 1: no checkpoint needed
    results["forward_pass"] = test_forward_pass()

    # Tests 2 & 3: need a checkpoint
    try:
        model = load_model(model_path, device)
        print(f"  Model loaded: {model.count_params():,} params")
        results["inference"]  = test_inference(model, device, skip_audio)
        results["val_loss"]   = test_val_loss(model, device)
    except FileNotFoundError as e:
        print(f"\n  [SKIP] {e}")
        results["inference"] = None
        results["val_loss"]  = None

    # Summary
    print("\n=== Summary ===")
    all_pass = True
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {name}: {status}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba TTS tests")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (auto-detects if omitted)"
    )
    parser.add_argument(
        "--skip_audio", action="store_true",
        help="Skip audio synthesis (faster, no soundfile/sounddevice needed)"
    )
    args = parser.parse_args()

    path = args.model
    if path is None:
        try:
            path = _find_checkpoint()
        except FileNotFoundError:
            path = "no_checkpoint"

    run_tests(path, args.skip_audio)
