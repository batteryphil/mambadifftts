"""
tts_inference.py — Mamba TTS Inference Pipeline

Converts text to speech using:
  1. Mamba TTS acoustic model (text → mel spectrogram)
  2. HiFi-GAN vocoder (mel → audio waveform)

Usage:
    python tts_inference.py --text "Hello, I am Jarvis." --out output.wav
    python tts_inference.py --text "Hello world"           # plays back
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mamba_tts import MambaDiffTTS, TTSConfig
from tts_data_builder import text_to_ids, VOCAB_SIZE, SAMPLE_RATE
from download_vocoder import load_hifigan


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "mamba_tts_best.pth"
DEFAULT_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_acoustic_model(
    model_path: Optional[str] = None,
    device: str = DEFAULT_DEVICE,
) -> MambaTTS:
    """
    Load the MambaTTS acoustic model from checkpoint.

    Falls back to best.pth, then the latest step checkpoint.
    """
    if model_path is None:
        if os.path.exists(DEFAULT_MODEL_PATH):
            model_path = DEFAULT_MODEL_PATH
        else:
            candidates = sorted(Path(".").glob("mamba_tts_step*.pth"))
            if candidates:
                model_path = str(candidates[-1])
            else:
                raise FileNotFoundError(
                    "No Mamba TTS checkpoint found. "
                    "Train first: python tts_trainer.py"
                )

    config = TTSConfig(
        vocab_size=VOCAB_SIZE,
        n_diff_steps=1000,
        ddim_steps=50,
    )
    model = MambaDiffTTS(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"  Acoustic model (Mamba Diffusion) loaded from: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Text → Mel
# ---------------------------------------------------------------------------

@torch.no_grad()
def text_to_mel(
    model: MambaDiffTTS,
    text:  str,
    device: str   = DEFAULT_DEVICE,
    ddim_steps: int = 50,
    eta: float    = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Run DDIM synthesis on input text.

    Returns mel spectrogram: (1, n_mels, T).
    ddim_steps : number of denoising steps (50 = fast, 200 = higher quality)
    eta        : 0 = deterministic DDIM, 1 = DDPM (stochastic)
    """
    text_ids = text_to_ids(text)
    if not text_ids:
        raise ValueError(f"Empty text after preprocessing: {repr(text)}")

    ids_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)

    # Override ddim_steps at inference time if desired
    model_raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_raw.config.ddim_steps = ddim_steps

    mel_out = model_raw.synthesize(
        ids_tensor, temperature=temperature, eta=eta
    )
    return mel_out   # (1, 80, T)


# ---------------------------------------------------------------------------
# Mel → Audio
# ---------------------------------------------------------------------------

@torch.no_grad()
def mel_to_wav(
    vocoder,
    mel: torch.Tensor,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Run HiFi-GAN vocoder to convert mel spectrogram to waveform.

    mel: (1, n_mels, T)
    Returns: numpy float32 array of shape (num_samples,)
    """
    mel = mel.to(device)
    wav = vocoder(mel)          # (1, 1, num_samples)
    wav = wav.squeeze().cpu().numpy()
    # Clamp to [-1, 1] range
    wav = np.clip(wav, -1.0, 1.0)
    return wav


# ---------------------------------------------------------------------------
# Save / Playback
# ---------------------------------------------------------------------------

def save_wav(wav: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 audio array to a .wav file via soundfile."""
    import soundfile as sf
    sf.write(path, wav, sample_rate, subtype="PCM_16")
    print(f"  Saved audio -> {path}")


def play_wav(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play audio via sounddevice (blocking)."""
    try:
        import sounddevice as sd
        print(f"  Playing {len(wav) / sample_rate:.2f}s of audio ...")
        sd.play(wav, samplerate=sample_rate)
        sd.wait()
    except ImportError:
        print("  sounddevice not installed; skipping playback.")


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    out_path: Optional[str] = None,
    model_path: Optional[str] = None,
    play: bool = True,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> np.ndarray:
    """
    Full TTS pipeline: text → DDIM diffusion → HiFi-GAN → audio.

    Args:
        text       : Input string to synthesize.
        out_path   : If given, save .wav to this path.
        model_path : Optional path to acoustic model checkpoint.
        play       : Whether to play back audio via sounddevice.
        ddim_steps : DDIM reverse diffusion steps (50 = fast, 200 = higher quality).
        eta        : 0 = deterministic DDIM, 1 = stochastic DDPM-like.
    """
    device = DEFAULT_DEVICE
    print(f"[Mamba Diffusion TTS] Text   : '{text}'")
    print(f"[Mamba Diffusion TTS] Device : {device}")
    print(f"[Mamba Diffusion TTS] DDIM steps: {ddim_steps}  eta={eta}")

    acoustic = load_acoustic_model(model_path, device)
    vocoder  = load_hifigan(device)

    print("  Step 1/2: Text → Mel (DDIM) ...")
    mel = text_to_mel(acoustic, text, device, ddim_steps=ddim_steps, eta=eta)
    print(f"  Mel shape: {mel.shape}  ({mel.shape[2] * 256 / SAMPLE_RATE:.2f}s)")

    print("  Step 2/2: Mel → Waveform (HiFi-GAN) ...")
    wav = mel_to_wav(vocoder, mel, device)
    print(f"  Waveform: {len(wav)} samples @ {SAMPLE_RATE} Hz")

    if out_path:
        save_wav(wav, out_path)
    if play:
        play_wav(wav)

    return wav


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mamba TTS: convert text to speech"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Input text to synthesize"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output .wav file path (optional)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to acoustic model checkpoint"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50,
        help="DDIM denoising steps (50=fast, 200=higher quality)"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM stochasticity (0=deterministic, 1=DDPM)"
    )
    parser.add_argument(
        "--no_play", action="store_true",
        help="Disable audio playback"
    )
    args = parser.parse_args()

    synthesize(
        text=args.text,
        out_path=args.out,
        model_path=args.model,
        play=not args.no_play,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
    )
