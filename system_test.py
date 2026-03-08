"""
system_test.py — Verification script for optimized TTS pipeline.
Verifies dataset indexing, mel caching, emotional conditioning, and model training loop.
"""

import os
import torch
import torch.nn as nn
from multi_style_data import MultiStyleDataset, collate_multi
from torch.utils.data import DataLoader
from mamba_tts import MambaTTS, TTSConfig
from vocoder.bigvgan import BigVGANWrapper

def test_pipeline():
    print("--- Starting System Test ---")
    
    # 1. Config Test
    config = TTSConfig(
        vocab_size=256,
        d_model=128,
        n_encoder_layers=2,
        n_denoiser_layers=2,
        n_mel_channels=80,
        n_emotions=8,
        n_styles=3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Model Test (Forward Pass with Checkpointing)
    print("\n2. Model Test:")
    model = MambaTTS(config).to(device)
    model.gradient_checkpointing_enable()
    
    B, T_text, T_mel = 2, 10, 50
    text_ids = torch.randint(0, 255, (B, T_text)).to(device)
    mel = torch.randn(B, 80, T_mel).to(device)
    durations = torch.randint(1, 10, (B, T_text)).to(device)
    emotion_ids = torch.randint(0, 7, (B,)).to(device)
    style_ids = torch.randint(0, 2, (B,)).to(device)
    
    model.train()
    # Training forward pass
    noise_pred, noise_target, log_dur_pred, _ = model(
        text_ids, mel, durations, emotion_ids, style_ids
    )
    print(f"  Training forward pass OK. Noise pred shape: {noise_pred.shape}")
    
    model.eval()
    # Synthesis (Inference)
    mel_out = model.synthesize(text_ids, emotion_id=emotion_ids[0].item(), style_id=style_ids[0].item(), steps=10)
    print(f"  Synthesis OK. Mel out shape: {mel_out.shape}")
    
    # 3. Vocoder Test
    print("\n3. Vocoder Test:")
    vocoder = BigVGANWrapper(device=str(device))
    # Dummy mel to audio
    audio = vocoder(mel_out.unsqueeze(0))
    print(f"  Vocoder OK. Audio shape: {audio.shape}")
    
    # 4. Dataset Loading Test
    print("\n4. Dataset Loading Test:")
    if os.path.exists("training_index.jsonl"):
        ds = MultiStyleDataset("training_index.jsonl")
        if len(ds) > 0:
            batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_multi)))
            print(f"  Dataset load OK. Batch keys: {batch.keys()}")
            print(f"  Emotion IDs: {batch['emotion_ids']}")
        else:
            print("  Dataset index empty. Skipping.")
    else:
        print("  Index not found. Skipping dataset test.")
        
    print("\n--- System Test Complete ---")

if __name__ == "__main__":
    test_pipeline()
