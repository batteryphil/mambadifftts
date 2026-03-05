"""
download_vocoder.py — Download Pretrained HiFi-GAN Vocoder

Downloads the universal HiFi-GAN checkpoint (trained on LJSpeech)
from Hugging Face Hub using the official huggingface_hub library.

Run:
    python download_vocoder.py
"""

import os
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

VOCODER_DIR = Path("./vocoder")
REPO_ID = "jaketae/hifigan-lj-v1"

def download_vocoder():
    """Download HiFi-GAN generator and config from HF Hub."""
    print(f"Downloading HiFi-GAN from {REPO_ID}...")
    VOCODER_DIR.mkdir(exist_ok=True)
    
    # Download files
    try:
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
        model_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
        
        # Copy/Symlink to our local vocoder dir
        import shutil
        shutil.copy(config_path, VOCODER_DIR / "config.json")
        shutil.copy(model_path, VOCODER_DIR / "generator.pth")
        
        print(f"Vocoder ready at: {VOCODER_DIR.resolve()}")
    except Exception as e:
        print(f"Error downloading vocoder: {e}")
        # Fallback to another repo if nvidia is gated
        REPO_ID_FALLBACK = "kan-bayashi/ljspeech_hifigan.v1"
        print(f"Attempting fallback to {REPO_ID_FALLBACK}...")
        try:
             config_path = hf_hub_download(repo_id=REPO_ID_FALLBACK, filename="config.json")
             model_path = hf_hub_download(repo_id=REPO_ID_FALLBACK, filename="pytorch_model.bin")
             shutil.copy(config_path, VOCODER_DIR / "config.json")
             shutil.copy(model_path, VOCODER_DIR / "generator.pth")
             print(f"Vocoder ready (fallback) at: {VOCODER_DIR.resolve()}")
        except Exception as e2:
             print(f"Failed fallback as well: {e2}")
             raise

def load_hifigan(device="cpu"):
    """Load the HiFi-GAN generator."""
    from mamba_tts import SinusoidalPE # just to check if we can import our modules
    
    config_path = VOCODER_DIR / "config.json"
    weights_path = VOCODER_DIR / "generator.pth"
    
    if not config_path.exists() or not weights_path.exists():
        download_vocoder()
        
    return _load_hifigan_inline(config_path, weights_path, device)

def _load_hifigan_inline(config_path, weights_path, device):
    """Minimal HiFi-GAN generator implementation."""
    import torch.nn as nn
    import torch.nn.functional as F

    with open(config_path) as f:
        h = json.load(f)

    LRELU_SLOPE = 0.1

    class ResBlock(nn.Module):
        def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
            super().__init__()
            self.convs1 = nn.ModuleList([
                nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=d*(kernel_size-1)//2))
                for d in dilations
            ])
            self.convs2 = nn.ModuleList([
                nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size-1)//2))
                for _ in dilations
            ])
        def forward(self, x):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = F.leaky_relu(c1(xt), LRELU_SLOPE)
                xt = c2(xt)
                x = xt + x
            return x

    class Generator(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.num_kernels = len(h["resblock_kernel_sizes"])
            self.num_upsamples = len(h["upsample_rates"])
            self.conv_pre = nn.utils.weight_norm(nn.Conv1d(80, h["upsample_initial_channel"], 7, 1, padding=3))
            self.ups = nn.ModuleList([
                nn.utils.weight_norm(nn.ConvTranspose1d(h["upsample_initial_channel"] // (2**i), 
                                                     h["upsample_initial_channel"] // (2**(i+1)), 
                                                     k, u, padding=(k-u)//2))
                for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"]))
            ])
            ch_list = [h["upsample_initial_channel"] // (2**(i+1)) for i in range(self.num_upsamples)]
            self.resblocks = nn.ModuleList([
                ResBlock(ch, k, tuple(d))
                for ch in ch_list
                for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])
            ])
            self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch_list[-1], 1, 7, 1, padding=3))

        def forward(self, x):
            x = self.conv_pre(x)
            for i, up in enumerate(self.ups):
                x = F.leaky_relu(x, LRELU_SLOPE)
                x = up(x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i*self.num_kernels+j](x)
                    else:
                        xs += self.resblocks[i*self.num_kernels+j](x)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = torch.tanh(self.conv_post(x))
            return x

        def remove_weight_norm(self):
            nn.utils.remove_weight_norm(self.conv_pre)
            nn.utils.remove_weight_norm(self.conv_post)
            for layer in self.ups: nn.utils.remove_weight_norm(layer)
            for rb in self.resblocks:
                for c in rb.convs1: nn.utils.remove_weight_norm(c)
                for c in rb.convs2: nn.utils.remove_weight_norm(c)

    generator = Generator(h).to(device)
    state = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state.get("generator", state), strict=False)
    generator.eval()
    generator.remove_weight_norm()
    return generator

if __name__ == "__main__":
    download_vocoder()
