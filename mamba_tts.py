"""
mamba_tts.py — Mamba Diffusion TTS Acoustic Model (Jarvisv5)

Architecture:
  Text → MambaEncoder → DurationPredictor → LengthRegulator
       → conditioning (B, L_mel, D)

  Training  (DDPM):
    1. Sample t ~ Uniform(0, T)
    2. Add noise: mel_noisy = sqrt(ᾱ_t)·mel + sqrt(1-ᾱ_t)·ε
    3. Predict: ε̂ = MambaDiffDenoiser(mel_noisy, conditioning, t)
    4. Loss: MSE(ε̂, ε)  +  α·MSE(log_dur_pred, log_dur_gt)

  Inference (DDIM, 50 steps):
    1. mel ~ N(0, I)
    2. For t in [T, T-T/50, ..., 0]: mel = DDIM_step(mel, ε̂, t)
    3. Return mel → HiFi-GAN → wav

Reuses MambaBlock from mamba_diffusion.py as the denoiser backbone.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_diffusion import MambaBlock


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TTSConfig:
    """Configuration for Mamba Diffusion TTS."""

    vocab_size:        int   = 256       # char vocabulary size
    d_model:           int   = 512       # hidden dimension
    n_encoder_layers:  int   = 4         # text encoder Mamba blocks
    n_denoiser_layers: int   = 6         # diffusion denoiser Mamba blocks
    n_mel_channels:    int   = 80        # mel filter banks
    max_text_len:      int   = 512
    max_mel_len:       int   = 1024
    dropout:           float = 0.1
    # Diffusion
    n_diff_steps:      int   = 1000      # DDPM training steps
    beta_start:        float = 1e-4
    beta_end:          float = 0.02
    ddim_steps:        int   = 50        # DDIM inference steps


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        """Init sinusoidal PE."""
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to x of shape (B, L, D)."""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Timestep Embedding (sinusoidal → MLP → D)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Converts integer diffusion timestep to a D-dimensional vector."""

    def __init__(self, d_model: int) -> None:
        """Init timestep embedding."""
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of scalar timesteps."""
        half = self.d_model // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer steps → (B, D) embedding."""
        return self.mlp(self._sinusoidal(t))


# ---------------------------------------------------------------------------
# Duration Predictor
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    """1-D CNN duration predictor (FastSpeech-style)."""

    def __init__(self, d_model: int, n_channels: int = 256, n_layers: int = 2) -> None:
        """Init duration predictor."""
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        in_ch = d_model
        for _ in range(n_layers):
            self.layers.append(nn.Conv1d(in_ch, n_channels, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(n_channels))
            in_ch = n_channels
        self.proj = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L) log-durations."""
        h = x.transpose(1, 2)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = layer(h)
        out = self.proj(h.transpose(1, 2)).squeeze(-1)
        return torch.clamp(out, min=-6.0, max=6.0)


# ---------------------------------------------------------------------------
# Length Regulator
# ---------------------------------------------------------------------------

class LengthRegulator(nn.Module):
    """Upsample encoder states by repeating per predicted duration."""

    def __init__(self) -> None:
        """Init (stateless)."""
        super().__init__()

    def forward(
        self,
        encoder_out: torch.Tensor,
        durations:   torch.Tensor,
        max_len:     Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        encoder_out : (B, L_text, D)
        durations   : (B, L_text) integer frame counts
        Returns (B, L_mel, D), actual_mel_len
        """
        outputs = []
        for b in range(encoder_out.size(0)):
            rep = torch.repeat_interleave(
                encoder_out[b], durations[b].clamp(min=0), dim=0
            )
            outputs.append(rep)

        if max_len is None:
            max_len = max(o.size(0) for o in outputs)

        padded = torch.zeros(
            encoder_out.size(0), max_len, encoder_out.size(2),
            device=encoder_out.device, dtype=encoder_out.dtype,
        )
        for b, o in enumerate(outputs):
            l = min(o.size(0), max_len)
            padded[b, :l] = o[:l]

        return padded, max_len


# ---------------------------------------------------------------------------
# Mamba Stack
# ---------------------------------------------------------------------------

class MambaStack(nn.Module):
    """N × MambaBlock with pre-LayerNorm + residual + dropout."""

    def __init__(self, d_model: int, n_layers: int, dropout: float = 0.1) -> None:
        """Init Mamba stack."""
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([
            nn.ModuleDict({
                "norm":  nn.LayerNorm(d_model),
                "mamba": MambaBlock(d_model),
                "drop":  nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)."""
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mamba"](x)
            x = layer["drop"](x) + residual
        return x


# ---------------------------------------------------------------------------
# Mamba Diffusion Denoiser
# ---------------------------------------------------------------------------

class MambaDiffDenoiser(nn.Module):
    """
    Denoiser: (noisy_mel, conditioning, t) → predicted noise ε̂

    At each Mamba layer:
      h ← LayerNorm(h + t_emb + cond)   # inject both signals
      h ← h + MambaBlock(h)
    """

    def __init__(self, d_model: int, n_mel: int, n_layers: int,
                 dropout: float = 0.1) -> None:
        """Init Mamba diffusion denoiser."""
        super().__init__()
        # Project mel frames (80-dim) into model space
        self.mel_proj   = nn.Linear(n_mel, d_model)
        self.time_emb   = TimestepEmbedding(d_model)
        self.mel_pe     = SinusoidalPE(d_model)

        self.layers: nn.ModuleList = nn.ModuleList([
            nn.ModuleDict({
                "norm":  nn.LayerNorm(d_model),
                "mamba": MambaBlock(d_model),
                "drop":  nn.Dropout(dropout),
            })
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj   = nn.Linear(d_model, n_mel)

    def forward(
        self,
        mel_noisy:    torch.Tensor,   # (B, n_mel, L_mel)
        conditioning: torch.Tensor,   # (B, L_mel, D)
        t:            torch.Tensor,   # (B,) integer timestep
    ) -> torch.Tensor:
        """Predict noise ε̂. Returns (B, n_mel, L_mel)."""
        # (B, n_mel, L_mel) → (B, L_mel, D)
        x = self.mel_proj(mel_noisy.transpose(1, 2))
        x = self.mel_pe(x)

        # Timestep embedding: (B, D) → (B, 1, D) for broadcasting
        t_emb = self.time_emb(t).unsqueeze(1)

        for layer in self.layers:
            # Inject time + conditioning at every layer (DiT-style)
            residual = x
            x = layer["norm"](x + t_emb + conditioning)
            x = layer["mamba"](x)
            x = layer["drop"](x) + residual

        x = self.final_norm(x)
        noise_pred = self.out_proj(x).transpose(1, 2)  # (B, n_mel, L_mel)
        return noise_pred


# ---------------------------------------------------------------------------
# DDPM / DDIM Scheduler
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    Linear-schedule DDPM forward process + DDIM reverse sampler.

    Training: forward_diffusion(x0, t) → (x_t, noise)
    Inference: ddim_sample(denoiser, conditioning, shape) → x0
    """

    def __init__(self, config: TTSConfig) -> None:
        """Build noise schedule buffers."""
        super().__init__()
        T    = config.n_diff_steps
        beta = torch.linspace(config.beta_start, config.beta_end, T)
        alpha          = 1.0 - beta
        alpha_bar      = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("beta",           beta)
        self.register_buffer("alpha",          alpha)
        self.register_buffer("alpha_bar",      alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",
                             torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))

        self.T          = T
        self.ddim_steps = config.ddim_steps

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to x0 at timestep t.

        x0 : (B, n_mel, L_mel) clean mel
        t  : (B,) integer in [0, T)
        Returns: x_t (noised), noise (target)
        """
        noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_oab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_oab * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser:     MambaDiffDenoiser,
        conditioning: torch.Tensor,       # (B, L_mel, D)
        shape:        Tuple,              # (B, n_mel, L_mel)
        device:       torch.device,
        temperature:  float = 1.0,
        eta:          float = 0.0,        # 0 = deterministic DDIM
    ) -> torch.Tensor:
        """
        DDIM reverse sampling.

        Returns clean mel estimate (B, n_mel, L_mel).
        """
        B = shape[0]

        # Build DDIM timestep sequence (evenly spaced)
        step = self.T // self.ddim_steps
        timesteps = list(range(0, self.T, step))[::-1]   # T-1 … 0

        x = torch.randn(shape, device=device) * temperature

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

            eps = denoiser(x, conditioning, t_batch)   # predicted noise

            ab_t  = self.alpha_bar[t_val]
            ab_s  = self.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # Predicted x0
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = torch.clamp(x0_pred, -15.0, 4.0)

            # DDIM update
            sigma    = eta * ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).sqrt()
            dir_xt   = (1 - ab_s - sigma ** 2).clamp(min=0).sqrt() * eps
            noise    = sigma * torch.randn_like(x) if eta > 0 else 0
            x        = ab_s.sqrt() * x0_pred + dir_xt + noise

        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class MambaDiffTTS(nn.Module):
    """
    Mamba Diffusion TTS acoustic model.

    Training forward: returns (noise_pred, log_dur_pred) for loss computation.
    Inference: call .synthesize(text_ids) → mel (B, n_mel, L_mel)
    """

    def __init__(self, config: TTSConfig = TTSConfig()) -> None:
        """Init MambaDiffTTS."""
        super().__init__()
        self.config = config
        D = config.d_model

        # --- Text Encoder ---
        self.text_embed = nn.Embedding(config.vocab_size, D, padding_idx=0)
        self.text_pe    = SinusoidalPE(D, max_len=config.max_text_len)
        self.encoder    = MambaStack(D, config.n_encoder_layers, config.dropout)

        # --- Duration ---
        self.dur_pred   = DurationPredictor(D)
        self.len_reg    = LengthRegulator()

        # --- Diffusion components ---
        self.diffusion  = GaussianDiffusion(config)
        self.denoiser   = MambaDiffDenoiser(
            d_model  = D,
            n_mel    = config.n_mel_channels,
            n_layers = config.n_denoiser_layers,
            dropout  = config.dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / zero-bias init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode(
        self,
        text_ids:     torch.Tensor,         # (B, L_text)
        mel_target:   Optional[torch.Tensor] = None,  # (B, n_mel, L_mel)
        durations_gt: Optional[torch.Tensor] = None,  # (B, L_text) int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        Encode text and produce conditioning via length regulator.

        Returns: (conditioning (B, L_mel, D), log_dur_pred (B, L_text), mel_len)
        """
        x       = self.text_embed(text_ids)
        x       = self.text_pe(x)
        enc_out = self.encoder(x)                      # (B, L_text, D)

        log_dur_pred = self.dur_pred(enc_out)          # (B, L_text)

        if durations_gt is not None:
            durs    = durations_gt
            max_len = mel_target.size(2) if mel_target is not None else None
        else:
            durs    = torch.clamp(
                torch.round(torch.exp(log_dur_pred) - 1).long(), min=1
            )
            max_len = None

        conditioning, mel_len = self.len_reg(enc_out, durs, max_len=max_len)
        return conditioning, log_dur_pred, mel_len

    def forward(
        self,
        text_ids:     torch.Tensor,
        mel_target:   torch.Tensor,          # (B, n_mel, L_mel)
        durations_gt: torch.Tensor,          # (B, L_text)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns:
            noise_pred   : (B, n_mel, L_mel)  predicted noise ε̂
            noise_target : (B, n_mel, L_mel)  actual sampled noise ε
            log_dur_pred : (B, L_text)
            mel_noisy    : (B, n_mel, L_mel)  for diagnostics
        """
        B = text_ids.size(0)
        conditioning, log_dur_pred, _ = self._encode(
            text_ids, mel_target, durations_gt
        )

        # Sample random timestep per batch item
        t = torch.randint(0, self.config.n_diff_steps, (B,), device=text_ids.device)

        # Forward diffusion: add noise
        mel_noisy, noise = self.diffusion.forward_diffusion(mel_target, t)

        # Predict noise
        noise_pred = self.denoiser(mel_noisy, conditioning, t)

        return noise_pred, noise, log_dur_pred, mel_noisy

    @torch.no_grad()
    def synthesize(
        self,
        text_ids:    torch.Tensor,          # (1, L_text)
        temperature: float = 1.0,
        eta:         float = 0.0,           # 0 = deterministic DDIM
    ) -> torch.Tensor:
        """
        Inference: text → mel via DDIM sampling.

        Returns (1, n_mel, L_mel) predicted mel spectrogram.
        """
        device = text_ids.device
        self.eval()

        conditioning, _, mel_len = self._encode(text_ids)
        shape = (text_ids.size(0), self.config.n_mel_channels, mel_len)

        mel = self.diffusion.ddim_sample(
            self.denoiser, conditioning, shape, device, temperature, eta
        )
        return mel

    def count_params(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiffTTSLoss(nn.Module):
    """
    DDPM noise-prediction MSE loss + duration MSE loss.
    """

    def __init__(self, dur_weight: float = 0.1) -> None:
        """Init loss. dur_weight controls duration loss contribution."""
        super().__init__()
        self.dur_weight = dur_weight

    def forward(
        self,
        noise_pred:   torch.Tensor,    # (B, n_mel, L_mel)
        noise_target: torch.Tensor,    # (B, n_mel, L_mel)
        log_dur_pred: torch.Tensor,    # (B, L_text)
        durations_gt: torch.Tensor,    # (B, L_text) int
        mel_mask:     Optional[torch.Tensor] = None,   # (B, L_mel) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, noise_loss, dur_loss)."""
        # Noise prediction MSE (main diffusion objective)
        noise_loss = F.mse_loss(noise_pred, noise_target, reduction="none")
        # Guard corrupt targets
        finite_mask = torch.isfinite(noise_target)
        noise_loss  = noise_loss * finite_mask.float()
        if mel_mask is not None:
            mask       = mel_mask.unsqueeze(1).float()   # (B, 1, L_mel)
            noise_loss = (noise_loss * mask).sum() / (mask.sum() * noise_pred.size(1) + 1e-8)
        else:
            noise_loss = noise_loss.mean()

        # Duration MSE in log space
        log_dur_gt = torch.log(durations_gt.float().clamp(min=1))
        dur_loss   = F.mse_loss(log_dur_pred, log_dur_gt)

        total = noise_loss + self.dur_weight * dur_loss
        return total, noise_loss, dur_loss


# ---------------------------------------------------------------------------
# Backward-compat aliases (so existing test scripts don't break)
# ---------------------------------------------------------------------------
MambaTTS = MambaDiffTTS
TTSLoss  = DiffTTSLoss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TTSConfig(
        d_model=256, n_encoder_layers=2, n_denoiser_layers=3,
        n_diff_steps=100, ddim_steps=10
    )
    model = MambaDiffTTS(cfg)
    print(f"MambaDiffTTS params: {model.count_params():,}")

    B, L_text, L_mel = 2, 20, 64
    text_ids    = torch.randint(1, 80, (B, L_text))
    durations   = torch.ones(B, L_text, dtype=torch.long) * (L_mel // L_text)
    mel_target  = torch.randn(B, 80, L_mel)

    noise_pred, noise, log_dur, mel_noisy = model(text_ids, mel_target, durations)
    print(f"noise_pred : {noise_pred.shape}")
    print(f"log_dur    : {log_dur.shape}")

    criterion = DiffTTSLoss()
    loss, nl, dl = criterion(noise_pred, noise, log_dur, durations)
    print(f"total={loss.item():.4f}  noise={nl.item():.4f}  dur={dl.item():.4f}")

    # DDIM inference
    mel_out = model.synthesize(text_ids[:1])
    print(f"synthesized mel: {mel_out.shape}")
    print("Sanity check passed ✓")
