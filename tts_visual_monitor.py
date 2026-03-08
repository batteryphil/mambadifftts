"""
tts_visual_monitor.py — Cyberpunk TTS Training Monitor

Reads tts_training_stats.json (written by tts_trainer_cpu.py) and displays:
  Panel 1 (top-left)  : Train loss curve + val loss points
  Panel 2 (top-right) : Live stats readout (step, loss breakdown, speed)
  Panel 3 (bottom-left): Smoothed mel loss trend (last 200 steps)
  Panel 4 (bottom-right): Latest mel spectrogram preview (if saved)

Refreshes every 5 seconds. Run from the Jarvisv5 directory.

Usage:
    python tts_visual_monitor.py
"""

import json
import os
import time
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headless-free on most Linux setups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATS_FILE   = "tts_training_stats.json"
MEL_GLOB     = "tts_mel_preview_step*.npy"   # saved by trainer every 500 steps
REFRESH_MS   = 5000  # poll every 5 seconds

# Cyberpunk palette
BG_DARK      = "#0A0A0A"
BG_PANEL     = "#111111"
CYAN         = "#00FFCC"
MAGENTA      = "#FF007F"
YELLOW       = "#FFE000"
GREY         = "#888888"
WHITE        = "#FFFFFF"
DIM          = "#333333"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _smooth(arr: list, window: int = 20) -> list:
    """Simple moving-average smoothing for a loss curve."""
    if len(arr) < 2:
        return arr
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window)
        out.append(float(np.mean(arr[lo: i + 1])))
    return out


def _load_stats() -> dict:
    """Load tts_training_stats.json; return empty dict on failure."""
    if not os.path.exists(STATS_FILE):
        return {}
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, PermissionError):
        return {}


def _latest_mel() -> np.ndarray | None:
    """Load the most recent mel preview npy file, if any."""
    files = sorted(glob.glob(MEL_GLOB))
    if not files:
        return None
    try:
        return np.load(files[-1])   # shape (80, T)
    except Exception:
        return None


def _elapsed_str(seconds: float) -> str:
    """Format seconds into h:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# ---------------------------------------------------------------------------
# Animation callback
# ---------------------------------------------------------------------------

_start_time = time.time()


def animate(frame: int,
            ax_loss: plt.Axes,
            ax_stats: plt.Axes,
            ax_trend: plt.Axes,
            ax_mel: plt.Axes) -> None:
    """Matplotlib FuncAnimation callback — called every REFRESH_MS ms."""
    stats        = _load_stats()
    train_loss   = stats.get("train_loss",  [])
    val_records  = stats.get("val_loss",    [])    # list of {"step":N,"loss":F}
    vram_usage   = stats.get("vram_usage",  [])
    current_step = stats.get("step",        0)
    elapsed      = time.time() - _start_time

    # Store the first global step we see to calculate session-relative speed
    if not hasattr(animate, "initial_step"):
        animate.initial_step = current_step

    # ------------------------------------------------------------------ #
    # Panel 1 — Full Loss Curve                                           #
    # ------------------------------------------------------------------ #
    ax_loss.clear()
    ax_loss.set_facecolor(BG_PANEL)

    if train_loss:
        steps_tr = range(1, len(train_loss) + 1)
        smoothed  = _smooth(train_loss, window=30)

        ax_loss.plot(steps_tr, train_loss,
                     color=CYAN, linewidth=1, alpha=0.25, label="_raw")
        ax_loss.plot(steps_tr, smoothed,
                     color=CYAN, linewidth=2.5, label="Train (smooth)", alpha=0.9)
        ax_loss.fill_between(steps_tr, smoothed,
                             color=CYAN, alpha=0.06)

        if val_records:
            val_steps  = [r["step"]  for r in val_records if isinstance(r, dict)]
            val_losses = [r["loss"]  for r in val_records if isinstance(r, dict)]
            ax_loss.scatter(val_steps, val_losses,
                            color=MAGENTA, s=60, zorder=5,
                            label="Val", marker="D", edgecolors=WHITE, linewidths=0.5)

        ax_loss.set_xlabel("Optimizer Step", color=GREY, fontsize=9)
        ax_loss.set_ylabel("Total Loss",     color=GREY, fontsize=9)
        ax_loss.legend(facecolor=BG_DARK, edgecolor=DIM,
                       labelcolor=WHITE, fontsize=8, loc="upper right")
    else:
        ax_loss.text(0.5, 0.5, "COLLECTING DATA...",
                     transform=ax_loss.transAxes,
                     ha="center", color=CYAN, fontsize=12, fontweight="bold")

    ax_loss.set_title("LEARNING CURVE", color=WHITE,
                      loc="left", pad=10, fontsize=11, fontweight="bold")
    ax_loss.grid(True, linestyle=":", alpha=0.15, color=WHITE)
    ax_loss.tick_params(colors=GREY, labelsize=8)
    for spine in ax_loss.spines.values():
        spine.set_edgecolor(DIM)

    # ------------------------------------------------------------------ #
    # Panel 2 — Stats Readout                                             #
    # ------------------------------------------------------------------ #
    ax_stats.clear()
    ax_stats.axis("off")
    ax_stats.set_facecolor(BG_PANEL)

    best_val = min((r["loss"] for r in val_records if isinstance(r, dict)),
                   default=float("nan"))
    cur_loss = train_loss[-1] if train_loss else float("nan")
    vram_cur = vram_usage[-1] if vram_usage else float("nan")
    vram_peak = max(vram_usage) if vram_usage else float("nan")

    # Correct samples/sec from delta step count and session elapsed
    delta_steps = current_step - animate.initial_step
    sps = (delta_steps * 128) / max(elapsed, 1)   # 128 = total samples per step (batch*accum)
    if delta_steps == 0 and elapsed > 10:
        # If we've been running for a while but no steps, show 0 instead of infinity/nonsense
        sps = 0.0

    stats_lines = [
        ("MAMBA TTS MONITOR v1",  MAGENTA, 13),
        ("─" * 26,                DIM,      9),
        (f"ARCHITECTURE  Mamba FastSpeech",  CYAN,   10),
        (f"BACKBONE      MambaBlock (×8)",   CYAN,   10),
        (f"VOCODER       HiFi-GAN",          CYAN,   10),
        ("─" * 26,                DIM,      9),
        (f"STEP          {current_step:,}",  WHITE,  11),
        (f"TRAIN LOSS    {cur_loss:.4f}",    YELLOW, 11),
        (f"BEST VAL      {best_val:.4f}" if not np.isnan(best_val)
                       else "BEST VAL      --",      MAGENTA, 11),
        ("─" * 26,                DIM,      9),
        (f"VRAM CUR      {vram_cur:.2f} GB", CYAN,   11),
        (f"VRAM PEAK     {vram_peak:.2f} GB", CYAN,   11),
        (f"SAMPLES/S     {sps:.1f}",         CYAN,   10),
        (f"ELAPSED       {_elapsed_str(elapsed)}", GREY, 10),
        ("─" * 26,                DIM,      9),
        ("CPU MODE      bf16 + compile",     GREY,   9),
        ("CKPT EVERY    200 steps",          GREY,   9),
    ]

    y = 0.97
    for text, color, fs in stats_lines:
        ax_stats.text(0.05, y, text,
                      transform=ax_stats.transAxes,
                      fontsize=fs, color=color,
                      verticalalignment="top", family="monospace")
        y -= 0.067

    # ------------------------------------------------------------------ #
    # Panel 3 — Mel Loss Short-Window Trend (last 200 steps)              #
    # ------------------------------------------------------------------ #
    ax_trend.clear()
    ax_trend.set_facecolor(BG_PANEL)

    tail = train_loss[-200:] if len(train_loss) > 200 else train_loss
    if tail:
        xs = range(max(0, current_step - len(tail)), current_step)
        smoothed_tail = _smooth(tail, window=10)
        ax_trend.plot(xs, tail,
                      color=YELLOW, linewidth=1, alpha=0.2)
        ax_trend.plot(xs, smoothed_tail,
                      color=YELLOW, linewidth=2, label="Loss (last 200 steps)")
        ax_trend.fill_between(xs, smoothed_tail, color=YELLOW, alpha=0.08)

        # Mark minimum
        min_val = min(smoothed_tail)
        min_idx = smoothed_tail.index(min_val)
        ax_trend.axhline(min_val, color=MAGENTA, linewidth=1,
                         linestyle="--", alpha=0.6, label=f"Min={min_val:.4f}")
        ax_trend.legend(facecolor=BG_DARK, edgecolor=DIM,
                        labelcolor=WHITE, fontsize=8, loc="upper right")
    else:
        ax_trend.text(0.5, 0.5, "AWAITING FIRST STEPS...",
                      transform=ax_trend.transAxes,
                      ha="center", color=YELLOW, fontsize=11)

    ax_trend.set_title("RECENT LOSS TREND  (last 200 steps)",
                        color=WHITE, loc="left", pad=10, fontsize=11, fontweight="bold")
    ax_trend.set_xlabel("Step", color=GREY, fontsize=9)
    ax_trend.set_ylabel("Loss", color=GREY, fontsize=9)
    ax_trend.grid(True, linestyle=":", alpha=0.15, color=WHITE)
    ax_trend.tick_params(colors=GREY, labelsize=8)
    for spine in ax_trend.spines.values():
        spine.set_edgecolor(DIM)

    # ------------------------------------------------------------------ #
    # Panel 4 — Latest Mel Spectrogram Preview                            #
    # ------------------------------------------------------------------ #
    ax_mel.clear()
    ax_mel.set_facecolor(BG_PANEL)
    mel = _latest_mel()

    if mel is not None:
        img = ax_mel.imshow(
            mel[::-1],          # flip: low freq at bottom
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        ax_mel.set_xlabel("Time Frames", color=GREY, fontsize=9)
        ax_mel.set_ylabel("Mel Bins",    color=GREY, fontsize=9)
        ax_mel.tick_params(colors=GREY, labelsize=8)
    else:
        ax_mel.text(0.5, 0.5,
                    "MEL PREVIEW\nAvailable after step 500",
                    transform=ax_mel.transAxes,
                    ha="center", va="center",
                    color=GREY, fontsize=11, linespacing=2.0)
        ax_mel.set_xticks([])
        ax_mel.set_yticks([])

    ax_mel.set_title("LATEST MEL SPECTROGRAM PREVIEW",
                      color=WHITE, loc="left", pad=10, fontsize=11, fontweight="bold")
    for spine in ax_mel.spines.values():
        spine.set_edgecolor(DIM)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def visual_monitor() -> None:
    """Build figure layout and start the animation loop."""
    plt.rcParams["figure.facecolor"] = BG_DARK
    plt.rcParams["axes.facecolor"]   = BG_PANEL
    plt.rcParams["savefig.facecolor"] = BG_DARK
    plt.rcParams["font.family"]       = "monospace"

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[2, 1],
        hspace=0.35, wspace=0.25,
    )

    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_trend = fig.add_subplot(gs[1, 0])
    ax_mel   = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "⬡  JARVIS TTS  ·  MAMBA ACOUSTIC ENGINE  ·  CPU TRAINING",
        color=MAGENTA, fontsize=15, fontweight="bold", y=0.98,
    )

    ani = FuncAnimation(
        fig, animate,
        fargs=(ax_loss, ax_stats, ax_trend, ax_mel),
        interval=REFRESH_MS,
        cache_frame_data=False,
    )

    print("TTS Visual Monitor running — close window to exit.")
    print(f"Watching: {os.path.abspath(STATS_FILE)}")
    plt.show()


if __name__ == "__main__":
    visual_monitor()
