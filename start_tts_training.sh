#!/usr/bin/env bash
# start_tts_training.sh — Launch full TTS pipeline (CPU-optimized)
#
# Steps:
#   1. Set LD_LIBRARY_PATH so mamba_scan.so can find libc10.so
#   2. Build LJSpeech dataset if not already done
#   3. Download HiFi-GAN vocoder if not already done
#   4. Launch CPU-optimized training
#
# Usage:
#   chmod +x start_tts_training.sh && ./start_tts_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- PyTorch shared library path ---
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# --- CPU thread pinning ---
NCORES="$(nproc)"
export OMP_NUM_THREADS="$NCORES"
export MKL_NUM_THREADS="$NCORES"
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

echo "============================================"
echo "  Mamba TTS Training — CPU Optimized"
echo "  Cores    : $NCORES"
echo "  Torch lib: $TORCH_LIB"
echo "  Working  : $SCRIPT_DIR"
echo "============================================"

# --- Verify C++ extension ---
echo "[1/4] Verifying mamba_scan extension ..."
python -c "
import mamba_scan, torch
import torch
B, L, D, N = 1, 16, 16, 8
x  = torch.randn(B, L, D)
dt = torch.rand(B, L, D) * 0.1
A  = -torch.rand(D, N)
Bp = torch.randn(B, L, N)
Cp = torch.randn(B, L, N)
Dp = torch.randn(D)
y = mamba_scan.ssm_scan_fwd(x, dt, A, Bp, Cp, Dp)
print(f'  SSM scan OK: output shape={y.shape}')
"

# --- Dataset ---
if [ ! -f "train_tts.pkl" ]; then
    echo "[2/4] Building LJSpeech dataset (downloads ~2.6 GB) ..."
    python tts_data_builder.py
else
    echo "[2/4] Dataset found (train_tts.pkl). Skipping download."
fi

# --- Vocoder ---
if [ ! -f "vocoder/generator.pth" ]; then
    echo "[3/4] Downloading HiFi-GAN vocoder ..."
    python download_vocoder.py
else
    echo "[3/4] Vocoder found. Skipping download."
fi

# --- Train ---
echo "[4/4] Starting CPU-optimized training ..."
echo "      Log: tts_training.log"
echo "      Stop with: Ctrl+C (checkpoints saved every 200 steps)"
echo ""

PYTHONUNBUFFERED=1 python tts_trainer_cpu.py "$@" 2>&1 | tee tts_training.log
