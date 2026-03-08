import torch
import mamba_scan
import os

def test_fp16():
    print("Testing FP16 Mamba Scan...")
    B, L, D, N = 1, 16, 128, 16
    device = "cuda"
    dtype = torch.half
    
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    dt = torch.randn(B, L, D, device=device, dtype=dtype).abs()
    A = -torch.randn(D, N, device=device, dtype=dtype).abs()
    B_params = torch.randn(B, L, N, device=device, dtype=dtype)
    C_params = torch.randn(B, L, N, device=device, dtype=dtype)
    D_params = torch.randn(D, device=device, dtype=dtype)
    
    print("Inputs prepared. Calling extension...")
    try:
        y = mamba_scan.ssm_scan_fwd(x, dt, A, B_params, C_params, D_params)
        print("Success! Output shape:", y.shape, "dtype:", y.dtype)
        assert y.shape == x.shape
        assert y.dtype == dtype
    except Exception as e:
        print("Failed with error:", e)

if __name__ == "__main__":
    if os.name == "nt":
        # Add DLL directory for Windows
        cuda_path = os.environ.get("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1")
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.exists(bin_path):
            os.add_dll_directory(bin_path)
    
    test_fp16()
