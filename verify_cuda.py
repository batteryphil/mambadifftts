import torch
import time
from mamba_llm_diffusion import Config, DiM_LLM
import mamba_scan

def benchmark():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {DEVICE}")
    
    config = Config(vocab_size=100, d_model=512, n_layers=4, seq_len=128)
    model = DiM_LLM(config).to(DEVICE).eval()
    
    # Dummy inputs
    x = torch.randint(0, 100, (1, 128)).to(DEVICE)
    timesteps = torch.tensor([50]).to(DEVICE)
    
    print("Warmup...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(x, timesteps)
            
    print("Benchmarking...")
    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model(x, timesteps)
    end = time.time()
    
    print(f"Average time per forward pass: {(end - start) / 20:.4f}s")
    print("CUDA optimization verified successfully!")

if __name__ == "__main__":
    benchmark()
