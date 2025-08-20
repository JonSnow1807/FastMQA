# test_cuda_working.py
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("FastMQA CUDA Test")
print("=" * 60)

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

# Try to import the CUDA module
try:
    import fastmqa_cuda
    print("✓ CUDA module imported successfully!")
    
    # Test the kernel with small inputs first
    batch_size, num_heads, seq_len, head_dim = 1, 4, 64, 64
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    
    print(f"\nTesting CUDA kernel with:")
    print(f"  Batch: {batch_size}, Heads: {num_heads}, Seq: {seq_len}, Dim: {head_dim}")
    
    output = fastmqa_cuda.forward(Q, K, V)
    print(f"✓ CUDA kernel executed!")
    print(f"  Output shape: {output.shape}")
    
    # Verify correctness
    print("\nVerifying correctness...")
    scale = 1.0 / (head_dim ** 0.5)
    K_expanded = K.expand(-1, num_heads, -1, -1)
    V_expanded = V.expand(-1, num_heads, -1, -1)
    scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    expected = torch.matmul(attn, V_expanded)
    
    diff = torch.abs(output - expected).max().item()
    print(f"Max difference from PyTorch: {diff:.6f}")
    if diff < 1e-3:
        print("✓ Results match PyTorch implementation!")
    
    # Quick benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = fastmqa_cuda.forward(Q, K, V)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 100 * 1000
    
    print(f"\nPerformance: {elapsed:.2f} ms per forward pass")
    
except ImportError as e:
    print(f"✗ Failed to import CUDA module: {e}")
    print("\nDebugging info:")
    import subprocess
    result = subprocess.run(['nm', '-D', 'fastmqa_cuda.cpython-310-x86_64-linux-gnu.so'], 
                          capture_output=True, text=True)
    print("Exported symbols containing 'launch':")
    for line in result.stdout.split('\n'):
        if 'launch' in line.lower():
            print(f"  {line}")