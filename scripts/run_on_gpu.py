# run_on_gpu.py
"""
Complete script to build and test FastMQA on a GPU machine.
Run this on Lightning AI, Colab, or any NVIDIA GPU machine.
"""

import os
import sys
import subprocess
import torch
import time

def setup_and_build():
    """Build the CUDA extension"""
    print("=" * 60)
    print("FastMQA GPU Setup")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available!")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name()}")
    print(f"✓ CUDA: {torch.version.cuda}")
    print(f"✓ PyTorch: {torch.__version__}")
    
    # Build the extension
    print("\nBuilding CUDA extension...")
    try:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True)
        print("✓ Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        return False

def benchmark_real_performance():
    """Run actual benchmarks on GPU"""
    sys.path.insert(0, 'python')
    from fastmqa import FastMQAttention
    
    print("\n" + "=" * 60)
    print("Real Performance Benchmarks")
    print("=" * 60)
    
    configs = [
        (1, 128, 64),
        (4, 256, 128),
        (8, 512, 128),
        (16, 1024, 128),
    ]
    
    for batch_size, seq_len, head_dim in configs:
        num_heads = 32
        
        # Create inputs
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
        K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
        V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
        
        # Benchmark CUDA kernel
        mqa_cuda = FastMQAttention(num_heads=num_heads, head_dim=head_dim, use_cuda=True)
        
        # Warmup
        for _ in range(10):
            _ = mqa_cuda(Q, K, V)
        torch.cuda.synchronize()
        
        # Time CUDA kernel
        start = time.perf_counter()
        for _ in range(100):
            _ = mqa_cuda(Q, K, V)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 100 * 1000
        
        # Benchmark PyTorch
        mqa_pytorch = FastMQAttention(num_heads=num_heads, head_dim=head_dim, use_cuda=False)
        
        # Warmup
        for _ in range(10):
            _ = mqa_pytorch(Q, K, V)
        torch.cuda.synchronize()
        
        # Time PyTorch
        start = time.perf_counter()
        for _ in range(100):
            _ = mqa_pytorch(Q, K, V)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        speedup = pytorch_time / cuda_time
        
        print(f"\nBatch={batch_size}, Seq={seq_len}, HeadDim={head_dim}")
        print(f"  CUDA Kernel: {cuda_time:.2f} ms")
        print(f"  PyTorch: {pytorch_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Memory comparison
        memory_mqa = (K.numel() + V.numel()) * 4 / (1024 * 1024)  # MB
        memory_mha = (K.numel() + V.numel()) * num_heads * 4 / (1024 * 1024)
        memory_reduction = 1 - (memory_mqa / memory_mha)
        print(f"  Memory Reduction: {memory_reduction:.1%}")

if __name__ == "__main__":
    if setup_and_build():
        benchmark_real_performance()
    else:
        print("\nFailed to build. Please check CUDA installation.")