# full_gpu_benchmark.py
import torch
import time
import sys
import os
import json
import numpy as np

# Fix import paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'python')

print("=" * 60)
print("FastMQA Complete GPU Benchmarks")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print()

# Import the CUDA module
import fastmqa_cuda

# Test configurations
configs = [
    (1, 128, 64, 32),    # Small
    (2, 256, 64, 32),    # Medium
    (4, 512, 128, 32),   # Large
    (8, 1024, 128, 32),  # XLarge
    (1, 2048, 128, 32),  # Long sequence
]

results = []

for batch_size, seq_len, head_dim, num_heads in configs:
    print(f"Testing B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
    
    # Create inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    
    # Warmup
    for _ in range(10):
        _ = fastmqa_cuda.forward(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark CUDA kernel
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        _ = fastmqa_cuda.forward(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()
    
    cuda_time = start_event.elapsed_time(end_event) / 50  # ms
    
    # Benchmark PyTorch baseline
    def pytorch_mqa(Q, K, V):
        scale = 1.0 / (head_dim ** 0.5)
        K_expanded = K.expand(-1, num_heads, -1, -1)
        V_expanded = V.expand(-1, num_heads, -1, -1)
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V_expanded)
    
    # Warmup PyTorch
    for _ in range(10):
        _ = pytorch_mqa(Q, K, V)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(50):
        _ = pytorch_mqa(Q, K, V)
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time = start_event.elapsed_time(end_event) / 50  # ms
    
    # Calculate metrics
    speedup = pytorch_time / cuda_time
    throughput = (batch_size * seq_len) / (cuda_time / 1000)  # tokens/sec
    
    # Memory usage
    memory_mqa = (K.numel() + V.numel()) * 4 / (1024**2)  # MB
    memory_mha = (K.numel() + V.numel()) * num_heads * 4 / (1024**2)  # MB
    memory_reduction = (1 - memory_mqa/memory_mha) * 100
    
    print(f"  CUDA: {cuda_time:.2f} ms")
    print(f"  PyTorch: {pytorch_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    print(f"  Memory Reduction: {memory_reduction:.1f}%")
    print()
    
    results.append({
        'batch_size': batch_size,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'num_heads': num_heads,
        'cuda_ms': cuda_time,
        'pytorch_ms': pytorch_time,
        'speedup': speedup,
        'throughput': throughput,
        'memory_reduction': memory_reduction
    })

# Save results
with open('gpu_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("Summary:")
print(f"Average Speedup: {np.mean([r['speedup'] for r in results]):.2f}x")
print(f"Average Memory Reduction: {np.mean([r['memory_reduction'] for r in results]):.1f}%")
print("Results saved to gpu_benchmark_results.json")

# Generate markdown table for README
print("\n## Benchmark Results (NVIDIA L4)\n")
print("| Config | CUDA (ms) | PyTorch (ms) | Speedup | Memory Reduction |")
print("|--------|-----------|--------------|---------|------------------|")
for r in results:
    print(f"| B{r['batch_size']}_S{r['seq_len']}_H{r['num_heads']} | {r['cuda_ms']:.2f} | {r['pytorch_ms']:.2f} | {r['speedup']:.2f}x | {r['memory_reduction']:.1f}% |")

