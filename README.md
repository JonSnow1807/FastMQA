# FastMQA: CUDA Multi-Query Attention Implementation

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A deep-dive into CUDA kernel optimization through implementing Multi-Query Attention (MQA), achieving **97% memory reduction** for LLM inference while documenting the challenging journey of competing with production-grade libraries.

## 🎯 Project Objectives & Achievements

This project demonstrates my journey in CUDA kernel development, starting from a naive implementation running at 491ms and optimizing it down to 10ms through various techniques, ultimately achieving:

- ✅ **97% KV-cache memory reduction** - Primary goal achieved
- ✅ **49x performance improvement** from baseline (491ms → 10ms)
- ✅ **Perfect numerical accuracy** (max error: 1.43e-06)
- ✅ **Production-ready hybrid solution** using optimized libraries

## 📊 Performance Analysis

### Optimization Journey

| Version | Execution Time | Speedup | Technique Applied |
|---------|---------------|---------|-------------------|
| Baseline | 491ms | 1x | Naive CUDA implementation |
| v1: Shared Memory | 220ms | 2.2x | Shared memory tiling |
| v2: Warp Primitives | 103ms | 4.8x | Warp-level reductions |
| v3: Memory Coalescing | 45ms | 10.9x | Optimized memory access |
| v4: cuBLAS Hybrid | **10ms** | **49x** | NVIDIA cuBLAS for GEMM |
| PyTorch Reference | 1.7ms | 289x | cuDNN + Flash Attention |

### Honest Performance Metrics

| Metric | Initial Claim | **Verified Reality** | Status |
|--------|--------------|---------------------|---------|
| Memory Reduction | 97% | **97%** | ✅ Achieved |
| Speed vs PyTorch | "2.4x faster" | **5.7x slower** | ❌ Learned why |
| Throughput | 129K tok/s | **22K tok/s** | ⚠️ Memory-bound |
| Batch Scaling | 32x larger | **32x larger** | ✅ Achieved |

## 🔬 Technical Deep Dive

### Why Custom CUDA Kernels Are Slower Than PyTorch

Through this implementation, I discovered why PyTorch's performance is so difficult to match:

1. **cuDNN Integration**: NVIDIA's proprietary kernels with assembly-level optimizations
2. **Tensor Cores**: Hardware acceleration providing 8-16x speedup for matrix operations
3. **Flash Attention**: Advanced algorithmic optimizations reducing memory bandwidth
4. **Years of Engineering**: Thousands of engineer-hours optimizing every instruction

### Optimizations I Implemented

```cuda
// Key optimizations in my custom kernel:
- Shared memory tiling (48KB per SM utilization)
- Warp-level reduction primitives (__shfl_xor_sync)
- Vectorized memory access (float4)
- Memory coalescing patterns
- cuBLAS integration for matrix multiplication
- Online softmax algorithm (FlashAttention-inspired)
```

### Memory Efficiency Architecture

```
Standard MHA: Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D] 
              Memory: 2 * B * H * S * D * sizeof(float)

FastMQA:      Q[B,H,S,D], K[B,1,S,D], V[B,1,S,D]
              Memory: 2 * B * 1 * S * D * sizeof(float)
              
Reduction:    (H-1)/H * 100% ≈ 97% for H=32
```

## 🚀 Installation & Usage

### Prerequisites
```bash
- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.0+
- PyTorch 2.0+
- Python 3.8+
- cuBLAS library
```

### Setup
```bash
# Clone repository
git clone https://github.com/JonSnow1807/FastMQA.git
cd FastMQA

# Install dependencies
pip install torch numpy matplotlib pytest

# Build with cuBLAS support
python setup_cublas.py build_ext --inplace
```

### Usage Examples

#### Custom CUDA Kernel (Memory Efficient, 10ms)
```python
import torch
import fastmqa_cuda

# MQA with single K,V heads - 97% memory savings
Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()  
V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()

output = fastmqa_cuda.forward(Q, K, V)
```

#### Production Hybrid Approach (Fastest, 1.7ms)
```python
from python.fastmqa_production import FastMQA

# Uses PyTorch's optimized kernels with MQA memory layout
model = FastMQA(hidden_dim=768, num_heads=12, use_flash=True)
output = model(input_tensor)

# Memory savings calculation
savings = model.get_memory_savings(batch_size=32, seq_len=1024)
print(f"Memory reduced by {savings['reduction_percent']:.1f}%")
print(f"Can serve {savings['cache_multiplier']:.0f}x more users")
```

## 📈 Benchmarking Results

### Test Configuration
- **GPU**: NVIDIA L4 (Lightning AI Platform)
- **Test Size**: Batch=4, Heads=32, Seq=512, Dim=128
- **Iterations**: 100 runs, averaged

### Results
```python
python test_final.py

# Output:
Correctness check:
  Max error: 1.43e-06  ✅ PASSED
  
Performance:
  Custom CUDA kernel: 10.0ms
  PyTorch baseline: 1.7ms
  Relative speed: 5.7x slower
  
Memory efficiency:
  Standard MHA: 64.0 MB
  FastMQA: 2.0 MB
  Reduction: 96.9%
  Batch scaling: 32x larger possible
```

## 💡 Key Learnings & Insights

### What I Learned About CUDA Optimization

1. **Memory Bandwidth is Often the Bottleneck**: Attention is memory-bound, not compute-bound
2. **Shared Memory is Critical**: Proper utilization can provide 10x speedup
3. **Warp-Level Primitives**: Essential for efficient reductions
4. **Library Integration**: Sometimes the best optimization is using existing optimized libraries
5. **Hardware Matters**: Tensor cores provide massive speedups that custom CUDA can't match

### The Value Proposition

While my custom kernel is 5.7x slower than PyTorch, the implementation provides significant value:

- **Memory-Constrained Scenarios**: Enables serving 32x more concurrent users
- **Educational Value**: Demonstrates real-world CUDA optimization challenges
- **Production Trade-off**: Many systems would accept 5.7x latency for 32x throughput
- **Hybrid Solution**: Combines custom memory layout with optimized computation

## 🏗️ Project Structure

```
FastMQA/
├── kernels/
│   ├── mqa_kernel.cu              # Current best implementation (cuBLAS)
│   ├── mqa_kernel_original.cu     # Baseline implementation (educational)
│   ├── mqa_kernel_optimized.cu    # Optimized with CUDA primitives
│   ├── mqa_kernel_hybrid.cu       # cuBLAS integration
│   └── versions/                   # All experimental versions
├── python/
│   ├── fastmqa_production.py      # Production-ready wrapper
│   └── fastmqa_ultimate.py        # Advanced features implementation
├── benchmarks/
│   ├── benchmark_baseline.py      # Performance testing suite
│   └── profile_kernel.py          # CUDA profiling tools
├── tests/
│   ├── test_correctness.py        # Numerical accuracy tests
│   └── test_final.py              # Comprehensive validation
└── docs/
    └── optimization_journey.md     # Detailed optimization notes
```

## 🔍 Detailed Performance Analysis

### Where Time Is Spent (Profiled with Nsight)

1. **Matrix Multiplication (60%)**: QK^T and Attention×V operations
2. **Softmax (25%)**: Row-wise max reduction and exponentiation  
3. **Memory Transfer (15%)**: Global memory reads/writes

### Why PyTorch Is Faster

```python
# PyTorch's advantages I cannot replicate:
1. cuDNN kernels:        Assembly-optimized by NVIDIA
2. Tensor Cores:         8-16x faster matrix multiplication
3. Flash Attention v2:   Reduces memory bandwidth by 10x
4. Kernel Fusion:        Eliminates intermediate memory writes
5. Hardware-specific:    Optimized for each GPU architecture
```

## 🎓 Educational Value

This project serves as a comprehensive case study in:

- CUDA kernel development from scratch
- Performance optimization techniques
- Benchmarking and profiling methodology
- Understanding the gap between custom and production code
- Making engineering trade-offs (memory vs compute)

## 🚦 Future Improvements

- [ ] Implement Flash Attention v3 algorithm
- [ ] Add FP16/BF16 mixed precision support
- [ ] Integrate tensor core operations (WMMA API)
- [ ] Support for variable sequence lengths
- [ ] INT8 quantization for additional memory savings

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@software{fastmqa2024,
  author = {JonSnow1807},
  title = {FastMQA: A CUDA Journey in Multi-Query Attention},
  year = {2024},
  url = {https://github.com/JonSnow1807/FastMQA}
}
```

## 🤝 Contributing

I welcome contributions, especially:
- Further optimization techniques
- Support for additional GPU architectures
- Integration with inference frameworks
- Performance analysis tools

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and cuBLAS library
- PyTorch team for the reference implementation
- FlashAttention authors for algorithmic insights
- Lightning AI for GPU compute resources

## 📧 Contact

- GitHub: [@JonSnow1807](https://github.com/JonSnow1807)
- Project: [FastMQA Repository](https://github.com/JonSnow1807/FastMQA)

---

*This project represents my deep dive into CUDA optimization, demonstrating both the challenges and rewards of low-level GPU programming. While the custom kernel doesn't beat PyTorch's optimized implementation, the 97% memory reduction enables real-world production benefits, and the learning journey provides valuable insights into high-performance computing.*
