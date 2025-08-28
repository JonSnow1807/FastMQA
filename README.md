# FastMQA: CUDA Multi-Query Attention Implementation

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive exploration of CUDA kernel optimization implementing Multi-Query Attention (MQA), achieving **97% memory reduction** for LLM inference. This project documents the complete journey from a naive 491ms implementation to a 10ms optimized kernel, providing valuable insights into the challenges of competing with production-grade libraries.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Achievements](#key-achievements)
- [Performance Analysis](#performance-analysis)
- [Technical Implementation](#technical-implementation)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Benchmarking Results](#benchmarking-results)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)

## Project Overview

Multi-Query Attention (MQA) is a memory optimization technique that shares key and value projections across all attention heads, reducing memory requirements by approximately 97% while maintaining model quality. This implementation explores the practical challenges of CUDA kernel development and the trade-offs between memory efficiency and computational speed.

### Why This Project Matters

In production LLM serving, memory is often the primary bottleneck limiting batch sizes and throughput. This implementation demonstrates that accepting a 5.7x computational slowdown can enable serving 32x more concurrent users - a worthwhile trade-off for many production scenarios.

## Key Achievements

| Achievement | Target | Result | Status |
|------------|--------|--------|--------|
| Memory Reduction | 95%+ | **97%** | ✅ Exceeded |
| Optimization from Baseline | 10x+ | **49x** (491ms → 10ms) | ✅ Exceeded |
| Numerical Accuracy | < 1e-4 error | **1.43e-06** | ✅ Exceeded |
| Beat PyTorch Speed | Faster | **5.7x slower** | ❌ Learned why |

### The Reality Check

Initial claims suggested this implementation would be faster than PyTorch. Through extensive optimization and benchmarking, I discovered why this is nearly impossible:
- PyTorch uses NVIDIA's proprietary cuDNN kernels
- Tensor cores provide 8-16x hardware acceleration
- Years of optimization by dedicated teams
- Access to low-level hardware features unavailable to CUDA developers

## Performance Analysis

### Optimization Journey

| Version | Implementation | Time (ms) | Speedup | Key Technique |
|---------|---------------|-----------|---------|---------------|
| v0 | Naive CUDA | 491 | 1x | Basic parallel computation |
| v1 | Shared Memory | 220 | 2.2x | 48KB shared memory tiles |
| v2 | Warp Primitives | 103 | 4.8x | `__shfl_xor_sync` reductions |
| v3 | Memory Coalescing | 45 | 10.9x | Vectorized float4 access |
| v4 | cuBLAS Integration | **10** | **49x** | NVIDIA's optimized GEMM |
| - | PyTorch (Reference) | 1.7 | 289x | cuDNN + Flash Attention |

### Performance Breakdown (Nsight Systems Profiling)

```
Matrix Multiplication: 60% (QK^T and Attention×V)
Softmax Computation:   25% (max reduction + exp)
Memory Transfers:      15% (global memory access)
```

### Memory Efficiency Analysis

```python
# Standard Multi-Head Attention
Memory = 2 * B * H * S * D * 4 bytes  # K and V for all heads
Example: B=32, H=32, S=2048, D=128 → 2048 MB

# Multi-Query Attention (This Implementation)
Memory = 2 * B * 1 * S * D * 4 bytes  # K and V shared
Example: B=32, H=32, S=2048, D=128 → 64 MB

Reduction: 96.9% (enables 32x larger batches)
```

## Technical Implementation

### Core Optimizations Implemented

1. **Shared Memory Tiling**
   ```cuda
   extern __shared__ float smem[];
   float* Q_tile = smem;
   float* KV_cache = &Q_tile[TILE_SIZE * HEAD_DIM];
   ```

2. **Warp-Level Primitives**
   ```cuda
   float warp_reduce_sum(float val) {
       for (int offset = 16; offset > 0; offset /= 2)
           val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
       return val;
   }
   ```

3. **Vectorized Memory Access**
   ```cuda
   float4 q_vec = *reinterpret_cast<float4*>(&Q[idx]);
   ```

4. **cuBLAS Integration**
   ```cuda
   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ...);
   ```

### Architecture Comparison

| Component | Standard MHA | FastMQA | Benefit |
|-----------|-------------|---------|---------|
| Q Tensor | [B, H, S, D] | [B, H, S, D] | - |
| K Tensor | [B, H, S, D] | [B, 1, S, D] | H× reduction |
| V Tensor | [B, H, S, D] | [B, 1, S, D] | H× reduction |
| KV Cache | B×H×S×D×8 | B×S×D×8 | 97% smaller |

## Installation & Usage

### Prerequisites

- NVIDIA GPU (Compute Capability ≥ 7.0)
- CUDA Toolkit 11.0+
- PyTorch 2.0+
- cuBLAS library
- Python 3.8+

### Installation

```bash
# Clone repository
git clone https://github.com/JonSnow1807/FastMQA.git
cd FastMQA

# Install dependencies
pip install torch numpy matplotlib pytest

# Build CUDA extension (choose one)
python setup.py build_ext --inplace        # Basic build
python setup_cublas.py build_ext --inplace # With cuBLAS (recommended)
```

### Usage

#### Option 1: Custom CUDA Kernel (10ms, 97% memory reduction)
```python
import torch
import fastmqa_cuda

batch_size, num_heads, seq_len, head_dim = 4, 32, 512, 128

# Create tensors with MQA structure
Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()  # Single head
V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()  # Single head

# Run custom kernel
output = fastmqa_cuda.forward(Q, K, V)
```

#### Option 2: Production Wrapper (1.7ms, uses PyTorch)
```python
from python.fastmqa_production import FastMQA

# Initialize model
model = FastMQA(
    hidden_dim=768,
    num_heads=12,
    use_flash=True  # Use Flash Attention if available
).cuda()

# Forward pass
output = model(input_tensor)

# Calculate memory savings
stats = model.get_memory_savings(batch_size=32, seq_len=2048)
print(f"Memory reduction: {stats['reduction_percent']:.1f}%")
print(f"Can serve {stats['cache_multiplier']}x more users")
```

#### Option 3: Benchmarking
```bash
# Test correctness and performance
python test_final.py

# Run comprehensive benchmarks
python benchmarks/benchmark_baseline.py

# Profile kernel performance
python benchmarks/profile_kernel.py
```

## Project Structure

```
FastMQA/
├── kernels/
│   ├── mqa_kernel.cu              # Main CUDA kernel (cuBLAS version)
│   ├── mqa_kernel_original.cu     # Baseline implementation (491ms)
│   ├── mqa_kernel_optimized.cu    # Optimized with primitives (45ms)
│   ├── mqa_extension.cpp          # PyTorch C++ bindings
│   └── versions/                   # Experimental versions archive
│       ├── mqa_kernel_warp.cu     # Warp-level optimizations
│       ├── mqa_kernel_tiled.cu    # Shared memory tiling
│       └── mqa_kernel_hybrid.cu   # cuBLAS integration
│
├── python/
│   ├── fastmqa.py                  # Python wrapper
│   ├── fastmqa_production.py       # Production implementation
│   └── fastmqa_ultimate.py         # Advanced features
│
├── benchmarks/
│   ├── benchmark_baseline.py       # Performance benchmarking
│   ├── profile_kernel.py          # CUDA profiling
│   └── results/                    # Benchmark outputs
│
├── tests/
│   ├── test_correctness.py        # Numerical validation
│   ├── test_memory.py             # Memory usage verification
│   └── test_final.py              # Comprehensive tests
│
├── scripts/
│   └── benchmark_sweep.py         # Parameter sweep analysis
│
└── docs/
    └── optimization_journey.md     # Detailed optimization notes
```

## Benchmarking Results

### Test Configuration
- **Hardware**: NVIDIA L4 GPU (Lightning AI Platform)
- **Software**: CUDA 12.1, PyTorch 2.0, cuBLAS 12.1
- **Test Size**: Batch=4, Heads=32, Seq=512, Dim=128

### Performance Results

```python
# Output from test_final.py
Correctness check:
  Max error: 1.43e-06  ✅ PASSED
  Mean error: 5.34e-08  ✅ PASSED

Performance:
  Custom CUDA kernel: 10.0 ms
  PyTorch baseline: 1.7 ms
  Relative speed: 5.7x slower

Memory efficiency:
  Standard MHA: 64.0 MB
  FastMQA: 2.0 MB
  Reduction: 96.9%
  Batch scaling: 32x larger possible
```

### Throughput Analysis

| Configuration | Custom Kernel | PyTorch | Memory Saved |
|--------------|--------------|---------|--------------|
| B=1, S=128 | 12.8K tok/s | 75.3K tok/s | 87.5% |
| B=4, S=512 | 22.0K tok/s | 126.0K tok/s | 96.9% |
| B=8, S=1024 | 18.5K tok/s | 105.0K tok/s | 96.9% |
| B=32, S=2048 | OOM | OOM with MHA | 96.9% |

## Lessons Learned

### Technical Insights

1. **Memory Bandwidth Dominates**: Attention is memory-bound, not compute-bound. Optimizing memory access patterns provides the largest speedups.

2. **Shared Memory is Critical**: Proper use of the 48KB shared memory per SM can provide 10x performance improvements.

3. **Warp-Level Operations**: Using `__shfl_xor_sync` for reductions is essential for efficiency.

4. **Library Integration**: Sometimes the best optimization is using existing optimized libraries (cuBLAS/cuDNN).

5. **Hardware Features Matter**: Tensor cores provide speedups that are impossible to achieve with standard CUDA cores.

### Engineering Trade-offs

- **Memory vs Speed**: 5.7x slower computation for 32x more concurrent users
- **Complexity vs Performance**: Simple cuBLAS calls outperform complex custom kernels
- **Development Time vs Optimization**: Months of optimization still can't match production libraries

### What Makes PyTorch Fast

1. **cuDNN**: Proprietary NVIDIA kernels with assembly-level optimizations
2. **Flash Attention**: Algorithmic improvements reducing memory bandwidth by 10x
3. **Tensor Cores**: Hardware matrix multiplication units (8-16x faster)
4. **Kernel Fusion**: Eliminating intermediate memory writes
5. **Auto-tuning**: Selecting optimal kernels for each GPU architecture

## Future Work

- [ ] **Flash Attention v3 Implementation**: Latest algorithmic improvements
- [ ] **Mixed Precision (FP16/BF16)**: 2x memory and computation savings
- [ ] **Tensor Core Integration**: Using WMMA API for matrix operations
- [ ] **Dynamic Sequence Length**: Support for variable-length sequences
- [ ] **INT8 Quantization**: Additional 4x memory reduction
- [ ] **Multi-GPU Support**: Distributed attention computation
- [ ] **Integration with vLLM/TGI**: Production inference frameworks

## Contributing

Contributions are welcome! Areas of particular interest:
- Further optimization techniques
- Support for newer GPU architectures (Hopper/Ada)
- Integration with inference frameworks
- Benchmarking on different hardware

## Citation

```bibtex
@software{fastmqa2024,
  author = {JonSnow1807},
  title = {FastMQA: CUDA Multi-Query Attention - A Journey in Memory Optimization},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/JonSnow1807/FastMQA}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NVIDIA**: CUDA toolkit, cuBLAS library, and Nsight profiling tools
- **PyTorch Team**: Reference implementation and benchmarking baseline
- **FlashAttention Authors**: Algorithmic insights and optimization strategies
- **Lightning AI**: GPU compute resources for development and testing

## Contact

- **GitHub**: [@JonSnow1807](https://github.com/JonSnow1807)
- **Repository**: [github.com/JonSnow1807/FastMQA](https://github.com/JonSnow1807/FastMQA)

---

*This project represents a deep exploration of CUDA optimization, demonstrating that while beating production libraries is extremely challenging, significant memory optimizations can provide real value in production systems. The 97% memory reduction enables practical benefits that outweigh the computational slowdown in memory-constrained serving scenarios.*
