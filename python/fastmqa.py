# python/fastmqa.py
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import warnings

try:
    # Try to import the compiled CUDA extension
    import fastmqa_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("FastMQA CUDA extension not built. Please run: python setup.py install")

class FastMQAttention(nn.Module):
    """
    Fast Multi-Query Attention implementation using custom CUDA kernels.
    
    Args:
        num_heads: Number of attention heads for queries
        head_dim: Dimension of each attention head
        use_cuda: Whether to use CUDA kernel (falls back to PyTorch if False)
    """
    
    def __init__(self, num_heads=32, head_dim=128, use_cuda=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.scale = 1.0 / (head_dim ** 0.5)
        
        if self.use_cuda and head_dim not in [64, 80, 128]:
            warnings.warn(f"Head dim {head_dim} not optimized. Using PyTorch fallback.")
            self.use_cuda = False
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch, num_heads, seq_len, head_dim]
            K: [batch, 1, seq_len, head_dim] (single head for MQA)
            V: [batch, 1, seq_len, head_dim] (single head for MQA)
            mask: Optional attention mask
        
        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Validate inputs
        assert K.shape[1] == 1, "K should have single head for MQA"
        assert V.shape[1] == 1, "V should have single head for MQA"
        assert head_dim == self.head_dim
        assert num_heads == self.num_heads
        
        if self.use_cuda and mask is None and seq_len <= 2048:
            # Use custom CUDA kernel
            output = torch.empty_like(Q)
            
            # Ensure contiguous memory layout
            Q_contig = Q.contiguous()
            K_contig = K.contiguous()
            V_contig = V.contiguous()
            
            # Call CUDA kernel
            fastmqa_cuda.forward(
                Q_contig, K_contig, V_contig, output,
                batch_size, num_heads, seq_len, head_dim
            )
            
            return output
        else:
            # Fallback to PyTorch implementation
            return self._pytorch_mqa(Q, K, V, mask)
    
    def _pytorch_mqa(self, Q, K, V, mask=None):
        """PyTorch fallback implementation"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Expand K and V to match Q's head dimension
        K_expanded = K.expand(-1, num_heads, -1, -1)
        V_expanded = V.expand(-1, num_heads, -1, -1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V_expanded)
        
        return output
    
    def profile(self, batch_size=4, seq_len=512):
        """Profile the kernel performance"""
        if not self.use_cuda:
            print("CUDA kernel not available. Cannot profile.")
            return
        
        # Create dummy inputs
        Q = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim).cuda()
        K = torch.randn(batch_size, 1, seq_len, self.head_dim).cuda()
        V = torch.randn(batch_size, 1, seq_len, self.head_dim).cuda()
        
        # Warmup
        for _ in range(10):
            _ = self.forward(Q, K, V)
        
        # Time the kernel
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = self.forward(Q, K, V)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 100
        
        # Calculate metrics
        total_flops = 2 * batch_size * self.num_heads * seq_len * seq_len * self.head_dim
        throughput = (total_flops / elapsed_ms) / 1e9  # GFLOPS
        
        print(f"FastMQA Kernel Performance:")
        print(f"  Batch Size: {batch_size}, Seq Len: {seq_len}")
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {throughput:.2f} GFLOPS")
        
        return elapsed_ms, throughput


# Convenience function for testing
def test_mqa():
    """Simple test to verify the implementation works"""
    print("Testing FastMQA implementation...")
    
    # Create model
    mqa = FastMQAttention(num_heads=8, head_dim=64, use_cuda=False)  # PyTorch only for now
    
    # Create inputs
    batch_size, seq_len = 2, 128
    Q = torch.randn(batch_size, 8, seq_len, 64)
    K = torch.randn(batch_size, 1, seq_len, 64)
    V = torch.randn(batch_size, 1, seq_len, 64)
    
    # Run forward pass
    output = mqa(Q, K, V)
    
    print(f"Input Q shape: {Q.shape}")
    print(f"Input K shape: {K.shape}")
    print(f"Input V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed! ✓")
    
    return output

if __name__ == "__main__":
    test_mqa()