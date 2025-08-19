# python/fastmqa.py
import torch
import torch.nn as nn
import warnings
import os

# Try to import the compiled CUDA extension
try:
    import fastmqa_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("FastMQA CUDA extension not built. Using PyTorch fallback.")

class FastMQAttention(nn.Module):
    def __init__(self, num_heads=32, head_dim=128, use_cuda=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()
        
        if use_cuda and not self.use_cuda:
            warnings.warn("CUDA requested but not available. Using PyTorch fallback.")
    
    def forward(self, Q, K, V, mask=None):
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Use CUDA kernel if available and no mask
        if self.use_cuda and mask is None:
            return fastmqa_cuda.forward(Q, K, V)
        else:
            return self._pytorch_fallback(Q, K, V, mask)
    
    def _pytorch_fallback(self, Q, K, V, mask=None):
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Expand K and V for MQA
        K_expanded = K.expand(-1, num_heads, -1, -1)
        V_expanded = V.expand(-1, num_heads, -1, -1)
        
        # Standard attention computation
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_expanded)
        
        return output

def test_cuda_kernel():
    """Test the actual CUDA kernel"""
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a GPU machine.")
        return
    
    print("Testing FastMQA CUDA Kernel...")
    
    # Create test inputs
    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    
    # Test CUDA kernel
    mqa_cuda = FastMQAttention(num_heads=num_heads, head_dim=head_dim, use_cuda=True)
    output_cuda = mqa_cuda(Q, K, V)
    
    # Test PyTorch fallback
    mqa_pytorch = FastMQAttention(num_heads=num_heads, head_dim=head_dim, use_cuda=False)
    output_pytorch = mqa_pytorch(Q.cpu(), K.cpu(), V.cpu()).cuda()
    
    # Compare results
    diff = torch.abs(output_cuda - output_pytorch).max().item()
    print(f"Max difference between CUDA and PyTorch: {diff:.6f}")
    
    if diff < 1e-3:
        print("✓ CUDA kernel produces correct results!")
    else:
        print("✗ CUDA kernel has accuracy issues")
    
    return output_cuda

if __name__ == "__main__":
    test_cuda_kernel()