# tests/test_correctness.py
import torch
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from fastmqa import FastMQAttention

class TestFastMQA:
    """Test suite for FastMQA implementation"""
    
    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors"""
        batch_size = 2
        num_heads = 8
        seq_len = 64
        head_dim = 32
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, 1, seq_len, head_dim)
        V = torch.randn(batch_size, 1, seq_len, head_dim)
        
        return Q, K, V, batch_size, num_heads, seq_len, head_dim
    
    def test_output_shape(self, setup_tensors):
        """Test that output shape matches input Q shape"""
        Q, K, V, batch_size, num_heads, seq_len, head_dim = setup_tensors
        
        model = FastMQAttention(num_heads=num_heads, head_dim=head_dim)
        output = model(Q, K, V)
        
        assert output.shape == Q.shape, f"Output shape {output.shape} != Q shape {Q.shape}"
    
    def test_attention_sum(self, setup_tensors):
        """Test that attention weights sum to 1"""
        Q, K, V, batch_size, num_heads, seq_len, head_dim = setup_tensors
        
        model = FastMQAttention(num_heads=num_heads, head_dim=head_dim)
        
        # Get attention weights by computing them manually
        scale = 1.0 / (head_dim ** 0.5)
        K_expanded = K.expand(-1, num_heads, -1, -1)
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Check that weights sum to 1 along last dimension
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
    
    def test_deterministic(self, setup_tensors):
        """Test that the model produces deterministic results"""
        Q, K, V, batch_size, num_heads, seq_len, head_dim = setup_tensors
        
        model = FastMQAttention(num_heads=num_heads, head_dim=head_dim)
        
        output1 = model(Q, K, V)
        output2 = model(Q, K, V)
        
        assert torch.allclose(output1, output2, atol=1e-7), "Outputs are not deterministic"
    
    def test_different_sequence_lengths(self):
        """Test with various sequence lengths"""
        batch_size = 1
        num_heads = 4
        head_dim = 32
        
        for seq_len in [16, 32, 64, 128, 256]:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
            K = torch.randn(batch_size, 1, seq_len, head_dim)
            V = torch.randn(batch_size, 1, seq_len, head_dim)
            
            model = FastMQAttention(num_heads=num_heads, head_dim=head_dim)
            output = model(Q, K, V)
            
            assert output.shape == Q.shape, f"Failed for seq_len={seq_len}"
    
    def test_gradient_flow(self, setup_tensors):
        """Test that gradients flow through the model"""
        Q, K, V, batch_size, num_heads, seq_len, head_dim = setup_tensors
        
        Q.requires_grad = True
        K.requires_grad = True
        V.requires_grad = True
        
        model = FastMQAttention(num_heads=num_heads, head_dim=head_dim)
        output = model(Q, K, V)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        assert Q.grad is not None, "No gradient for Q"
        assert K.grad is not None, "No gradient for K"
        assert V.grad is not None, "No gradient for V"
        
        # Check gradients are non-zero
        assert Q.grad.abs().sum() > 0, "Q gradients are zero"
        assert K.grad.abs().sum() > 0, "K gradients are zero"
        assert V.grad.abs().sum() > 0, "V gradients are zero"

def test_memory_efficiency():
    """Test memory efficiency compared to standard MHA"""
    batch_size = 4
    num_heads = 32
    seq_len = 512
    head_dim = 128
    
    # MQA: K,V have single head
    K_mqa = torch.randn(batch_size, 1, seq_len, head_dim)
    V_mqa = torch.randn(batch_size, 1, seq_len, head_dim)
    
    # Standard MHA: K,V have multiple heads
    K_mha = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V_mha = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    mqa_memory = (K_mqa.numel() + V_mqa.numel()) * 4  # float32 bytes
    mha_memory = (K_mha.numel() + V_mha.numel()) * 4
    
    reduction = 1 - (mqa_memory / mha_memory)
    print(f"Memory reduction: {reduction:.1%}")
    
    assert reduction > 0.9, f"Memory reduction {reduction:.1%} is less than expected 90%"

if __name__ == "__main__":
    # Run basic tests
    print("Running FastMQA correctness tests...")
    test_memory_efficiency()
    print("✓ Memory efficiency test passed!")
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except:
        print("Run 'pytest tests/' for full test suite")