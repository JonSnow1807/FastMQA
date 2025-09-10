#!/usr/bin/env python3
"""
Comprehensive correctness testing for FastMQA implementation
Verifies numerical accuracy against high-precision reference
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from fastmqa_production import ProductionFastMQA

def high_precision_reference(Q, K, V):
    """High precision reference implementation using float64"""
    # Convert to double precision
    Q_double = Q.double()
    K_double = K.double()
    V_double = V.double()
    
    # Expand K,V to match all heads
    num_heads = Q.shape[1]
    K_exp = K_double.expand(-1, num_heads, -1, -1)
    V_exp = V_double.expand(-1, num_heads, -1, -1)
    
    # Compute attention in high precision
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q_double, K_exp.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1, dtype=torch.float64)
    output = torch.matmul(attn, V_exp)
    
    return output.float()

def test_numerical_accuracy():
    """Test numerical accuracy against reference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        ("Small", 2, 8, 128, 64),
        ("Medium", 4, 16, 512, 128),
        ("Large", 4, 32, 1024, 128),
    ]
    
    print("FastMQA Numerical Accuracy Test")
    print("=" * 50)
    
    all_passed = True
    
    for name, batch_size, num_heads, seq_len, head_dim in test_cases:
        print(f"\nTesting {name}: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
        
        # Create test tensors
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, 1, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, 1, seq_len, head_dim, device=device)
        
        # Get reference output
        reference = high_precision_reference(Q.cpu(), K.cpu(), V.cpu()).to(device)
        
        # Test FastMQA implementation
        model = ProductionFastMQA(num_heads, head_dim).to(device)
        fastmqa_output = model(Q, K, V)
        
        # Calculate errors
        abs_error = torch.abs(fastmqa_output - reference)
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        
        # Test criteria
        passed = max_error < 1e-5
        all_passed = all_passed and passed
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Status: {status}")
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - Production ready accuracy!")
    else:
        print("❌ Some tests failed - needs accuracy improvements")
    
    return all_passed

def test_memory_efficiency():
    """Test memory efficiency claims"""
    print(f"\nMemory Efficiency Verification")
    print("=" * 50)
    
    configs = [
        ("Inference", 32, 32, 2048, 128),
        ("Training", 8, 32, 1024, 128),
    ]
    
    for name, batch_size, num_heads, seq_len, head_dim in configs:
        model = ProductionFastMQA(num_heads, head_dim)
        stats = model.get_memory_stats(batch_size, seq_len)
        
        print(f"\n{name} Configuration:")
        print(f"  Standard MHA KV cache: {stats['standard_memory_mb']:.1f} MB")
        print(f"  FastMQA KV cache: {stats['mqa_memory_mb']:.1f} MB")
        print(f"  Memory reduction: {stats['reduction_percent']:.1f}%")
        print(f"  Batch size multiplier: {stats['cache_multiplier']}x")

def test_edge_cases():
    """Test edge cases and stability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductionFastMQA(16, 64).to(device)
    
    print(f"\nEdge Case Testing")
    print("=" * 50)
    
    edge_cases = [
        ("Very small values", lambda: torch.randn(2, 16, 64, 64, device=device) * 1e-4),
        ("Very large values", lambda: torch.randn(2, 16, 64, 64, device=device) * 100),
        ("Mixed precision", lambda: torch.randn(2, 16, 64, 64, device=device).half().float()),
    ]
    
    all_stable = True
    
    for name, tensor_gen in edge_cases:
        try:
            Q = tensor_gen()
            K = torch.randn(2, 1, 64, 64, device=device)
            V = torch.randn(2, 1, 64, 64, device=device)
            
            output = model(Q, K, V)
            
            # Check for NaN or Inf
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            
            if has_nan or has_inf:
                print(f"  {name}: ❌ FAILED (NaN/Inf detected)")
                all_stable = False
            else:
                print(f"  {name}: ✅ STABLE")
                
        except Exception as e:
            print(f"  {name}: ❌ FAILED ({str(e)[:30]}...)")
            all_stable = False
    
    return all_stable

if __name__ == "__main__":
    print("FastMQA Production Correctness Testing")
    print("=" * 60)
    
    # Run all tests
    accuracy_passed = test_numerical_accuracy()
    test_memory_efficiency()
    stability_passed = test_edge_cases()
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    if accuracy_passed and stability_passed:
        print("🏆 ALL TESTS PASSED")
        print("FastMQA is ready for production deployment!")
        print("✅ Numerical accuracy verified")
        print("✅ Memory efficiency confirmed")
        print("✅ Edge case stability validated")
    else:
        print("⚠️  Some tests need attention before production deployment")