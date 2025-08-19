# lightning_setup.py
"""
Setup script for running FastMQA on Lightning AI Studios
This script handles GPU setup and actual CUDA kernel compilation
"""

import os
import sys
import subprocess

def setup_lightning_env():
    """Setup Lightning AI environment for CUDA development"""
    
    print("=" * 60)
    print("FastMQA Lightning AI Setup")
    print("=" * 60)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: {torch.cuda.get_device_name()}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
        else:
            print("✗ CUDA not available in this environment")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Install required packages
    print("\nInstalling requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Build CUDA extension
    print("\nBuilding CUDA extension...")
    try:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True)
        print("✓ CUDA extension built successfully!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to build CUDA extension")
        print("  Note: This is expected if not on a GPU machine")
        return False

def run_gpu_benchmarks():
    """Run benchmarks on GPU"""
    print("\n" + "=" * 60)
    print("Running GPU Benchmarks")
    print("=" * 60)
    
    # Import and run benchmarks
    sys.path.append('python')
    from benchmark import MQABenchmark
    
    benchmark = MQABenchmark(device='cuda')
    benchmark.run_benchmarks()

def test_cuda_kernel():
    """Test the actual CUDA kernel"""
    print("\n" + "=" * 60)
    print("Testing CUDA Kernel")
    print("=" * 60)
    
    import torch
    sys.path.append('python')
    from fastmqa import FastMQAttention
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 4, 32, 512, 128
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    K = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    V = torch.randn(batch_size, 1, seq_len, head_dim).cuda()
    
    # Test with CUDA kernel
    mqa = FastMQAttention(num_heads=num_heads, head_dim=head_dim, use_cuda=True)
    output = mqa(Q, K, V)
    
    print(f"✓ CUDA kernel executed successfully!")
    print(f"  Output shape: {output.shape}")
    
    # Profile performance
    mqa.profile(batch_size=batch_size, seq_len=seq_len)

if __name__ == "__main__":
    print("To run this on Lightning AI:")
    print("1. Create a new Lightning Studio with GPU (T4 or better)")
    print("2. Clone the repository:")
    print("   git clone https://github.com/JonSnow1807/FastMQA.git")
    print("3. Navigate to the directory:")
    print("   cd FastMQA")
    print("4. Run this setup script:")
    print("   python lightning_setup.py")
    print()
    
    if "LIGHTNING_CLOUD_APP_ID" in os.environ:
        print("Detected Lightning AI environment!")
        if setup_lightning_env():
            test_cuda_kernel()
            run_gpu_benchmarks()
    else:
        print("Not running in Lightning AI. Please run this script in a Lightning Studio.")
        print("\nFor local testing without GPU, run:")
        print("  python python/benchmark.py")