# setup.py
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Check for CUDA
if not os.path.exists('/usr/local/cuda'):
    print("Warning: CUDA not found in standard location. Adjust CUDA path if needed.")

setup(
    name='fastmqa',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'fastmqa_cuda',
            ['kernels/mqa_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '--use_fast_math',
                    '--ptxas-options=-v',
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch>=2.0.0'],
)