# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    print("Warning: CUDA not available. Building CPU-only version.")
    ext_modules = []
else:
    ext_modules = [
        CUDAExtension(
            name='fastmqa_cuda',
            sources=['kernels/mqa_extension.cpp', 'kernels/mqa_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }
        )
    ]

setup(
    name='fastmqa',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch>=2.0.0'],
)