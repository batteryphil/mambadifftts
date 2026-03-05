from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# CPU-optimized build flags
# -O3          : maximum optimization
# -march=native: use all available CPU ISA (AVX2 / AVX-512 if present)
# -fopenmp     : enable OpenMP parallelism (used by at::parallel_for)
# -ffast-math  : allow reassociation + reciprocal estimates (safe for SSM)
# -funroll-loops: unroll the N and D inner loops
# -std=c++17   : required by LibTorch headers

extra_cxx = [
    "-O3",
    "-std=c++17",
    "-march=native",
    "-fopenmp",
    "-ffast-math",
    "-funroll-loops",
]

setup(
    name="mamba_scan",
    ext_modules=[
        CppExtension(
            name="mamba_scan",
            sources=["mamba_scan.cpp"],
            extra_compile_args={"cxx": extra_cxx},
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
