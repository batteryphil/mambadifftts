from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Build flags
# On Windows, we need to use MSVC-style flags (/O2, /std:c++17) instead of GCC-style (-O3, -std=c++17)
extra_cxx = ["/O2", "/std:c++17"]
# nvcc flags: allow unsupported compiler (Visual Studio 2022/2025) and suppress STL version mismatch
extra_cuda = [
    "-O3",
    "-allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
]

setup(
    name="mamba_scan",
    ext_modules=[
        CUDAExtension(
            name="mamba_scan",
            sources=["mamba_scan.cpp", "mamba_scan_cuda.cu"],
            extra_compile_args={
                "cxx": extra_cxx,
                "nvcc": extra_cuda
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
