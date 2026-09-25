# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v"],
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
