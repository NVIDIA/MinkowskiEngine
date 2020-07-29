import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = ["-g"]

USE_NINJA = os.getenv("USE_NINJA") == "0"
HERE = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    CppExtension(
        "torch_test",
        ["coords_types.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
]

# if torch.cuda.is_available() and CUDA_HOME is not None:
#     extension = CUDAExtension(
#         'torch_test_cpp_extension.cuda', [
#             'cuda_extension.cpp',
#             'cuda_extension_kernel.cu',
#             'cuda_extension_kernel2.cu',
#         ],
#         extra_compile_args={'cxx': CXX_FLAGS,
#                             'nvcc': ['-O2']})
#     ext_modules.append(extension)

setup(
    name="torch_test",
    packages=["torch_test"],
    ext_modules=ext_modules,
    include_dirs=[
        os.path.join(HERE, "../../src"),
        os.path.join(HERE, "../../src/3rdparty"),
        os.path.join(CUDA_HOME, "include"),
    ],
    test_suite="tests",
    cmdclass={"build_ext": BuildExtension},
)
