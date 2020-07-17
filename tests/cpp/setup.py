import sys
from sys import argv, platform
import torch.cuda
import os
import subprocess
from setuptools import setup
import unittest
from pathlib import Path

from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def run_command(*args):
    subprocess.check_call(args)


def _argparse(pattern, argv, is_flag=True):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern in arg]
        if len(arr) == 0:  # not found
            return False, argv
        else:
            assert "=" in arr[0], f"{arr[0]} requires a value."
            argv.remove(arr[0])
            return arr[0].split("=")[1], argv


SOURCE_SETS = {
    "convolution_cpu": [
        CppExtension,
        ["convolution_test.cpp"],
        ["math_functions.cpp", "coordinate_map_manager.cpp", "convolution_cpu.cpp"],
        ["-DCPU_ONLY"],
    ],
    "convolution_gpu": [
        CUDAExtension,
        ["convolution_test.cu"],
        [
            "math_functions.cpp",
            "coordinate_map_manager.cu",
            "convolution_gpu.cu",
            "coordinate_map_gpu.cu",
            "convolution_kernel.cu",
        ],
        [],
    ],
    "coordinate_map_manager_cpu": [
        CppExtension,
        ["coordinate_map_manager_cpu_test.cpp"],
        ["coordinate_map_manager.cpp"],
        ["-DCPU_ONLY"],
    ],
    "coordinate_map_manager_gpu": [
        CUDAExtension,
        ["coordinate_map_manager_gpu_test.cu"],
        ["coordinate_map_manager.cu", "coordinate_map_gpu.cu"],
        [],
    ],
    "coordinate_map_key": [CppExtension, ["coordinate_map_key_test.cpp"], [], [],],
    "coordinate_map_cpu": [CppExtension, ["coordinate_map_cpu_test.cpp"], [], [],],
    "coordinate_map_gpu": [
        CUDAExtension,
        ["coordinate_map_gpu_test.cu"],
        ["coordinate_map_gpu.cu"],
        [],
    ],
    "coordinate": [CppExtension, ["coordinate_test.cpp"], [], []],
    "kernel_region_cpu": [CppExtension, ["kernel_region_cpu_test.cpp"], [], []],
    "kernel_region_gpu": [
        CUDAExtension,
        ["kernel_region_gpu_test.cu"],
        ["coordinate_map_gpu.cu"],
        [],
    ],
    "type": [CppExtension, ["type_test.cpp"], [], []],
}

test_target, argv = _argparse("--test", argv, False)
no_debug, argv = _argparse("--nodebug", argv)

USE_NINJA = os.getenv("USE_NINJA") == "0"
HERE = Path(os.path.dirname(__file__)).absolute()
SRC_PATH = HERE.parent.parent / "src"
CXX = os.environ["CXX"]

assert test_target in SOURCE_SETS.keys()

if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = ["-fopenmp"]

NVCC_FLAGS = [f"-ccbin={CXX}", "--extended-lambda"]

if not no_debug:
    CXX_FLAGS += ["-g", "-DDEBUG"]
    NVCC_FLAGS += ["-g", "-DDEBUG"]
else:
    CXX_FLAGS += ["-O3"]
    NVCC_FLAGS += ["-O3"]

Extension = SOURCE_SETS[test_target][0]
CURR_TEST_FILES = SOURCE_SETS[test_target][1:3]
ARGS = SOURCE_SETS[test_target][3]
CXX_FLAGS += ARGS
NVCC_FLAGS += ARGS

ext_modules = [
    Extension(
        name="MinkowskiEngineTest._C",
        # ["type_test.cpp", "],
        sources=[
            *[str(HERE / test_file) for test_file in CURR_TEST_FILES[0]],
            *[str(SRC_PATH / src_file) for src_file in CURR_TEST_FILES[1]],
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS,},
        libraries=["openblas"],
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
    name="MinkowskiEngineTest",
    packages=[],
    ext_modules=ext_modules,
    include_dirs=[
        str(SRC_PATH),
        str(SRC_PATH / "3rdparty"),
        os.path.join(CUDA_HOME, "include"),
    ],
    test_suite="setup.suite",
    cmdclass={"build_ext": BuildExtension},
)
