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

if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = ["-g", "-fopenmp"]


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
    "coordinate_map_key": [
        CppExtension,
        ["coordinate_map_key_test.cpp"],
        ["coordinate_map_key.cpp"],
    ],
    "coordinate_map_cpu": [CppExtension, ["coordinate_map_cpu_test.cpp"], [],],
    "coordinate_map_gpu": [
        CUDAExtension,
        ["coordinate_map_gpu_test.cu"],
        ["coordinate_map_gpu.cu"],
    ],
    "coordinate": [CppExtension, ["coordinate_test.cpp"], []],
    "kernel_region_cpu": [CppExtension, ["kernel_region_cpu_test.cpp"], []],
    "kernel_region_gpu": [CUDAExtension, ["kernel_region_gpu_test.cu"], []],
    "type": [CppExtension, ["type_test.cpp"], []],
}

test_target, argv = _argparse("--test", argv, False)
assert test_target in SOURCE_SETS.keys()

USE_NINJA = os.getenv("USE_NINJA") == "0"
HERE = Path(os.path.dirname(__file__)).absolute()
SRC_PATH = HERE.parent.parent / "src"
OBJ_DIR = HERE / "objs"
ME_OBJ_DIR = OBJ_DIR / "ME"
CXX = os.environ["CXX"]

CURR_TEST_FILES = SOURCE_SETS[test_target][1:]
Extension = SOURCE_SETS[test_target][0]

CXX_FLAGS += ["-DDEBUG"]

ext_modules = [
    Extension(
        name="MinkowskiEngineTest._C",
        # ["type_test.cpp", "],
        sources=[
            *[str(HERE / test_file) for test_file in CURR_TEST_FILES[0]],
            *[str(SRC_PATH / src_file) for src_file in CURR_TEST_FILES[1]],
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": ["-O2", f"-ccbin={CXX}", "--extended-lambda", "-DDEBUG"],
        },
        # library_dirs=[str(OBJ_DIR), str(ME_OBJ_DIR)],
        # libraries=["coordinate_map_key"],
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
