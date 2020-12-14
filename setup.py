r"""
Parse additional arguments along with the setup.py arguments such as install, build, distribute, sdist, etc.


Usage:

  python setup.py install <additional_flags>..<additional_flags> <additional_arg>=<value>..<additional_arg>=<value>

  export CXX=<C++ compiler>; python setup.py install <additional_flags>..<additional_flags> <additional_arg>=<value>..<additional_arg>=<value>


Examples:

  python setup.py install --force_cuda --cuda_home=/usr/local/cuda
  export CXX=g++7; python setup.py install --force_cuda --cuda_home=/usr/local/cuda


Additional flags:

  --cpu_only: Force building only a CPU version. However, if
      torch.cuda.is_available() is False, it will default to CPU_ONLY.

  --force_cuda: If torch.cuda.is_available() is false, but you have a working
      nvcc, compile cuda files. --force_cuda will supercede --cpu_only.


Additional arguments:

  --blas=<value> : type of blas library to use for CPU matrix multiplications.
      Options: [openblas, mkl, atlas, blas]. By default, it will use the first
      numpy blas library it finds.

  --cuda_home=<value> : a directory that contains <value>/bin/nvcc and
      <value>/lib64/libcudart.so. By default, use
      `torch.utils.cpp_extension._find_cuda_home()`.

  --blas_include_dirs=<comma_separated_values> : additional include dirs. Only
      activated when --blas=<value> is set.

  --blas_library_dirs=<comma_separated_values> : additional library dirs. Only
      activated when --blas=<value> is set.
"""
import sys

if sys.version_info < (3, 6):
    sys.stdout.write(
        "Minkowski Engine requires Python 3.6 or higher. Please use anaconda https://www.anaconda.com/distribution/ for an isolated python environment.\n"
    )
    sys.exit(1)

try:
    import torch
except ImportError:
    raise ImportError("Pytorch not found. Please install pytorch first.")

import codecs
import os
import re
import subprocess
from sys import argv, platform
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc

if platform == "win32":
    raise ImportError("Windows is currently not supported.")
elif platform == "darwin":
    # Set the distutils to use clang instead of g++ for valid std
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang"

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


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


print("argv: ")
print(argv)
# For cpu only build
CPU_ONLY, argv = _argparse("--cpu_only", argv)
CPU_ONLY = CPU_ONLY or not torch.cuda.is_available()
KEEP_OBJS, argv = _argparse("--keep_objs", argv)
FORCE_CUDA, argv = _argparse("--force_cuda", argv)
print("CPU_ONLY: ")
print(CPU_ONLY)
print("FORCE_CUDA: ")
print(FORCE_CUDA)

# args with return value
CUDA_HOME, argv = _argparse("--cuda_home", argv, False)
BLAS, argv = _argparse("--blas", argv, False)
BLAS_INCLUDE_DIRS, argv = _argparse("--blas_include_dirs", argv, False)
BLAS_LIBRARY_DIRS, argv = _argparse("--blas_library_dirs", argv, False)

Extension = CUDAExtension
compile_args = [
    "make",
    "-j%d" % min(os.cpu_count(), 12),  # parallel compilation
    "PYTHON=" + sys.executable,  # curr python
]

extra_compile_args = []
#extra_compile_args = ["-Wno-deprecated-declarations"]
extra_link_args = []
libraries = ["minkowski"]

# extra_compile_args+=['-g']  # Uncomment for debugging
if CPU_ONLY and not FORCE_CUDA:
    print("--------------------------------")
    print("| WARNING: CPU_ONLY build set  |")
    print("--------------------------------")
    compile_args += ["CPU_ONLY=1"]
    extra_compile_args += ["-DCPU_ONLY"]
    Extension = CppExtension
else:
    # system python installation
    libraries.append("cusparse")
    libraries.append("cudadevrt")

if not (CUDA_HOME is False):  # False when not set, str otherwise
    print(f"Using CUDA_HOME={CUDA_HOME}")
    compile_args += [f"CUDA_HOME={CUDA_HOME}"]

if KEEP_OBJS:
    print("\nUsing built objects")

BLAS_LIST = ["openblas", "mkl", "atlas", "blas"]
if not (BLAS is False):  # False only when not set, str otherwise
    assert BLAS in BLAS_LIST
    if BLAS == "mkl":
        libraries.append("mkl_rt")
    else:
        libraries.append(BLAS)
    if not (BLAS_INCLUDE_DIRS is False):
        compile_args += [f"BLAS_INCLUDE_DIRS={BLAS_INCLUDE_DIRS}"]
    if not (BLAS_LIBRARY_DIRS is False):
        extra_link_args += [f"-Wl,-rpath,{BLAS_LIBRARY_DIRS}"]
else:
    # find the default BLAS library
    import numpy.distutils.system_info as sysinfo

    # Search blas in this order
    for blas in BLAS_LIST:
        if "libraries" in sysinfo.get_info(blas):
            BLAS = blas
            libraries += sysinfo.get_info(blas)["libraries"]
            break
    else:
        # BLAS not found
        raise ImportError(
            ' \
\nBLAS not found from numpy.distutils.system_info.get_info. \
\nPlease specify BLAS with: python setup.py install --blas=openblas" \
\nfor more information, please visit https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation'
        )

print(f"\nUsing BLAS={BLAS}")

compile_args += ["BLAS=" + BLAS]

if "darwin" in platform:
    extra_compile_args += ["-stdlib=libc++"]

if not KEEP_OBJS:
    run_command("make", "clean")

run_command(*compile_args)

'''
print("extra_compile_args: ")
print(extra_compile_args)
print("extra_link_args: ")
print(extra_link_args)
extra_compile_args = {
        #'cxx': ['-DBATCH_FIRST=1',],
        'cxx': ['-DBATCH_FIRST=1', '-MMD', '-MP', '-ffast-math', '-funsafe-math-optimizations', '-fno-math-errno', '-DBATCH_FIRST=1', '-fopenmp', '-fPIC', '-fwrapv', '-std=c++14', '-DNDEBUG', '-O3', '-DTORCH_API_INCLUDE_EXTENSION_H', '-DTORCH_EXTENSION_NAME=minkowski', '-D_GLIBCXX_USE_CXX11_ABI=0', '-Wall', '-Wcomment', '-Wno-sign-compare', '-Wno-deprecated-declarations',],
                        'nvcc': ['-DBATCH_FIRST=1', '-arch=sm_61', '-rdc=true', '--compiler-options', '-fPIC'],
                        'nvcclink': ['-arch=sm_61', '--device-link', '--compiler-options', '-fPIC'],
        }
#extra_link_args = ['-pthread', '--device-link', '--compiler-options', '-fPIC', '-Wall', '-Wcomment', '-Wno-sign-compare', '-Wno-deprecated-declarations']
extra_link_args = ['-pthread', '-fPIC', '-Wall', '-Wcomment', '-Wno-sign-compare', '-Wno-deprecated-declarations']
'''
# Python interface
setup(
    name="MinkowskiEngine",
    version=find_version("MinkowskiEngine", "__init__.py"),
    install_requires=["torch", "numpy"],
    packages=["MinkowskiEngine", "MinkowskiEngine.utils", "MinkowskiEngine.modules"],
    package_dir={"MinkowskiEngine": "./MinkowskiEngine"},
    ext_modules=[
        Extension(
            name="MinkowskiEngineBackend",
            include_dirs=[here, get_python_inc() + "/.."],
            library_dirs=["objs"],
            sources=["pybind/minkowski.cpp",
                ],
            libraries=libraries + ["cudart", "cudadevrt"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    author="Christopher Choy",
    author_email="chrischoy@ai.stanford.edu",
    description="a convolutional neural network library for sparse tensors",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/MinkowskiEngine",
    keywords=[
        "pytorch",
        "Minkowski Engine",
        "Sparse Tensor",
        "Convolutional Neural Networks",
        "3D Vision",
        "Deep Learning",
    ],
    zip_safe=False,
    classifiers=[
        # https://pypi.org/classifiers/
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
)

'''
                     "src/convolution.cpp",
                     "src/math_functions.cpp",
                     "src/coordsmap.cpp",
                     "src/gpu_coordsmap.cpp",
                     "src/pooling_max.cpp",
                     "src/coords_key.cpp",
                     "src/pooling_avg.cpp",
                     "src/pooling_global_avg.cpp",
                     "src/quantization.cpp",
                     "src/pooling_global_max.cpp",
                     "src/pruning.cpp",
                     "src/3rdparty/gpu_coords_map/include/cuda_unordered_map.cpp",
                     "src/broadcast.cpp",
                     "src/coords_manager.cpp",
                     "src/gpu_coords_manager.cpp",
                     "src/region.cpp",
                     "src/pooling_transpose.cpp",
                     "src/convolution_transpose.cpp",
                     "src/union.cpp",
                     "src/pooling_avg.cu",
                     "src/union.cu",
                     "src/pooling_max.cu",
                     "src/math_functions.cu",
                     "src/pruning.cu",
                     "src/3rdparty/gpu_coords_map/include/slab_hash/slab_hash.cu",
                     "src/3rdparty/gpu_coords_map/include/slab_hash/slab_alloc.cu",
                     "src/3rdparty/gpu_coords_map/include/coordinate.cu",
                     "src/broadcast.cu",
                     "src/gpu.cu",
                     "src/convolution.cu",
                ],
'''
