import sys
if sys.version_info < (3, 6):
    sys.stdout.write(
        "Minkowski Engine requires Python 3.6 or higher. Please use anaconda https://www.anaconda.com/distribution/ for isolated python environment.\n"
    )
    sys.exit(1)

try:
    import torch
except ImportError:
    raise ImportError('Pytorch not found. Please install pytorch first.')

import codecs
import os
import re
import subprocess
from sys import argv, platform
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc

if platform == 'win32':
    raise ImportError('Windows is currently not supported.')
elif platform == 'darwin':
    # Set the distutils to use clang instead of g++ for valid std
    os.environ['CC'] = '/usr/local/opt/llvm/bin/clang'
    os.environ['CXX'] = '/usr/local/opt/llvm/bin/clang'

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def run_command(*args):
    subprocess.call(args)


# For cpu only build
CPU_ONLY = '--cpu_only' in argv
KEEP_OBJS = '--keep_objs' in argv
BLAS = [arg for arg in argv if '--blas' in arg]

Extension = CUDAExtension
compile_args = [
    'make',
    '-j%d' % min(os.cpu_count(), 12),  # parallel compilation
    'PYTHON=' + sys.executable,  # curr python
]
extra_compile_args = ['-Wno-deprecated-declarations']
libraries = ['minkowski']

# extra_compile_args+=['-g']  # Uncomment for debugging
if CPU_ONLY:
    print('\nCPU_ONLY build')
    argv.remove('--cpu_only')
    compile_args += ['CPU_ONLY=1']
    extra_compile_args += ['-DCPU_ONLY']
    Extension = CppExtension
else:
    # system python installation
    libraries.append('cusparse')

if KEEP_OBJS:
    print('\nUsing built objects')
    argv.remove('--keep_objs')

if len(BLAS) > 0:
    BLAS = BLAS[0]
    argv.remove(BLAS)
    BLAS = BLAS.split('=')[1]
    assert BLAS in ['openblas', 'mkl', 'atlas', 'blas']
else:
    # find the default BLAS library
    import numpy.distutils.system_info as sysinfo
    # Search blas in this order
    for blas in ['openblas', 'atlas', 'mkl', 'blas']:
        if 'libraries' in sysinfo.get_info(blas):
            BLAS = blas
            libraries += sysinfo.get_info(blas)['libraries']
            break
    else:
        # BLAS not found
        raise ImportError(' \
\nBLAS not found from numpy.distutils.system_info.get_info. \
\nPlease specify BLAS with: python setup.py install --blas=openblas" \
\nPlease visit https://github.com/StanfordVL/MinkowskiEngine/wiki/Installation \
for more detail.')

print(f'\nUsing BLAS={BLAS}')

compile_args += ['BLAS=' + BLAS]

if 'darwin' in platform:
    extra_compile_args += ['-stdlib=libc++']

if not KEEP_OBJS:
    run_command('make', 'clean')

run_command(*compile_args)

# Python interface
setup(
    name='MinkowskiEngine',
    version=find_version('MinkowskiEngine', '__init__.py'),
    install_requires=['torch', 'numpy'],
    packages=[
        'MinkowskiEngine', 'MinkowskiEngine.utils', 'MinkowskiEngine.modules'
    ],
    package_dir={'MinkowskiEngine': './MinkowskiEngine'},
    ext_modules=[
        Extension(
            name='MinkowskiEngineBackend',
            include_dirs=['./', get_python_inc() + "/.."],
            library_dirs=['objs'],
            sources=[
                'pybind/minkowski.cpp',
            ],
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Christopher Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='a convolutional neural network library for sparse tensors',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/StanfordVL/MinkowskiEngine',
    keywords=[
        'pytorch', 'Minkowski Engine', 'Sparse Tensor',
        'Convolutional Neural Networks', '3D Vision', 'Deep Learning'
    ],
    zip_safe=False,
    python_requires='>=3.6')
