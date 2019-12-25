import codecs
import os
import re
import subprocess
from sys import argv
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc

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

Extension = CUDAExtension
compile_args = ['make', '-j%d' % min(os.cpu_count(), 12)]
extra_compile_args = ['-Wno-deprecated-declarations']
# extra_compile_args+=['-g']  # Uncomment for debugging
if CPU_ONLY:
    argv.remove('--cpu_only')
    compile_args += ['CPU_ONLY=1']
    extra_compile_args += ['-DCPU_ONLY']
    Extension = CppExtension

if KEEP_OBJS:
    argv.remove('--keep_objs')

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
            sources=[
                'pybind/minkowski.cpp',
            ],
            libraries=[
                'minkowski',
                'openblas',  # for other blas, replace openblas
            ],
            library_dirs=['objs'],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Christopher Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Minkowski Engine, a Sparse Tensor Library for Neural Networks',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/StanfordVL/MinkowskiEngine',
    keywords=[
        'pytorch', 'Minkowski Engine', 'Sparse Tensor',
        'Convolutional Neural Networks', '3D Vision', 'Deep Learning'
    ],
    zip_safe=False,
    python_requires='>=3.6')
