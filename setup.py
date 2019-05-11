import codecs
import os
import re
import time
from subprocess import Popen, PIPE, STDOUT
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

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


def run_command(cmd):
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT,
                    shell=True, encoding='utf8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
        time.sleep(0.1)

    rc = process.poll()
    return rc


run_command('make -j%d' % min(os.cpu_count(), 12))

# Python interface
setup(
    name='MinkowskiEngine',
    version=find_version('MinkowskiEngine', '__init__.py'),
    install_requires=['torch'],
    packages=[
        'MinkowskiEngine', 'MinkowskiEngine.utils', 'MinkowskiEngine.modules'
    ],
    package_dir={'MinkowskiEngine': './MinkowskiEngine'},
    ext_modules=[
        CUDAExtension(
            name='MinkowskiEngineBackend',
            include_dirs=['./', get_python_inc() + "/.."],
            sources=[
                'pybind/minkowski.cpp',
            ],
            libraries=['minkowski', 'openblas'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Christopher B. Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Minkowski Engine, a Sparse Tensor Library for Neural Networks',
    keywords='Minkowski Engine Sparse Tensor Library Convolutional Neural Networks',
    url='https://github.com/chrischoy/MinkowskiEngine',
    zip_safe=False,
)
