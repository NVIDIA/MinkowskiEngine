import os
import time
from subprocess import Popen, PIPE
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc


def run_command(cmd):
    process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
        time.sleep(0.1)

    rc = process.poll()
    return rc


run_command('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='MinkowskiEngine',
    version='0.2.2',
    install_requires=['torch'],
    packages=[
        'MinkowskiEngine', 'MinkowskiEngine.utils'
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
