import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

from distutils.sysconfig import get_python_inc

os.system('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='MinkowskiEngine',
    version='0.2.0',
    install_requires=['torch'],
    packages=['MinkowskiEngine', 'MinkowskiEngine.utils'],
    package_dir={'MinkowskiEngine': './'},
    ext_modules=[
        CUDAExtension(
            name='MinkowskiEngineBackend',
            include_dirs=['./', get_python_inc() + "/.."],  # For sparse hash from conda
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
    keywords=
    'Minkowski Engine Sparse Tensor Library Convolutional Neural Networks',
    url='https://github.com/chrischoy/MinkowskiEngine',
    zip_safe=False,
)
