from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

import numpy.distutils.system_info as si
ai = si.atlas_info()

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
            include_dirs=['./'],
            sources=[
                'pybind/minkowski.cpp',
            ],
            libraries=['minkowski'],
            library_dirs=['objs', *ai.get_lib_dirs()],
            extra_link_args=['-lminkowski -lcblas -latlas'],
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
