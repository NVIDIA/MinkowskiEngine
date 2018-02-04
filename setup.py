from os import system
from setuptools import setup, find_packages
assert system('make') == 0
setup(
    name='SparseConvolutionEngine',
    version='0.1.0',
    install_requires=['torch', 'cffi'],
    packages=['SparseConvolutionEngine'],
    package_dir={'SparseConvolutionEngine': './'},
    package_data={
        'SparseConvolutionEngine': [
            'SparseConvolutionEngineFFI/__init__.py',
            'SparseConvolutionEngineFFI/_SparseConvolutionEngineFFI.so',
            'SparseConvolutionEngineFFI/libsparse.so',
        ],
    },
    author='Christopher B. Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Sparse Convolution Engine',
    keywords='Sparse Convolution Neural Network',
    url='https://github.com/chrischoy/SparseConvolutionEngine',
    zip_safe=False,
)
