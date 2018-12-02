from os import system
from setuptools import setup, find_packages
assert system('make') == 0
setup(
    name='MinkowskiEngine',
    version='0.1.0',
    install_requires=['torch', 'cffi'],
    packages=['MinkowskiEngine'],
    package_dir={'MinkowskiEngine': './'},
    package_data={
        'MinkowskiEngine': [
            'MinkowskiEngineFFI/__init__.py',
            'MinkowskiEngineFFI/_MinkowskiEngineFFI.so',
            'MinkowskiEngineFFI/libminkowski.so',
        ],
    },
    author='Christopher B. Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Autodiff Sparse Tensor Library',
    keywords='Minkowski Engine Neural Network',
    url='https://github.com/chrischoy/MinkowskiEngine',
    zip_safe=False,
)
