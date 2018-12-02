import os
import torch
from os import path as osp
from torch.utils.ffi import create_extension

torch_dir = os.path.dirname(torch.__file__)
this_dir = osp.dirname(osp.realpath(__file__))

ffi = create_extension(
    'MinkowskiEngineFFI',
    headers=['ffi/minkowski.h'],
    sources=['ffi/minkowski.c'],
    include_dirs=[this_dir],
    relative_to=__file__,
    define_macros=[('WITH_CUDA', None)],
    with_cuda=True,
    libraries=['minkowski'],
    library_dirs=[this_dir + '/MinkowskiEngineFFI'],
    extra_compile_args=['-std=c99'],
    extra_link_args=["-Wl,-rpath=$ORIGIN"],
)

# If the rpath is not working correctly, use readelf -d
# SparseConvolutionEngineFFI/_SparseConvolutionEngineFFI.so to check the rpath
ffi.build()
