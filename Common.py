import cffi

import collections
import numpy as np
from enum import Enum

import torch
import SparseConvolutionEngineFFI as SCE

ffi = cffi.FFI()


def convert_to_long_tensor(arg, dimension):
    if isinstance(arg, torch.LongTensor):
        assert arg.numel() == dimension
        return arg

    if isinstance(arg, (collections.Sequence, np.ndarray)):
        tmp = torch.LongTensor([i for i in arg])
        assert tmp.numel() == dimension
    elif isinstance(arg, str):
        raise ValueError('Input must be a scalar or a sequence')
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = torch.LongTensor([int(arg) for i in range(dimension)])
    else:
        raise ValueError('Input must be a scalar or a sequence')

    return tmp


class RegionType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    HYPERCROSS = 1, 'HYPERCROSS'
    CUSTOM = 2, 'CUSTOM'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Metadata(object):
    def __init__(self, D, ptr=0):
        self.D = D
        self.ffi = ffi.new('void *[1]')
        SCE.write_ffi_ptr(ptr, self.ffi)

    def clear(self):
        """
        Clear all coordinates and convolution maps
        """
        SCE.clear(self.D, self.ffi)
