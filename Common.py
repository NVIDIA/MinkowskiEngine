import cffi

import collections
import numpy as np
from enum import Enum
from itertools import product

import torch
import SparseConvolutionEngineFFI as SCE

ffi = cffi.FFI()


def convert_to_int_tensor(arg, dimension):
    if isinstance(arg, torch.IntTensor):
        assert arg.numel() == dimension
        return arg

    if isinstance(arg, (collections.Sequence, np.ndarray)):
        tmp = torch.IntTensor([i for i in arg])
        assert tmp.numel() == dimension
    elif isinstance(arg, str):
        raise ValueError('Input must be a scalar or a sequence')
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = torch.IntTensor([int(arg) for i in range(dimension)])
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


def convert_region_type(region_type, pixel_dist, kernel_size, dilation,
                        region_offset, dimension):
    if region_type == RegionType.HYPERCUBE:
        assert torch.unique(kernel_size).numel() == 1
        assert torch.unique(dilation).numel() == 1

        # Convolution kernel with even numbered kernel size not defined.
        if (kernel_size % 2).prod() == 1:  # Odd
            kernel_volume = int(torch.prod(kernel_size))
        elif (kernel_size % 2).sum() == 0:  # Even
            iter_args = []
            for d in range(dimension):
                off = (dilation[d] * pixel_dist[d] * torch.arange(
                    kernel_size[d]).int()).tolist()
                iter_args.append(off)
            region_offset = list(product(*iter_args))
            region_offset = torch.IntTensor(region_offset)
            kernel_volume = region_offset.size(0)
            region_type = RegionType.CUSTOM
        else:
            raise ValueError('All edges must have the same length.')
    elif region_type == RegionType.HYPERCROSS:
        assert (kernel_size % 2).prod() == 1
        # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        kernel_volume = int(torch.sum(kernel_size - 1) + 1)
    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel() > 0
        assert region_offset.size(1) == dimension
        kernel_volume = int(region_offset.size(0))
    else:
        raise NotImplementedError()
    return region_type, region_offset, kernel_volume


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
