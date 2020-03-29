# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import math
from collections import Sequence
import numpy as np
from enum import Enum
from itertools import product
from typing import Union

import torch

from torch.nn import Module

import MinkowskiEngineBackend as MEB


class GlobalPoolingMode(Enum):
    """
    Define the global pooling mode
    """
    AUTO = 0, 'AUTO'
    INDEX_SELECT = 1, 'INDEX_SELECT'
    SPARSE = 2, 'SPARSE'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class RegionType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    HYPERCROSS = 1, 'HYPERCROSS'
    CUSTOM = 2, 'CUSTOM'
    HYBRID = 3, 'HYBRID'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


def convert_to_int_list(arg: Union[int, Sequence, np.ndarray, torch.Tensor],
                        dimension: int):
    if isinstance(arg, list):
        assert len(arg) == dimension
        return arg

    if isinstance(arg, (Sequence, np.ndarray, torch.Tensor)):
        tmp = [i for i in arg]
        assert len(tmp) == dimension
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = [int(arg) for i in range(dimension)]
    else:
        raise ValueError('Input must be a scalar or a sequence')

    return tmp


def convert_to_int_tensor(
        arg: Union[int, Sequence, np.ndarray, torch.IntTensor], dimension: int):
    if isinstance(arg, torch.IntTensor):
        assert arg.numel() == dimension
        return arg

    if isinstance(arg, (Sequence, np.ndarray)):
        tmp = torch.IntTensor([i for i in arg])
        assert tmp.numel() == dimension
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = torch.IntTensor([int(arg) for i in range(dimension)])
    else:
        raise ValueError('Input must be a scalar or a sequence')

    return tmp


def prep_args(tensor_stride: Union[int, Sequence, np.ndarray, torch.IntTensor],
              stride: Union[int, Sequence, np.ndarray, torch.IntTensor],
              kernel_size: Union[int, Sequence, np.ndarray, torch.IntTensor],
              dilation: Union[int, Sequence, np.ndarray, torch.IntTensor],
              region_type: Union[int, RegionType],
              D=-1):
    assert torch.prod(
        kernel_size > 0
    ), f"kernel_size must be a positive integer, provided {kernel_size}"
    assert D > 0, f"dimension must be a positive integer, {D}"
    tensor_stride = convert_to_int_tensor(tensor_stride, D)
    stride = convert_to_int_tensor(stride, D)
    kernel_size = convert_to_int_tensor(kernel_size, D)
    dilation = convert_to_int_tensor(dilation, D)
    region_type = int(region_type)
    return tensor_stride, stride, kernel_size, dilation, region_type,


def get_postfix(tensor: torch.Tensor):
    postfix = 'GPU' if tensor.is_cuda else 'CPU'
    if isinstance(tensor, torch.DoubleTensor) or isinstance(
            tensor, torch.cuda.DoubleTensor):
        postfix += 'd'
    else:
        postfix += 'f'
    return postfix


def get_kernel_volume(region_type, kernel_size, region_offset, axis_types,
                      dimension):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPERCUBE, HYPERCROSS with odd kernel sizes cannot
    use center=False.
    """
    if region_type == RegionType.HYPERCUBE:
        assert region_offset is None, "Region offset must be None when region_type is given"
        assert axis_types is None, "Axis types must be None when region_type is given"
        # Typical convolution kernel
        assert torch.prod(kernel_size > 0) == 1

        # Convolution kernel with even numbered kernel size not defined.
        kernel_volume = int(torch.prod(kernel_size))

    elif region_type == RegionType.HYPERCROSS:
        assert torch.prod(kernel_size > 0) == 1, "kernel_size must be positive"
        assert (
            kernel_size %
            2).prod() == 1, "kernel_size must be odd for region_type HYPERCROSS"
        # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        kernel_volume = int(torch.sum(kernel_size - 1) + 1)

    elif region_type == RegionType.HYBRID:
        assert region_offset is None, \
            "region_offset must be None when region_type is HYBRID"
        kernel_size_list = kernel_size.tolist()
        kernel_volume = 1
        # First HYPERCUBE
        for axis_type, curr_kernel_size, d in \
                zip(axis_types, kernel_size_list, range(dimension)):
            if axis_type == RegionType.HYPERCUBE:
                kernel_volume *= curr_kernel_size

        # Second, HYPERCROSS
        for axis_type, curr_kernel_size, d in \
                zip(axis_types, kernel_size_list, range(dimension)):
            if axis_type == RegionType.HYPERCROSS:
                kernel_volume += (curr_kernel_size - 1)

    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel(
        ) > 0, "region_offset must be non empty when region_type is CUSTOM"
        assert region_offset.size(
            1
        ) == dimension, "region_offset must have the same dimension as the network"
        kernel_volume = int(region_offset.size(0))

    else:
        raise NotImplementedError()

    return kernel_volume


def convert_region_type(
        region_type: RegionType,
        tensor_stride: Union[Sequence, np.ndarray, torch.IntTensor],
        kernel_size: Union[Sequence, np.ndarray, torch.IntTensor],
        up_stride: Union[Sequence, np.ndarray, torch.IntTensor],
        dilation: Union[Sequence, np.ndarray, torch.IntTensor],
        region_offset: Union[Sequence, np.ndarray, torch.IntTensor],
        axis_types: Union[Sequence, np.ndarray, torch.IntTensor],
        dimension: int,
        center: bool = True):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPERCUBE, HYPERCROSS with odd kernel sizes cannot
    use center=False.

    up_stride: stride for conv_transpose, otherwise set it as 1
    """
    if region_type == RegionType.HYPERCUBE:
        assert region_offset is None, "Region offset must be None when region_type is given"
        assert axis_types is None, "Axis types must be None when region_type is given"
        # Typical convolution kernel
        assert torch.prod(kernel_size > 0) == 1
        # assert torch.unique(dilation).numel() == 1
        kernel_volume = int(torch.prod(kernel_size))

    elif region_type == RegionType.HYPERCROSS:
        assert torch.prod(kernel_size > 0) == 1, "kernel_size must be positive"
        assert (
            kernel_size %
            2).prod() == 1, "kernel_size must be odd for region_type HYPERCROSS"
        # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        kernel_volume = int(torch.sum(kernel_size - 1) + 1)

    elif region_type == RegionType.HYBRID:
        assert region_offset is None, \
            "region_offset must be None when region_type is HYBRID"
        region_offset = [[
            0,
        ] * dimension]
        kernel_size_list = kernel_size.tolist()
        # First HYPERCUBE
        for axis_type, curr_kernel_size, d in \
                zip(axis_types, kernel_size_list, range(dimension)):
            new_offset = []
            if axis_type == RegionType.HYPERCUBE:
                for offset in region_offset:
                    for curr_offset in range(curr_kernel_size):
                        off_center = int(
                            math.floor(
                                (curr_kernel_size - 1) / 2)) if center else 0
                        offset = offset.copy()  # Do not modify the original
                        # Exclude the coord (0, 0, ..., 0)
                        if curr_offset == off_center:
                            continue
                        offset[d] = (curr_offset - off_center) * \
                            dilation[d] * (tensor_stride[d] / up_stride[d])
                        new_offset.append(offset)
            region_offset.extend(new_offset)

        # Second, HYPERCROSS
        for axis_type, curr_kernel_size, d in \
                zip(axis_types, kernel_size_list, range(dimension)):
            new_offset = []
            if axis_type == RegionType.HYPERCROSS:
                for curr_offset in range(curr_kernel_size):
                    off_center = int(math.floor(
                        (curr_kernel_size - 1) / 2)) if center else 0
                    offset = [
                        0,
                    ] * dimension
                    # Exclude the coord (0, 0, ..., 0)
                    if curr_offset == off_center:
                        continue
                    offset[d] = (curr_offset - off_center) * \
                        dilation[d] * (tensor_stride[d] / up_stride[d])
                    new_offset.append(offset)
            region_offset.extend(new_offset)

        # Convert to CUSTOM type
        region_type = RegionType.CUSTOM
        region_offset = torch.IntTensor(region_offset)
        kernel_volume = int(region_offset.size(0))

    elif region_type == RegionType.CUSTOM:
        assert region_offset.numel(
        ) > 0, "region_offset must be non empty when region_type is CUSTOM"
        assert region_offset.size(
            1
        ) == dimension, "region_offset must have the same dimension as the network"
        kernel_volume = int(region_offset.size(0))
        assert isinstance(
            region_offset.dtype,
            torch.IntTensor), "region_offset must be a torch.IntTensor."
    else:
        raise NotImplementedError()

    if region_offset is None:
        region_offset = torch.IntTensor()

    return region_type, region_offset, kernel_volume


class KernelGenerator:

    def __init__(self,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 is_transpose=False,
                 region_type=RegionType.HYPERCUBE,
                 region_offsets=None,
                 axis_types=None,
                 dimension=-1):
        r"""
            :attr:`region_type` (RegionType, optional): defines the kernel
            shape. Please refer to MinkowskiEngine.Comon for details.

            :attr:`region_offset` (torch.IntTensor, optional): when the
            :attr:`region_type` is :attr:`RegionType.CUSTOM`, the convolution
            kernel uses the provided `region_offset` to define offsets. It
            should be a matrix of size :math:`N \times D` where :math:`N` is
            the number of offsets and :math:`D` is the dimension of the
            space.

            :attr:`axis_types` (list of RegionType, optional): If given, it
            uses different methods to create a kernel for each axis. e.g., when
            it is `[RegionType.HYPERCUBE, RegionType.HYPERCUBE,
            RegionType.HYPERCROSS]`, the kernel would be rectangular for the
            first two dimensions and cross shaped for the thrid dimension.
        """
        assert dimension > 0
        assert isinstance(region_type, RegionType)

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.cache = {}
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offsets = region_offsets
        self.axis_types = axis_types
        self.dimension = dimension
        self.kernel_volume = get_kernel_volume(region_type, kernel_size,
                                               region_offsets, axis_types,
                                               dimension)

    def get_kernel(self, tensor_stride, is_transpose):
        assert len(tensor_stride) == self.dimension
        if tuple(tensor_stride) not in self.cache:
            up_stride = self.stride \
                if is_transpose else torch.Tensor([1, ] * self.dimension)

            self.cache[tuple(tensor_stride)] = convert_region_type(
                self.region_type, tensor_stride, self.kernel_size, up_stride,
                self.dilation, self.region_offsets, self.axis_types,
                self.dimension)

        return self.cache[tuple(tensor_stride)]


class MinkowskiModuleBase(Module):
    pass


def get_minkowski_function(name, variable):
    fn_name = name + get_postfix(variable)
    if hasattr(MEB, fn_name):
        return getattr(MEB, fn_name)
    else:
        if variable.is_cuda:
            raise ValueError(
                f"Function {fn_name} not available. Please compile MinkowskiEngine where `torch.cuda.is_available()` is `True`."
            )
        else:
            raise ValueError(f"Function {fn_name} not available.")
