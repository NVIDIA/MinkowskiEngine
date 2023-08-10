# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
from collections import namedtuple
from collections.abc import Sequence
from functools import reduce
import numpy as np
from typing import Union

import torch
from MinkowskiCommon import convert_to_int_list
from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType
from MinkowskiCoordinateManager import CoordinateManager


def get_kernel_volume(region_type, kernel_size, region_offset, axis_types, dimension):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPER_CUBE, HYPER_CROSS with odd kernel sizes cannot
    use center=False.
    """
    if region_type == RegionType.HYPER_CUBE:
        assert reduce(
            lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
        ), "kernel_size must be positive"
        assert (
            region_offset is None
        ), "Region offset must be None when region_type is given"
        assert axis_types is None, "Axis types must be None when region_type is given"
        # Typical convolution kernel

        # Convolution kernel with even numbered kernel size not defined.
        kernel_volume = torch.prod(torch.IntTensor(kernel_size)).item()

    elif region_type == RegionType.HYPER_CROSS:
        assert reduce(
            lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
        ), "kernel_size must be positive"
        assert (
            torch.IntTensor(kernel_size) % 2
        ).prod().item() == 1, "kernel_size must be odd for region_type HYPER_CROSS"
        # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        kernel_volume = (torch.sum(torch.IntTensor(kernel_size) - 1) + 1).item()

    # elif region_type == RegionType.HYBRID:
    #     assert reduce(
    #         lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
    #     ), "kernel_size must be positive"
    #     assert (
    #         region_offset is None
    #     ), "region_offset must be None when region_type is HYBRID"
    #     kernel_size_list = kernel_size.tolist()
    #     kernel_volume = 1
    #     # First HYPER_CUBE
    #     for axis_type, curr_kernel_size, d in zip(
    #         axis_types, kernel_size_list, range(dimension)
    #     ):
    #         if axis_type == RegionType.HYPER_CUBE:
    #             kernel_volume *= curr_kernel_size

    #     # Second, HYPER_CROSS
    #     for axis_type, curr_kernel_size, d in zip(
    #         axis_types, kernel_size_list, range(dimension)
    #     ):
    #         if axis_type == RegionType.HYPER_CROSS:
    #             kernel_volume += curr_kernel_size - 1

    elif region_type == RegionType.CUSTOM:
        assert (
            region_offset.numel() > 0
        ), "region_offset must be non empty when region_type is CUSTOM"
        assert (
            region_offset.size(1) == dimension
        ), "region_offset must have the same dimension as the network"
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
    center: bool = True,
):
    """
    when center is True, the custom region_offset will be centered at the
    origin. Currently, for HYPER_CUBE, HYPER_CROSS with odd kernel sizes cannot
    use center=False.

    up_stride: stride for conv_transpose, otherwise set it as 1
    """
    if region_type == RegionType.HYPER_CUBE:
        if isinstance(region_offset, torch.Tensor):
            assert (
                region_offset.numel() == 0
            ), "Region offset must be empty when region_type is given"
        else:
            assert (
                region_offset is None
            ), "Region offset must be None when region_type is given"

        assert axis_types is None, "Axis types must be None when region_type is given"
        # Typical convolution kernel
        assert reduce(
            lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
        ), "kernel_size must be positive"
        # assert torch.unique(dilation).numel() == 1
        kernel_volume = reduce(lambda k1, k2: k1 * k2, kernel_size)

    elif region_type == RegionType.HYPER_CROSS:
        assert reduce(
            lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
        ), "kernel_size must be positive"
        assert (
            kernel_size % 2
        ).prod() == 1, "kernel_size must be odd for region_type HYPER_CROSS"
        # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        kernel_volume = (
            reduce(lambda k1, k2: k1 + k2, map(lambda k: k - 1, kernel_size)) + 1
        )

    elif region_type == RegionType.HYBRID:
        assert reduce(
            lambda k1, k2: k1 > 0 and k2 > 0, kernel_size
        ), "kernel_size must be positive"
        if isinstance(region_offset, torch.Tensor):
            assert (
                region_offset.numel() == 0
            ), "Region offset must be empty when region_type is given"
        else:
            assert (
                region_offset is None
            ), "Region offset must be None when region_type is given"

        region_offset = [
            [
                0,
            ]
            * dimension
        ]
        kernel_size_list = kernel_size.tolist()
        # First HYPER_CUBE
        for axis_type, curr_kernel_size, d in zip(
            axis_types, kernel_size_list, range(dimension)
        ):
            new_offset = []
            if axis_type == RegionType.HYPER_CUBE:
                for offset in region_offset:
                    for curr_offset in range(curr_kernel_size):
                        off_center = (
                            int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                        )
                        offset = offset.copy()  # Do not modify the original
                        # Exclude the coord (0, 0, ..., 0)
                        if curr_offset == off_center:
                            continue
                        offset[d] = (
                            (curr_offset - off_center)
                            * dilation[d]
                            * (tensor_stride[d] / up_stride[d])
                        )
                        new_offset.append(offset)
            region_offset.extend(new_offset)

        # Second, HYPER_CROSS
        for axis_type, curr_kernel_size, d in zip(
            axis_types, kernel_size_list, range(dimension)
        ):
            new_offset = []
            if axis_type == RegionType.HYPER_CROSS:
                for curr_offset in range(curr_kernel_size):
                    off_center = (
                        int(math.floor((curr_kernel_size - 1) / 2)) if center else 0
                    )
                    offset = [
                        0,
                    ] * dimension
                    # Exclude the coord (0, 0, ..., 0)
                    if curr_offset == off_center:
                        continue
                    offset[d] = (
                        (curr_offset - off_center)
                        * dilation[d]
                        * (tensor_stride[d] / up_stride[d])
                    )
                    new_offset.append(offset)
            region_offset.extend(new_offset)

        # Convert to CUSTOM type
        region_type = RegionType.CUSTOM
        region_offset = torch.IntTensor(region_offset)
        kernel_volume = int(region_offset.size(0))

    elif region_type == RegionType.CUSTOM:
        assert (
            region_offset.numel() > 0
        ), "region_offset must be non empty when region_type is CUSTOM"
        assert (
            region_offset.size(1) == dimension
        ), "region_offset must have the same dimension as the network"
        kernel_volume = int(region_offset.size(0))
        assert isinstance(
            region_offset.dtype, torch.IntTensor
        ), "region_offset must be a torch.IntTensor."
    else:
        raise NotImplementedError()

    if region_offset is None:
        region_offset = torch.IntTensor()

    return region_type, region_offset, kernel_volume


class KernelGenerator:
    __slots__ = (
        "cache",
        "kernel_size",
        "kernel_stride",
        "kernel_dilation",
        "region_type",
        "region_offsets",
        "axis_types",
        "dimension",
        "kernel_volume",
        "requires_strided_coordinates",
        "expand_coordinates",
    )

    def __init__(
        self,
        kernel_size=-1,
        stride=1,
        dilation=1,
        is_transpose: bool = False,
        region_type: RegionType = RegionType.HYPER_CUBE,
        region_offsets: torch.Tensor = None,
        expand_coordinates: bool = False,
        axis_types=None,
        dimension=-1,
    ):
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
        it is `[RegionType.HYPER_CUBE, RegionType.HYPER_CUBE,
        RegionType.HYPER_CROSS]`, the kernel would be rectangular for the
        first two dimensions and cross shaped for the thrid dimension.
        """
        assert dimension > 0
        assert isinstance(region_type, RegionType)

        kernel_size = convert_to_int_list(kernel_size, dimension)
        kernel_stride = convert_to_int_list(stride, dimension)
        kernel_dilation = convert_to_int_list(dilation, dimension)

        self.cache = {}
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_dilation = kernel_dilation
        self.region_type = region_type
        self.region_offsets = region_offsets if region_offsets else torch.IntTensor()
        self.axis_types = axis_types
        self.dimension = dimension
        self.kernel_volume = get_kernel_volume(
            region_type, kernel_size, region_offsets, axis_types, dimension
        )
        self.requires_strided_coordinates = reduce(
            lambda s1, s2: s1 == 1 and s2 == 1, kernel_stride
        )
        self.expand_coordinates = expand_coordinates

    def get_kernel(self, tensor_stride, is_transpose):
        assert len(tensor_stride) == self.dimension
        if tuple(tensor_stride) not in self.cache:
            up_stride = (
                self.stride
                if is_transpose
                else torch.Tensor(
                    [
                        1,
                    ]
                    * self.dimension
                )
            )

            self.cache[tuple(tensor_stride)] = convert_region_type(
                self.region_type,
                tensor_stride,
                self.kernel_size,
                up_stride,
                self.kernel_dilation,
                self.region_offsets,
                self.axis_types,
                self.dimension,
            )

        return self.cache[tuple(tensor_stride)]

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(kernel_size={self.kernel_size}, kernel_stride={self.kernel_stride}, kernel_dilation={self.kernel_dilation}, "
            + f"region_type={self.region_type}, expand_coordinates={self.expand_coordinates}, dimension={self.dimension})"
        )


class KernelRegion(
    namedtuple(
        "KernelRegion",
        (
            "kernel_size",
            "kernel_stride",
            "kernel_dilation",
            "region_type",
            "offset",
            "D",
        ),
    )
):
    """adding functionality to a named tuple"""

    __slots__ = ()

    def __init__(
        self,
        kernel_size,
        kernel_stride,
        kernel_dilation,
        region_type,
        offset,
        dimension,
    ):
        kernel_size = convert_to_int_list(kernel_size, dimension)
        kernel_stride = convert_to_int_list(kernel_stride, dimension)
        kernel_dilation = convert_to_int_list(kernel_dilation, dimension)
        super(KernelRegion, self).__init__(
            kernel_size, kernel_stride, kernel_dilation, region_type, offset, dimension
        )

    def __str__(self):
        return "kernel_size:{self.kernel_size}, kernel_stride:{self.kernel_stride}, region_type:{self.region_type}"


def save_ctx(
    ctx,  # function object context
    kernel_generator: KernelGenerator,
    in_coords_key: CoordinateMapKey,
    out_coords_key: CoordinateMapKey,
    coordinate_manager: CoordinateManager,
):
    ctx.kernel_generator = kernel_generator
    ctx.in_coordinate_map_key = in_coords_key
    ctx.out_coordinate_map_key = out_coords_key
    ctx.coordinate_manager = coordinate_manager
    return ctx
