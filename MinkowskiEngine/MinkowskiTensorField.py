# Copyright (c) 2020 NVIDIA CORPORATION.
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
import os
import numpy as np
from collections.abc import Sequence
from typing import Union, List, Tuple

import torch
from MinkowskiCommon import convert_to_int_list, StrideType
from MinkowskiEngineBackend._C import (
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
    CoordinateMapKey,
    CoordinateMapType,
)
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiTensor import (
    SparseTensorOperationMode,
    SparseTensorQuantizationMode,
    Tensor,
    sparse_tensor_operation_mode,
    global_coordinate_manager,
    set_global_coordinate_manager,
    COORDINATE_MANAGER_DIFFERENT_ERROR,
    COORDINATE_KEY_DIFFERENT_ERROR,
)
from MinkowskiSparseTensor import SparseTensor
from sparse_matrix_functions import MinkowskiSPMMFunction, MinkowskiSPMMAverageFunction
from MinkowskiPooling import MinkowskiDirectMaxPoolingFunction


def create_splat_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    r"""Create splat coordinates. splat coordinates could have duplicate coordinates."""
    dimension = coordinates.shape[1] - 1
    region_offset = [
        [
            0,
        ]
        * (dimension + 1)
    ]
    for d in reversed(range(1, dimension + 1)):
        new_offset = []
        for offset in region_offset:
            offset = offset.copy()  # Do not modify the original
            offset[d] = 1
            new_offset.append(offset)
        region_offset.extend(new_offset)
    region_offset = torch.IntTensor(region_offset).to(coordinates.device)
    coordinates = torch.floor(coordinates).int().unsqueeze(1) + region_offset.unsqueeze(
        0
    )
    return coordinates.reshape(-1, dimension + 1)


class TensorField(Tensor):
    def __init__(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor = None,
        # optional coordinate related arguments
        tensor_stride: StrideType = 1,
        coordinate_field_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        quantization_mode: SparseTensorQuantizationMode = SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        # optional manager related arguments
        allocator_type: GPUMemoryAllocatorType = None,
        minkowski_algorithm: MinkowskiAlgorithm = None,
        requires_grad=None,
        device=None,
    ):
        r"""

        Args:
            :attr:`features` (:attr:`torch.FloatTensor`,
            :attr:`torch.DoubleTensor`, :attr:`torch.cuda.FloatTensor`, or
            :attr:`torch.cuda.DoubleTensor`): The features of a sparse
            tensor.

            :attr:`coordinates` (:attr:`torch.IntTensor`): The coordinates
            associated to the features. If not provided, :attr:`coordinate_map_key`
            must be provided.

            :attr:`tensor_stride` (:attr:`int`, :attr:`list`,
            :attr:`numpy.array`, or :attr:`tensor.Tensor`): The tensor stride
            of the current sparse tensor. By default, it is 1.

            :attr:`coordinate_field_map_key`
            (:attr:`MinkowskiEngine.CoordinateMapKey`): When the coordinates
            are already cached in the MinkowskiEngine, we could reuse the same
            coordinate map by simply providing the coordinate map key. In most
            case, this process is done automatically. When you provide a
            `coordinate_field_map_key`, `coordinates` will be be ignored.

            :attr:`coordinate_manager`
            (:attr:`MinkowskiEngine.CoordinateManager`): The MinkowskiEngine
            manages all coordinate maps using the `_C.CoordinateMapManager`. If
            not provided, the MinkowskiEngine will create a new computation
            graph. In most cases, this process is handled automatically and you
            do not need to use this.

            :attr:`quantization_mode`
            (:attr:`MinkowskiEngine.SparseTensorQuantizationMode`): Defines how
            continuous coordinates will be quantized to define a sparse tensor.
            Please refer to :attr:`SparseTensorQuantizationMode` for details.

            :attr:`allocator_type`
            (:attr:`MinkowskiEngine.GPUMemoryAllocatorType`): Defines the GPU
            memory allocator type. By default, it uses the c10 allocator.

            :attr:`minkowski_algorithm`
            (:attr:`MinkowskiEngine.MinkowskiAlgorithm`): Controls the mode the
            minkowski engine runs, Use
            :attr:`MinkowskiAlgorithm.MEMORY_EFFICIENT` if you want to reduce
            the memory footprint. Or use
            :attr:`MinkowskiAlgorithm.SPEED_OPTIMIZED` if you want to make it
            run fasterat the cost of more memory.

            :attr:`requires_grad` (:attr:`bool`): Set the requires_grad flag.

            :attr:`device` (:attr:`torch.device`): Set the device the sparse
            tensor is defined.
        """
        # Type checks
        assert isinstance(features, torch.Tensor), "Features must be a torch.Tensor"
        assert (
            features.ndim == 2
        ), f"The feature should be a matrix, The input feature is an order-{features.ndim} tensor."
        assert isinstance(quantization_mode, SparseTensorQuantizationMode)
        assert quantization_mode in [
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            SparseTensorQuantizationMode.MAX_POOL,
        ], "invalid quantization mode"

        self.quantization_mode = quantization_mode

        if coordinates is not None:
            assert isinstance(coordinates, torch.Tensor)
        if coordinate_field_map_key is not None:
            assert isinstance(coordinate_field_map_key, CoordinateMapKey)
            assert coordinate_manager is not None, "Must provide coordinate_manager if coordinate_field_map_key is provided"
            assert coordinates is None, "Must not provide coordinates if coordinate_field_map_key is provided"
        if coordinate_manager is not None:
            assert isinstance(coordinate_manager, CoordinateManager)
        if coordinates is None and (
            coordinate_field_map_key is None or coordinate_manager is None
        ):
            raise ValueError(
                "Either coordinates or (coordinate_field_map_key, coordinate_manager) pair must be provided."
            )

        Tensor.__init__(self)

        # To device
        if device is not None:
            features = features.to(device)
            if coordinates is not None:
                # assertion check for the map key done later
                coordinates = coordinates.to(device)

        self._D = (
            coordinates.size(1) - 1 if coordinates is not None else coordinate_manager.D
        )
        ##########################
        # Setup CoordsManager
        ##########################
        if coordinate_manager is None:
            # If set to share the coords man, use the global coords man
            if (
                sparse_tensor_operation_mode()
                == SparseTensorOperationMode.SHARE_COORDINATE_MANAGER
            ):
                coordinate_manager = global_coordinate_manager()
                if coordinate_manager is None:
                    coordinate_manager = CoordinateManager(
                        D=self._D,
                        coordinate_map_type=CoordinateMapType.CUDA
                        if coordinates.is_cuda
                        else CoordinateMapType.CPU,
                        allocator_type=allocator_type,
                        minkowski_algorithm=minkowski_algorithm,
                    )
                    set_global_coordinate_manager(coordinate_manager)
            else:
                coordinate_manager = CoordinateManager(
                    D=coordinates.size(1) - 1,
                    coordinate_map_type=CoordinateMapType.CUDA
                    if coordinates.is_cuda
                    else CoordinateMapType.CPU,
                    allocator_type=allocator_type,
                    minkowski_algorithm=minkowski_algorithm,
                )
        self._manager = coordinate_manager

        ##########################
        # Initialize coords
        ##########################
        # Coordinate Management
        if coordinates is not None:
            assert (
                features.shape[0] == coordinates.shape[0]
            ), "The number of rows in features and coordinates must match."

            assert (
                features.is_cuda == coordinates.is_cuda
            ), "Features and coordinates must have the same backend."

            coordinate_field_map_key = CoordinateMapKey(
                convert_to_int_list(tensor_stride, self._D), ""
            )
            coordinate_field_map_key = self._manager.insert_field(
                coordinates.float(), convert_to_int_list(tensor_stride, self._D), ""
            )
        else:
            assert (
                coordinate_field_map_key.is_key_set()
            ), "The coordinate field map key must be valid."

        if requires_grad is not None:
            features.requires_grad_(requires_grad)

        self._F = features
        self._C = coordinates
        self.coordinate_field_map_key = coordinate_field_map_key
        self._batch_rows = None
        self._inverse_mapping = {}
        self._splat = {}

    @property
    def coordinate_key(self):
        return self.coordinate_field_map_key

    @property
    def C(self):
        r"""The alias of :attr:`coords`."""
        return self.coordinates

    @property
    def coordinates(self):
        r"""
        The coordinates of the current sparse tensor. The coordinates are
        represented as a :math:`N \times (D + 1)` dimensional matrix where
        :math:`N` is the number of points in the space and :math:`D` is the
        dimension of the space (e.g. 3 for 3D, 4 for 3D + Time). Additional
        dimension of the column of the matrix C is for batch indices which is
        internally treated as an additional spatial dimension to disassociate
        different instances in a batch.
        """
        if self._C is None:
            self._C = self._get_coordinate_field()
        return self._C

    @property
    def _batchwise_row_indices(self):
        if self._batch_rows is None:
            _, self._batch_rows = self._manager.origin_field_map(
                self.coordinate_field_map_key
            )
        return self._batch_rows

    def _get_coordinate_field(self):
        return self._manager.get_coordinate_field(self.coordinate_field_map_key)

    def sparse(
        self,
        tensor_stride: Union[int, Sequence, np.array] = 1,
        coordinate_map_key: CoordinateMapKey = None,
        quantization_mode: SparseTensorQuantizationMode = None,
    ):
        r"""Converts the current sparse tensor field to a sparse tensor."""
        if quantization_mode is None:
            quantization_mode = self.quantization_mode
        assert (
            quantization_mode != SparseTensorQuantizationMode.SPLAT_LINEAR_INTERPOLATION
        ), "Please use .splat() for splat quantization."

        if coordinate_map_key is None:
            tensor_stride = convert_to_int_list(tensor_stride, self.D)

            coordinate_map_key, (
                unique_index,
                inverse_mapping,
            ) = self._manager.field_to_sparse_insert_and_map(
                self.coordinate_field_map_key,
                tensor_stride,
            )
            N_rows = len(unique_index)
        else:
            # sparse index, field index
            inverse_mapping, unique_index = self._manager.field_to_sparse_map(
                self.coordinate_field_map_key,
                coordinate_map_key,
            )
            N_rows = self._manager.size(coordinate_map_key)

        assert N_rows > 0, f"Invalid out coordinate map key. Found {N_row} elements."

        if len(inverse_mapping) == 0:
            # When the input has the same shape as the output
            self._inverse_mapping[coordinate_map_key] = torch.arange(
                len(self._F),
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            return SparseTensor(
                self._F,
                coordinate_map_key=coordinate_map_key,
                coordinate_manager=self._manager,
            )

        # Create features
        if quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_SUM:
            N = len(self._F)
            cols = torch.arange(
                N,
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            vals = torch.ones(N, dtype=self._F.dtype, device=self._F.device)
            size = torch.Size([N_rows, len(inverse_mapping)])
            features = MinkowskiSPMMFunction().apply(
                inverse_mapping, cols, vals, size, self._F
            )
        elif quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE:
            N = len(self._F)
            cols = torch.arange(
                N,
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            size = torch.Size([N_rows, len(inverse_mapping)])
            features = MinkowskiSPMMAverageFunction().apply(
                inverse_mapping, cols, size, self._F
            )
        elif quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE:
            features = self._F[unique_index]
        elif quantization_mode == SparseTensorQuantizationMode.MAX_POOL:
            N = len(self._F)
            in_map = torch.arange(
                N,
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            features = MinkowskiDirectMaxPoolingFunction().apply(
                in_map, inverse_mapping, self._F, N_rows
            )
        else:
            # No quantization
            raise ValueError("Invalid quantization mode")

        self._inverse_mapping[coordinate_map_key] = inverse_mapping

        return SparseTensor(
            features,
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=self._manager,
        )

    def splat(self):
        r"""
        For slice, use Y.slice(X) where X is the tensor field and Y is the
        resulting sparse tensor.
        """
        splat_coordinates = create_splat_coordinates(self.C)
        (coordinate_map_key, _) = self._manager.insert_and_map(splat_coordinates)
        N_rows = self._manager.size(coordinate_map_key)

        tensor_map, field_map, weights = self._manager.interpolation_map_weight(
            coordinate_map_key, self._C
        )
        # features
        N = len(self._F)
        assert weights.dtype == self._F.dtype
        size = torch.Size([N_rows, N])
        # Save the results for slice
        self._splat[coordinate_map_key] = (tensor_map, field_map, weights, size)
        features = MinkowskiSPMMFunction().apply(
            tensor_map, field_map, weights, size, self._F
        )
        return SparseTensor(
            features,
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=self._manager,
        )

    def inverse_mapping(self, sparse_tensor_map_key: CoordinateMapKey):
        if sparse_tensor_map_key not in self._inverse_mapping:
            if not self._manager.exists_field_to_sparse(
                self.coordinate_field_map_key, sparse_tensor_map_key
            ):
                sparse_keys = self.coordinate_manager.field_to_sparse_keys(
                    self.coordinate_field_map_key
                )
                one_key = None
                if len(sparse_keys) > 0:
                    for key in sparse_keys:
                        if np.prod(key.get_tensor_stride()) == 1:
                            one_key = key
                else:
                    one_key = CoordinateMapKey(
                        [
                            1,
                        ]
                        * self.D,
                        "",
                    )

                if one_key not in self._inverse_mapping:
                    (
                        _,
                        self._inverse_mapping[one_key],
                    ) = self._manager.get_field_to_sparse_map(
                        self.coordinate_field_map_key, one_key
                    )
                _, stride_map = self.coordinate_manager.stride_map(
                    one_key, sparse_tensor_map_key
                )
                field_map = self._inverse_mapping[one_key]
                self._inverse_mapping[sparse_tensor_map_key] = stride_map[field_map]
            else:
                # Extract the mapping
                (
                    _,
                    self._inverse_mapping[sparse_tensor_map_key],
                ) = self._manager.get_field_to_sparse_map(
                    self.coordinate_field_map_key, sparse_tensor_map_key
                )
        return self._inverse_mapping[sparse_tensor_map_key]

    def _is_same_key(self, other):
        assert isinstance(other, self.__class__)
        assert self._manager == other._manager, COORDINATE_MANAGER_DIFFERENT_ERROR
        assert (
            self.coordinate_field_map_key == other.coordinate_field_map_key
        ), COORDINATE_KEY_DIFFERENT_ERROR

    def _binary_functor(self, other, binary_fn):
        assert isinstance(other, (self.__class__, torch.Tensor))
        if isinstance(other, self.__class__):
            self._is_same_key(other)
            return self.__class__(
                binary_fn(self._F, other.F),
                coordinate_map_key=self.coordinate_map_key,
                coordinate_manager=self._manager,
            )
        else:  # when it is a torch.Tensor
            return self.__class__(
                binary_fn(self._F, other),
                coordinate_field_map_key=self.coordinate_map_key,
                coordinate_manager=self._manager,
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + os.linesep
            + "  coordinates="
            + str(self.C)
            + os.linesep
            + "  features="
            + str(self.F)
            + os.linesep
            + "  coordinate_field_map_key="
            + str(self.coordinate_field_map_key)
            + os.linesep
            + "  coordinate_manager="
            + str(self._manager)
            + "  spatial dimension="
            + str(self._D)
            + ")"
        )

    __slots__ = (
        "_C",
        "_F",
        "_D",
        "coordinate_field_map_key",
        "_manager",
        "quantization_mode",
        "_inverse_mapping",
        "_batch_rows",
        "_splat",
    )
