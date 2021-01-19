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
from collections import Sequence
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
from sparse_matrix_functions import MinkowskiSPMMFunction


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
        ], "invalid quantization mode"

        self.quantization_mode = quantization_mode

        if coordinates is not None:
            assert isinstance(coordinates, torch.Tensor)
        if coordinate_field_map_key is not None:
            assert isinstance(coordinate_field_map_key, CoordinateMapKey)
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
            batch_inds = torch.unique(self._C[:, 0])
            self._batch_rows = [self._C[:, 0] == b for b in batch_inds]
        return self._batch_rows

    def _get_coordinate_field(self):
        return self._manager.get_coordinate_field(self.coordinate_field_map_key)

    def sparse(
        self, tensor_stride: Union[int, Sequence, np.array] = 1, quantization_mode=None
    ):
        r"""Converts the current sparse tensor field to a sparse tensor."""
        if quantization_mode is None:
            quantization_mode = self.quantization_mode

        tensor_stride = convert_to_int_list(tensor_stride, self.D)

        sparse_tensor_key, (
            unique_index,
            inverse_mapping,
        ) = self._manager.field_to_sparse_insert_and_map(
            self.coordinate_field_map_key,
            tensor_stride,
        )

        self._inverse_mapping[sparse_tensor_key] = inverse_mapping

        if self.quantization_mode in [
            SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ]:
            spmm = MinkowskiSPMMFunction()
            N = len(self._F)
            cols = torch.arange(
                N,
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            vals = torch.ones(N, dtype=self._F.dtype, device=self._F.device)
            size = torch.Size([len(unique_index), len(inverse_mapping)])
            features = spmm.apply(inverse_mapping, cols, vals, size, self._F)
            if (
                self.quantization_mode
                == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
            ):
                nums = spmm.apply(
                    inverse_mapping,
                    cols,
                    vals,
                    size,
                    vals.reshape(N, 1),
                )
                features /= nums
        elif self.quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE:
            features = self._F[unique_index]
        else:
            # No quantization
            raise ValueError("Invalid quantization mode")

        sparse_tensor = SparseTensor(
            features,
            coordinate_map_key=sparse_tensor_key,
            coordinate_manager=self._manager,
        )

        return sparse_tensor

    def inverse_mapping(self, sparse_tensor_map_key: CoordinateMapKey):
        if sparse_tensor_map_key not in self._inverse_mapping:
            if not self._manager.exists_field_to_sparse(
                self.coordinate_field_map_key, sparse_tensor_map_key
            ):
                sparse_keys = self.coordinate_manager.field_to_sparse_keys(
                    self.coordinate_field_map_key
                )
                one_key = None
                for key in sparse_keys:
                    if np.prod(key.get_tensor_stride()) == 1:
                        one_key = key

                if one_key is not None:
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
                    raise ValueError(
                        f"The field to sparse tensor mapping does not exists for the key: {sparse_tensor_map_key}. Please run TensorField.sparse() before you call slice."
                    )
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
    )
