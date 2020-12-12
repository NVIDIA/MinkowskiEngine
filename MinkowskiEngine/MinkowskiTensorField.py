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
import warnings
import torch

from MinkowskiCommon import StrideType
from MinkowskiEngineBackend._C import (
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
    CoordinateMapKey,
)
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiTensor import (
    SparseTensorQuantizationMode,
    Tensor,
)
from MinkowskiSparseTensor import SparseTensor
from sparse_matrix_functions import MinkowskiSPMMFunction


class TensorField(Tensor):
    def __init__(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor = None,
        inverse_mapping: torch.Tensor = None,
        # optional coordinate related arguments
        tensor_stride: StrideType = 1,
        coordinate_map_key: CoordinateMapKey = None,
        coordinate_field_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        quantization_mode: SparseTensorQuantizationMode = SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        # optional manager related arguments
        allocator_type: GPUMemoryAllocatorType = None,
        minkowski_algorithm: MinkowskiAlgorithm = None,
        device=None,
    ):

        # To device
        if device is not None:
            features = features.to(device)
            coordinates = coordinates.to(device)

        assert quantization_mode in [
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        ], "invalid quantization mode"

        # A tensor field is a shallow wrapper on a sparse tensor, but keeps the original data for element-wise operations
        Tensor.__init__(
            self,
            features,
            coordinates,
            tensor_stride,
            coordinate_map_key,
            coordinate_manager,
            SparseTensorQuantizationMode.NO_QUANTIZATION,
            allocator_type,
            minkowski_algorithm,
            device,
        )

        # overwrite the current quantization mode
        self.quantization_mode = quantization_mode
        if inverse_mapping is not None:
            self.inverse_mapping = inverse_mapping

        self.coordinate_field_map_key = coordinate_field_map_key
        if coordinate_field_map_key is None:
            assert coordinates is not None
            self._CC = coordinates.float()
            self.coordinate_field_map_key = self._manager.insert_field(
                self._CC, *self.coordinate_map_key.get_key()
            )

    def initialize_coordinates(self, coordinates, features, coordinate_map_key):

        if not isinstance(coordinates, (torch.IntTensor, torch.cuda.IntTensor)):
            int_coordinates = torch.floor(coordinates).int()
        else:
            int_coordinates = coordinates

        (
            self.coordinate_map_key,
            (unique_index, self.inverse_mapping),
        ) = self._manager.insert_and_map(int_coordinates, *coordinate_map_key.get_key())
        self.unique_index = unique_index.long()
        int_coordinates = int_coordinates[self.unique_index]

        if self.quantization_mode in [
            SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ]:
            spmm = MinkowskiSPMMFunction()
            N = len(features)
            cols = torch.arange(
                N, dtype=self.inverse_mapping.dtype, device=self.inverse_mapping.device,
            )
            vals = torch.ones(N, dtype=features.dtype, device=features.device)
            size = torch.Size([len(self.unique_index), len(self.inverse_mapping)])
            features = spmm.apply(self.inverse_mapping, cols, vals, size, features)
            if (
                self.quantization_mode
                == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
            ):
                nums = spmm.apply(
                    self.inverse_mapping, cols, vals, size, vals.reshape(N, 1),
                )
                features /= nums
        elif self.quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE:
            features = features[self.unique_index]
        else:
            # No quantization
            pass

        return int_coordinates, features, coordinate_map_key

    @property
    def C(self):
        r"""The alias of :attr:`coords`.
        """
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
        if not hasattr(self, '_CC') or self._CC is None:
            self._CC = self._get_coordinate_field()
        return self._CC

    def _get_coordinate_field(self):
        return self._manager.get_coordinate_field(self.coordinate_field_map_key)

    def sparse(self):
        r"""Converts the current sparse tensor field to a sparse tensor."""
        spmm = MinkowskiSPMMFunction()
        N = len(self._F)
        assert N == len(self.inverse_mapping), "invalid inverse mapping"
        cols = torch.arange(
            N, dtype=self.inverse_mapping.dtype, device=self.inverse_mapping.device,
        )
        vals = torch.ones(N, dtype=self._F.dtype, device=self._F.device)
        size = torch.Size(
            [self._manager.size(self.coordinate_map_key), len(self.inverse_mapping)]
        )
        features = spmm.apply(self.inverse_mapping, cols, vals, size, self._F)
        # int_inverse_mapping = self.inverse_mapping.int()
        if self.quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE:
            nums = spmm.apply(self.inverse_mapping, cols, vals, size, vals.reshape(N, 1),)
            features /= nums

        return SparseTensor(
            features,
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self.coordinate_manager,
        )

    __slots__ = (
        "_C",
        "_CC",
        "_F",
        "_D",
        "coordinate_map_key",
        "coordinate_field_map_key",
        "_manager",
        "unique_index",
        "inverse_mapping",
        "quantization_mode",
        "_batch_rows",
    )
