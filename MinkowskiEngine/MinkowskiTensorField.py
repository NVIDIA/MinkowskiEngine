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
import torch

from MinkowskiCommon import convert_to_int_list, StrideType, MinkowskiModuleBase
from MinkowskiEngineBackend._C import (
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
    CoordinateMapType,
    CoordinateMapKey,
)
from MinkowskiCoordinateManager import (
    CoordinateManager,
    _allocator_type,
)
from MinkowskiSparseTensor import (
    _global_coordinate_manager,
    _sparse_tensor_operation_mode,
    SparseTensor,
    SparseTensorQuantizationMode,
)
from sparse_matrix_functions import spmm as _spmm


class TensorField(SparseTensor):
    def __init__(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor = None,
        inverse_mapping: torch.Tensor = None,
        # optional coordinate related arguments
        tensor_stride: StrideType = 1,
        coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        quantization_mode: SparseTensorQuantizationMode = SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        # optional manager related arguments
        allocator_type: GPUMemoryAllocatorType = None,
        minkowski_algorithm: MinkowskiAlgorithm = None,
        device=None,
    ):

        assert quantization_mode in [
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            SparseTensorQuantizationMode.UNWEIGHTED_SUM,
        ], "invalid quantization mode"

        # A tensor field is a shallow wrapper on a sparse tensor, but keeps the original data for element-wise operations
        SparseTensor.__init__(
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

    def sparse(self):
        r"""Converts the current sparse tensor field to a sparse tensor."""
        N = len(self._F)
        assert N == len(self.inverse_mapping), "invalid inverse mapping"
        cols = torch.arange(
            N, dtype=self.inverse_mapping.dtype, device=self.inverse_mapping.device,
        )
        vals = torch.ones(N, dtype=self._F.dtype, device=self._F.device)
        size = torch.Size(
            [self._manager.size(self.coordinate_map_key), len(self.inverse_mapping)]
        )
        features = _spmm(self.inverse_mapping, cols, vals, size, self._F)
        # int_inverse_mapping = self.inverse_mapping.int()
        if self.quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE:
            nums = _spmm(self.inverse_mapping, cols, vals, size, vals.reshape(N, 1),)
            features /= nums

        return SparseTensor(
            features,
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self.coordinate_manager,
        )
