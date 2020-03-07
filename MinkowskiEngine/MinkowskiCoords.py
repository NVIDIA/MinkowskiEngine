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
import os
import numpy as np
from collections import Sequence
from typing import Union

import torch
from Common import convert_to_int_list, convert_to_int_tensor, prep_args
import MinkowskiEngineBackend as MEB

CPU_COUNT = os.cpu_count()
if 'OMP_NUM_THREADS' in os.environ:
    CPU_COUNT = int(os.environ['OMP_NUM_THREADS'])


class CoordsKey():

    def __init__(self, D):
        self.D = D
        self.CPPCoordsKey = MEB.CoordsKey()
        self.CPPCoordsKey.setDimension(D)

    def isKeySet(self):
        return self.CPPCoordsKey.isKeySet()

    def setKey(self, key):
        self.CPPCoordsKey.setKey(key)

    def getKey(self):
        return self.CPPCoordsKey.getKey()

    def setTensorStride(self, tensor_stride):
        tensor_stride = convert_to_int_list(tensor_stride, self.D)
        self.CPPCoordsKey.setTensorStride(tensor_stride)

    def getTensorStride(self):
        return self.CPPCoordsKey.getTensorStride()

    def __repr__(self):
        return str(self.CPPCoordsKey)

    def __eq__(self, other):
        assert isinstance(other, CoordsKey)
        return self.getKey() == other.getKey()


class CoordsManager():

    def __init__(self, num_threads: int = -1, D: int = -1):
        if D < 1:
            raise ValueError(f"Invalid dimension {D}")
        self.D = D
        CPPCoordsManager = MEB.CoordsManager
        if num_threads < 0:
            num_threads = CPU_COUNT
        coords_man = CPPCoordsManager(num_threads)
        self.CPPCoordsManager = coords_man

    def initialize(self,
                   coords: torch.IntTensor,
                   coords_key: CoordsKey,
                   force_creation: bool = False,
                   force_remap: bool = False,
                   allow_duplicate_coords: bool = False) -> torch.LongTensor:
        assert isinstance(coords_key, CoordsKey)
        mapping = torch.LongTensor()
        self.CPPCoordsManager.initializeCoords(coords, mapping,
                                               coords_key.CPPCoordsKey,
                                               force_creation, force_remap,
                                               allow_duplicate_coords)
        return mapping

    def create_coords_key(self,
                          coords: torch.IntTensor,
                          tensor_stride: int = 1,
                          force_creation: bool = False,
                          force_remap: bool = False,
                          allow_duplicate_coords: bool = False) -> CoordsKey:
        coords_key = CoordsKey(self.D)
        coords_key.setTensorStride(tensor_stride)
        mapping = self.initialize(
            coords,
            coords_key,
            force_creation=True,
            force_remap=True,
            allow_duplicate_coords=True)
        # Set the tensor stride
        tensor_stride = convert_to_int_list(tensor_stride, self.D)
        coords_key.setTensorStride(tensor_stride)

        return coords_key

    def stride(self,
               coords_key: CoordsKey,
               stride: Union[int, Sequence, np.ndarray, torch.Tensor],
               force_creation: bool = False):
        assert isinstance(coords_key, CoordsKey)
        stride = convert_to_int_list(stride, self.D)

        strided_key = CoordsKey(self.D)
        tensor_stride = coords_key.getTensorStride()
        strided_key.setTensorStride(
            [t * s for t, s in zip(tensor_stride, stride)])

        strided_key.setKey(
            self.CPPCoordsManager.createStridedCoords(coords_key.getKey(),
                                                      tensor_stride, stride,
                                                      force_creation))
        return strided_key

    def reduce(self):
        origin_key = CoordsKey(self.D)
        origin_key.setTensorStride(convert_to_int_list(0, self.D))
        origin_key.setKey(self.CPPCoordsManager.createOriginCoords(self.D))
        return origin_key

    def transposed_stride(
            self,
            coords_key: CoordsKey,
            stride: Union[int, Sequence, np.ndarray, torch.Tensor],
            kernel_size: Union[int, Sequence, np.ndarray, torch.Tensor],
            dilation: Union[int, Sequence, np.ndarray, torch.Tensor],
            force_creation: bool = False):
        assert isinstance(coords_key, CoordsKey)
        stride = convert_to_int_list(stride, self.D)
        kernel_size = convert_to_int_list(kernel_size, self.D)
        dilation = convert_to_int_list(dilation, self.D)
        region_type = 0
        region_offset = torch.IntTensor()

        strided_key = CoordsKey(self.D)
        tensor_stride = coords_key.getTensorStride()
        strided_key.setTensorStride(
            [int(t / s) for t, s in zip(tensor_stride, stride)])

        strided_key.setKey(
            self.CPPCoordsManager.createTransposedStridedRegionCoords(
                coords_key.getKey(), coords_key.getTensorStride(), stride,
                kernel_size, dilation, region_type, region_offset,
                force_creation))
        return strided_key

    def _get_coords_key(self, key_or_tensor_strides):
        assert isinstance(key_or_tensor_strides, CoordsKey) or \
            isinstance(key_or_tensor_strides, (Sequence, np.ndarray, torch.IntTensor, int)), \
            f"The input must be either a CoordsKey or tensor_stride of type (int, list, tuple, array, Tensor). Invalid: {key_or_tensor_strides}"
        if isinstance(key_or_tensor_strides, CoordsKey):
            # Do nothing and return the input
            return key_or_tensor_strides
        else:
            tensor_strides = convert_to_int_list(key_or_tensor_strides, self.D)
            key = self.CPPCoordsManager.getCoordsKey(tensor_strides)
            coords_key = CoordsKey(self.D)
            coords_key.setKey(key)
            coords_key.setTensorStride(tensor_strides)
            return coords_key

    def get_coords(self, coords_key_or_tensor_strides):
        coords_key = self._get_coords_key(coords_key_or_tensor_strides)
        coords = torch.IntTensor()
        self.CPPCoordsManager.getCoords(coords, coords_key.CPPCoordsKey)
        return coords

    def get_batch_size(self):
        return self.CPPCoordsManager.getBatchSize()

    def get_batch_indices(self):
        return self.CPPCoordsManager.getBatchIndices()

    def set_origin_coords_key(self, coords_key: CoordsKey):
        self.CPPCoordsManager.setOriginCoordsKey(coords_key.CPPCoordsKey)

    def get_row_indices_per_batch(self, coords_key, out_coords_key=None):
        r"""Return a list of lists of row indices per batch.

        The corresponding batch indices are accessible by `get_batch_indices`.

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coords=coordinates)
           row_indices = sp_tensor.coords_man.get_row_indices_per_batch(sp_tensor.coords_key)

        """
        assert isinstance(coords_key, CoordsKey)
        if out_coords_key is None:
            out_coords_key = CoordsKey(self.D)
        return self.CPPCoordsManager.getRowIndicesPerBatch(
            coords_key.CPPCoordsKey, out_coords_key.CPPCoordsKey)

    def get_row_indices_at(self, coords_key, batch_index):
        r"""Return an torch.LongTensor of row indices for the specified batch index

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coords=coordinates)
           row_indices = sp_tensor.coords_man.get_row_indices_at(sp_tensor.coords_key, batch_index)

        """
        assert isinstance(coords_key, CoordsKey)
        out_coords_key = CoordsKey(self.D)
        return self.CPPCoordsManager.getRowIndicesAtBatchIndex(
            coords_key.CPPCoordsKey, out_coords_key.CPPCoordsKey, batch_index)

    def get_kernel_map(self,
                       in_key_or_tensor_strides,
                       out_key_or_tensor_strides,
                       stride=1,
                       kernel_size=3,
                       dilation=1,
                       region_type=0,
                       region_offset=None,
                       is_transpose=False,
                       is_pool=False,
                       on_gpu=False):
        r"""Get kernel in-out maps for the specified coords keys or tensor strides.

        """
        # region type 1 iteration with kernel_size 1 is invalid
        assert kernel_size > 0, "Invalid kernel size."
        if kernel_size == 1:
            region_type = 0

        if isinstance(in_key_or_tensor_strides, CoordsKey):
            in_tensor_strides = in_key_or_tensor_strides.getTensorStride()
        else:
            in_tensor_strides = in_key_or_tensor_strides
        if region_offset is None:
            region_offset = torch.IntTensor()

        in_coords_key = self._get_coords_key(in_key_or_tensor_strides)
        out_coords_key = self._get_coords_key(out_key_or_tensor_strides)

        tensor_strides = convert_to_int_tensor(in_tensor_strides, self.D)
        strides = convert_to_int_tensor(stride, self.D)
        kernel_sizes = convert_to_int_tensor(kernel_size, self.D)
        dilations = convert_to_int_tensor(dilation, self.D)
        D = in_coords_key.D
        tensor_strides, strides, kernel_sizes, dilations, region_type = prep_args(
            tensor_strides, strides, kernel_sizes, dilations, region_type, D)
        kernel_map_fn = self.CPPCoordsManager.getKernelMapGPU \
            if on_gpu else self.CPPCoordsManager.getKernelMapGPU
        kernel_map = kernel_map_fn(
            convert_to_int_list(tensor_strides, D),  #
            convert_to_int_list(strides, D),  #
            convert_to_int_list(kernel_sizes, D),  #
            convert_to_int_list(dilations, D),  #
            region_type,
            region_offset,
            in_coords_key.CPPCoordsKey,
            out_coords_key.CPPCoordsKey,
            is_transpose,
            is_pool)

        return kernel_map

    def get_coords_map(self, in_key_or_tensor_strides,
                       out_key_or_tensor_strides):
        r"""Extract input coords indices that maps to output coords indices.

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coords=coordinates)
           out_sp_tensor = stride_2_conv(sp_tensor)

           cm = sp_tensor.coords_man
           # cm = out_sp_tensor.coords_man  # doesn't matter which tensor you pick
           ins, outs = cm.get_coords_map(1,  # in stride
                                         2)  # out stride
           for i, o in zip(ins, outs):
              print(f"{i} -> {o}")

        """
        in_coords_key = self._get_coords_key(in_key_or_tensor_strides)
        out_coords_key = self._get_coords_key(out_key_or_tensor_strides)

        return self.CPPCoordsManager.getCoordsMap(in_coords_key.CPPCoordsKey,
                                                  out_coords_key.CPPCoordsKey)

    def get_union_map(self, in_keys, out_key: CoordsKey):
        r"""Extract input coords indices that maps to output coords indices.

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coords=coordinates)
           out_sp_tensor = stride_2_conv(sp_tensor)

           cm = sp_tensor.coords_man
           # cm = out_sp_tensor.coords_man  # doesn't matter which tensor you pick
           ins, outs = cm.get_coords_map(1,  # in stride
                                         2)  # out stride
           for i, o in zip(ins, outs):
              print(f"{i} -> {o}")

        """
        return self.CPPCoordsManager.getUnionMap(
            [key.CPPCoordsKey for key in in_keys], out_key.CPPCoordsKey)

    def get_coords_size_by_coords_key(self, coords_key):
        assert isinstance(coords_key, CoordsKey)
        return self.CPPCoordsManager.getCoordsSize(coords_key.CPPCoordsKey)

    def get_mapping_by_tensor_strides(self, in_tensor_strides,
                                      out_tensor_strides):
        in_key = self._get_coords_key(in_tensor_strides)
        out_key = self._get_coords_key(out_tensor_strides)
        return self.get_mapping_by_coords_key(in_key, out_key)

    def permute_label(self,
                      label,
                      max_label,
                      target_tensor_stride,
                      label_tensor_stride=1):
        if target_tensor_stride == label_tensor_stride:
            return label

        label_coords_key = self._get_coords_key(label_tensor_stride)
        target_coords_key = self._get_coords_key(target_tensor_stride)

        permutation = self.get_mapping_by_coords_key(label_coords_key,
                                                     target_coords_key)
        nrows = self.get_coords_size_by_coords_key(target_coords_key)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def print_diagnostics(self, coords_key: CoordsKey):
        assert isinstance(coords_key, CoordsKey)
        self.CPPCoordsManager.printDiagnostics(coords_key.CPPCoordsKey)

    def __repr__(self):
        return str(self.CPPCoordsManager)


def save_ctx(
        ctx,  # function object context
        tensor_stride: torch.IntTensor,
        stride: torch.IntTensor,
        kernel_size: torch.IntTensor,
        dilation: torch.IntTensor,
        region_type: int,
        in_coords_key: CoordsKey,
        out_coords_key: CoordsKey,
        coords_man: CoordsManager):
    ctx.tensor_stride = tensor_stride
    ctx.stride = stride
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.region_type = region_type
    ctx.in_coords_key = in_coords_key
    ctx.out_coords_key = out_coords_key
    ctx.coords_man = coords_man
    return ctx
