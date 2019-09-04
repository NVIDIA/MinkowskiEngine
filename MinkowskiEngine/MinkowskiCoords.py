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
import numpy as np
import torch
from Common import convert_to_int_list
import MinkowskiEngineBackend as MEB


class CoordsKey():

    def __init__(self, D):
        self.D = D
        self.CPPCoordsKey = getattr(MEB, f'PyCoordsKey')()
        self.CPPCoordsKey.setDimension(D)

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


class CoordsManager():

    def __init__(self, D=-1):
        if D < 1:
            raise ValueError(f"Invalid dimension {D}")
        self.D = D
        CPPCoordsManager = getattr(MEB, f'PyCoordsManagerint32')
        coords_man = CPPCoordsManager()
        self.CPPCoordsManager = coords_man

    def initialize(self, coords, coords_key, enforce_creation=False):
        assert isinstance(coords_key, CoordsKey)
        self.CPPCoordsManager.initializeCoords(coords, coords_key.CPPCoordsKey,
                                               enforce_creation)

    def initialize_enforce(self, coords, coords_key):
        assert isinstance(coords_key, CoordsKey)
        self.CPPCoordsManager.initializeCoords(coords, coords_key.CPPCoordsKey,
                                               True)

    def get_coords_key(self, tensor_strides):
        tensor_strides = convert_to_int_list(tensor_strides, self.D)
        key = self.CPPCoordsManager.getCoordsKey(tensor_strides)
        coords_key = CoordsKey(self.D)
        coords_key.setKey(key)
        coords_key.setTensorStride(tensor_strides)
        return coords_key

    def get_coords(self, coords_key):
        assert isinstance(coords_key, CoordsKey)
        coords = torch.IntTensor()
        self.CPPCoordsManager.getCoords(coords, coords_key.CPPCoordsKey)
        return coords

    def get_row_indices_per_batch(self, coords_key):
        r"""
        return a list of unique batch indices, and a list of lists of row indices per batch.

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coords=coordinates)
           batch_indices, list_of_row_indices = sp_tensor.coords_man.get_row_indices_per_batch(sp_tensor.coords_key)

        """
        assert isinstance(coords_key, CoordsKey)
        out_key = CoordsKey(self.D)
        return self.CPPCoordsManager.getRowIndicesPerBatch(
            coords_key.CPPCoordsKey, out_key.CPPCoordsKey)

    def get_coo_broadcast_coords(self, coords_key, transpose=False):
        _, list_of_row_indices = self.get_row_indices_per_batch(coords_key)
        coos = []
        for batch_ind, row_inds in enumerate(list_of_row_indices):
            if transpose:
                coo = torch.LongTensor([[
                    batch_ind,
                ] * len(row_inds), row_inds])
            else:
                coo = torch.LongTensor([row_inds, [
                    batch_ind,
                ] * len(row_inds)])
            coos.append(coo)

        return torch.cat(coos, dim=1)

    def get_kernel_map(self,
                       in_tensor_strides,
                       out_tensor_strides,
                       stride=1,
                       kernel_size=3,
                       dilation=1,
                       region_type=0,
                       is_transpose=False):
        in_coords_key = self.get_coords_key(in_tensor_strides)
        out_coords_key = self.get_coords_key(out_tensor_strides)

        tensor_strides = convert_to_int_list(in_tensor_strides, self.D)
        strides = convert_to_int_list(stride, self.D)
        kernel_sizes = convert_to_int_list(kernel_size, self.D)
        dilations = convert_to_int_list(dilation, self.D)

        kernel_map = torch.IntTensor()
        self.CPPCoordsManager.getKernelMap(
            kernel_map, tensor_strides, strides, kernel_sizes, dilations,
            region_type, in_coords_key.CPPCoordsKey,
            out_coords_key.CPPCoordsKey, is_transpose)
        return kernel_map

    def get_coords_size_by_coords_key(self, coords_key):
        assert isinstance(coords_key, CoordsKey)
        return self.CPPCoordsManager.getCoordsSize(coords_key.CPPCoordsKey)

    def get_mapping_by_tensor_strides(self, in_tensor_strides,
                                      out_tensor_strides):
        in_key = self.get_coords_key(in_tensor_strides)
        out_key = self.get_coords_key(out_tensor_strides)
        return self.get_mapping_by_coords_key(in_key, out_key)

    def permute_label(self,
                      label,
                      max_label,
                      target_tensor_stride,
                      label_tensor_stride=1):
        """
        """
        if target_tensor_stride == label_tensor_stride:
            return label

        label_coords_key = self.get_coords_key(label_tensor_stride)
        target_coords_key = self.get_coords_key(target_tensor_stride)

        permutation = self.get_mapping_by_coords_key(label_coords_key,
                                                     target_coords_key)
        nrows = self.get_coords_size_by_coords_key(target_coords_key)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def __repr__(self):
        return str(self.CPPCoordsManager)
