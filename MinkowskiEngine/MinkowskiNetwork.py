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
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import convert_to_int_tensor


class MinkowskiNetwork(nn.Module, ABC):
    """
    MinkowskiNetwork: an abstract class for sparse convnets.

    Note: All modules that use the same coordinates must use the same net_metadata
    """

    def __init__(self, D):
        super(MinkowskiNetwork, self).__init__()
        self.D = D

    @abstractmethod
    def forward(self, x):
        pass

    def init(self, x):
        """
        Initialize coordinates if it does not exist
        """
        nrows = self.get_nrows(1)
        if nrows < 0:
            if isinstance(x, SparseTensor):
                self.initialize_coords(x.coords_man)
            else:
                raise ValueError('Initialize input coordinates')
        elif nrows != x.F.size(0):
            raise ValueError('Input size does not match the coordinate size')

    def get_index_map(self, coords, tensor_stride):
        r"""
        Get the current coords (with duplicates) index map.

        If `tensor_stride > 1`, use

        .. code-block:: python

           coords = torch.cat(((coords[:, :D] / tensor_stride) * tensor_stride, coords[:, D:]), dim=1)

        """
        assert isinstance(coords, torch.IntTensor), "Coord must be IntTensor"
        index_map = torch.IntTensor()
        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        success = MEB.get_index_map(coords.contiguous(), index_map,
                                    tensor_stride, self.D,
                                    self.net_metadata.ffi)
        if success < 0:
            raise ValueError('get_index_map failed')
        return index_map

    def permute_label(self, label, max_label, tensor_stride):
        if tensor_stride == 1 or np.prod(tensor_stride) == 1:
            return label

        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        permutation = self.get_permutation(tensor_stride, 1)
        nrows = self.get_nrows(tensor_stride)

        label = label.contiguous().numpy()
        permutation = permutation.numpy()

        counter = np.zeros((nrows, max_label), dtype='int32')
        np.add.at(counter, (permutation, label), 1)
        return torch.from_numpy(np.argmax(counter, 1))

    def permute_feature(self, feat, tensor_stride, dtype=np.float32):
        tensor_stride = convert_to_int_tensor(tensor_stride, self.D)
        permutation = self.get_permutation(tensor_stride, 1)
        nrows = self.get_nrows(tensor_stride)

        feat_np = feat.contiguous().numpy()
        warped_feat = np.zeros((nrows, feat.size(1)), dtype=dtype)
        counter = np.zeros((nrows, 1), dtype='int32')
        for j in range(feat.size(1)):
            np.add.at(warped_feat, (permutation, j), feat_np[:, j])
        np.add.at(counter, permutation, 1)
        warped_feat = warped_feat / counter
        return torch.from_numpy(warped_feat)
