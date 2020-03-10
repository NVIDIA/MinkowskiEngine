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
import torch
import numpy as np
from collections import Sequence
import MinkowskiEngineBackend as MEB


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def quantize(coords):
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert coords.dtype == np.int32, f"Invalid coords type {coords.dtype} != np.int32"
        return MEB.quantize_np(coords.astype(np.int32))
    else:
        # Type check done inside
        return MEB.quantize_th(coords.int())


def quantize_label(coords, labels, ignore_label):
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert isinstance(labels, np.ndarray)
        assert coords.dtype == np.int32, f"Invalid coords type {coords.dtype} != np.int32"
        assert labels.dtype == np.int32, f"Invalid label type {labels.dtype} != np.int32"
        return MEB.quantize_label_np(coords, labels, ignore_label)
    else:
        assert isinstance(labels, torch.Tensor)
        # Type check done inside
        return MEB.quantize_label_th(coords, labels.int(), ignore_label)


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=-100,
                    return_index=False,
                    quantization_size=None):
    r"""Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        :attr:`coords` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        matrix of size :math:`N \times D` where :math:`N` is the number of
        points in the :math:`D` dimensional space.

        :attr:`feats` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a
        matrix of size :math:`N \times D_F` where :math:`N` is the number of
        points and :math:`D_F` is the dimension of the features. Must have the
        same container as `coords` (i.e. if `coords` is a torch.Tensor, `feats`
        must also be a torch.Tensor).

        :attr:`labels` (:attr:`numpy.ndarray` or :attr:`torch.IntTensor`,
        optional): integer labels associated to eah coordinates.  Must have the
        same container as `coords` (i.e. if `coords` is a torch.Tensor,
        `labels` must also be a torch.Tensor). For classification where a set
        of points are mapped to one label, do not feed the labels.

        :attr:`ignore_label` (:attr:`int`, optional): the int value of the
        IGNORE LABEL.
        :attr:`torch.nn.CrossEntropyLoss(ignore_index=ignore_label)`

        :attr:`return_index` (:attr:`bool`, optional): True if you want the indices of the
        quantized coordinates. False by default.

        :attr:`quantization_size` (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyperrectangle of of the grid cell.

     Example::

        >>> # Segmentation
        >>> criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        >>> coords, feats, labels = MinkowskiEngine.utils.sparse_quantize(
        >>>     coords, feats, labels, ignore_label=-100, quantization_size=0.1)
        >>> output = net(MinkowskiEngine.SparseTensor(feats, coords))
        >>> loss = criterion(output.F, labels.long())
        >>>
        >>> # Classification
        >>> criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        >>> coords, feats = MinkowskiEngine.utils.sparse_quantize(coords, feats)
        >>> output = net(MinkowskiEngine.SparseTensor(feats, coords))
        >>> loss = criterion(output.F, labels.long())


    """
    assert isinstance(coords, np.ndarray) or isinstance(coords, torch.Tensor), \
        'Coords must be either np.array or torch.Tensor.'

    use_label = labels is not None
    use_feat = feats is not None

    assert coords.ndim == 2, \
        "The coordinates must be a 2D matrix. The shape of the input is " + str(coords.shape)

    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]

    if use_label:
        assert coords.shape[0] == len(labels)

    dimension = coords.shape[1]
    # Quantize the coordinates
    if quantization_size is not None:
        if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
            assert len(
                quantization_size
            ) == dimension, "Quantization size and coordinates size mismatch."
            if isinstance(coords, np.ndarray):
                quantization_size = np.array([i for i in quantization_size])
                discrete_coords = np.floor(coords / quantization_size)
            else:
                quantization_size = torch.Tensor([i for i in quantization_size])
                discrete_coords = (coords / quantization_size).floor()

        elif np.isscalar(quantization_size):  # Assume that it is a scalar

            if quantization_size == 1:
                discrete_coords = coords
            else:
                discrete_coords = np.floor(coords / quantization_size)
        else:
            raise ValueError('Not supported type for quantization_size.')
    else:
        discrete_coords = coords

    discrete_coords = np.floor(discrete_coords)
    if isinstance(coords, np.ndarray):
        discrete_coords = discrete_coords.astype(np.int32)
    else:
        discrete_coords = discrete_coords.int()

    # Return values accordingly
    if use_label:
        mapping, colabels = quantize_label(discrete_coords, labels,
                                           ignore_label)

        if return_index:
            return mapping, colabels
        else:
            if use_feat:
                return discrete_coords[mapping], feats[mapping], colabels
            else:
                return discrete_coords[mapping], colabels

    else:
        mapping = quantize(discrete_coords)
        if len(mapping) > 0:
            if return_index:
                return mapping
            else:
                if use_feat:
                    return discrete_coords[mapping], feats[mapping]
                else:
                    return discrete_coords[mapping]

        else:
            if return_index:
                if isinstance(discrete_coords, np.ndarray):
                    return np.arange(len(discrete_coords))
                else:
                    return torch.arange(len(discrete_coords), dtype=torch.long)
            else:
                if use_feat:
                    return discrete_coords, feats
                else:
                    return discrete_coords
