import torch
import numpy as np
from collections import Sequence

import MinkowskiEngineBackend as MEB


def fnv_hash_vec(arr):
    """
    FNV64-1A

    Given a numpy array of N X D, generate hash values using the same hash
    function used for ME
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    FNV64-1A

    Given a numpy array of N X D, generate hash values using the same hash
    function used for ME
    """
    assert arr.ndim == 2
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64)

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    return_index=False,
                    hash_type='ravel',
                    quantization_size=1):
    r"""Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        coords (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a matrix of size
        :math:`N \times D` where :math:`N` is the number of points in the
        :math:`D` dimensional space.

        feats (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a matrix of size
        :math:`N \times D_F` where :math:`N` is the number of points and
        :math:`D_F` is the dimension of the features.

        labels (:attr:`numpy.ndarray`, optional): labels associated to eah coordinates.

        ignore_label (:attr:`int`, optional): the int value of the IGNORE LABEL.

        return_index (:attr:`bool`, optional): True if you want the indices of the
        quantized coordinates. False by default.

        hash_type (:attr:`str`, optional): Hash function used for quantization. Either
        `ravel` or `fnv`. `ravel` by default.

        quantization_size (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyperrectangle of of the grid cell.

    .. note::
        Please check `examples/indoor.py` for the usage.

    """
    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert hash_type in [
        'ravel', 'fnv'
    ], "Invalid hash_type. Either ravel, or fnv allowed. You put hash_type=" + hash_type
    assert coords.ndim == 2
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(quantization_size) == dimension, "Quantization size and coordinates size mismatch."
        quantization_size = [i for i in quantization_size]
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [int(quantization_size) for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coords)
    else:
        key = fnv_hash_vec(discrete_coords)

    if use_label:
        inds, labels = MEB.SparseVoxelization(key, labels.astype(np.int32),
                                              ignore_label, use_label)
        if return_index:
            return inds, labels
        else:
            return discrete_coords[inds], feats[inds], labels
    else:
        _, inds = np.unique(key, return_index=True)
        if return_index:
            return inds
        else:
            if use_feat:
                return discrete_coords[inds], feats[inds]
            else:
                return discrete_coords[inds]
