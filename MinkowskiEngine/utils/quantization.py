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
from collections.abc import Sequence
import MinkowskiEngineBackend._C as MEB
from typing import Union, Tuple
from MinkowskiCommon import convert_to_int_list


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
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
    r"""Returns a unique index map and an inverse index map.

    Args:
        :attr:`coords` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        matrix of size :math:`N \times D` where :math:`N` is the number of
        points in the :math:`D` dimensional space.

    Returns:
        :attr:`unique_map` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        list of indices that defines unique coordinates.
        :attr:`coords[unique_map]` is the unique coordinates.

        :attr:`inverse_map` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        list of indices that defines the inverse map that recovers the original
        coordinates.  :attr:`coords[unique_map[inverse_map]] == coords`

    Example::

       >>> unique_map, inverse_map = quantize(coords)
       >>> unique_coords = coords[unique_map]
       >>> print(unique_coords[inverse_map] == coords)  # True, ..., True
       >>> print(coords[unique_map[inverse_map]] == coords)  # True, ..., True

    """
    assert isinstance(coords, np.ndarray) or isinstance(
        coords, torch.Tensor
    ), "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert (
            coords.dtype == np.int32
        ), f"Invalid coords type {coords.dtype} != np.int32"
        return MEB.quantize_np(coords.astype(np.int32))
    else:
        # Type check done inside
        return MEB.quantize_th(coords.int())


def quantize_label(coords, labels, ignore_label):
    assert isinstance(coords, np.ndarray) or isinstance(
        coords, torch.Tensor
    ), "Invalid coords type"
    if isinstance(coords, np.ndarray):
        assert isinstance(labels, np.ndarray)
        assert (
            coords.dtype == np.int32
        ), f"Invalid coords type {coords.dtype} != np.int32"
        assert (
            labels.dtype == np.int32
        ), f"Invalid label type {labels.dtype} != np.int32"
        return MEB.quantize_label_np(coords, labels, ignore_label)
    else:
        assert isinstance(labels, torch.Tensor)
        # Type check done inside
        return MEB.quantize_label_th(coords, labels.int(), ignore_label)


def _auto_floor(array):
    assert isinstance(
        array, (np.ndarray, torch.Tensor)
    ), "array must be either np.array or torch.Tensor."

    if isinstance(array, np.ndarray):
        return np.floor(array)
    else:
        return torch.floor(array)


def sparse_quantize(
    coordinates,
    features=None,
    labels=None,
    ignore_label=-100,
    return_index=False,
    return_inverse=False,
    return_maps_only=False,
    quantization_size=None,
    device="cpu",
):
    r"""Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        :attr:`coordinates` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a
        matrix of size :math:`N \times D` where :math:`N` is the number of
        points in the :math:`D` dimensional space.

        :attr:`features` (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a
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

        :attr:`return_index` (:attr:`bool`, optional): set True if you want the
        indices of the quantized coordinates. False by default.

        :attr:`return_inverse` (:attr:`bool`, optional): set True if you want
        the indices that can recover the discretized original coordinates.
        False by default. `return_index` must be True when `return_reverse` is True.

        :attr:`return_maps_only` (:attr:`bool`, optional): if set, return the
        unique_map or optionally inverse map, but not the coordinates. Can be
        used if you don't care about final coordinates or if you use
        device==cuda and you don't need coordinates on GPU. This returns either
        unique_map alone or (unique_map, inverse_map) if return_inverse is set.

        :attr:`quantization_size` (attr:`float`, optional): if set, will use
        the quanziation size to define the smallest distance between
        coordinates.

        :attr:`device` (attr:`str`, optional): Either 'cpu' or 'cuda'.

        Example::

           >>> unique_map, inverse_map = sparse_quantize(discrete_coords, return_index=True, return_inverse=True)
           >>> unique_coords = discrete_coords[unique_map]
           >>> print(unique_coords[inverse_map] == discrete_coords)  # True

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
    assert isinstance(
        coordinates, (np.ndarray, torch.Tensor)
    ), "Coords must be either np.array or torch.Tensor."

    use_label = labels is not None
    use_feat = features is not None

    assert (
        coordinates.ndim == 2
    ), "The coordinates must be a 2D matrix. The shape of the input is " + str(
        coordinates.shape
    )

    if return_inverse:
        assert return_index, "return_reverse must be set with return_index"

    if use_feat:
        assert features.ndim == 2
        assert coordinates.shape[0] == features.shape[0]

    if use_label:
        assert coordinates.shape[0] == len(labels)

    dimension = coordinates.shape[1]
    # Quantize the coordinates
    if quantization_size is not None:
        if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
            assert (
                len(quantization_size) == dimension
            ), "Quantization size and coordinates size mismatch."
            if isinstance(coordinates, np.ndarray):
                quantization_size = np.array([i for i in quantization_size])
            else:
                quantization_size = torch.Tensor([i for i in quantization_size])
            discrete_coordinates = _auto_floor(coordinates / quantization_size)

        elif np.isscalar(quantization_size):  # Assume that it is a scalar

            if quantization_size == 1:
                discrete_coordinates = _auto_floor(coordinates)
            else:
                discrete_coordinates = _auto_floor(coordinates / quantization_size)
        else:
            raise ValueError("Not supported type for quantization_size.")
    else:
        discrete_coordinates = _auto_floor(coordinates)

    if isinstance(coordinates, np.ndarray):
        discrete_coordinates = discrete_coordinates.astype(np.int32)
    else:
        discrete_coordinates = discrete_coordinates.int()

    if (type(device) == str and device == "cpu") or (type(device) == torch.device and device.type == "cpu"):
        manager = MEB.CoordinateMapManagerCPU()
    elif (type(device) == str and "cuda" in device) or (type(device) == torch.device and device.type == "cuda"):
        manager = MEB.CoordinateMapManagerGPU_c10()
    else:
        raise ValueError("Invalid device. Only `cpu`, `cuda` or torch.device supported.")

    # Return values accordingly
    if use_label:
        if isinstance(coordinates, np.ndarray):
            unique_map, inverse_map, colabels = MEB.quantize_label_np(
                discrete_coordinates, labels, ignore_label
            )
        else:
            assert (
                not discrete_coordinates.is_cuda
            ), "Quantization with label requires cpu tensors."
            assert not labels.is_cuda, "Quantization with label requires cpu tensors."
            unique_map, inverse_map, colabels = MEB.quantize_label_th(
                discrete_coordinates, labels, ignore_label
            )
        return_args = [discrete_coordinates[unique_map]]
        if use_feat:
            return_args.append(features[unique_map])
        # Labels
        return_args.append(colabels)
        # Additional return args
        if return_index:
            return_args.append(unique_map)
        if return_inverse:
            return_args.append(inverse_map)

        if len(return_args) == 1:
            return return_args[0]
        else:
            return tuple(return_args)
    else:
        tensor_stride = [1 for i in range(discrete_coordinates.shape[1] - 1)]
        discrete_coordinates = (
            discrete_coordinates.to(device)
            if isinstance(discrete_coordinates, torch.Tensor)
            else torch.from_numpy(discrete_coordinates).to(device)
        )
        _, (unique_map, inverse_map) = manager.insert_and_map(
            discrete_coordinates, tensor_stride, ""
        )
        if return_maps_only:
            if return_inverse:
                return unique_map, inverse_map
            else:
                return unique_map

        return_args = [discrete_coordinates[unique_map]]
        if use_feat:
            return_args.append(features[unique_map])
        if return_index:
            return_args.append(unique_map)
        if return_inverse:
            return_args.append(inverse_map)

        if len(return_args) == 1:
            return return_args[0]
        else:
            return tuple(return_args)


def unique_coordinate_map(
    coordinates: torch.Tensor,
    tensor_stride: Union[int, Sequence, np.ndarray] = 1,
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    r"""Returns the unique indices and the inverse indices of the coordinates.

    :attr:`coordinates`: `torch.Tensor` (Int tensor. `CUDA` if
    coordinate_map_type == `CoordinateMapType.GPU`) that defines the
    coordinates.

    Example::

       >>> coordinates = torch.IntTensor([[0, 0], [0, 0], [0, 1], [0, 2]])
       >>> unique_map, inverse_map = unique_coordinates_map(coordinates)
       >>> coordinates[unique_map] # unique coordinates
       >>> torch.all(coordinates == coordinates[unique_map][inverse_map]) # True

    """
    assert coordinates.ndim == 2, "Coordinates must be a matrix"
    assert isinstance(coordinates, torch.Tensor)
    if not coordinates.is_cuda:
        manager = MEB.CoordinateMapManagerCPU()
    else:
        manager = MEB.CoordinateMapManagerGPU_c10()
    tensor_stride = convert_to_int_list(tensor_stride, coordinates.shape[-1] - 1)
    key, (unique_map, inverse_map) = manager.insert_and_map(
        coordinates, tensor_stride, ""
    )
    return unique_map, inverse_map
