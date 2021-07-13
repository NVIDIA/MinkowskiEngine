# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
import torch
import copy
from enum import Enum

from MinkowskiEngineBackend._C import CoordinateMapKey


class SparseTensorOperationMode(Enum):
    r"""Enum class for SparseTensor internal instantiation modes.

    :attr:`SEPARATE_COORDINATE_MANAGER`: always create a new coordinate manager.

    :attr:`SHARE_COORDINATE_MANAGER`: always use the globally defined coordinate
    manager. Must clear the coordinate manager manually by
    :attr:`MinkowskiEngine.SparseTensor.clear_global_coordinate_manager`.

    """
    SEPARATE_COORDINATE_MANAGER = 0
    SHARE_COORDINATE_MANAGER = 1


class SparseTensorQuantizationMode(Enum):
    r"""
    `RANDOM_SUBSAMPLE`: Subsample one coordinate per each quantization block randomly.
    `UNWEIGHTED_AVERAGE`: average all features within a quantization block equally.
    `UNWEIGHTED_SUM`: sum all features within a quantization block equally.
    `NO_QUANTIZATION`: No quantization is applied. Should not be used for normal operation.
    `MAX_POOL`: Voxel-wise max pooling is applied.
    `SPLAT_LINEAR_INTERPOLATION`: Splat features using N-dimensional linear interpolation to 2^N neighbors.
    """
    RANDOM_SUBSAMPLE = 0
    UNWEIGHTED_AVERAGE = 1
    UNWEIGHTED_SUM = 2
    NO_QUANTIZATION = 3
    MAX_POOL = 4
    SPLAT_LINEAR_INTERPOLATION = 5


_sparse_tensor_operation_mode = SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER
_global_coordinate_manager = None

COORDINATE_MANAGER_DIFFERENT_ERROR = "SparseTensors must share the same coordinate manager for this operation. Please refer to the SparseTensor creation API (https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html) to share the coordinate manager, or set the sparse tensor operation mode with `set_sparse_tensor_operation_mode` to share it by default."
COORDINATE_KEY_DIFFERENT_ERROR = "SparseTensors must have the same coordinate_map_key."


def set_sparse_tensor_operation_mode(operation_mode: SparseTensorOperationMode):
    r"""Define the sparse tensor coordinate manager operation mode.

    By default, a :attr:`MinkowskiEngine.SparseTensor.SparseTensor`
    instantiation creates a new coordinate manager that is not shared with
    other sparse tensors. By setting this function with
    :attr:`MinkowskiEngine.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER`, you
    can share the coordinate manager globally with other sparse tensors.
    However, you must explicitly clear the coordinate manger after use. Please
    refer to :attr:`MinkowskiEngine.clear_global_coordinate_manager`.

    Args:
        :attr:`operation_mode`
        (:attr:`MinkowskiEngine.SparseTensorOperationMode`): The operation mode
        for the sparse tensor coordinate manager. By default
        :attr:`MinkowskiEngine.SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER`.

    Example:

        >>> import MinkowskiEngine as ME
        >>> ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        >>> ...
        >>> a = ME.SparseTensor(...)
        >>> b = ME.SparseTensor(...)  # coords_man shared
        >>> ...  # one feed forward and backward
        >>> ME.clear_global_coordinate_manager()  # Must use to clear the coordinates after one forward/backward

    """
    assert isinstance(
        operation_mode, SparseTensorOperationMode
    ), f"Input must be an instance of SparseTensorOperationMode not {operation_mode}"
    global _sparse_tensor_operation_mode
    _sparse_tensor_operation_mode = operation_mode


def sparse_tensor_operation_mode() -> SparseTensorOperationMode:
    r"""Return the current sparse tensor operation mode."""
    global _sparse_tensor_operation_mode
    return copy.deepcopy(_sparse_tensor_operation_mode)


def global_coordinate_manager():
    r"""Return the current global coordinate manager"""
    global _global_coordinate_manager
    return _global_coordinate_manager


def set_global_coordinate_manager(coordinate_manager):
    r"""Set the global coordinate manager.

    :attr:`MinkowskiEngine.CoordinateManager` The coordinate manager which will
    be set to the global coordinate manager.
    """
    global _global_coordinate_manager
    _global_coordinate_manager = coordinate_manager


def clear_global_coordinate_manager():
    r"""Clear the global coordinate manager cache.

    When you use the operation mode:
    :attr:`MinkowskiEngine.SparseTensor.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER`,
    you must explicitly clear the coordinate manager after each feed forward/backward.
    """
    global _global_coordinate_manager
    _global_coordinate_manager = None


class Tensor:
    r"""A sparse tensor class. Can be accessed via
    :attr:`MinkowskiEngine.SparseTensor`.

    The :attr:`SparseTensor` class is the basic tensor in MinkowskiEngine. For
    the definition of a sparse tensor, please visit `the terminology page
    <https://nvidia.github.io/MinkowskiEngine/terminology.html#sparse-tensor>`_.
    We use the COOrdinate (COO) format to save a sparse tensor `[1]
    <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_. This
    representation is simply a concatenation of coordinates in a matrix
    :math:`C` and associated features :math:`F`.

    .. math::

       \mathbf{C} = \begin{bmatrix}
       b_1    & x_1^1  & x_1^2  & \cdots & x_1^D  \\
       \vdots & \vdots & \vdots & \ddots & \vdots \\
       b_N    & x_N^1  & x_N^2  & \cdots & x_N^D
       \end{bmatrix}, \; \mathbf{F} = \begin{bmatrix}
       \mathbf{f}_1^T\\
       \vdots\\
       \mathbf{f}_N^T
       \end{bmatrix}

    where :math:`\mathbf{x}_i \in \mathcal{Z}^D` is a :math:`D`-dimensional
    coordinate and :math:`b_i \in \mathcal{Z}_+` denotes the corresponding
    batch index. :math:`N` is the number of non-zero elements in the sparse
    tensor, each with the coordinate :math:`(b_i, x_i^1, x_i^1, \cdots,
    x_i^D)`, and the associated feature :math:`\mathbf{f}_i`. Internally, we
    handle the batch index as an additional spatial dimension.

    Example::

        >>> coords, feats = ME.utils.sparse_collate([coords_batch0, coords_batch1], [feats_batch0, feats_batch1])
        >>> A = ME.SparseTensor(features=feats, coordinates=coords)
        >>> B = ME.SparseTensor(features=feats, coordinate_map_key=A.coordiante_map_key, coordinate_manager=A.coordinate_manager)
        >>> C = ME.SparseTensor(features=feats, coordinates=coords, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        >>> D = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=2)

    .. warning::

       To use the GPU-backend for coordinate management, the
       :attr:`coordinates` must be a torch tensor on GPU. Applying `to(device)`
       after a :attr:`MinkowskiEngine.SparseTensor` initialization with a CPU
       `coordinates` will waste time and computation for creating a CPU
       CoordinateMap since GPU CoordinateMap will be created from scratch.

    .. warning::

       Before MinkowskiEngine version 0.4, we put the batch indices on the last
       column. Thus, direct manipulation of coordinates will be incompatible
       with the latest versions. Instead, please use
       :attr:`MinkowskiEngine.utils.batched_coordinates` or
       :attr:`MinkowskiEngine.utils.sparse_collate` to create batched
       coordinates.

       Also, to access coordinates or features batch-wise, use the functions
       :attr:`coordinates_at(batch_index : int)`, :attr:`features_at(batch_index : int)` of
       a sparse tensor. Or to access all batch-wise coordinates and features,
       `decomposed_coordinates`, `decomposed_features`,
       `decomposed_coordinates_and_features` of a sparse tensor.

       Example::

           >>> coords, feats = ME.utils.sparse_collate([coords_batch0, coords_batch1], [feats_batch0, feats_batch1])
           >>> A = ME.SparseTensor(features=feats, coordinates=coords)
           >>> coords_batch0 = A.coordinates_at(batch_index=0)
           >>> feats_batch1 = A.features_at(batch_index=1)
           >>> list_of_coords, list_of_featurs = A.decomposed_coordinates_and_features

    """

    @property
    def coordinate_manager(self):
        return self._manager

    @property
    def tensor_stride(self):
        return self.coordinate_map_key.get_tensor_stride()

    @tensor_stride.setter
    def tensor_stride(self, p):
        r"""
        This function is not recommended to be used directly.
        """
        raise SyntaxError("Direct modification of tensor_stride is not permitted")

    def _get_coordinates(self):
        return self._manager.get_coordinates(self.coordinate_map_key)

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
            self._C = self._get_coordinates()
        return self._C

    @property
    def coordinate_key(self):
        raise NotImplementedError("Tensor interface does not have coordinate_key")

    @C.setter
    def C(self):
        raise SyntaxError("Direct modification of coordinates is not permitted")

    @coordinates.setter
    def coordinates(self):
        raise SyntaxError("Direct modification of coordinates is not permitted")

    @property
    def F(self):
        r"""The alias of :attr:`feats`."""
        return self._F

    @property
    def features(self):
        r"""
        The features of the current sparse tensor. The features are :math:`N
        \times D_F` where :math:`N` is the number of points in the space and
        :math:`D_F` is the dimension of each feature vector. Please refer to
        :attr:`coords` to access the associated coordinates.
        """
        return self._F

    @property
    def _batchwise_row_indices(self):
        if self._batch_rows is None:
            _, self._batch_rows = self._manager.origin_map(self.coordinate_map_key)
        return self._batch_rows

    @property
    def _sorted_batchwise_row_indices(self):
        if self._sorted_batch_rows is None:
            batch_rows = self._batchwise_row_indices
            with torch.no_grad():
                self._sorted_batch_rows = [t.sort()[0] for t in batch_rows]
        return self._sorted_batch_rows

    @property
    def decomposition_permutations(self):
        r"""Returns a list of indices per batch that where indices defines the permutation of the batch-wise decomposition.

        Example::

            >>> # coords, feats, labels are given. All follow the same order
            >>> stensor = ME.SparseTensor(feats, coords)
            >>> conv = ME.MinkowskiConvolution(in_channels=3, out_nchannel=3, kernel_size=3, dimension=3)
            >>> list_of_featurs = stensor.decomposed_features
            >>> list_of_permutations = stensor.decomposition_permutations
            >>> # list_of_features == [feats[inds] for inds in list_of_permutations]
            >>> list_of_decomposed_labels = [labels[inds] for inds in list_of_permutations]
            >>> for curr_feats, curr_labels in zip(list_of_features, list_of_decomposed_labels):
            >>>     loss += torch.functional.mse_loss(curr_feats, curr_labels)
        """
        return self._batchwise_row_indices

    @property
    def decomposed_coordinates(self):
        r"""Returns a list of coordinates per batch.

        Returns a list of torch.IntTensor :math:`C \in \mathcal{R}^{N_i
        \times D}` coordinates per batch where :math:`N_i` is the number of non
        zero elements in the :math:`i`th batch index in :math:`D` dimensional
        space.

        .. note::

           The order of coordinates is non-deterministic within each batch. Use
           :attr:`decomposed_coordinates_and_features` to retrieve both
           coordinates features with the same order. To retrieve the order the
           decomposed coordinates is generated, use
           :attr:`decomposition_permutations`.

        """
        return [self.C[row_inds, 1:] for row_inds in self._batchwise_row_indices]

    def coordinates_at(self, batch_index):
        r"""Return coordinates at the specified batch index.

        Returns a torch.IntTensor :math:`C \in \mathcal{R}^{N_i
        \times D}` coordinates at the specified batch index where :math:`N_i`
        is the number of non zero elements in the :math:`i`th batch index in
        :math:`D` dimensional space.

        .. note::

           The order of coordinates is non-deterministic within each batch. Use
           :attr:`decomposed_coordinates_and_features` to retrieve both
           coordinates features with the same order. To retrieve the order the
           decomposed coordinates is generated, use
           :attr:`decomposition_permutations`.

        """
        return self.C[self._batchwise_row_indices[batch_index], 1:]

    @property
    def decomposed_features(self):
        r"""Returns a list of features per batch.

        Returns a list of torch.Tensor :math:`C \in \mathcal{R}^{N_i
        \times N_F}` features per batch where :math:`N_i` is the number of non
        zero elements in the :math:`i`th batch index in :math:`D` dimensional
        space.

        .. note::

           The order of features is non-deterministic within each batch. Use
           :attr:`decomposed_coordinates_and_features` to retrieve both
           coordinates features with the same order. To retrieve the order the
           decomposed features is generated, use
           :attr:`decomposition_permutations`.

        """
        return [self._F[row_inds] for row_inds in self._batchwise_row_indices]

    def features_at(self, batch_index):
        r"""Returns a feature matrix at the specified batch index.

        Returns a torch.Tensor :math:`C \in \mathcal{R}^{N
        \times N_F}` feature matrix :math:`N` is the number of non
        zero elements in the specified batch index and :math:`N_F` is the
        number of channels.

        .. note::

           The order of features is non-deterministic within each batch. Use
           :attr:`decomposed_coordinates_and_features` to retrieve both
           coordinates features with the same order. To retrieve the order the
           decomposed features is generated, use
           :attr:`decomposition_permutations`.

        """
        return self._F[self._batchwise_row_indices[batch_index]]

    def coordinates_and_features_at(self, batch_index):
        r"""Returns a coordinate and feature matrix at the specified batch index.

        Returns a coordinate and feature matrix at the specified `batch_index`.
        The coordinate matrix is a torch.IntTensor :math:`C \in \mathcal{R}^{N
        \times D}` where :math:`N` is the number of non zero elements in the
        specified batch index in :math:`D` dimensional space. The feature
        matrix is a torch.Tensor :math:`C \in \mathcal{R}^{N \times N_F}`
        matrix :math:`N` is the number of non zero elements in the specified
        batch index and :math:`N_F` is the number of channels.

        .. note::

           The order of features is non-deterministic within each batch. To
           retrieve the order the decomposed features is generated, use
           :attr:`decomposition_permutations`.

        """
        row_inds = self._batchwise_row_indices[batch_index]
        return self.C[row_inds, 1:], self._F[row_inds]

    @property
    def decomposed_coordinates_and_features(self):
        r"""Returns a list of coordinates and a list of features per batch.abs

        .. note::

           The order of decomposed coordinates and features is
           non-deterministic within each batch. To retrieve the order the
           decomposed features is generated, use
           :attr:`decomposition_permutations`.

        """
        row_inds_list = self._batchwise_row_indices
        return (
            [self.C[row_inds, 1:] for row_inds in row_inds_list],
            [self._F[row_inds] for row_inds in row_inds_list],
        )

    @property
    def dimension(self):
        r"""Alias of attr:`D`"""
        return self._D

    @dimension.setter
    def dimension(self):
        raise SyntaxError("Direct modification not permitted")

    @property
    def D(self):
        r"""Alias of attr:`D`"""
        return self._D

    @D.setter
    def D(self):
        raise SyntaxError("Direct modification not permitted")

    @property
    def requires_grad(self):
        return self._F.requires_grad

    def requires_grad_(self, requires_grad: bool = True):
        self._F.requires_grad_(requires_grad)

    def float(self):
        self._F = self._F.float()
        return self

    def double(self):
        self._F = self._F.double()
        return self

    def __len__(self):
        return len(self._F)

    def size(self):
        return self._F.size()

    @property
    def shape(self):
        return self._F.shape

    @property
    def device(self):
        return self._F.device

    @property
    def dtype(self):
        return self._F.dtype

    def detach(self):
        self._F = self._F.detach()
        return self

    def get_device(self):
        return self._F.get_device()

    def _is_same_key(self, other):
        assert isinstance(other, self.__class__)
        assert self._manager == other._manager, COORDINATE_MANAGER_DIFFERENT_ERROR
        assert (
            self.coordinate_map_key == other.coordinate_map_key
        ), COORDINATE_KEY_DIFFERENT_ERROR

    # Operation overloading
    def __iadd__(self, other):
        self._is_same_key(other)
        self._F += other.F
        return self

    def __isub__(self, other):
        self._is_same_key(other)
        self._F -= other.F
        return self

    def __imul__(self, other):
        self._is_same_key(other)
        self._F *= other.F
        return self

    def __idiv__(self, other):
        self._is_same_key(other)
        self._F /= other.F
        return self

    def _binary_functor(self, other, binary_fn):
        assert isinstance(other, (self.__class__, torch.Tensor))
        if isinstance(other, self.__class__):
            assert self._manager == other._manager, COORDINATE_MANAGER_DIFFERENT_ERROR

            if self.coordinate_map_key == other.coordinate_map_key:
                return self.__class__(
                    binary_fn(self._F, other.F),
                    coordinate_map_key=self.coordinate_map_key,
                    coordinate_manager=self._manager,
                )
            else:
                # Generate union maps
                out_key = CoordinateMapKey(
                    self.coordinate_map_key.get_coordinate_size()
                )
                union_maps = self.coordinate_manager.union_map(
                    [self.coordinate_map_key, other.coordinate_map_key], out_key
                )
                N_out = self.coordinate_manager.size(out_key)
                out_F = torch.zeros(
                    (N_out, self._F.size(1)), dtype=self.dtype, device=self.device
                )
                out_F[union_maps[0][1]] = self._F[union_maps[0][0]]
                out_F[union_maps[1][1]] = binary_fn(
                    out_F[union_maps[1][1]], other._F[union_maps[1][0]]
                )
                return self.__class__(
                    out_F, coordinate_map_key=out_key, coordinate_manager=self._manager
                )
        else:  # when it is a torch.Tensor
            return self.__class__(
                binary_fn(self._F, other),
                coordinate_map_key=self.coordinate_map_key,
                coordinate_manager=self._manager,
            )

    def __add__(self, other):
        r"""
        Add its feature with the corresponding feature of the other
        :attr:`MinkowskiEngine.SparseTensor` or a :attr:`torch.Tensor`
        element-wise. For coordinates that exist on one sparse tensor but not
        on the other, features of the counterpart that do not exist will be set
        to 0.
        """
        return self._binary_functor(other, lambda x, y: x + y)

    def __sub__(self, other):
        r"""
        Subtract the feature of the other :attr:`MinkowskiEngine.SparseTensor`
        or a :attr:`torch.Tensor` from its corresponding feature element-wise.
        For coordinates that exist on one sparse tensor but not on the other,
        features of the counterpart that do not exist will be set to 0.
        """
        return self._binary_functor(other, lambda x, y: x - y)

    def __mul__(self, other):
        r"""
        Multiply its feature of with the corresponding feature of the other
        :attr:`MinkowskiEngine.SparseTensor` or a :attr:`torch.Tensor`
        element-wise. For coordinates that exist on one sparse tensor but not
        on the other, features of the counterpart that do not exist will be set
        to 0.
        """
        return self._binary_functor(other, lambda x, y: x * y)

    def __truediv__(self, other):
        r"""
        Divide its feature by the corresponding feature of the other
        :attr:`MinkowskiEngine.SparseTensor` or a :attr:`torch.Tensor`
        element-wise. For coordinates that exist on one sparse tensor but not
        on the other, features of the counterpart that do not exist will be set
        to 0.
        """
        return self._binary_functor(other, lambda x, y: x / y)

    def __power__(self, power):
        return self.__class__(
            self._F ** power,
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self._manager,
        )

    __slots__ = (
        "_C",
        "_F",
        "_D",
        "coordinate_map_key",
        "_manager",
        "unique_index",
        "inverse_mapping",
        "quantization_mode",
        "_batch_rows",
    )
