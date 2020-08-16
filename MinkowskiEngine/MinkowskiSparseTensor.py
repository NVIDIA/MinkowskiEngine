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
import warnings
import torch
import copy
from enum import Enum

from MinkowskiCommon import convert_to_int_list, StrideType
from MinkowskiEngineBackend._C import (
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
    CoordinateMapType,
    CoordinateMapKey,
)
from MinkowskiCoordinateManager import CoordinateManager
from sparse_matrix_functions import spmm as _spmm


class SparseTensorOperationMode(Enum):
    r"""Enum class for SparseTensor internal instantiation modes.

    :attr:`SEPARATE_COORDINATE_MANAGER`: always create a new coordinate manager.

    :attr:`SHARE_COORDINATE_MANAGER`: always use the globally defined coordinate
    manager. Must clear the coordinate manager manually by
    :attr:`MinkowskiEngine.SparseTensor.clear_global_coordinate_mananager`.

    """
    SEPARATE_COORDINATE_MANAGER = 0
    SHARE_COORDINATE_MANAGER = 1


class SparseTensorQuantizationMode(Enum):
    r"""
    `RANDOM_SUBSAMPLE`: Subsample one coordinate per each quantization block randomly.
    `UNWEIGHTED_AVERAGE`: average all features within a quantization block equally.
    `UNWEIGHTED_SUM`: sum all features within a quantization block equally.
    `NO_QUANTIZATION`: No quantization is applied. Should not be used for normal operation.
    """
    RANDOM_SUBSAMPLE = 0
    UNWEIGHTED_AVERAGE = 1
    UNWEIGHTED_SUM = 2
    NO_QUANTIZATION = 3


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
    refer to :attr:`MinkowskiEngine.clear_global_coordinate_mananager`.

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
        >>> ME.clear_global_coordinate_mananager()  # Must use to clear the coordinates after one forward/backward

    """
    assert isinstance(
        operation_mode, SparseTensorOperationMode
    ), f"Input must be an instance of SparseTensorOperationMode not {operation_mode}"
    global _sparse_tensor_operation_mode
    _sparse_tensor_operation_mode = operation_mode


def sparse_tensor_operation_mode():
    global _sparse_tensor_operation_mode
    return copy.deepcopy(_sparse_tensor_operation_mode)


def clear_global_coordinate_mananager():
    r"""Clear the global coordinate manager cache.

    When you use the operation mode:
    :attr:`MinkowskiEngine.SparseTensor.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER`,
    you must explicitly clear the coordinate manager after each feed forward/backward.
    """
    global _global_coordinate_manager
    _global_coordinate_manager = None


class SparseTensor:
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
           >>> A = ME.SparseTensor(feats=feats, coords=coords)
           >>> coords_batch0 = A.coordinates_at(batch_index=0)
           >>> feats_batch1 = A.features_at(batch_index=1)
           >>> list_of_coords, list_of_featurs = A.decomposed_coordinates_and_features

    """

    def __init__(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor = None,
        # optional coordinate related arguments
        tensor_stride: StrideType = 1,
        coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        quantization_mode: SparseTensorQuantizationMode = SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        # optional manager related arguments
        allocator_type: GPUMemoryAllocatorType = None,
        minkowski_algorithm: MinkowskiAlgorithm = None,
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

            :attr:`coordinate_map_key`
            (:attr:`MinkowskiEngine.CoordinateMapKey`): When the coordinates
            are already cached in the MinkowskiEngine, we could reuse the same
            coordinate map by simply providing the coordinate map key. In most
            case, this process is done automatically. When you provide a
            `coordinate_map_key`, `coordinates` will be be ignored.

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

            :attr:`tensor_stride` (:attr:`int`, :attr:`list`,
            :attr:`numpy.array`, or :attr:`tensor.Tensor`): The tensor stride
            of the current sparse tensor. By default, it is 1.

        """
        # Type checks
        assert isinstance(features, torch.Tensor), "Features must be a torch.Tensor"
        assert (
            features.ndim == 2
        ), f"The feature should be a matrix, The input feature is an order-{features.ndim} tensor."
        assert isinstance(quantization_mode, SparseTensorQuantizationMode)
        self.quantization_mode = quantization_mode

        if coordinates is not None:
            assert isinstance(coordinates, torch.Tensor)
        if coordinate_map_key is not None:
            assert isinstance(coordinate_map_key, CoordinateMapKey)
        if coordinate_manager is not None:
            assert isinstance(coordinate_manager, CoordinateManager)

        # To device
        if device is not None:
            features = features.to(device)
            coordinates = coordinates.to(device)

        # Coordinate Management
        self._D = 0  # coordinate size - 1
        if coordinates is None and (
            coordinate_map_key is None or coordinate_manager is None
        ):
            raise ValueError(
                "Either coordinates or (coordinate_map_key, coordinate_manager) pair must be provided."
            )
        elif coordinates is not None:
            if not isinstance(coordinates, (torch.IntTensor, torch.cuda.IntTensor)):
                warnings.warn(
                    "coordinates implicitly converted to torch.IntTensor. "
                    + "To remove this warning, use `.int()` to convert the "
                    + "coords into an torch.IntTensor"
                )
                coordinates = torch.floor(coordinates).int()
            assert (
                features.shape[0] == coordinates.shape[0]
            ), "The number of rows in features and coordinates must match."
            self._D = coordinates.size(1) - 1

            coordinate_map_key = CoordinateMapKey(
                convert_to_int_list(tensor_stride, self._D), ""
            )
        else:
            # not (coordinate_map_key is None or coordinate_manager is None)
            self._D = coordinate_manager.D

        ##########################
        # Setup CoordsManager
        ##########################
        if coordinate_manager is None:
            # If set to share the coords man, use the global coords man
            global _sparse_tensor_operation_mode, _global_coordinate_manager
            if (
                _sparse_tensor_operation_mode
                == SparseTensorOperationMode.SHARE_COORDINATE_MANAGER
            ):
                if _global_coordinate_manager is None:
                    _global_coordinate_manager = CoordinateManager(
                        D=self._D,
                        coordinate_map_type=CoordinateMapType.CUDA
                        if coordinates.is_cuda
                        else CoordinateMapType.CPU,
                        allocator_type=allocator_type,
                    )
                coordinate_manager = _global_coordinate_manager
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
        if coordinates is not None:
            assert (
                features.is_cuda == coordinates.is_cuda
            ), "Features and coordinates must have the same backend."
            (
                self.coordinate_map_key,
                (unique_index, self.inverse_mapping),
            ) = self._manager.insert_and_map(coordinates, *coordinate_map_key.get_key())
            self.unique_index = unique_index.long()
            coordinates = coordinates[self.unique_index]

            if self.quantization_mode in [
                SparseTensorQuantizationMode.UNWEIGHTED_SUM,
                SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            ]:
                N = len(features)
                cols = torch.arange(
                    N,
                    dtype=self.inverse_mapping.dtype,
                    device=self.inverse_mapping.device,
                )
                vals = torch.ones(N, dtype=features.dtype, device=features.device)
                size = torch.Size([len(self.unique_index), len(self.inverse_mapping)])
                features = _spmm(self.inverse_mapping, cols, vals, size, features)
                # int_inverse_mapping = self.inverse_mapping.int()
                if (
                    self.quantization_mode
                    == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
                ):
                    nums = _spmm(
                        self.inverse_mapping, cols, vals, size, vals.reshape(N, 1),
                    )
                    features /= nums
            elif (
                self.quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
            ):
                features = features[self.unique_index]
            else:
                # No quantization
                pass

        elif coordinate_map_key is not None:
            assert (
                coordinate_map_key.is_key_set()
            ), "The coordinate key must be a valid key."
            self.coordinate_map_key = coordinate_map_key

        self._F = features
        self._C = coordinates
        self._batch_rows = None

    @property
    def _batchwise_row_indices(self):
        if self._batch_rows is None:
            _, self._batch_rows = self._manager.origin_map(self.coordinate_map_key)
        return self._batch_rows

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
        p = convert_to_int_list(p, self._D)
        self.coordinate_map_key.set_tensor_stride(p)

    def _get_coordinates(self):
        return self._manager.get_coordinates(self.coordinate_map_key)

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
        if self._C is None:
            self._C = self._get_coordinates()
        return self._C

    @C.setter
    def C(self):
        raise SyntaxError("Direct modification of coordinates is not permitted")

    @coordinates.setter
    def coordinates(self):
        raise SyntaxError("Direct modification of coordinates is not permitted")

    @property
    def decomposed_coordinates(self):
        r"""Returns a list of coordinates per batch.

        Returns a list of torch.IntTensor :math:`C \in \mathcal{R}^{N_i
        \times D}` coordinates per batch where :math:`N_i` is the number of non
        zero elements in the :math:`i`th batch index in :math:`D` dimensional
        space.
        """
        return [self.C[row_inds, 1:] for row_inds in self._batchwise_row_indices]

    def coordinates_at(self, batch_index):
        r"""Return coordinates at the specified batch index.

        Returns a torch.IntTensor :math:`C \in \mathcal{R}^{N_i
        \times D}` coordinates at the specified batch index where :math:`N_i`
        is the number of non zero elements in the :math:`i`th batch index in
        :math:`D` dimensional space.
        """
        return self.C[self._batchwise_row_indices[batch_index], 1:]

    @property
    def F(self):
        r"""The alias of :attr:`feats`.
        """
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
    def decomposed_features(self):
        r"""Returns a list of features per batch.

        Returns a list of torch.Tensor :math:`C \in \mathcal{R}^{N_i
        \times N_F}` features per batch where :math:`N_i` is the number of non
        zero elements in the :math:`i`th batch index in :math:`D` dimensional
        space.
        """
        return [self._F[row_inds] for row_inds in self._batchwise_row_indices]

    def features_at(self, batch_index):
        r"""Returns a feature matrix at the specified batch index.

        Returns a torch.Tensor :math:`C \in \mathcal{R}^{N
        \times N_F}` feature matrix :math:`N` is the number of non
        zero elements in the specified batch index and :math:`N_F` is the
        number of channels.
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
        """
        row_inds = self._batchwise_row_indices[batch_index]
        return self.C[row_inds, 1:], self._F[row_inds]

    @property
    def decomposed_coordinates_and_features(self):
        r"""Returns a list of coordinates and a list of features per batch.abs

        """
        row_inds_list = self._batchwise_row_indices
        return (
            [self.C[row_inds, 1:] for row_inds in row_inds_list],
            [self._F[row_inds] for row_inds in row_inds_list],
        )

    @property
    def dimension(self):
        r"""Alias of attr:`D`
        """
        return self._D

    @dimension.setter
    def dimension(self):
        raise SyntaxError("Direct modification not permitted")

    @property
    def D(self):
        r"""Alias of attr:`D`
        """
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

    def set_tensor_stride(self, s):
        ss = convert_to_int_list(s, self._D)
        self.coordinate_map_key.set_tensor_stride(ss)

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
            + "  coordinate_map_key="
            + str(self.coordinate_map_key)
            + os.linesep
            + "  coordinate_manager="
            + str(self._manager)
            + "  spatial dimension="
            + str(self._D)
            + ")"
        )

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

    def get_device(self):
        return self._F.get_device()

    def _is_same_key(self, other):
        assert isinstance(other, SparseTensor)
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
        assert isinstance(other, (SparseTensor, torch.Tensor))
        if isinstance(other, SparseTensor):
            assert self._manager == other._manager, COORDINATE_MANAGER_DIFFERENT_ERROR

            if self.coordinate_map_key == other.coordinate_map_key:
                return SparseTensor(
                    binary_fn(self._F, other.F),
                    coordinate_map_key=self.coordinate_map_key,
                    coordinate_manager=self._manager,
                )
            else:
                # Generate union maps
                out_key = CoordinateMapKey(self._manager.D)
                ins, outs = self._manager.get_union_map(
                    (self.coordinate_map_key, other.coordinate_map_key), out_key
                )
                N_out = self._manager.get_coords_size_by_coordinate_map_key(out_key)
                out_F = torch.zeros(
                    (N_out, self._F.size(1)), dtype=self.dtype, device=self.device
                )
                out_F[outs[0]] = self._F[ins[0]]
                out_F[outs[1]] = binary_fn(out_F[outs[1]], other._F[ins[1]])
                return SparseTensor(
                    out_F, coordinate_map_key=out_key, coords_manager=self._manager
                )
        else:  # when it is a torch.Tensor
            return SparseTensor(
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
        return SparseTensor(
            self._F ** power,
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self._manager,
        )

    # Conversion functions
    def sparse(self, min_coords=None, max_coords=None, contract_coords=True):
        r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch sparse
        tensor.

        Args:
            :attr:`min_coords` (torch.IntTensor, optional): The min
            coordinates of the output sparse tensor. Must be divisible by the
            current :attr:`tensor_stride`.

            :attr:`max_coords` (torch.IntTensor, optional): The max coordinates
            of the output sparse tensor (inclusive). Must be divisible by the
            current :attr:`tensor_stride`.

            :attr:`contract_coords` (bool, optional): Given True, the output
            coordinates will be divided by the tensor stride to make features
            contiguous.

        Returns:
            :attr:`spare_tensor` (torch.sparse.Tensor): the torch sparse tensor
            representation of the self in `[Batch Dim, Spatial Dims..., Feature
            Dim]`. The coordinate of each feature can be accessed via
            `min_coord + tensor_stride * [the coordinate of the dense tensor]`.

            :attr:`min_coords` (torch.IntTensor): the D-dimensional vector
            defining the minimum coordinate of the output sparse tensor. If
            :attr:`contract_coords` is True, the :attr:`min_coords` will also
            be contracted.

            :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
            defining the stride between tensor elements.

        """

        if min_coords is not None:
            assert isinstance(min_coords, torch.IntTensor)
            assert min_coords.numel() == self._D
        if max_coords is not None:
            assert isinstance(max_coords, torch.IntTensor)
            assert min_coords.numel() == self._D

        def torch_sparse_Tensor(coords, feats, size=None):
            if size is None:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats)
                else:
                    raise ValueError("Feature type not supported.")
            else:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats, size)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats, size)
                else:
                    raise ValueError("Feature type not supported.")

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride)

        # New coordinates
        coords = self.C
        coords, batch_indices = coords[:, 1:], coords[:, 0]

        # TODO, batch first
        if min_coords is None:
            min_coords, _ = coords.min(0, keepdim=True)
        elif min_coords.ndim == 1:
            min_coords = min_coords.unsqueeze(0)

        assert (
            min_coords % tensor_stride
        ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

        if max_coords is not None:
            if max_coords.ndim == 1:
                max_coords = max_coords.unsqueeze(0)
            assert (
                max_coords % tensor_stride
            ).sum() == 0, (
                "The maximum coordinates must be divisible by the tensor stride."
            )

        coords -= min_coords

        if coords.ndim == 1:
            coords = coords.unsqueeze(1)
        if batch_indices.ndim == 1:
            batch_indices = batch_indices.unsqueeze(1)

        # return the contracted tensor
        if contract_coords:
            coords = coords // tensor_stride
            if max_coords is not None:
                max_coords = max_coords // tensor_stride
            min_coords = min_coords // tensor_stride

        new_coords = torch.cat((batch_indices, coords), dim=1).long()

        size = None
        if max_coords is not None:
            size = max_coords - min_coords + 1  # inclusive
            # Squeeze to make the size one-dimensional
            size = size.squeeze()

            max_batch = max(self._manager.get_batch_indices())
            size = torch.Size([max_batch + 1, *size, self.F.size(1)])

        sparse_tensor = torch_sparse_Tensor(
            new_coords.t().to(self.F.device), self.F, size
        )
        tensor_stride = torch.IntTensor(self.tensor_stride)
        return sparse_tensor, min_coords, tensor_stride

    def dense(self, shape=None, min_coordinate=None, contract_stride=True):
        r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch dense
        tensor.

        Args:
            :attr:`shape` (torch.Size, optional): The size of the output tensor.

            :attr:`min_coordinate` (torch.IntTensor, optional): The min
            coordinates of the output sparse tensor. Must be divisible by the
            current :attr:`tensor_stride`. If 0 is given, it will use the origin for the min coordinate.

            :attr:`contract_stride` (bool, optional): The output coordinates
            will be divided by the tensor stride to make features spatially
            contiguous. True by default.

        Returns:
            :attr:`tensor` (torch.Tensor): the torch tensor with size `[Batch
            Dim, Feature Dim, Spatial Dim..., Spatial Dim]`. The coordinate of
            each feature can be accessed via `min_coordinate + tensor_stride *
            [the coordinate of the dense tensor]`.

            :attr:`min_coordinate` (torch.IntTensor): the D-dimensional vector
            defining the minimum coordinate of the output tensor.

            :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
            defining the stride between tensor elements.

        """
        if min_coordinate is not None:
            assert isinstance(min_coordinate, torch.IntTensor)
            assert min_coordinate.numel() == self._D
        if shape is not None:
            assert isinstance(shape, torch.Size)
            assert len(shape) == self._D + 2  # batch and channel
            if shape[1] != self._F.size(1):
                shape = torch.Size([shape[0], self._F.size(1), *[s for s in shape[2:]]])

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride)

        # New coordinates
        batch_indices = self.C[:, 0]

        # TODO, batch first
        if min_coordinate is None:
            min_coordinate, _ = self.C.min(0, keepdim=True)
            min_coordinate = min_coordinate[:, 1:]
            coords = self.C[:, 1:] - min_coordinate
        elif isinstance(min_coordinate, int) and min_coordinate == 0:
            coords = self.C[:, 1:]
        else:
            if min_coordinate.ndim == 1:
                min_coordinate = min_coordinate.unsqueeze(0)
            coords = self.C[:, 1:] - min_coordinate

        assert (
            min_coordinate % tensor_stride
        ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

        if coords.ndim == 1:
            coords = coords.unsqueeze(1)

        # return the contracted tensor
        if contract_stride:
            coords = coords // tensor_stride

        nchannels = self.F.size(1)
        if shape is None:
            size = coords.max(0)[0] + 1
            shape = torch.Size([batch_indices.max() + 1, nchannels, *size.numpy()])

        dense_F = torch.zeros(shape, dtype=self.F.dtype, device=self.F.device)

        tcoords = coords.t().long()
        batch_indices = batch_indices.long()
        exec(
            "dense_F[batch_indices, :, "
            + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
            + "] = self.F"
        )

        tensor_stride = torch.IntTensor(self.tensor_stride)
        return dense_F, min_coordinate, tensor_stride

    def slice(self, X, slicing_mode=0):
        r"""

        Args:
           :attr:`X` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor
           that discretized the original input.

           :attr:`slicing_mode`: For future updates.

        Returns:
           :attr:`tensor_field` (:attr:`MinkowskiEngine.TensorField`): the
           resulting tensor field contains features on the continuous
           coordinates that generated the input X.

        Example::

           >>> # coords, feats from a data loader
           >>> print(len(coords))  # 227742
           >>> sinput = ME.SparseTensor(coords=coords, feats=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
           >>> print(len(sinput))  # 161890 quantization results in fewer voxels
           >>> soutput = network(sinput)
           >>> print(len(soutput))  # 161890 Output with the same resolution
           >>> ofield = soutput.slice(sinput)
           >>> assert isinstance(ofield, ME.TensorField)
           >>> len(ofield) == len(coords)  # recovers the original ordering and length
           >>> assert isinstance(ofield.F, torch.Tensor)  # .F returns the features
        """
        # Currently only supports unweighted slice.
        assert X.quantization_mode in [
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ], "slice only available for sparse tensors with quantization RANDOM_SUBSAMPLE or UNWEIGHTED_AVERAGE"
        assert (
            X.coordinate_map_key == self.coordinate_map_key
        ), "Slice can only be applied on the same coordinates (coordinate_map_key)"
        from MinkowskiTensorField import TensorField

        return TensorField(
            self.F[X.inverse_mapping],
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self.coordinate_manager,
            inverse_mapping=X.inverse_mapping,
            quantization_mode=X.quantization_mode,
        )

    def cat_slice(self, X, slicing_mode=0):
        r"""

        Args:
           :attr:`X` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor
           that discretized the original input.

           :attr:`slicing_mode`: For future updates.

        Returns:
           :attr:`tensor_field` (:attr:`MinkowskiEngine.TensorField`): the
           resulting tensor field contains the concatenation of features on the
           original continuous coordinates that generated the input X and the
           self.

        Example::

           >>> # coords, feats from a data loader
           >>> print(len(coords))  # 227742
           >>> sinput = ME.SparseTensor(coords=coords, feats=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
           >>> print(len(sinput))  # 161890 quantization results in fewer voxels
           >>> soutput = network(sinput)
           >>> print(len(soutput))  # 161890 Output with the same resolution
           >>> ofield = soutput.cat_slice(sinput)
           >>> assert soutput.F.size(1) + sinput.F.size(1) == ofield.F.size(1)  # concatenation of features
        """
        # Currently only supports unweighted slice.
        assert X.quantization_mode in [
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ], "slice only available for sparse tensors with quantization RANDOM_SUBSAMPLE or UNWEIGHTED_AVERAGE"
        assert (
            X.coordinate_map_key == self.coordinate_map_key
        ), "Slice can only be applied on the same coordinates (coordinate_map_key)"
        from MinkowskiTensorField import TensorField

        return TensorField(
            torch.cat((self.F[X.inverse_mapping], X.F), dim=1),
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self.coordinate_manager,
            inverse_mapping=X.inverse_mapping,
            quantization_mode=X.quantization_mode,
        )

    def features_at_coords(self, query_coords: torch.Tensor):
        r"""Extract features at the specified coordinate matrix.

        Args:
           :attr:`query_coords` (:attr:`torch.IntTensor`): a coordinate matrix
           of size :math:`N \times (D + 1)` where :math:`D` is the size of the
           spatial dimension.

        Returns:
           :attr:`query_feats` (:attr:`torch.Tensor`): a feature matrix of size
           :math:`N \times D_F` where :math:`D_F` is the number of channels in
           the feature. Features for the coordinates that are not found, it will be zero.

           :attr:`valid_rows` (:attr:`list`): a list of row indices that
           contain valid values. The rest of the rows that are not found in the
           `query_feats` will be 0.

        """
        cm = self._manager

        self_key = self.coordinate_map_key
        query_key = cm.create_coordinate_map_key(query_coords)

        self_indices, query_indices = cm.get_kernel_map(
            self_key, query_key, kernel_size=1
        )
        query_feats = torch.zeros(
            (len(query_coords), self._F.size(1)), dtype=self.dtype, device=self.device
        )

        if len(self_indices[0]) > 0:
            query_feats[query_indices[0]] = self._F[self_indices[0]]
        return query_feats, query_indices[0]

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


def _get_coordinate_map_key(
    input: SparseTensor,
    coordinates: torch.Tensor = None,
    tensor_stride: StrideType = 1,
):
    r"""Process coords according to its type.
    """
    if coordinates is not None:
        assert isinstance(coordinates, (CoordinateMapKey, torch.Tensor, SparseTensor))
        if isinstance(coordinates, torch.Tensor):
            coordinate_map_key = input._manager.create_coordinate_map_key(
                coordinates, tensor_stride=tensor_stride
            )
        elif isinstance(coordinates, SparseTensor):
            coordinate_map_key = coordinates.coordinate_map_key
        else:  # CoordinateMapKey type due to the previous assertion
            coordinate_map_key = coordinates
    else:  # coordinates is None
        coordinate_map_key = CoordinateMapKey(
            input.coordinate_map_key.get_coordinate_size()
        )
    return coordinate_map_key
