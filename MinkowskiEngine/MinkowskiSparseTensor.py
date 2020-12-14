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
import torch
import warnings

from MinkowskiCommon import convert_to_int_list, StrideType
from MinkowskiEngineBackend._C import CoordinateMapKey
from MinkowskiTensor import (
    SparseTensorQuantizationMode,
    Tensor,
)
from sparse_matrix_functions import MinkowskiSPMMFunction


class SparseTensor(Tensor):
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
        >>> D = ME.SparseTensor(features=feats, coordinates=coords, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        >>> E = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=2)

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

    def initialize_coordinates(self, coordinates, features, coordinate_map_key):
        if not isinstance(coordinates, (torch.IntTensor, torch.cuda.IntTensor)):
            warnings.warn(
                "coordinates implicitly converted to torch.IntTensor. "
                + "To remove this warning, use `.int()` to convert the "
                + "coords into an torch.IntTensor"
            )
            coordinates = torch.floor(coordinates).int()

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
            spmm = MinkowskiSPMMFunction()
            N = len(features)
            cols = torch.arange(
                N,
                dtype=self.inverse_mapping.dtype,
                device=self.inverse_mapping.device,
            )
            vals = torch.ones(N, dtype=features.dtype, device=features.device)
            size = torch.Size([len(self.unique_index), len(self.inverse_mapping)])
            features = spmm.apply(self.inverse_mapping, cols, vals, size, features)
            if (
                self.quantization_mode
                == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
            ):
                nums = spmm.apply(
                    self.inverse_mapping,
                    cols,
                    vals,
                    size,
                    vals.reshape(N, 1),
                )
                features /= nums
        elif self.quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE:
            features = features[self.unique_index]
        else:
            # No quantization
            pass

        return coordinates, features, coordinate_map_key

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
           >>> tfield = ME.TensorField(coords=coords, feats=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
           >>> print(len(tfield))  # 227742
           >>> sinput = tfield.sparse() # 161890 quantization results in fewer voxels
           >>> soutput = MinkUNet(sinput)
           >>> print(len(soutput))  # 161890 Output with the same resolution
           >>> ofield = soutput.slice(tfield)
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

        if isinstance(X, TensorField):
            return TensorField(
                self.F[X.inverse_mapping],
                coordinate_map_key=X.coordinate_map_key,
                coordinate_field_map_key=X.coordinate_field_map_key,
                coordinate_manager=X.coordinate_manager,
                inverse_mapping=X.inverse_mapping,
                quantization_mode=X.quantization_mode,
            )
        else:
            return TensorField(
                self.F[X.inverse_mapping],
                coordinates=self.C[X.inverse_mapping],
                coordinate_map_key=X.coordinate_map_key,
                coordinate_manager=X.coordinate_manager,
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

        features = torch.cat((self.F[X.inverse_mapping], X.F), dim=1)
        if isinstance(X, TensorField):
            return TensorField(
                features,
                coordinate_map_key=X.coordinate_map_key,
                coordinate_field_map_key=X.coordinate_field_map_key,
                coordinate_manager=X.coordinate_manager,
                inverse_mapping=X.inverse_mapping,
                quantization_mode=X.quantization_mode,
            )
        else:
            return TensorField(
                features,
                coordinates=self.C[X.inverse_mapping],
                coordinate_map_key=X.coordinate_map_key,
                coordinate_manager=X.coordinate_manager,
                inverse_mapping=X.inverse_mapping,
                quantization_mode=X.quantization_mode,
            )

    def features_at_coordinates(self, query_coordinates: torch.Tensor):
        r"""Extract features at the specified continuous coordinate matrix.

        Args:
           :attr:`query_coordinates` (:attr:`torch.FloatTensor`): a coordinate
           matrix of size :math:`N \times (D + 1)` where :math:`D` is the size
           of the spatial dimension.

        Returns:
           :attr:`queried_features` (:attr:`torch.Tensor`): a feature matrix of
           size :math:`N \times D_F` where :math:`D_F` is the number of
           channels in the feature. For coordinates not present in the current
           sparse tensor, corresponding feature rows will be zeros.
        """
        from MinkowskiInterpolation import MinkowskiInterpolationFunction

        assert (
            self.dtype == query_coordinates.dtype
        ), "Invalid query_coordinates dtype. use {self.dtype}"

        assert (
            query_coordinates.device == self.device
        ), "query coordinates device ({query_coordinates.device}) does not match the sparse tensor device ({self.device})."
        return MinkowskiInterpolationFunction().apply(
            self._F,
            query_coordinates,
            self.coordinate_map_key,
            None,
            self.coordinate_manager,
        )[0]

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
    expand_coordinates: bool = False,
):
    r"""Returns the coordinates map key."""
    if coordinates is not None and not expand_coordinates:
        assert isinstance(coordinates, (CoordinateMapKey, torch.Tensor, SparseTensor))
        if isinstance(coordinates, torch.Tensor):
            assert coordinates.ndim == 2
            coordinate_map_key = CoordinateMapKey(
                convert_to_int_list(tensor_stride, coordinates.size(1) - 1), ""
            )

            (
                coordinate_map_key,
                (unique_index, inverse_mapping),
            ) = input._manager.insert_and_map(
                coordinates, *coordinate_map_key.get_key()
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
