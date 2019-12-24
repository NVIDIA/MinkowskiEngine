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
import warnings
import torch

from Common import convert_to_int_list
from MinkowskiCoords import CoordsKey, CoordsManager


class SparseTensor():
    r"""A sparse tensor class. Can be accessed via
    :attr:`MinkowskiEngine.SparseTensor`.

    The :attr:`SparseTensor` class is the basic tensor in MinkowskiEngine. We
    use the COOrdinate (COO) format to save a sparse tensor `[1]
    <http://groups.csail.mit.edu/commit/papers/2016/parker-thesis.pdf>`_. This
    representation is simply a concatenation of coordinates in a matrix
    :math:`C` and associated features :math:`F`.

    .. math::

       \mathbf{C} = \begin{bmatrix}
       x_1^1   & x_1^2  & \cdots & x_1^D  & b_1    \\
        \vdots & \vdots & \ddots & \vdots & \vdots \\
       x_N^1   & x_N^2  & \cdots & x_N^D  & b_N
       \end{bmatrix}, \; \mathbf{F} = \begin{bmatrix}
       \mathbf{f}_1^T\\
       \vdots\\
       \mathbf{f}_N^T
       \end{bmatrix}

    In the above equation, we use a :math:`D`-dimensional space and :math:`N`
    number of points, each with the coordinate :math:`(x_i^1, x_i^1, \cdots,
    x_i^D)`, and the associated feature :math:`\mathbf{f}_i`. :math:`b_i`
    indicates the mini-batch index to disassociate instances within the same
    batch.  Internally, we handle the batch index as an additional spatial
    dimension.

    .. warning::

       From the version 0.4, we will put the batch indices on the first column
       to be consistent with the standard neural network packages.

       Please use :attr:`MinkowskiEngine.utils.batched_coordinates` or
       :attr:`MinkowskiEngine.utils.sparse_collate` when creating coordinates
       to make your code to generate batched coordinates automatically that are
       compatible with the latest version of Minkowski Engine.

       .. math::

          \mathbf{C} = \begin{bmatrix}
          b_1    & x_1^1   & x_1^2  & \cdots & x_1^D    \\
          \vdots &    \vdots & \vdots & \ddots & \vdots \\
          b_N    & x_N^1   & x_N^2  & \cdots & x_N^D
          \end{bmatrix}, \; \mathbf{F} = \begin{bmatrix}
          \mathbf{f}_1^T\\
          \vdots\\
          \mathbf{f}_N^T
          \end{bmatrix}

    """

    def __init__(self,
                 feats,
                 coords=None,
                 coords_key=None,
                 coords_manager=None,
                 force_creation=False,
                 allow_duplicate_coords=False,
                 tensor_stride=1):
        r"""

        Args:
            :attr:`feats` (:attr:`torch.FloatTensor`,
            :attr:`torch.DoubleTensor`, :attr:`torch.cuda.FloatTensor`, or
            :attr:`torch.cuda.DoubleTensor`): The features of the sparse
            tensor.

            :attr:`coords` (:attr:`torch.IntTensor`): The coordinates
            associated to the features. If not provided, :attr:`coords_key`
            must be provided.

            :attr:`coords_key` (:attr:`MinkowskiEngine.CoordsKey`): When the
            coordinates are already cached in the MinkowskiEngine, we could
            reuse the same coordinates by simply providing the coordinate hash
            key. In most case, this process is done automatically. If you
            provide one, make sure you understand what you are doing.

            :attr:`coords_manager` (:attr:`MinkowskiEngine.CoordsManager`): The
            MinkowskiEngine creates a dynamic computation graph and all
            coordinates inside the same computation graph are managed by a
            CoordsManager object. If not provided, the MinkowskiEngine will
            create a new computation graph. In most cases, this process is
            handled automatically and you do not need to use this. When you use
            it, make sure you understand what you are doing.

            :attr:`force_creation` (:attr:`bool`): Force creation of the
            coordinates. This allows generating a new set of coordinates even
            when there exists another set of coordinates with the same
            tensor stride. This could happen when you manually feed the same
            :attr:`coords_manager`.

            :attr:`allow_duplicate_coords` (:attr:`bool`): Allow duplicate
            coordinates when creating the sparse tensor. Internally, it will
            generate a new unique set of coordinates and use features of at the
            corresponding unique coordinates. In general, setting
            `allow_duplicate_coords=True` is not recommended as it could hide
            obvious errors in your data loading and preprocessing steps. Please
            refer to the quantization and data loading tutorial on `here
            <https://stanfordvl.github.io/MinkowskiEngine/demo/training.html>`_
            for more details.

            :attr:`tensor_stride` (:attr:`int`, :attr:`list`,
            :attr:`numpy.array`, or :attr:`tensor.Tensor`): The tensor stride
            of the current sparse tensor. By default, it is 1.

        """
        assert isinstance(feats,
                          torch.Tensor), "Features must be a torch.Tensor"

        if coords is None and coords_key is None:
            raise ValueError('Either coords or coords_key must be provided')

        if coords_key is None:
            assert coords_manager is not None or coords is not None
            D = -1
            if coords_manager is None:
                D = coords.size(1) - 1
            else:
                D = coords_manager.D
            coords_key = CoordsKey(D)
            coords_key.setTensorStride(convert_to_int_list(tensor_stride, D))
        else:
            assert isinstance(coords_key, CoordsKey)

        if coords is not None:
            assert isinstance(coords, torch.Tensor), \
                "Coordinate must be of type torch.Tensor"

            if not isinstance(coords, torch.IntTensor):
                warnings.warn(
                    'Coords implicitly converted to torch.IntTensor. ' +
                    'To remove this warning, use `.int()` to convert the ' +
                    'coords into an torch.IntTensor')
                coords = coords.int()

            assert feats.shape[0] == coords.shape[0], \
                "Number of rows in features and coordinates do not match."

            coords = coords.contiguous()

        if coords_manager is None:
            assert coords is not None, "Initial coordinates must be given"
            D = coords.size(1) - 1
            coords_manager = CoordsManager(D=D)
            self.mapping = coords_manager.initialize(
                coords,
                coords_key,
                force_creation=force_creation,
                force_remap=allow_duplicate_coords,
                allow_duplicate_coords=allow_duplicate_coords)
            if len(self.mapping) > 0:
                coords = coords[self.mapping]
                feats = feats[self.mapping]
        else:
            assert isinstance(coords_manager, CoordsManager)

            if not coords_key.isKeySet():
                assert coords is not None
                self.mapping = coords_manager.initialize(
                    coords,
                    coords_key,
                    force_creation=force_creation,
                    force_remap=allow_duplicate_coords,
                    allow_duplicate_coords=allow_duplicate_coords)
                if len(self.mapping) > 0:
                    coords = coords[self.mapping]
                    feats = feats[self.mapping]

        self._F = feats.contiguous()
        self._C = coords
        self.coords_key = coords_key
        self.coords_man = coords_manager

    @property
    def tensor_stride(self):
        return self.coords_key.getTensorStride()

    @tensor_stride.setter
    def tensor_stride(self, p):
        r"""
        This function is not recommended to be used directly.
        """
        p = convert_to_int_list(p, self.D)
        self.coords_key.setTensorStride(p)

    def _get_coords(self):
        return self.coords_man.get_coords(self.coords_key)

    @property
    def C(self):
        r"""The alias of :attr:`coords`.
        """
        return self.coords

    @property
    def coords(self):
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
            self._C = self._get_coords()
        return self._C

    @property
    def F(self):
        r"""The alias of :attr:`feats`.
        """
        return self._F

    @property
    def feats(self):
        r"""
        The features of the current sparse tensor. The features are :math:`N
        \times D_F` where :math:`N` is the number of points in the space and
        :math:`D_F` is the dimension of each feature vector. Please refer to
        :attr:`coords` to access the associated coordinates.
        """
        return self._F

    @property
    def D(self):
        r"""
        The spatial dimension of the sparse tensor. This is equal to the number
        of columns of :attr:`C` minus 1.
        """
        return self.coords_key.D

    def stride(self, s):
        ss = convert_to_int_list(s)
        tensor_strides = self.coords_key.getTensorStride()
        self.coords_key.setTensorStride(
            [s * p for s, p in zip(ss, tensor_strides)])

    def __add__(self, other):
        return SparseTensor(
            self._F + (other.F if isinstance(other, SparseTensor) else other),
            coords_key=self.coords_key,
            coords_manager=self.coords_man)

    def __power__(self, power):
        return SparseTensor(
            self._F**power,
            coords_key=self.coords_key,
            coords_manager=self.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '(' + os.linesep \
            + '  Coords=' + str(self.C) + os.linesep \
            + '  Feats=' + str(self.F) + os.linesep \
            + '  coords_key=' + str(self.coords_key) \
            + '  tensor_stride=' + str(self.coords_key.getTensorStride()) + os.linesep \
            + '  coords_man=' + str(self.coords_man) + ')'

    def __len__(self):
        return len(self._F)

    def size(self):
        return self._F.size()

    @property
    def shape(self):
        return self._F.shape

    def to(self, device):
        self._F = self._F.to(device)
        return self

    def cpu(self):
        self._F = self._F.cpu()
        return self

    @property
    def device(self):
        return self._F.device

    @property
    def dtype(self):
        return self._F.dtype

    def get_device(self):
        return self._F.get_device()

    def getKey(self):
        return self.coords_key

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
            assert min_coords.numel() == self.D
        if max_coords is not None:
            assert isinstance(max_coords, torch.IntTensor)
            assert min_coords.numel() == self.D

        def torch_sparse_Tensor(coords, feats, size=None):
            if size is None:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats)
                else:
                    raise ValueError('Feature type not supported.')
            else:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats, size)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats, size)
                else:
                    raise ValueError('Feature type not supported.')

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride)

        # New coordinates
        coords = self.C
        coords, batch_indices = coords[:, :-1], coords[:, -1]

        # TODO, batch first
        if min_coords is None:
            min_coords, _ = coords.min(0, keepdim=True)
        elif min_coords.ndim == 1:
            min_coords = min_coords.unsqueeze(0)

        assert (min_coords % tensor_stride).sum() == 0, \
            "The minimum coordinates must be divisible by the tensor stride."

        if max_coords is not None:
            if max_coords.ndim == 1:
                max_coords = max_coords.unsqueeze(0)
            assert (max_coords % tensor_stride).sum() == 0, \
                "The maximum coordinates must be divisible by the tensor stride."

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

            max_batch = batch_indices.max().item()
            size = torch.Size([max_batch + 1, *size, self.F.size(1)])

        sparse_tensor = torch_sparse_Tensor(new_coords.t().to(self.F.device),
                                            self.F, size)
        tensor_stride = torch.IntTensor(self.tensor_stride)
        return sparse_tensor, min_coords, tensor_stride

    def dense(self, min_coords=None, max_coords=None, contract_coords=True):
        r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch dense
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
            assert min_coords.numel() == self.D
        if max_coords is not None:
            assert isinstance(max_coords, torch.IntTensor)
            assert min_coords.numel() == self.D

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride)

        # New coordinates
        coords = self.C
        coords, batch_indices = coords[:, :-1], coords[:, -1]

        # TODO, batch first
        if min_coords is None:
            min_coords, _ = coords.min(0, keepdim=True)
        elif min_coords.ndim == 1:
            min_coords = min_coords.unsqueeze(0)

        assert (min_coords % tensor_stride).sum() == 0, \
            "The minimum coordinates must be divisible by the tensor stride."

        if max_coords is not None:
            if max_coords.ndim == 1:
                max_coords = max_coords.unsqueeze(0)
            assert (max_coords % tensor_stride).sum() == 0, \
                "The maximum coordinates must be divisible by the tensor stride."

        coords -= min_coords

        if coords.ndim == 1:
            coords = coords.unsqueeze(1)

        # return the contracted tensor
        if contract_coords:
            coords = coords // tensor_stride
            if max_coords is not None:
                max_coords = max_coords // tensor_stride
            min_coords = min_coords // tensor_stride

        size = None
        nchannels = self.F.size(1)
        if max_coords is not None:
            size = max_coords - min_coords + 1  # inclusive
            # Squeeze to make the size one-dimensional
            size = size.squeeze()

            max_batch = batch_indices.max().item()
            size = torch.Size([max_batch + 1, nchannels, *size])
        else:
            size = coords.max(0)[0] + 1
            max_batch = batch_indices.max().item()
            size = torch.Size([max_batch + 1, nchannels, *size.numpy()])

        dense_F = torch.zeros(size, dtype=self.F.dtype, device=self.F.device)

        tcoords = coords.t().long()
        batch_indices = batch_indices.long()
        exec("dense_F[batch_indices, :, " +
             ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))]) +
             "] = self.F")

        tensor_stride = torch.IntTensor(self.tensor_stride)
        return dense_F, min_coords, tensor_stride
