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

    """

    def __init__(self,
                 feats,
                 coords=None,
                 coords_key=None,
                 coords_manager=None,
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
            MinkowskiEngine creates a dynamic computation graph using an input
            coordinates. If not provided, the MinkowskiEngine will create a new
            computation graph, so make sure to provide the same
            :attr:`CoordsManager` when you want to use the same computation
            graph. To use a sparse tensor within the same computation graph
            that you are using before, feed the :attr:`CoordsManager` of the
            sparse tensor that you want to use by
            :attr:`sparse_tensor.coords_man`. In most cases, this process is
            handled automatically. When you use it, make sure you understand
            what you are doing.

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
            coords_manager.initialize(coords, coords_key)
        else:
            assert isinstance(coords_manager, CoordsManager)

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
            + '  Feats=' + str(self.F) + os.linesep \
            + '  coords_key=' + str(self.coords_key) + os.linesep \
            + '  tensor_stride=' + str(self.coords_key.getTensorStride()) + os.linesep \
            + '  coords_man=' + str(self.coords_man) + ')'

    def __len__(self):
        return len(self._F)

    def size(self):
        return self._F.size()

    def to(self, device):
        self._F = self._F.to(device)
        return self

    def cpu(self):
        self._F = self._F.cpu()
        return self

    def get_device(self):
        return self._F.get_device()

    def getKey(self):
        return self.coords_key
