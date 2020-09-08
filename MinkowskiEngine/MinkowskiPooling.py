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
from typing import Union

import torch
from torch.autograd import Function

from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType, PoolingMode
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiKernelGenerator import KernelGenerator, save_ctx
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)


class MinkowskiLocalPoolingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        pooling_mode: PoolingMode,
        kernel_generator: KernelGenerator,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )

        ctx.input_features = input_features
        ctx = save_ctx(
            ctx,
            kernel_generator,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        )
        ctx.pooling_mode = pooling_mode

        fw_fn = get_minkowski_function("LocalPoolingForward", input_features)
        out_feat, num_nonzero = fw_fn(
            ctx.input_features,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            pooling_mode,
            ctx.in_coordinate_map_key,
            ctx.out_coordinate_map_key,
            ctx.coordinate_manager._manager,
        )
        ctx.num_nonzero = num_nonzero
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        bw_fn = get_minkowski_function("LocalPoolingBackward", grad_out_feat)
        grad_in_feat = bw_fn(
            ctx.input_features,
            grad_out_feat,
            ctx.num_nonzero,
            ctx.kernel_generator.kernel_size,
            ctx.kernel_generator.kernel_stride,
            ctx.kernel_generator.kernel_dilation,
            ctx.kernel_generator.region_type,
            ctx.kernel_generator.region_offsets,
            ctx.pooling_mode,
            ctx.in_coordinate_map_key,
            ctx.out_coordinate_map_key,
            ctx.coordinate_manager._manager,
        )
        return (
            grad_in_feat,
            None,
            None,
            None,
            None,
            None,
        )


class MinkowskiPoolingBase(MinkowskiModuleBase):

    __slots__ = (
        "is_transpose",
        "kernel_generator",
        "pooling_mode",
        "dimension",
        "pooling",
    )

    def __init__(
        self,
        kernel_size,
        stride=1,
        dilation=1,
        kernel_generator=None,
        is_transpose=False,
        pooling_mode=PoolingMode.LOCAL_AVG_POOLING,
        dimension=-1,
    ):
        super(MinkowskiPoolingBase, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
            )

        self.is_transpose = is_transpose
        self.kernel_generator = kernel_generator
        self.pooling_mode = pooling_mode
        self.dimension = dimension
        self.pooling = MinkowskiLocalPoolingFunction()

    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.IntTensor, CoordinateMapKey, SparseTensor] = None,
    ):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coordinates` ((`torch.IntTensor`, `MinkowskiEngine.CoordsKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Get a new coordinate map key or extract one from the coordinates
        out_coordinate_map_key = _get_coordinate_map_key(input, coordinates)
        outfeat = self.pooling.apply(
            input.F,
            self.pooling_mode,
            self.kernel_generator,
            input.coordinate_map_key,
            out_coordinate_map_key,
            input._manager,
        )

        return SparseTensor(
            outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        s = "(kernel_size={}, stride={}, dilation={})".format(
            self.kernel_size, self.stride, self.dilation
        )
        return self.__class__.__name__ + s


class MinkowskiAvgPooling(MinkowskiPoolingBase):
    r"""Average input features within a kernel.

    .. math::

        \mathbf{y}_\mathbf{u} = \frac{1}{|\mathcal{N}^D(\mathbf{u},
        \mathcal{C}^\text{in})|} \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u},
        \mathcal{C}^\text{in})} \mathbf{x}_{\mathbf{u} + \mathbf{i}}
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    For each output :math:`\mathbf{u}` in :math:`\mathcal{C}^\text{out}`,
    average input features.

    .. note::

        An average layer first computes the cardinality of the input features,
        the number of input features for each output, and divide the sum of the
        input features by the cardinality. For a dense tensor, the cardinality
        is a constant, the volume of a kernel. However, for a sparse tensor, the
        cardinality varies depending on the number of input features per output.
        Thus, the average pooling for a sparse tensor is not equivalent to the
        conventional average pooling layer for a dense tensor. Please refer to
        the :attr:`MinkowskiSumPooling` for the equivalent layer.

    .. note::

       The engine will generate the in-out mapping corresponding to a
       pooling function faster if the kernel sizes is equal to the stride
       sizes, e.g. `kernel_size = [2, 1], stride = [2, 1]`.

       If you use a U-network architecture, use the transposed version of
       the same function for up-sampling. e.g. `pool =
       MinkowskiSumPooling(kernel_size=2, stride=2, D=D)`, then use the
       `unpool = MinkowskiPoolingTranspose(kernel_size=2, stride=2, D=D)`.

    """

    def __init__(
        self,
        kernel_size=-1,
        stride=1,
        dilation=1,
        kernel_generator=None,
        dimension=None,
    ):
        r"""a high-dimensional sparse average pooling layer.

        Args:
            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): define custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. warning::

           Custom kernel shapes are not supported when kernel_size == stride.

        """
        is_transpose = False
        MinkowskiPoolingBase.__init__(
            self,
            kernel_size,
            stride,
            dilation,
            kernel_generator,
            is_transpose,
            pooling_mode=PoolingMode.LOCAL_AVG_POOLING,
            dimension=dimension,
        )


class MinkowskiSumPooling(MinkowskiPoolingBase):
    r"""Sum all input features within a kernel.

    .. math::

        \mathbf{y}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u},
        \mathcal{C}^\text{in})} \mathbf{x}_{\mathbf{u} + \mathbf{i}}
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    For each output :math:`\mathbf{u}` in :math:`\mathcal{C}^\text{out}`,
    average input features.

    .. note::

        An average layer first computes the cardinality of the input features,
        the number of input features for each output, and divide the sum of the
        input features by the cardinality. For a dense tensor, the cardinality
        is a constant, the volume of a kernel. However, for a sparse tensor, the
        cardinality varies depending on the number of input features per output.
        Thus, averaging the input features with the cardinality may not be
        equivalent to the conventional average pooling for a dense tensor.
        This layer provides an alternative that does not divide the sum by the
        cardinality.

    .. note::

       The engine will generate the in-out mapping corresponding to a
       pooling function faster if the kernel sizes is equal to the stride
       sizes, e.g. `kernel_size = [2, 1], stride = [2, 1]`.

       If you use a U-network architecture, use the transposed version of
       the same function for up-sampling. e.g. `pool =
       MinkowskiSumPooling(kernel_size=2, stride=2, D=D)`, then use the
       `unpool = MinkowskiPoolingTranspose(kernel_size=2, stride=2, D=D)`.


    """

    def __init__(
        self, kernel_size, stride=1, dilation=1, kernel_generator=None, dimension=None
    ):
        r"""a high-dimensional sum pooling layer

        Args:
            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): define custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. warning::

           Custom kernel shapes are not supported when kernel_size == stride.

        """
        is_transpose = False
        MinkowskiPoolingBase.__init__(
            self,
            kernel_size,
            stride,
            dilation,
            kernel_generator,
            is_transpose,
            pooling_mode=PoolingMode.LOCAL_SUM_POOLING,
            dimension=dimension,
        )


class MinkowskiMaxPooling(MinkowskiPoolingBase):
    r"""A max pooling layer for a sparse tensor.

    .. math::

        y^c_\mathbf{u} = \max_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u},
        \mathcal{C}^\text{in})} x^c_{\mathbf{u} + \mathbf{i}} \; \text{for} \;
        \mathbf{u} \in \mathcal{C}^\text{out}

    where :math:`y^c_\mathbf{u}` is a feature at channel :math:`c` and a
    coordinate :math:`\mathbf{u}`.

    .. note::

       The engine will generate the in-out mapping corresponding to a
       pooling function faster if the kernel sizes is equal to the stride
       sizes, e.g. `kernel_size = [2, 1], stride = [2, 1]`.

       If you use a U-network architecture, use the transposed version of
       the same function for up-sampling. e.g. `pool =
       MinkowskiSumPooling(kernel_size=2, stride=2, D=D)`, then use the
       `unpool = MinkowskiPoolingTranspose(kernel_size=2, stride=2, D=D)`.

    """

    def __init__(
        self, kernel_size, stride=1, dilation=1, kernel_generator=None, dimension=None
    ):
        r"""a high-dimensional max pooling layer for sparse tensors.

        Args:
            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): define custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. warning::

           Custom kernel shapes are not supported when kernel_size == stride.

        """

        MinkowskiPoolingBase.__init__(
            self,
            kernel_size,
            stride,
            dilation,
            kernel_generator,
            is_transpose=False,
            pooling_mode=PoolingMode.LOCAL_MAX_POOLING,
            dimension=dimension,
        )


class MinkowskiPoolingTransposeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features,
        tensor_stride=1,
        stride=1,
        kernel_size=-1,
        dilation=1,
        region_type=-1,
        region_offset=None,
        average=False,
        in_coords_key=None,
        out_coords_key=None,
        coords_manager=None,
    ):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type, in_coords_key.D
        )

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx = save_ctx(
            ctx,
            tensor_stride,
            stride,
            kernel_size,
            dilation,
            region_type,
            in_coords_key,
            out_coords_key,
            coords_manager,
        )
        D = in_coords_key.D
        fw_fn = get_minkowski_function("PoolingTransposeForward", input_features)
        fw_fn(
            ctx.in_feat,
            out_feat,
            ctx.num_nonzero,
            convert_to_int_list(ctx.tensor_stride, D),
            convert_to_int_list(ctx.stride, D),
            convert_to_int_list(ctx.kernel_size, D),
            convert_to_int_list(ctx.dilation, D),
            region_type,
            region_offset,
            ctx.in_coords_key.CPPCoordsKey,
            ctx.out_coords_key.CPPCoordsKey,
            ctx.coords_man.CPPCoordsManager,
        )
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function("PoolingTransposeBackward", grad_out_feat)
        bw_fn(
            ctx.in_feat,
            grad_in_feat,
            grad_out_feat,
            ctx.num_nonzero,
            convert_to_int_list(ctx.tensor_stride, D),
            convert_to_int_list(ctx.stride, D),
            convert_to_int_list(ctx.kernel_size, D),
            convert_to_int_list(ctx.dilation, D),
            ctx.region_type,
            ctx.in_coords_key.CPPCoordsKey,
            ctx.out_coords_key.CPPCoordsKey,
            ctx.coords_man.CPPCoordsManager,
        )
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingTranspose(MinkowskiPoolingBase):
    r"""A pooling transpose layer for a sparse tensor.

    Unpool the features and divide it by the number of non zero elements that
    contributed.
    """

    def __init__(
        self, kernel_size, stride, dilation=1, kernel_generator=None, dimension=None
    ):
        r"""a high-dimensional unpooling layer for sparse tensors.

        Args:
            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): define custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        is_transpose = True
        MinkowskiPoolingBase.__init__(
            self,
            kernel_size,
            stride,
            dilation,
            kernel_generator,
            is_transpose,
            average=False,
            dimension=dimension,
        )
        self.pooling = MinkowskiPoolingTransposeFunction()

    def forward(
        self,
        input: SparseTensor,
        coords: Union[torch.IntTensor, CoordinateMapKey, SparseTensor] = None,
    ):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coords` ((`torch.IntTensor`, `MinkowskiEngine.CoordsKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = self.kernel_generator.get_kernel(
            input.tensor_stride, self.is_transpose
        )

        # Get a new coords key or extract one from the coords
        out_coords_key = _get_coords_key(input, coords)

        output = self.pooling.apply(
            input.F,
            input.tensor_stride,
            self.stride,
            self.kernel_size,
            self.dilation,
            self.region_type_,
            self.region_offset_,
            self.average,
            input.coords_key,
            out_coords_key,
            input.coords_man,
        )

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man
        )


class MinkowskiGlobalPoolingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        pooling_mode: PoolingMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        ctx.input_features = input_features
        ctx.in_coords_key = in_coordinate_map_key
        ctx.out_coords_key = out_coordinate_map_key
        ctx.coordinate_manager = coordinate_manager
        ctx.pooling_mode = pooling_mode

        fw_fn = get_minkowski_function("GlobalPoolingForward", input_features)
        out_feat, num_nonzero = fw_fn(
            input_features,
            pooling_mode,
            ctx.in_coords_key,
            ctx.out_coords_key,
            ctx.coordinate_manager._manager,
        )

        ctx.num_nonzero = num_nonzero

        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        bw_fn = get_minkowski_function("GlobalPoolingBackward", grad_out_feat)
        grad_in_feat = bw_fn(
            ctx.input_features,
            grad_out_feat,
            ctx.num_nonzero,
            ctx.pooling_mode,
            ctx.in_coords_key,
            ctx.out_coords_key,
            ctx.coordinate_manager._manager,
        )
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalPooling(MinkowskiModuleBase):
    r"""Pool all input features to one output.

    .. math::

        \mathbf{y} = \frac{1}{|\mathcal{C}^\text{in}|} \sum_{\mathbf{i} \in
        \mathcal{C}^\text{in}} \mathbf{x}_{\mathbf{i}}

    """

    def __init__(self, mode: PoolingMode):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        Args:
            :attr:`mode` (PoolingMode):

        """
        super(MinkowskiGlobalPooling, self).__init__()
        assert isinstance(
            mode, PoolingMode
        ), f"Mode must be an instance of PoolingMode. mode={mode}"

        self.pooling_mode = mode
        self.pooling = MinkowskiGlobalPoolingFunction()

    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.IntTensor, CoordinateMapKey, SparseTensor] = None,
    ):
        # Get a new coordinate map key or extract one from the coordinates
        out_coordinate_map_key = _get_coordinate_map_key(input, coordinates)
        output = self.pooling.apply(
            input.F,
            self.pooling_mode,
            input.coordinate_map_key,
            out_coordinate_map_key,
            input._manager,
        )

        return SparseTensor(
            output,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        return self.__class__.__name__ + f"(mode={str(self.pooling_mode)})"


class MinkowskiGlobalSumPooling(MinkowskiGlobalPooling):
    def __init__(self, mode=PoolingMode.GLOBAL_SUM_POOLING_KERNEL):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        MinkowskiGlobalPooling.__init__(self, mode=mode)


class MinkowskiGlobalAvgPooling(MinkowskiGlobalPooling):
    def __init__(self, mode=PoolingMode.GLOBAL_AVG_POOLING_KERNEL):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        MinkowskiGlobalPooling.__init__(self, mode=mode)


class MinkowskiGlobalMaxPooling(MinkowskiGlobalPooling):
    r"""Max pool all input features to one output feature at the origin.

    .. math::

        \mathbf{y} = \frac{1}{|\mathcal{C}^\text{in}|} \max_{\mathbf{i} \in
        \mathcal{C}^\text{in}} \mathbf{x}_{\mathbf{i}}

    """

    def __init__(self, mode=PoolingMode.GLOBAL_MAX_POOLING_KERNEL):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        """
        MinkowskiGlobalPooling.__init__(self, mode=mode)
