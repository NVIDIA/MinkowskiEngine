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
from torch.autograd import Function

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import KernelGenerator, RegionType, GlobalPoolingMode, \
    MinkowskiModuleBase, \
    convert_to_int_list, convert_to_int_tensor, \
    prep_args, save_ctx, get_postfix
from MinkowskiCoords import CoordsKey


class MinkowskiMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                tensor_stride=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()
        max_index = input_features.new().int()

        ctx.max_index = max_index

        fw_fn = getattr(MEB, 'MaxPoolingForward' + get_postfix(input_features))
        fw_fn(input_features, out_feat, max_index,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'MaxPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.max_index,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class MinkowskiAvgPoolingFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''

    @staticmethod
    def forward(ctx,
                input_features,
                tensor_stride=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                average=True,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)
        ctx.use_avg = average

        D = in_coords_key.D
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = getattr(MEB, 'AvgPoolingForward' + get_postfix(input_features))
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'AvgPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingBase(MinkowskiModuleBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 kernel_generator=None,
                 out_coords_key=None,
                 is_transpose=False,
                 average=True,
                 dimension=-1):
        super(MinkowskiPoolingBase, self).__init__()
        if out_coords_key is not None:
            assert isinstance(out_coords_key, CoordsKey)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)
        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1:
            raise ValueError('Trivial input output mapping')

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension)

        self.is_transpose = is_transpose
        self.average = average
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.out_coords_key = out_coords_key
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        output = self.pooling.apply(input.F, input.tensor_stride, self.stride,
                                    self.kernel_size, self.dilation,
                                    self.region_type_, self.region_offset_,
                                    self.average, input.coords_key,
                                    out_coords_key, input.coords_man)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        s = '(kernel_size={}, stride={}, dilation={})'.format(
            self.kernel_size, self.stride, self.dilation)
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

    def __init__(self,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
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

            :attr:`out_coords_key` (:attr:`MinkowskiEngine.CoordsKey`,
            optional): when given, the network uses the specific coordinates
            for the output coordinates.  It must be a type of
            :attr:`MinkowskiEngine.CoordsKey`.

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
            out_coords_key,
            is_transpose,
            average=True,
            dimension=dimension)
        self.pooling = MinkowskiAvgPoolingFunction()


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

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
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

            :attr:`out_coords_key` (:attr:`MinkowskiEngine.CoordsKey`,
            optional): when given, the network uses the specific coordinates
            for the output coordinates.  It must be a type of
            :attr:`MinkowskiEngine.CoordsKey`.

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
            out_coords_key,
            is_transpose,
            average=False,
            dimension=dimension)
        self.pooling = MinkowskiAvgPoolingFunction()


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

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
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

            :attr:`out_coords_key` (:attr:`MinkowskiEngine.CoordsKey`,
            optional): when given, the network uses the specific coordinates
            for the output coordinates.  It must be a type of
            :attr:`MinkowskiEngine.CoordsKey`.

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
            out_coords_key,
            is_transpose=False,
            dimension=dimension)
        self.pooling = MinkowskiMaxPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        output = self.pooling.apply(input.F, input.tensor_stride, self.stride,
                                    self.kernel_size, self.dilation,
                                    self.region_type_, self.region_offset_,
                                    input.coords_key, out_coords_key,
                                    input.coords_man)
        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)


class MinkowskiPoolingTransposeFunction(Function):

    @staticmethod
    def forward(ctx,
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
                coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)
        D = in_coords_key.D
        fw_fn = getattr(MEB,
                        'PoolingTransposeForward' + get_postfix(input_features))
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB,
                        'PoolingTransposeBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingTranspose(MinkowskiPoolingBase):
    r"""A pooling transpose layer for a sparse tensor.

    Unpool the features and divide it by the number of non zero elements that
    contributed.
    """

    def __init__(self,
                 kernel_size,
                 stride,
                 dilation=1,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
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

            :attr:`out_coords_key` (:attr:`MinkowskiEngine.CoordsKey`,
            optional): when given, the network uses the specific coordinates
            for the output coordinates.  It must be a type of
            :attr:`MinkowskiEngine.CoordsKey`.

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
            out_coords_key,
            is_transpose,
            average=False,
            dimension=dimension)
        self.pooling = MinkowskiPoolingTransposeFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        output = self.pooling.apply(input.F, input.tensor_stride, self.stride,
                                    self.kernel_size, self.dilation,
                                    self.region_type_, self.region_offset_,
                                    self.average, input.coords_key,
                                    out_coords_key, input.coords_man)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)


class MinkowskiGlobalPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                average=True,
                mode=GlobalPoolingMode.AUTO,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert isinstance(mode, GlobalPoolingMode), \
            f"Mode must be an instance of GlobalPoolingMode, {mode}"

        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key

        ctx.in_feat = input_features
        ctx.average = average
        ctx.coords_manager = coords_manager
        ctx.mode = mode.value

        fw_fn = getattr(MEB,
                        'GlobalPoolingForward' + get_postfix(input_features))
        out_feat, num_nonzero = fw_fn(ctx.in_feat,
                                      ctx.in_coords_key.CPPCoordsKey,
                                      ctx.out_coords_key.CPPCoordsKey,
                                      ctx.coords_manager.CPPCoordsManager,
                                      ctx.average, ctx.mode)

        ctx.num_nonzero = num_nonzero

        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        bw_fn = getattr(MEB,
                        'GlobalPoolingBackward' + get_postfix(grad_out_feat))
        grad_in_feat = bw_fn(ctx.in_feat, grad_out_feat, ctx.num_nonzero,
                             ctx.in_coords_key.CPPCoordsKey,
                             ctx.out_coords_key.CPPCoordsKey,
                             ctx.coords_manager.CPPCoordsManager, ctx.average)
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalPooling(MinkowskiModuleBase):
    r"""Pool all input features to one output.

    .. math::

        \mathbf{y} = \frac{1}{|\mathcal{C}^\text{in}|} \sum_{\mathbf{i} \in
        \mathcal{C}^\text{in}} \mathbf{x}_{\mathbf{i}}

    """

    def __init__(self, average=True, mode=GlobalPoolingMode.AUTO, dimension=-1):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        Args:
            :attr:`average` (bool): when True, return the averaged output. If
            not, return the sum of all input features.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        super(MinkowskiGlobalPooling, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"
        assert isinstance(mode, GlobalPoolingMode), \
            f"Mode must be an instance of GlobalPoolingMode. mode={mode}"

        self.mode = mode
        self.average = average
        self.dimension = dimension
        self.pooling = MinkowskiGlobalPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pooling.apply(input.F, self.average, self.mode,
                                    input.coords_key, out_coords_key,
                                    input.coords_man)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + "(average=" + str(self.average) + ")"


class MinkowskiGlobalMaxPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key

        ctx.in_feat = input_features
        out_feat = input_features.new()

        max_index = input_features.new().int()

        ctx.max_index = max_index
        ctx.coords_manager = coords_manager

        fw_fn = getattr(MEB,
                        'GlobalMaxPoolingForward' + get_postfix(input_features))
        fw_fn(ctx.in_feat, out_feat, ctx.max_index,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = getattr(MEB,
                        'GlobalMaxPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.max_index,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalMaxPooling(MinkowskiModuleBase):
    r"""Max pool all input features to one output feature at the origin.

    .. math::

        \mathbf{y} = \frac{1}{|\mathcal{C}^\text{in}|} \max_{\mathbf{i} \in
        \mathcal{C}^\text{in}} \mathbf{x}_{\mathbf{i}}

    """

    def __init__(self, dimension=-1):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        Args:

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        super(MinkowskiGlobalMaxPooling, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.dimension = dimension
        self.pooling = MinkowskiGlobalMaxPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pooling.apply(input.F, input.coords_key, out_coords_key,
                                    input.coords_man)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__
