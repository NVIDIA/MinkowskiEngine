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

from SparseTensor import SparseTensor
from MinkowskiConvolution import MinkowskiConvolutionBase, MinkowskiConvolutionFunction, MinkowskiConvolutionTransposeFunction
from MinkowskiCoords import CoordsKey


class MinkowskiConvolutionOnCoords(MinkowskiConvolutionBase):
    r"""The generalized sparse convolution on a set of specified output coordinates.


    .. math::

        \mathbf{x}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u}, K,
        \mathcal{C}^\text{in})} W_\mathbf{i} \mathbf{x}_{\mathbf{i} +
        \mathbf{u}} \;\text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    where :math:`K` is the kernel size and :math:`\mathcal{N}^D(\mathbf{u}, K,
    \mathcal{C}^\text{in})` is the set of offsets that are at most :math:`\left
    \lceil{\frac{1}{2}(K - 1)} \right \rceil` away from :math:`\mathbf{u}`
    definied in :math:`\mathcal{S}^\text{in}`.

    .. note::
        For even :math:`K`, the kernel offset :math:`\mathcal{N}^D`
        implementation is different from the above definition. The offsets
        range from :math:`\mathbf{i} \in [0, K)^D, \; \mathbf{i} \in
        \mathbb{Z}_+^D`.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 dimension=None):
        r"""a generalized sparse convolution layer.

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

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

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            has_bias,
            kernel_generator,
            is_transpose=False,
            dimension=dimension)
        self.reset_parameters()
        self.conv = MinkowskiConvolutionFunction()

    def forward(self, input, coords, tensor_stride=1):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        assert tensor_stride > -1
        assert isinstance(coords, CoordsKey) or \
            isinstance(coords, torch.IntTensor) or \
            isinstance(coords, SparseTensor)

        if isinstance(coords, torch.IntTensor):
            coords_key = CoordsKey(input.D)
            coords_key.setTensorStride(tensor_stride)
            mapping = input.coords_man.initialize(
                coords,
                coords_key,
                force_creation=True,
                force_remap=True,
                allow_duplicate_coords=True)
        elif isinstance(coords, SparseTensor):
            coords_key = coords.coords_key
        else:  # CoordsKey type due to previous assertions
            coords_key = coords

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        outfeat = self.conv.apply(input.F, self.kernel, input.tensor_stride,
                                  self.stride, self.kernel_size, self.dilation,
                                  self.region_type_, self.region_offset_,
                                  input.coords_key, coords_key,
                                  input.coords_man)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=coords_key, coords_manager=input.coords_man)


class MinkowskiConvolutionTransposeOnCoords(MinkowskiConvolutionBase):
    r"""A generalized sparse transposed convolution or deconvolution layer on a set of specified output coordinates.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 generate_new_coords=False,
                 dimension=None):
        r"""a generalized sparse transposed convolution layer.

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size that defines
            upsampling rate. If non-identity is used, the output coordinates
            will be :attr:`tensor_stride` / :attr:`stride` apart.  When a list is
            given, the length must be D; each element will be used for stride
            size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`generate_new_coords` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by defaul.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. note:
            TODO: support `kernel_size` > `stride`.

        """
        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            has_bias,
            kernel_generator,
            is_transpose=True,
            dimension=dimension)
        self.reset_parameters(True)
        self.generate_new_coords = generate_new_coords
        self.conv = MinkowskiConvolutionTransposeFunction()

    def forward(self, input, coords, tensor_stride=1):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        assert tensor_stride > -1
        assert isinstance(coords, CoordsKey) or \
            isinstance(coords, torch.IntTensor) or \
            isinstance(coords, SparseTensor)

        if isinstance(coords, torch.IntTensor):
            coords_key = CoordsKey(input.D)
            coords_key.setTensorStride(tensor_stride)
            mapping = input.coords_man.initialize(
                coords,
                coords_key,
                force_creation=True,
                force_remap=True,
                allow_duplicate_coords=True)
        elif isinstance(coords, SparseTensor):
            coords_key = coords.coords_key
        else:  # CoordsKey type due to previous assertions
            coords_key = coords

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        outfeat = self.conv.apply(input.F, self.kernel, input.tensor_stride,
                                  self.stride, self.kernel_size, self.dilation,
                                  self.region_type_, self.region_offset_,
                                  self.generate_new_coords, input.coords_key,
                                  coords_key, input.coords_man)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=coords_key, coords_manager=input.coords_man)
