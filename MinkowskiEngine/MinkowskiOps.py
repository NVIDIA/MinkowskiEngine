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
from typing import Union
import numpy as np

import torch
from torch.nn.modules import Module
from MinkowskiSparseTensor import SparseTensor
from MinkowskiTensor import (
    COORDINATE_MANAGER_DIFFERENT_ERROR,
    COORDINATE_KEY_DIFFERENT_ERROR,
)
from MinkowskiTensorField import TensorField
from MinkowskiCommon import MinkowskiModuleBase
from MinkowskiEngineBackend._C import CoordinateMapKey


class MinkowskiLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: Union[SparseTensor, TensorField]):
        output = self.linear(input.F)
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        s = "(in_features={}, out_features={}, bias={})".format(
            self.linear.in_features,
            self.linear.out_features,
            self.linear.bias is not None,
        )
        return self.__class__.__name__ + s


def _tuple_operator(*sparse_tensors, operator):
    if len(sparse_tensors) == 1:
        assert isinstance(sparse_tensors[0], (tuple, list))
        sparse_tensors = sparse_tensors[0]

    assert (
        len(sparse_tensors) > 1
    ), f"Invalid number of inputs. The input must be at least two len(sparse_tensors) > 1"

    if isinstance(sparse_tensors[0], SparseTensor):
        device = sparse_tensors[0].device
        coordinate_manager = sparse_tensors[0].coordinate_manager
        coordinate_map_key = sparse_tensors[0].coordinate_map_key
        for s in sparse_tensors:
            assert isinstance(
                s, SparseTensor
            ), "Inputs must be either SparseTensors or TensorFields."
            assert (
                device == s.device
            ), f"Device must be the same. {device} != {s.device}"
            assert (
                coordinate_manager == s.coordinate_manager
            ), COORDINATE_MANAGER_DIFFERENT_ERROR
            assert coordinate_map_key == s.coordinate_map_key, (
                COORDINATE_KEY_DIFFERENT_ERROR
                + str(coordinate_map_key)
                + " != "
                + str(s.coordinate_map_key)
            )
        tens = []
        for s in sparse_tensors:
            tens.append(s.F)
        return SparseTensor(
            operator(tens),
            coordinate_map_key=coordinate_map_key,
            coordinate_manager=coordinate_manager,
        )
    elif isinstance(sparse_tensors[0], TensorField):
        device = sparse_tensors[0].device
        coordinate_manager = sparse_tensors[0].coordinate_manager
        coordinate_field_map_key = sparse_tensors[0].coordinate_field_map_key
        for s in sparse_tensors:
            assert isinstance(
                s, TensorField
            ), "Inputs must be either SparseTensors or TensorFields."
            assert (
                device == s.device
            ), f"Device must be the same. {device} != {s.device}"
            assert (
                coordinate_manager == s.coordinate_manager
            ), COORDINATE_MANAGER_DIFFERENT_ERROR
            assert coordinate_field_map_key == s.coordinate_field_map_key, (
                COORDINATE_KEY_DIFFERENT_ERROR
                + str(coordinate_field_map_key)
                + " != "
                + str(s.coordinate_field_map_key)
            )
        tens = []
        for s in sparse_tensors:
            tens.append(s.F)
        return TensorField(
            operator(tens),
            coordinate_field_map_key=coordinate_field_map_key,
            coordinate_manager=coordinate_manager,
        )
    else:
        raise ValueError(
            "Invalid data type. The input must be either a list of sparse tensors or a list of tensor fields."
        )


def cat(*sparse_tensors):
    r"""Concatenate sparse tensors

    Concatenate sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To concatenate sparse tensors
    with different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_mananger=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.cat(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """
    return _tuple_operator(*sparse_tensors, operator=lambda xs: torch.cat(xs, dim=-1))


def _sum(*sparse_tensors):
    r"""Sum sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.sum(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_sum(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        return tmp

    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_sum(xs))


def mean(*sparse_tensors):
    r"""Sum sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.mean(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_mean(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        return tmp / len(xs)

    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_mean(xs))


def var(*sparse_tensors):
    r"""Sum sparse tensor features

    Sum all sparse tensor features. All sparse tensors must have the same
    `coordinate_map_key` (the same coordinates). To sum sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coordinate_manager=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.var(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """

    def return_var(xs):
        tmp = xs[0] + xs[1]
        for x in xs[2:]:
            tmp += x
        mean = tmp / len(xs)
        var = (xs[0] - mean) ** 2
        for x in xs[1:]:
            var += (x - mean) ** 2
        return var / len(xs)

    return _tuple_operator(*sparse_tensors, operator=lambda xs: return_var(xs))


def dense_coordinates(shape: Union[list, torch.Size]):
    """
    coordinates = dense_coordinates(tensor.shape)
    """
    r"""
    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use 
    """
    spatial_dim = len(shape) - 2
    assert (
        spatial_dim > 0
    ), "Invalid shape. Shape must be batch x channel x spatial dimensions."

    # Generate coordinates
    size = [i for i in shape]
    B = size[0]
    coordinates = torch.from_numpy(
        np.stack(
            [
                s.reshape(-1)
                for s in np.meshgrid(
                    np.linspace(0, B - 1, B),
                    *(np.linspace(0, s - 1, s) for s in size[2:]),
                    indexing="ij",
                )
            ],
            1,
        )
    ).int()
    return coordinates


def to_sparse(dense_tensor: torch.Tensor, coordinates: torch.Tensor = None):
    r"""Converts a (differentiable) dense tensor to a sparse tensor.

    Assume the input to have BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use `dense_coordinates` to cache the coordinates.
    Please refer to tests/python/dense.py for usage

    Example::

       >>> dense_tensor = torch.rand(3, 4, 5, 6, 7, 8)  # BxCxD1xD2xD3xD4
       >>> dense_tensor.requires_grad = True
       >>> stensor = to_sparse(dense_tensor)

    """
    spatial_dim = dense_tensor.ndim - 2
    assert (
        spatial_dim > 0
    ), "Invalid shape. Shape must be batch x channel x spatial dimensions."

    if coordinates is None:
        coordinates = dense_coordinates(dense_tensor.shape)

    feat_tensor = dense_tensor.permute(0, *(2 + i for i in range(spatial_dim)), 1)
    return SparseTensor(
        feat_tensor.reshape(-1, dense_tensor.size(1)),
        coordinates,
        device=dense_tensor.device,
    )


class MinkowskiToSparseTensor(MinkowskiModuleBase):
    r"""Converts a (differentiable) dense tensor or a :attr:`MinkowskiEngine.TensorField` to a :attr:`MinkowskiEngine.SparseTensor`.

    For dense tensor, the input must have the BxCxD1xD2x....xDN format.

    If the shape of the tensor do not change, use `dense_coordinates` to cache the coordinates.
    Please refer to tests/python/dense.py for usage.

    Example::

       >>> # Differentiable dense torch.Tensor to sparse tensor.
       >>> dense_tensor = torch.rand(3, 4, 11, 11, 11, 11)  # BxCxD1xD2x....xDN
       >>> dense_tensor.requires_grad = True

       >>> # Since the shape is fixed, cache the coordinates for faster inference
       >>> coordinates = dense_coordinates(dense_tensor.shape)

       >>> network = nn.Sequential(
       >>>     # Add layers that can be applied on a regular pytorch tensor
       >>>     nn.ReLU(),
       >>>     MinkowskiToSparseTensor(coordinates=coordinates),
       >>>     MinkowskiConvolution(4, 5, kernel_size=3, dimension=4),
       >>>     MinkowskiBatchNorm(5),
       >>>     MinkowskiReLU(),
       >>> )

       >>> for i in range(5):
       >>>   print(f"Iteration: {i}")
       >>>   soutput = network(dense_tensor)
       >>>   soutput.F.sum().backward()
       >>>   soutput.dense(shape=dense_tensor.shape)

    """

    def __init__(self, coordinates: torch.Tensor = None):
        MinkowskiModuleBase.__init__(self)
        self.coordinates = coordinates

    def forward(self, input: Union[TensorField, torch.Tensor]):
        if isinstance(input, TensorField):
            return input.sparse()
        elif isinstance(input, torch.Tensor):
            # dense tensor to sparse tensor conversion
            return to_sparse(input, self.coordinates)
        else:
            raise ValueError(
                "Unsupported type. Only TensorField and torch.Tensor are supported"
            )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinkowskiToDenseTensor(MinkowskiModuleBase):
    r"""Converts a (differentiable) sparse tensor to a torch tensor.

    The return type has the BxCxD1xD2x....xDN format.

    Example::

       >>> dense_tensor = torch.rand(3, 4, 11, 11, 11, 11)  # BxCxD1xD2x....xDN
       >>> dense_tensor.requires_grad = True

       >>> # Since the shape is fixed, cache the coordinates for faster inference
       >>> coordinates = dense_coordinates(dense_tensor.shape)

       >>> network = nn.Sequential(
       >>>     # Add layers that can be applied on a regular pytorch tensor
       >>>     nn.ReLU(),
       >>>     MinkowskiToSparseTensor(coordinates=coordinates),
       >>>     MinkowskiConvolution(4, 5, stride=2, kernel_size=3, dimension=4),
       >>>     MinkowskiBatchNorm(5),
       >>>     MinkowskiReLU(),
       >>>     MinkowskiConvolutionTranspose(5, 6, stride=2, kernel_size=3, dimension=4),
       >>>     MinkowskiToDenseTensor(
       >>>         dense_tensor.shape
       >>>     ),  # must have the same tensor stride.
       >>> )

       >>> for i in range(5):
       >>>     print(f"Iteration: {i}")
       >>>     output = network(dense_tensor) # returns a regular pytorch tensor
       >>>     output.sum().backward()

    """

    def __init__(self, shape: torch.Size = None):
        MinkowskiModuleBase.__init__(self)
        self.shape = shape

    def forward(self, input: SparseTensor):
        # dense tensor to sparse tensor conversion
        dense_tensor, _, _ = input.dense(shape=self.shape)
        return dense_tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinkowskiStackCat(torch.nn.Sequential):
    def forward(self, x):
        return cat([module(x) for module in self])


class MinkowskiStackSum(torch.nn.Sequential):
    def forward(self, x):
        return _sum([module(x) for module in self])


class MinkowskiStackMean(torch.nn.Sequential):
    def forward(self, x):
        return mean([module(x) for module in self])


class MinkowskiStackVar(torch.nn.Sequential):
    def forward(self, x):
        return var([module(x) for module in self])
