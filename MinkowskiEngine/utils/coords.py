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

from MinkowskiSparseTensor import SparseTensor


def get_coords_map(x, y):
    r"""Get mapping between sparse tensor 1 and sparse tensor 2.

    Args:
        :attr:`x` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor with
        `x.tensor_stride` <= `y.tensor_stride`.

        :attr:`y` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor with
        `x.tensor_stride` <= `y.tensor_stride`.

    Returns:
        :attr:`x_indices` (:attr:`torch.LongTensor`): the indices of x that
        corresponds to the returned indices of y.

        :attr:`x_indices` (:attr:`torch.LongTensor`): the indices of y that
        corresponds to the returned indices of x.

    Example::

        .. code-block:: python

           sp_tensor = ME.SparseTensor(features, coordinates=coordinates)
           out_sp_tensor = stride_2_conv(sp_tensor)

           ins, outs = get_coords_map(sp_tensor, out_sp_tensor)
           for i, o in zip(ins, outs):
              print(f"{i} -> {o}")

    """
    assert isinstance(x, SparseTensor)
    assert isinstance(y, SparseTensor)
    assert (
        x.coords_man == y.coords_man
    ), "X and Y are using different CoordinateManagers. Y must be derived from X through strided conv/pool/etc."
    return x.coords_man.get_coords_map(x.coords_key, y.coords_key)
