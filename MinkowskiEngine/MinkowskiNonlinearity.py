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
from torch.nn import Module

from SparseTensor import SparseTensor


class MinkowskiModuleBase(Module):
    MODULE = None

    def __init__(self, *args, **kwargs):
        super(MinkowskiModuleBase, self).__init__()
        self.module = self.MODULE(*args, **kwargs)

    def forward(self, input):
        output = self.module(input.F)
        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinkowskiReLU(MinkowskiModuleBase):
    MODULE = torch.nn.ReLU


class MinkowskiPReLU(MinkowskiModuleBase):
    MODULE = torch.nn.PReLU


class MinkowskiELU(MinkowskiModuleBase):
    MODULE = torch.nn.ELU


class MinkowskiSELU(MinkowskiModuleBase):
    MODULE = torch.nn.SELU


class MinkowskiCELU(MinkowskiModuleBase):
    MODULE = torch.nn.CELU


class MinkowskiDropout(MinkowskiModuleBase):
    MODULE = torch.nn.Dropout


class MinkowskiThreshold(MinkowskiModuleBase):
    MODULE = torch.nn.Threshold


class MinkowskiSigmoid(MinkowskiModuleBase):
    MODULE = torch.nn.Sigmoid


class MinkowskiTanh(MinkowskiModuleBase):
    MODULE = torch.nn.Tanh


class MinkowskiSoftmax(MinkowskiModuleBase):
    MODULE = torch.nn.Softmax
