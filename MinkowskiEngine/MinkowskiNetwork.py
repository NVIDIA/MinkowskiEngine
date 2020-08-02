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
from abc import ABC, abstractmethod

import torch.nn as nn

from MinkowskiSparseTensor import SparseTensor


class MinkowskiNetwork(nn.Module, ABC):
    """
    MinkowskiNetwork: an abstract class for sparse convnets.

    Note: All modules that use the same coordinates must use the same net_metadata
    """

    def __init__(self, D):
        super(MinkowskiNetwork, self).__init__()
        self.D = D

    @abstractmethod
    def forward(self, x):
        pass

    def init(self, x):
        """
        Initialize coordinates if it does not exist
        """
        nrows = self.get_nrows(1)
        if nrows < 0:
            if isinstance(x, SparseTensor):
                self.initialize_coords(x.coords_man)
            else:
                raise ValueError('Initialize input coordinates')
        elif nrows != x.F.size(0):
            raise ValueError('Input size does not match the coordinate size')
