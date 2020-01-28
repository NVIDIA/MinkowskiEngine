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
__version__ = "0.4.1"

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Must be imported first to load all required shared libs
import torch

from SparseTensor import SparseTensor, SparseTensorOperationMode, \
    set_sparse_tensor_operation_mode, sparse_tensor_operation_mode, clear_global_coords_man

from Common import RegionType, convert_to_int_tensor, convert_region_type, \
    MinkowskiModuleBase, KernelGenerator, GlobalPoolingMode

from MinkowskiCoords import CoordsKey, CoordsManager

from MinkowskiConvolution import MinkowskiConvolutionFunction, MinkowskiConvolution, \
    MinkowskiConvolutionTransposeFunction, MinkowskiConvolutionTranspose

from MinkowskiPooling import MinkowskiAvgPoolingFunction, MinkowskiAvgPooling, \
    MinkowskiSumPooling, \
    MinkowskiPoolingTransposeFunction, MinkowskiPoolingTranspose, \
    MinkowskiGlobalPoolingFunction, MinkowskiGlobalPooling, \
    MinkowskiGlobalSumPooling, MinkowskiGlobalAvgPooling, \
    MinkowskiGlobalMaxPoolingFunction, MinkowskiGlobalMaxPooling, \
    MinkowskiMaxPoolingFunction, MinkowskiMaxPooling

from MinkowskiBroadcast import MinkowskiBroadcastFunction, \
    MinkowskiBroadcast, MinkowskiBroadcastConcatenation, \
    MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication, OperationType

from MinkowskiNonlinearity import MinkowskiReLU, MinkowskiSigmoid, MinkowskiSoftmax, \
    MinkowskiPReLU, MinkowskiELU, MinkowskiSELU, MinkowskiCELU, MinkowskiDropout, \
    MinkowskiThreshold, MinkowskiTanh

from MinkowskiNormalization import MinkowskiBatchNorm, MinkowskiSyncBatchNorm, \
    MinkowskiInstanceNorm, MinkowskiInstanceNormFunction, MinkowskiStableInstanceNorm

from MinkowskiPruning import MinkowskiPruning, MinkowskiPruningFunction

from MinkowskiUnion import MinkowskiUnion, MinkowskiUnionFunction

from MinkowskiNetwork import MinkowskiNetwork

import MinkowskiOps

from MinkowskiOps import MinkowskiLinear, cat

import MinkowskiFunctional

import MinkowskiEngine.utils as utils
import MinkowskiEngine.modules as modules
