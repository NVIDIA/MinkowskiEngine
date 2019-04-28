__version__ = "0.2.2"

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Must be imported first to load all required shared libs
import torch

from SparseTensor import SparseTensor

from Common import RegionType, convert_to_int_tensor, convert_region_type, \
    MinkowskiModuleBase, KernelGenerator

from MinkowskiCoords import CoordsKey, CoordsManager, initialize_nthreads

from MinkowskiConvolution import MinkowskiConvolutionFunction, MinkowskiConvolution, \
    MinkowskiConvolutionTransposeFunction, MinkowskiConvolutionTranspose

from MinkowskiPooling import MinkowskiAvgPoolingFunction, MinkowskiAvgPooling, \
    MinkowskiSumPooling, \
    MinkowskiPoolingTransposeFunction, MinkowskiPoolingTranspose, \
    MinkowskiGlobalPoolingFunction, MinkowskiGlobalPooling, \
    MinkowskiMaxPoolingFunction, MinkowskiMaxPooling

from MinkowskiBroadcast import MinkowskiBroadcastFunction, \
    MinkowskiBroadcast, MinkowskiBroadcastAddition, \
    MinkowskiBroadcastMultiplication, OperationType

from MinkowskiNonlinearity import MinkowskiReLU, MinkowskiSigmoid, MinkowskiSoftmax, \
    MinkowskiPReLU, MinkowskiSELU, MinkowskiCELU, MinkowskiDropout, MinkowskiThreshold, \
    MinkowskiTanh


from MinkowskiNormalization import MinkowskiBatchNorm, MinkowskiInstanceNorm, \
    MinkowskiInstanceNormFunction, MinkowskiStableInstanceNorm

from MinkowskiPruning import MinkowskiPruning, MinkowskiPruningFunction

from MinkowskiNetwork import MinkowskiNetwork

import MinkowskiOps

from MinkowskiOps import MinkowskiLinear, cat

import MinkowskiFunctional

from .utils import *
from .utils import gradcheck
