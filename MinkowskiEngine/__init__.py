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
__version__ = "0.5.4"

import os
import sys
import warnings

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Force OMP_NUM_THREADS setup
if os.cpu_count() > 16 and "OMP_NUM_THREADS" not in os.environ:
    warnings.warn(
        " ".join(
            [
                "The environment variable `OMP_NUM_THREADS` not set. MinkowskiEngine will automatically set `OMP_NUM_THREADS=16`.",
                "If you want to set `OMP_NUM_THREADS` manually, please export it on the command line before running a python script.",
                "e.g. `export OMP_NUM_THREADS=12; python your_program.py`.",
                "It is recommended to set it below 24.",
            ]
        )
    )
    os.environ["OMP_NUM_THREADS"] = str(16)

# Must be imported first to load all required shared libs
import torch

from diagnostics import print_diagnostics

from MinkowskiEngineBackend._C import (
    MinkowskiAlgorithm,
    CoordinateMapKey,
    GPUMemoryAllocatorType,
    CoordinateMapType,
    RegionType,
    PoolingMode,
    BroadcastMode,
    is_cuda_available,
    cuda_version,
    cudart_version,
    get_gpu_memory_info,
)

from MinkowskiKernelGenerator import (
    KernelRegion,
    KernelGenerator,
    convert_region_type,
    get_kernel_volume,
)

from MinkowskiTensor import (
    SparseTensorOperationMode,
    SparseTensorQuantizationMode,
    set_sparse_tensor_operation_mode,
    sparse_tensor_operation_mode,
    global_coordinate_manager,
    set_global_coordinate_manager,
    clear_global_coordinate_manager,
)

from MinkowskiSparseTensor import SparseTensor

from MinkowskiTensorField import TensorField

from MinkowskiCommon import (
    convert_to_int_tensor,
    MinkowskiModuleBase,
)

from MinkowskiCoordinateManager import (
    set_memory_manager_backend,
    set_gpu_allocator,
    CoordsManager,
    CoordinateManager,
)

from MinkowskiConvolution import (
    MinkowskiConvolutionFunction,
    MinkowskiConvolution,
    MinkowskiConvolutionTransposeFunction,
    MinkowskiConvolutionTranspose,
    MinkowskiGenerativeConvolutionTranspose,
)


from MinkowskiChannelwiseConvolution import MinkowskiChannelwiseConvolution

from MinkowskiPooling import (
    MinkowskiLocalPoolingFunction,
    MinkowskiSumPooling,
    MinkowskiAvgPooling,
    MinkowskiMaxPooling,
    MinkowskiLocalPoolingTransposeFunction,
    MinkowskiPoolingTranspose,
    MinkowskiGlobalPoolingFunction,
    MinkowskiGlobalPooling,
    MinkowskiGlobalSumPooling,
    MinkowskiGlobalAvgPooling,
    MinkowskiGlobalMaxPooling,
    MinkowskiDirectMaxPoolingFunction,
)

from MinkowskiBroadcast import (
    MinkowskiBroadcastFunction,
    MinkowskiBroadcastAddition,
    MinkowskiBroadcastMultiplication,
    MinkowskiBroadcast,
    MinkowskiBroadcastConcatenation,
)

from MinkowskiNonlinearity import (
    MinkowskiELU,
    MinkowskiHardshrink,
    MinkowskiHardsigmoid,
    MinkowskiHardtanh,
    MinkowskiHardswish,
    MinkowskiLeakyReLU,
    MinkowskiLogSigmoid,
    MinkowskiPReLU,
    MinkowskiReLU,
    MinkowskiReLU6,
    MinkowskiRReLU,
    MinkowskiSELU,
    MinkowskiCELU,
    MinkowskiGELU,
    MinkowskiSigmoid,
    MinkowskiSiLU,
    MinkowskiSoftplus,
    MinkowskiSoftshrink,
    MinkowskiSoftsign,
    MinkowskiTanh,
    MinkowskiTanhshrink,
    MinkowskiThreshold,
    MinkowskiSoftmin,
    MinkowskiSoftmax,
    MinkowskiLogSoftmax,
    MinkowskiAdaptiveLogSoftmaxWithLoss,
    MinkowskiDropout,
    MinkowskiAlphaDropout,
    MinkowskiSinusoidal,
)

from MinkowskiNormalization import (
    MinkowskiBatchNorm,
    MinkowskiSyncBatchNorm,
    MinkowskiInstanceNorm,
    MinkowskiInstanceNormFunction,
    MinkowskiStableInstanceNorm,
)


from MinkowskiPruning import MinkowskiPruning, MinkowskiPruningFunction

from MinkowskiUnion import MinkowskiUnion, MinkowskiUnionFunction

from MinkowskiInterpolation import (
    MinkowskiInterpolation,
    MinkowskiInterpolationFunction,
)

from MinkowskiNetwork import MinkowskiNetwork

import MinkowskiOps

from MinkowskiOps import (
    MinkowskiLinear,
    MinkowskiToSparseTensor,
    MinkowskiToDenseTensor,
    MinkowskiToFeature,
    MinkowskiStackCat,
    MinkowskiStackSum,
    MinkowskiStackMean,
    MinkowskiStackVar,
    cat,
    mean,
    var,
    to_sparse,
    to_sparse_all,
    dense_coordinates,
)

from MinkowskiOps import _sum as sum

import MinkowskiFunctional

import MinkowskiEngine.utils as utils

import MinkowskiEngine.modules as modules

from sparse_matrix_functions import (
    spmm,
    MinkowskiSPMMFunction,
    MinkowskiSPMMAverageFunction,
)


if not is_cuda_available():
    warnings.warn(
        " ".join(
            [
                "The MinkowskiEngine was compiled with CPU_ONLY flag.",
                "If you want to compile with CUDA support, make sure `torch.cuda.is_available()` is True when you install MinkowskiEngine.",
            ]
        )
    )
