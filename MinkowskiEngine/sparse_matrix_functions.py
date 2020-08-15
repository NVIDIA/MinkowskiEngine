# Copyright (c) 2020 NVIDIA CORPORATION.
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

import MinkowskiEngineBackend._C as MEB


def spmm(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    size: torch.Size,
    mat: torch.Tensor,
    cuda_spmm_alg: int = 1,
):
    if mat.is_cuda:
        assert rows.is_cuda and cols.is_cuda and vals.is_cuda
        if MEB.cuda_version() < 11000:
            rows = rows.int()
            cols = cols.int()
            return MEB.coo_spmm_int32(
                rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg
            )
        else:
            if rows.dtype == torch.int32:
                return MEB.coo_spmm_int32(
                    rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg
                )
            else:
                return MEB.coo_spmm_int64(
                    rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg
                )
    else:
        COO = torch.stack((rows, cols), 0,)
        torchSparseTensor = None
        if vals.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif vals.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError("Unsupported data type")

        sp = torchSparseTensor(COO, vals, size)
        return sp.matmul(mat)
