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
from torch.autograd import Function

import MinkowskiEngineBackend._C as MEB


def spmm(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    size: torch.Size,
    mat: torch.Tensor,
    return_num_nonzero: bool = False,
    cuda_spmm_alg: int = 1,
):

    assert len(rows) == len(cols), "Invalid length"
    assert len(rows) == len(vals), "Invalid length"
    assert vals.dtype == mat.dtype, "dtype mismatch"
    assert vals.device == mat.device, "device mismatch"
    if mat.is_cuda:
        assert (
            rows.is_cuda and cols.is_cuda and vals.is_cuda
        ), "All inputs must be on cuda"
        rows = rows.int()
        cols = cols.int()
        result, num_nonzero = MEB.coo_spmm_int32(
            rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg, return_num_nonzero
        )

        # WARNING: TODO: not sorting the vals. Should not be used for generic SPMM
        # coosort only supports int32
        # return MEB.coo_spmm_int64(
        #     rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg
        # )
    else:
        COO = torch.stack(
            (rows, cols),
            0,
        ).long()
        torchSparseTensor = None
        if vals.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif vals.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f"Unsupported data type: {vals.dtype}")

        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)
        if return_num_nonzero:
            num_nonzero = sp.matmul(torch.ones((size[1], 1), dtype=vals.dtype))

    if return_num_nonzero:
        return result, num_nonzero
    else:
        return result


class MinkowskiSPMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        rows: torch.Tensor,
        cols: torch.Tensor,
        vals: torch.Tensor,
        size: torch.Size,
        mat: torch.Tensor,
        cuda_spmm_alg: int = 1,
    ):
        ctx.misc_args = size, cuda_spmm_alg
        ctx.save_for_backward(rows, cols, vals)
        mat = mat.contiguous()
        return spmm(
            rows,
            cols,
            vals,
            size,
            mat,
            return_num_nonzero=False,
            cuda_spmm_alg=cuda_spmm_alg,
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        size, cuda_spmm_alg = ctx.misc_args
        rows, cols, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = grad.contiguous()
        grad = spmm(
            cols,
            rows,
            vals,
            new_size,
            grad,
            return_num_nonzero=False,
            cuda_spmm_alg=cuda_spmm_alg,
        )
        return (
            None,
            None,
            None,
            None,
            grad,
            None,
        )
