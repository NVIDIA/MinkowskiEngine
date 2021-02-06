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

EPS = 1e-10


def spmm(
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    size: torch.Size,
    mat: torch.Tensor,
    is_sorted: bool = False,
    cuda_spmm_alg: int = 1,
) -> torch.Tensor:

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
        result = MEB.coo_spmm_int32(
            rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg, is_sorted
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

    return result


def spmm_average(
    rows: torch.Tensor,
    cols: torch.Tensor,
    size: torch.Size,
    mat: torch.Tensor,
    cuda_spmm_alg: int = 1,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    assert len(rows) == len(cols), "Invalid length"
    if mat.is_cuda:
        assert rows.is_cuda and cols.is_cuda, "All inputs must be on cuda"
        rows = rows.int()
        cols = cols.int()
        result, COO, vals = MEB.coo_spmm_average_int32(
            rows, cols, size[0], size[1], mat, cuda_spmm_alg
        )

        # WARNING: TODO: not sorting the vals. Should not be used for generic SPMM
        # coosort only supports int32
        # return MEB.coo_spmm_int64(
        #     rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg
        # )
    else:
        # fmt: off
        rows, sort_ind = torch.sort(rows)
        cols = cols[sort_ind]
        COO = torch.stack((rows, cols), 0,).long()
        # Vals
        _, inverse_ind, counts = torch.unique(rows, return_counts=True, return_inverse=True)
        vals = (1 / counts[inverse_ind]).to(mat.dtype)
        # fmt: on
        torchSparseTensor = None
        if mat.dtype == torch.float64:
            torchSparseTensor = torch.sparse.DoubleTensor
        elif mat.dtype == torch.float32:
            torchSparseTensor = torch.sparse.FloatTensor
        else:
            raise ValueError(f"Unsupported data type: {mat.dtype}")
        sp = torchSparseTensor(COO, vals, size)
        result = sp.matmul(mat)

    return result, COO, vals


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
        result = spmm(
            rows,
            cols,
            vals,
            size,
            mat,
            is_sorted=False,
            cuda_spmm_alg=cuda_spmm_alg,
        )
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        size, cuda_spmm_alg = ctx.misc_args
        rows, cols, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(
            cols,
            rows,
            vals,
            new_size,
            grad,
            is_sorted=False,
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


class MinkowskiSPMMAverageFunction(Function):
    @staticmethod
    def forward(
        ctx,
        rows: torch.Tensor,
        cols: torch.Tensor,
        size: torch.Size,
        mat: torch.Tensor,
        cuda_spmm_alg: int = 1,
    ):
        ctx.misc_args = size, cuda_spmm_alg
        result, COO, vals = spmm_average(
            rows,
            cols,
            size,
            mat,
            cuda_spmm_alg=cuda_spmm_alg,
        )
        ctx.save_for_backward(COO, vals)
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        size, cuda_spmm_alg = ctx.misc_args
        COO, vals = ctx.saved_tensors
        new_size = torch.Size([size[1], size[0]])
        grad = spmm(
            COO[1],
            COO[0],
            vals,
            new_size,
            grad,
            is_sorted=False,
            cuda_spmm_alg=cuda_spmm_alg,
        )
        return (
            None,
            None,
            None,
            grad,
            None,
        )
