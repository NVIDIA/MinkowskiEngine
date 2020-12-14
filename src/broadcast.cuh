/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#ifndef BROADCAST_CUH
#define BROADCAST_CUH

#include <array>
#include <cusparse_v2.h>
#include <vector>
#include <torch/extension.h>

#include "gpu.cuh"
#include "gpu_memory_manager.hpp"
#include "math_functions.hpp"
#include "types.hpp"

namespace minkowski {

template <typename Dtype, typename Itype>
void BroadcastForwardKernelGPU(const Dtype *d_in_feat, int in_nrows,
                               const Dtype *d_in_feat_global,
                               int in_nrows_global, Dtype *d_out_feat,
                               int nchannel, int op,
    const vector<at::Tensor>& in_maps, const vector<at::Tensor>& out_maps,
                               cusparseHandle_t cushandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void BroadcastBackwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel, int op,
    const vector<at::Tensor>& in_maps, const vector<at::Tensor>& out_maps,
    cusparseHandle_t cushandle, cudaStream_t stream);

} // namespace minkowski

#endif
