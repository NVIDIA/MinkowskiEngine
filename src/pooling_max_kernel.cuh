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
#ifndef POOLING_MAX_CUH
#define POOLING_MAX_CUH

#include <array>
#include <vector>

#include "allocators.cuh"
#include "gpu.cuh"
#include "kernel_map.cuh"
#include "math_functions.cuh"
#include "types.hpp"

namespace minkowski {

template <typename Dtype, typename MaskItype, typename MapItype>
void max_pool_forward_pointer_kernel_gpu(
    MapItype *d_in_map,     // this will be sorted
    MapItype *d_out_map,    // this will be sorted
    size_t const nmap,      // map size
    Dtype const *d_in_feat, //
    Dtype *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    MaskItype *d_max_index, //
    bool const is_sorted    //
);

template <typename Dtype, typename MapItype, typename ByteAllocator>
void MaxPoolingForwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_out_feat, size_t const out_nrows,
    int *d_max_index, size_t const nchannel,
    gpu_kernel_map<MapItype, ByteAllocator> const &kernel_map,
    ByteAllocator &allocator, cudaStream_t stream);

template <typename Dtype, typename MaskItype>
void MaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, size_t const in_nrows,
                                 const Dtype *d_grad_out_feat,
                                 size_t const out_nrows,
                                 const MaskItype *d_max_index,
                                 size_t const nchannel);

} // end namespace minkowski

#endif // end POOLING_MAX_CUH
