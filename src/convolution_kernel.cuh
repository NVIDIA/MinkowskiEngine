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
#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include <array>
#include <vector>

#include "gpu.cuh"
#include "kernel_map.cuh"
#include "math_functions.cuh"
#include "types.hpp"

namespace minkowski {

template <typename Dtype, typename Itype, typename ByteAllocator>
void ConvolutionForwardKernelGPU(
    Dtype const *d_in_feat,                      //
    default_types::size_type const in_nchannel,  //
    Dtype *d_out_feat,                           //
    default_types::size_type const out_nchannel, //
    Dtype *d_kernel, gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    default_types::size_type const in_nrows,      //
    default_types::size_type const out_nrows,     //
    ByteAllocator &allocator,                     //
    MinkowskiAlgorithm::Mode const algo_index,    //
    ConvolutionMode::Type const convolution_mode, //
    cublasHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype, typename ByteAllocator>
void ConvolutionBackwardKernelGPU(
    Dtype const *d_in_feat,                      //
    Dtype *d_grad_in_feat,                       //
    default_types::size_type const in_nchannel,  //
    Dtype const *d_grad_out_feat,                //
    default_types::size_type const out_nchannel, //
    Dtype const *d_kernel,                       //
    Dtype *d_grad_kernel,                        //
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    default_types::size_type const in_nrows,      //
    default_types::size_type const out_nrows,     //
    ByteAllocator &allocator,                     //
    MinkowskiAlgorithm::Mode const algo_index,    //
    ConvolutionMode::Type const convolution_mode, //
    cublasHandle_t cuhandle, cudaStream_t stream);
} // end namespace minkowski

#endif // end CONVOLUTION_CUH
