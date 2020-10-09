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
#ifndef INTERPOLATION_KERNEL
#define INTERPOLATION_KERNEL

#include "math_functions.hpp"

#include <limits>

namespace minkowski {

/**
 * CPU pooling function. The p_out_feat must be initialized and set to 0.
 * p_num_nonzero is set to 0 inside this function.
 *
 * TODO consistent memset
 */
template <typename Dtype, typename Wtype, typename Itype>
void InterpolationForwardKernelCPU(Dtype const *const p_in_feat,
                                   Dtype *p_out_feat,           //
                                   uint32_t const nchannel,     //
                                   Itype const *const in_maps,  //
                                   Itype const *const out_maps, //
                                   Wtype const *const weights,  //
                                   uint32_t const nnz) {
  const Dtype *p_curr_in;
  Dtype *p_curr_out;

  // Set all values to - Dtype min
  // std::fill(p_out_feat, p_out_feat + nnz * nchannel, 0);

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (uint32_t i = 0; i < nnz; ++i) {
    // Define current pointers
    p_curr_in = p_in_feat + in_maps[i] * nchannel;
    p_curr_out = p_out_feat + out_maps[i] * nchannel;
    cpu_axpy<Dtype>(nchannel, (Dtype)weights[i], p_curr_in, p_curr_out);
  }
}

template <typename Dtype, typename Wtype, typename Itype>
void InterpolationBackwardKernelCPU(Dtype *p_grad_in_feat,
                                    uint32_t const in_nrows,
                                    uint32_t const nchannel, //
                                    Dtype const *const p_grad_out_feat,
                                    Itype const *const in_maps, //
                                    Itype const *const out_maps,
                                    Wtype const *const weights,
                                    uint32_t const nnz) {
  Dtype *p_curr_grad_in;
  Dtype const *p_curr_grad_out;

  // cleanup gradients
  // std::fill(p_grad_in_feat, p_grad_in_feat + in_nrows * nchannel, 0);

  for (uint32_t i = 0; i < nnz; ++i) {
    // Define current pointers
    p_curr_grad_in = p_grad_in_feat + in_maps[i] * nchannel;
    p_curr_grad_out = p_grad_out_feat + out_maps[i] * nchannel;

    cpu_axpy<Dtype>(nchannel, (Dtype)weights[i], p_curr_grad_out,
                    p_curr_grad_in);
  }
}

template void
InterpolationForwardKernelCPU<float, float, int>(float const *const p_in_feat,
                                                 float *p_out_feat,          //
                                                 uint32_t const nchannel,    //
                                                 int const *const in_maps,   //
                                                 int const *const out_maps,  //
                                                 float const *const weights, //
                                                 uint32_t const nnz);

template void
InterpolationForwardKernelCPU<double, float, int>(double const *const p_in_feat,
                                                  double *p_out_feat,         //
                                                  uint32_t const nchannel,    //
                                                  int const *const in_maps,   //
                                                  int const *const out_maps,  //
                                                  float const *const weights, //
                                                  uint32_t const nnz);

template void InterpolationBackwardKernelCPU<float, float, int>(
    float *p_grad_in_feat,   //
    uint32_t const in_nrows, //
    uint32_t const nchannel, //
    float const *const p_grad_out_feat,
    int const *const in_maps,   //
    int const *const out_maps,  //
    float const *const weights, //
    uint32_t const nnz);

template void InterpolationBackwardKernelCPU<double, float, int>(
    double *p_grad_in_feat,  //
    uint32_t const in_nrows, //
    uint32_t const nchannel, //
    double const *const p_grad_out_feat,
    int const *const in_maps,   //
    int const *const out_maps,  //
    float const *const weights, //
    uint32_t const nnz);

} // namespace minkowski

#endif // end INTERPOLATION_KERNEL
