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
#ifndef CPU_CONVOLUTION
#define CPU_CONVOLUTION

#include "math_functions.hpp"
#include "types.hpp"

namespace minkowski {

template <typename Dtype, typename Itype>
void ConvolutionForwardKernelCPU(const Dtype *p_in_feat, int in_nchannel,
                                 Dtype *p_out_feat, int out_nchannel,
                                 const Dtype *p_kernel,
                                 const cpu_in_maps &in_maps,
                                 const cpu_out_maps &out_maps) {
  int kernel_volume, n_active_in_volume, row;
  std::vector<Dtype> input_buffer, output_buffer;

  // Number of weights
  kernel_volume = in_maps.size();

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  // for (auto &current_in2out : in2out) {
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_maps[k].size();
    if (n_active_in_volume == 0)
      continue;

    input_buffer.resize(n_active_in_volume * in_nchannel);
    output_buffer.resize(n_active_in_volume * out_nchannel);

    // Gather all features (im2col)
    for (row = 0; row < n_active_in_volume; row++)
      std::memcpy(&input_buffer[row * in_nchannel],
                  p_in_feat + in_maps[k][row] * in_nchannel,
                  sizeof(Dtype) * in_nchannel);

    // C := alpha*op(A)*op(B) + beta*C
    cpu_gemm<Dtype>(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    out_nchannel,                              // M
                    n_active_in_volume,                        // N
                    in_nchannel,                               // K
                    1,                                         // alpha
                    &p_kernel[k * in_nchannel * out_nchannel], // A
                    &input_buffer[0],                          // B
                    0,                                         // beta
                    &output_buffer[0]);                        // C

    // Put it back to the correct index
    for (row = 0; row < n_active_in_volume; row++) {
      Dtype *dst = &p_out_feat[out_maps[k][row] * out_nchannel];
      Dtype *src = &output_buffer[row * out_nchannel];
      cpu_add<Dtype>(out_nchannel, src, dst, dst);
    }
  }
}

template <typename Dtype, typename Itype>
void ConvolutionBackwardKernelCPU(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                                  int in_nchannel, const Dtype *p_grad_out_feat,
                                  int out_nchannel, const Dtype *p_kernel,
                                  Dtype *p_grad_kernel,
                                  const cpu_in_maps &in_maps,
                                  const cpu_out_maps &out_maps) {
  int kernel_volume, n_active_in_volume, row;
  std::vector<Dtype> input_buffer, output_buffer;

  // Number of weights
  kernel_volume = in_maps.size();

  // for (auto &current_in2out : in2out) {
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_maps[k].size();
    if (n_active_in_volume == 0)
      continue;

    input_buffer.resize(n_active_in_volume * in_nchannel);
    output_buffer.resize(n_active_in_volume * out_nchannel);

    // Gather all features for a matrix multiplication (im2col)
    for (row = 0; row < n_active_in_volume; row++)
      std::memcpy(&output_buffer[row * out_nchannel],
                  &p_grad_out_feat[out_maps[k][row] * out_nchannel],
                  sizeof(Dtype) * out_nchannel);

    cpu_gemm<Dtype>(CblasColMajor, CblasTrans, CblasNoTrans,
                    in_nchannel,                               // M
                    n_active_in_volume,                        // N
                    out_nchannel,                              // K
                    1,                                         // alpha
                    &p_kernel[k * in_nchannel * out_nchannel], // A
                    &output_buffer[0],                         // B
                    0,                                         // beta
                    &input_buffer[0]                           // C
    );

    // Accumulate gradients back to the input grad feat
    for (row = 0; row < n_active_in_volume; row++) {
      Dtype *src = &input_buffer[row * in_nchannel];
      Dtype *dst = &p_grad_in_feat[in_maps[k][row] * in_nchannel];
      cpu_add<Dtype>(in_nchannel, src, dst, dst);
    }

    // Compute gradient for kernel
    for (row = 0; row < n_active_in_volume; row++)
      std::memcpy(&input_buffer[row * in_nchannel],
                  p_in_feat + in_maps[k][row] * in_nchannel,
                  sizeof(Dtype) * in_nchannel);

    cpu_gemm<Dtype>(CblasColMajor, CblasNoTrans, CblasTrans,
                    out_nchannel,                                  // M
                    in_nchannel,                                   // N
                    n_active_in_volume,                            // K
                    1,                                             // alpha
                    &output_buffer[0],                             // A
                    &input_buffer[0],                              // B
                    1,                                             // beta
                    &p_grad_kernel[k * in_nchannel * out_nchannel] // C
    );
  }
}

} // end namespace minkowski

#endif // CPU_CONVOLUTION
