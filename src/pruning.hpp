/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef CPU_PRUNING
#define CPU_PRUNING

#include "types.hpp"

namespace minkowski {

template <typename Dtype>
void PruningForwardKernelCPU(const Dtype *p_in_feat, Dtype *p_out_feat,
                             int const nchannel, const cpu_in_maps &in_maps,
                             const cpu_out_maps &out_maps) {
  const Dtype *p_curr_in;
  Dtype *p_curr_out;
  auto const &in_map = in_maps[0];
  auto const &out_map = out_maps[0];
  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (uint32_t row = 0; row < in_map.size(); row++) {
    // Define current pointers
    p_curr_in = p_in_feat + in_map[row] * nchannel;
    p_curr_out = p_out_feat + out_map[row] * nchannel;
    std::memcpy(p_curr_out, p_curr_in, nchannel * sizeof(Dtype));
  }
}

template <typename Dtype>
void PruningBackwardKernelCPU(Dtype *p_grad_in_feat,
                              const Dtype *p_grad_out_feat, int const nchannel,
                              const cpu_in_maps &in_maps,
                              const cpu_out_maps &out_maps) {
  Dtype *p_curr_grad_in;
  const Dtype *p_curr_grad_out;
  auto const &in_map = in_maps[0];
  auto const &out_map = out_maps[0];

  for (uint32_t row = 0; row < in_map.size(); row++) {
    // Define current pointers
    p_curr_grad_in = p_grad_in_feat + in_map[row] * nchannel;
    p_curr_grad_out = p_grad_out_feat + out_map[row] * nchannel;
    std::memcpy(p_curr_grad_in, p_curr_grad_out, nchannel * sizeof(Dtype));
  }
}

} // end namespace minkowski

#endif // CPU_PRUNING
