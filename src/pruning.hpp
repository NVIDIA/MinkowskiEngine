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

#include "common.hpp"

namespace minkowski {

template <typename Dtype, typename Itype>
void PruningForwardKernelCPU(const Dtype *p_in_feat, Dtype *p_out_feat,
                             int nchannel, const InOutMaps<Itype> &in_map,
                             const InOutMaps<Itype> &out_map) {
  int row;
  const Dtype *p_curr_in;
  Dtype *p_curr_out;
  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (row = 0; row < in_map[0].size(); row++) {
    // Define current pointers
    p_curr_in = p_in_feat + in_map[0][row] * nchannel;
    p_curr_out = p_out_feat + out_map[0][row] * nchannel;
    std::memcpy(p_curr_out, p_curr_in, nchannel * sizeof(Dtype));
  }
}

template <typename Dtype, typename Itype>
void PruningBackwardKernelCPU(Dtype *p_grad_in_feat,
                              const Dtype *p_grad_out_feat, int nchannel,
                              const InOutMaps<Itype> &in_map,
                              const InOutMaps<Itype> &out_map) {
  int row;
  Dtype *p_curr_grad_in;
  const Dtype *p_curr_grad_out;
  for (row = 0; row < in_map[0].size(); row++) {
    // Define current pointers
    p_curr_grad_in = p_grad_in_feat + in_map[0][row] * nchannel;
    p_curr_grad_out = p_grad_out_feat + out_map[0][row] * nchannel;
    std::memcpy(p_curr_grad_in, p_curr_grad_out, nchannel * sizeof(Dtype));
  }
}

} // end namespace minkowski

#endif // CPU_PRUNING
