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
#ifndef CPU_POOLING_AVG
#define CPU_POOLING_AVG

#include "math_functions.hpp"

#include <limits>

namespace minkowski {

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingForwardKernelCPU(const Dtype *p_in_feat,
                                       Dtype *p_out_feat, Dtype *p_num_nonzero,
                                       int nchannel,
                                       const InOutMaps<Itype> &in_map,
                                       const InOutMaps<Itype> &out_map,
                                       int out_nrows, const bool use_avg) {
  int kernel_volume, n_active_in_volume, row, j, k;
  const Dtype *p_curr_in;
  Dtype *p_curr_out;
  Dtype *p_curr_num_nonzero;

  // Number of weights
  kernel_volume = in_map.size();

  // Set all values to - Dtype min
  if (use_avg)
    std::fill(p_num_nonzero, p_num_nonzero + out_nrows, 0);
  std::fill(p_out_feat, p_out_feat + out_nrows * nchannel, 0);

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    // Use the entire for loop inside to reduce branching
    if (use_avg) {
      for (row = 0; row < n_active_in_volume; row++) {
        // Define current pointers
        p_curr_in = p_in_feat + in_map[k][row] * nchannel;
        p_curr_out = p_out_feat + out_map[k][row] * nchannel;
        p_curr_num_nonzero = p_num_nonzero + out_map[k][row];
        (*p_curr_num_nonzero)++;
        cpu_add<Dtype>(nchannel, p_curr_in, p_curr_out, p_curr_out);
      }
    } else {
      for (row = 0; row < n_active_in_volume; row++) {
        // Define current pointers
        p_curr_in = p_in_feat + in_map[k][row] * nchannel;
        p_curr_out = p_out_feat + out_map[k][row] * nchannel;
        cpu_add<Dtype>(nchannel, p_curr_in, p_curr_out, p_curr_out);
      }
    }
  }

  // Average
  if (use_avg) {
    p_curr_out = p_out_feat;
    p_curr_num_nonzero = p_num_nonzero;
    for (row = 0; row < out_nrows; row++) {
      for (j = 0; j < nchannel; j++) {
        if (*p_curr_num_nonzero > 0)
          *p_curr_out /= *p_curr_num_nonzero;
        p_curr_out++;
      }
      p_curr_num_nonzero++;
    }
  }
}

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingBackwardKernelCPU(
    Dtype *p_grad_in_feat, int in_nrows, const Dtype *p_grad_out_feat,
    const Dtype *p_num_nonzero, int nchannel, const InOutMaps<Itype> &in_map,
    const InOutMaps<Itype> &out_map, const bool use_avg) {
  int kernel_volume, n_active_in_volume, row, j, k;
  Dtype *p_curr_grad_in, curr_num_nonzero;
  const Dtype *p_curr_grad_out;

  // Number of weights
  kernel_volume = in_map.size();

  // cleanup gradients
  std::fill(p_grad_in_feat, p_grad_in_feat + in_nrows * nchannel, 0);

  for (k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    for (row = 0; row < n_active_in_volume; row++) {
      // Define current pointers
      p_curr_grad_in = p_grad_in_feat + in_map[k][row] * nchannel;
      p_curr_grad_out = p_grad_out_feat + out_map[k][row] * nchannel;

      // To speed up, create if outside for loop
      if (use_avg) {
        curr_num_nonzero = p_num_nonzero[out_map[k][row]];
        for (j = 0; j < nchannel; j++) {
          if (curr_num_nonzero > 0)
            *p_curr_grad_in += *p_curr_grad_out / curr_num_nonzero;
          p_curr_grad_in++;
          p_curr_grad_out++;
        }
      } else {
        for (j = 0; j < nchannel; j++) {
          *p_curr_grad_in += *p_curr_grad_out;
          p_curr_grad_in++;
          p_curr_grad_out++;
        }
      }
    }
  }
}

} // namespace minkowski

#endif // end CPU_POOLING_AVG
