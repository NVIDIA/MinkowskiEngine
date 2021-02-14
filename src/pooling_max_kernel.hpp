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
#ifndef CPU_POOLING_MAX
#define CPU_POOLING_MAX

#include "math_functions.hpp"

#include <limits>
#include <omp.h>

namespace minkowski {

template <typename Dtype, typename MaskItype, typename MapItype>
void max_pooling_forward_pointer_kernel_cpu(Dtype const *p_in_feat,
                                            Dtype *p_out_feat,
                                            MaskItype *p_mask_index,
                                            size_t const nchannel,
                                            MapItype const *const p_in_maps,  //
                                            MapItype const *const p_out_maps, //
                                            size_t const map_size) {
  Dtype const *p_curr_in;
  Dtype *p_curr_out;
  MaskItype *p_curr_mask_index;

  for (size_t i = 0; i < map_size; ++i) {
    // Define current pointers
    MapItype in_offset = p_in_maps[i] * nchannel;
    p_curr_in = p_in_feat + in_offset;
    p_curr_out = p_out_feat + p_out_maps[i] * nchannel;
    p_curr_mask_index = p_mask_index + p_out_maps[i] * nchannel;

    for (size_t j = 0; j < nchannel; j++) {
      if (*p_curr_out < *p_curr_in) {
        *p_curr_out = *p_curr_in;
        *p_curr_mask_index = in_offset + j;
      }
      // forward all pointers
      p_curr_in++;
      p_curr_out++;
      p_curr_mask_index++;
    }
  }
}

template <typename Dtype, typename MaskItype, typename MapItype>
void MaxPoolingForwardKernelCPU(Dtype const *p_in_feat, Dtype *p_out_feat,
                                MaskItype *p_mask_index, int const nchannel,
                                cpu_in_maps const &in_maps,   //
                                cpu_out_maps const &out_maps, //
                                int const out_nrows) {
  int kernel_volume = in_maps.size();

  // Set all values to - Dtype min
  std::fill(p_mask_index, p_mask_index + out_nrows * nchannel, -1);
  std::fill(p_out_feat, p_out_feat + out_nrows * nchannel,
            -std::numeric_limits<Dtype>::max());

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (int k = 0; k < kernel_volume; ++k) {
    int n_active_in_volume = in_maps[k].size();
    if (n_active_in_volume == 0)
      continue;

    max_pooling_forward_pointer_kernel_cpu<Dtype, MaskItype, MapItype>(
        p_in_feat, p_out_feat, p_mask_index, nchannel, in_maps[k].data(),
        out_maps[k].data(), n_active_in_volume);
  }
}

template <typename Dtype, typename MaskItype>
void MaxPoolingBackwardKernelCPU(Dtype *p_grad_in_feat, size_t const in_nrows,
                                 Dtype const *p_grad_out_feat,
                                 size_t const out_nrows,
                                 MaskItype const *p_mask_index,
                                 size_t const nchannel) {
  const Dtype *p_curr_grad_out;
  const MaskItype *p_curr_mask_index;

  // cleanup gradients
  // std::fill(p_grad_in_feat, p_grad_in_feat + in_nrows * nchannel, 0);

  p_curr_grad_out = p_grad_out_feat;
  p_curr_mask_index = p_mask_index;

  for (int row = 0; row < out_nrows; row++) {
    for (int j = 0; j < nchannel; j++) {
      // Accumulate gradients
      p_grad_in_feat[*p_curr_mask_index] += *p_curr_grad_out;
      p_curr_grad_out++;
      p_curr_mask_index++;
    }
  }
}

} // end namespace minkowski

#endif // CPU_POOLING_MAX
