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
#ifndef CPU_BROADCAST
#define CPU_BROADCAST

#include "math_functions.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace minkowski {

template <typename Dtype, typename Itype>
void BroadcastForwardKernelCPU(const Dtype *p_in_feat, uint32_t in_nrows,
                               const Dtype *p_in_feat_global,
                               uint32_t in_nrows_global, Dtype *p_out_feat,
                               uint32_t nchannel, BroadcastMode::Type const op,
                               const cpu_in_maps &in_maps,
                               const cpu_out_maps &glob_maps) {
  Dtype *p_curr_out_feat;
  const Dtype *p_curr_in_feat_global;

  // Compute the size
  uint32_t num_map = 0;
  for (const auto &in_map : in_maps)
    num_map += in_map.size();
  ASSERT(num_map == in_nrows, "The number of in-out map,", num_map,
         " mismatches the number of features,", in_nrows);

  // Copy all in_feat to out_feat
  std::memcpy(p_out_feat, p_in_feat, sizeof(Dtype) * in_nrows * nchannel);

  // To speed up, put switch outside for loops
  switch (op) {
  case BroadcastMode::ELEMENTWISE_ADDITON: // +
    for (uint32_t k = 0; k < in_maps.size(); ++k) {
      for (uint32_t row = 0; row < in_maps[k].size(); ++row) {
        p_curr_out_feat = p_out_feat + in_maps[k][row] * nchannel;
        p_curr_in_feat_global = p_in_feat_global + glob_maps[k][row] * nchannel;
        cpu_add<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                       p_curr_out_feat);
      }
    }
    break;
  case BroadcastMode::ELEMENTWISE_MULTIPLICATION: // *
    for (uint32_t k = 0; k < in_maps.size(); ++k) {
      for (uint32_t row = 0; row < in_maps[k].size(); ++row) {
        p_curr_out_feat = p_out_feat + in_maps[k][row] * nchannel;
        p_curr_in_feat_global = p_in_feat_global + glob_maps[k][row] * nchannel;
        cpu_mul<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                       p_curr_out_feat);
      }
    }
    break;
  /*
  case 2: // division
    for (int k = 0; k < in_maps.size(); ++k) {
      for (int row = 0; row < in_maps[k].size(); ++row) {
        p_curr_out_feat = p_out_feat + in_maps[k][row] * nchannel;
        p_curr_in_feat_global = p_in_feat_global + glob_maps[k][row] * nchannel;
        cpu_div<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                       p_curr_out_feat);
      }
    }
    break;
  */
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }
}

template <typename Dtype, typename Itype>
void BroadcastBackwardKernelCPU(const Dtype *p_in_feat,                   //
                                Dtype *p_grad_in_feat, uint32_t in_nrows, //
                                const Dtype *p_in_feat_global,
                                Dtype *p_grad_in_feat_global,
                                uint32_t in_nrows_global,     //
                                const Dtype *p_grad_out_feat, //
                                uint32_t nchannel,
                                BroadcastMode::Type const op, //
                                const cpu_in_maps &in_maps,
                                const cpu_out_maps &glob_maps) {
  Dtype *p_curr_grad_in_feat, *p_curr_grad_in_feat_global;
  const Dtype *p_curr_in_feat_global, *p_curr_in_feat, *p_curr_grad_out_feat;

  // Assume that the memory is cleared
  /*
  // Clear grad memory
  std::memset(p_grad_in_feat_global, 0,
              sizeof(Dtype) * in_nrows_global * nchannel);
  */

  // Initialize the grad_in_feat as grad_out_feat
  std::memcpy(p_grad_in_feat, p_grad_out_feat,
              sizeof(Dtype) * in_nrows * nchannel);

  // To speed up, put switch outside for loops
  switch (op) {
  case BroadcastMode::ELEMENTWISE_ADDITON: // +
    // For p_grad_in_feat, copy all grad_out
    for (uint32_t k = 0; k < in_maps.size(); ++k) {
      for (uint32_t row = 0; row < in_maps[k].size(); ++row) {
        p_curr_grad_out_feat = p_grad_out_feat + in_maps[k][row] * nchannel;
        p_curr_grad_in_feat_global =
            p_grad_in_feat_global + glob_maps[k][row] * nchannel;
        cpu_add<Dtype>(nchannel, p_curr_grad_out_feat,
                       p_curr_grad_in_feat_global, p_curr_grad_in_feat_global);
      }
    }
    break;
  case BroadcastMode::ELEMENTWISE_MULTIPLICATION: // *
    for (uint32_t k = 0; k < in_maps.size(); ++k) {
      for (uint32_t row = 0; row < in_maps[k].size(); ++row) {
        // In feat global
        p_curr_in_feat = p_in_feat + in_maps[k][row] * nchannel;
        p_curr_grad_in_feat = p_grad_in_feat + in_maps[k][row] * nchannel;
        p_curr_grad_in_feat_global =
            p_grad_in_feat_global + glob_maps[k][row] * nchannel;
        p_curr_grad_out_feat = p_grad_out_feat + in_maps[k][row] * nchannel;
        p_curr_in_feat_global = p_in_feat_global + glob_maps[k][row] * nchannel;

        // In feat
        cpu_mul<Dtype>(nchannel, p_curr_in_feat_global, p_curr_grad_out_feat,
                       p_curr_grad_in_feat);
        // In feat glob
        for (uint32_t j = 0; j < nchannel; j++) {
          p_curr_grad_in_feat_global[j] +=
              p_curr_grad_out_feat[j] * p_curr_in_feat[j];
        }
      }
    }
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }
}

} // namespace minkowski

#endif
