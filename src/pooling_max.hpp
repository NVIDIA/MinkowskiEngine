#ifndef CPU_POOLING_MAX
#define CPU_POOLING_MAX

#include "math_functions.hpp"

#include <limits>

template <typename Dtype, typename Itype>
void MaxPoolingForwardKernelCPU(const Dtype *p_in_feat, Dtype *p_out_feat,
                                Itype *p_mask_index, int nchannel,
                                const InOutMapPerKernel<Itype> &in_map,
                                const InOutMapPerKernel<Itype> &out_map,
                                int out_nrows) {
  int kernel_volume, n_active_in_volume, row, j, k;
  const Dtype *p_curr_in;
  Dtype *p_curr_out;
  Itype *p_curr_mask_index;

  // Number of weights
  kernel_volume = in_map.size();

  // Set all values to - Dtype min
  std::fill(p_mask_index, p_mask_index + out_nrows * nchannel, -1);
  std::fill(p_out_feat, p_out_feat + out_nrows * nchannel,
            -std::numeric_limits<Dtype>::max());

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    for (row = 0; row < n_active_in_volume; row++) {
      // Define current pointers
      p_curr_in = p_in_feat + in_map[k][row] * nchannel;
      p_curr_out = p_out_feat + out_map[k][row] * nchannel;
      p_curr_mask_index = p_mask_index + out_map[k][row] * nchannel;

      for (j = 0; j < nchannel; j++) {
        if (*p_curr_out < *p_curr_in) {
          *p_curr_out = *p_curr_in;
          *p_curr_mask_index = in_map[k][row] * nchannel + j;
        }
        p_curr_in++;
        p_curr_out++;
        p_curr_mask_index++;
      }
    }
  }
}

template <typename Dtype, typename Itype>
void MaxPoolingBackwardKernelCPU(Dtype *p_grad_in_feat, int in_nrows,
                                 const Dtype *p_grad_out_feat, int out_nrows,
                                 const Itype *p_mask_index, int nchannel,
                                 const InOutMapPerKernel<Itype> &in_map,
                                 const InOutMapPerKernel<Itype> &out_map) {
  const Dtype *p_curr_grad_out;
  const Itype *p_curr_mask_index;

  // cleanup gradients
  std::fill(p_grad_in_feat, p_grad_in_feat + in_nrows * nchannel, 0);

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

#endif
