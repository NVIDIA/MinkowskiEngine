#ifndef CPU_POOLING
#define CPU_POOLING

#include "src/math_functions.hpp"
#include <limits>

template <typename Dtype, typename Itype>
void SparseMaxPoolingForward(const Dtype *p_in_feat, Dtype *p_out_feat,
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
void SparseMaxPoolingBackward(Dtype *p_grad_in_feat, int in_nrows,
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

template <typename Dtype, typename Itype>
void SparseNonzeroAvgPoolingForward(const Dtype *p_in_feat, Dtype *p_out_feat,
                                    Dtype *p_num_nonzero, int nchannel,
                                    const InOutMapPerKernel<Itype> &in_map,
                                    const InOutMapPerKernel<Itype> &out_map,
                                    int out_nrows, bool use_avg) {
  int kernel_volume, n_active_in_volume, row, j, k;
  const Dtype *p_curr_in;
  Dtype *p_curr_out;
  Dtype *p_curr_num_nonzero;

  // Number of weights
  kernel_volume = in_map.size();

  // Set all values to - Dtype min
  std::fill(p_num_nonzero, p_num_nonzero + out_nrows, 0);
  std::fill(p_out_feat, p_out_feat + out_nrows * nchannel, 0);

  // Iterate through each spatial kernel out of filter_volume spatial kernels
  for (k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    for (row = 0; row < n_active_in_volume; row++) {
      // Define current pointers
      p_curr_in = p_in_feat + in_map[k][row] * nchannel;
      p_curr_out = p_out_feat + out_map[k][row] * nchannel;
      p_curr_num_nonzero = p_num_nonzero + out_map[k][row];
      (*p_curr_num_nonzero)++;
      cpu_add<Dtype>(nchannel, p_curr_in, p_curr_out, p_curr_out);
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
void SparseNonzeroAvgPoolingBackward(Dtype *p_grad_in_feat, int in_nrows,
                                     const Dtype *p_grad_out_feat,
                                     int out_nrows, const Dtype *p_num_nonzero,
                                     int nchannel,
                                     const InOutMapPerKernel<Itype> &in_map,
                                     const InOutMapPerKernel<Itype> &out_map,
                                     bool use_avg) {
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
      curr_num_nonzero = p_num_nonzero[out_map[k][row]];

      // To speed up, create if outside for loop
      if (use_avg) {
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

#endif
