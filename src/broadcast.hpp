#ifndef CPU_BROADCAST
#define CPU_BROADCAST

#include "math_functions.hpp"
#include "utils.hpp"

template <typename Dtype, typename Itype>
void BroadcastForwardKernelCPU(const Dtype *p_in_feat, int in_nrows,
                               const Dtype *p_in_feat_global,
                               int in_nrows_global, Dtype *p_out_feat,
                               int nchannel, int op,
                               const InOutMapPerKernel<Itype> &in_map,
                               const InOutMapPerKernel<Itype> &glob_map) {
  Dtype *p_curr_out_feat;
  const Dtype *p_curr_in_feat_global;

  // Copy all in_feat to out_feat
  std::memcpy(p_out_feat, p_in_feat, sizeof(Dtype) * in_nrows * nchannel);

  if (in_map.size() != 1)
    throw std::invalid_argument("InOut map must have one kernel for Broadcast");

  if (in_map[0].size() != in_nrows)
    throw std::invalid_argument("Invalid in_map");

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    for (int row = 0; row < in_nrows; row++) {
      p_curr_out_feat = p_out_feat + in_map[0][row] * nchannel;
      p_curr_in_feat_global = p_in_feat_global + glob_map[0][row] * nchannel;
      cpu_add<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                     p_curr_out_feat);
    }
    break;
  case 1: // *
    for (int row = 0; row < in_nrows; row++) {
      p_curr_out_feat = p_out_feat + in_map[0][row] * nchannel;
      p_curr_in_feat_global = p_in_feat_global + glob_map[0][row] * nchannel;
      cpu_mul<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                     p_curr_out_feat);
    }
    break;
  case 2: // division
    for (int row = 0; row < in_nrows; row++) {
      p_curr_out_feat = p_out_feat + in_map[0][row] * nchannel;
      p_curr_in_feat_global = p_in_feat_global + glob_map[0][row] * nchannel;
      cpu_div<Dtype>(nchannel, p_curr_in_feat_global, p_curr_out_feat,
                     p_curr_out_feat);
    }
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }
}

template <typename Dtype, typename Itype>
void BroadcastBackwardKernelCPU(const Dtype *p_in_feat, Dtype *p_grad_in_feat,
                                int in_nrows, const Dtype *p_in_feat_global,
                                Dtype *p_grad_in_feat_global,
                                int in_nrows_global,
                                const Dtype *p_grad_out_feat, int nchannel,
                                int op, const InOutMapPerKernel<Itype> &in_map,
                                const InOutMapPerKernel<Itype> &glob_map) {
  Dtype *p_curr_grad_in_feat, *p_curr_grad_in_feat_global;
  const Dtype *p_curr_in_feat_global, *p_curr_in_feat, *p_curr_grad_out_feat;

  // Clear grad memory
  std::memset(p_grad_in_feat_global, 0,
              sizeof(Dtype) * in_nrows_global * nchannel);
  // Initialize the grad_in_feat as grad_out_feat
  std::memcpy(p_grad_in_feat, p_grad_out_feat,
              sizeof(Dtype) * in_nrows * nchannel);

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    // For p_grad_in_feat, copy all grad_out
    for (int row = 0; row < in_nrows; row++) {
      p_curr_grad_out_feat = p_grad_out_feat + in_map[0][row] * nchannel;
      p_curr_grad_in_feat_global =
          p_grad_in_feat_global + glob_map[0][row] * nchannel;
      cpu_add<Dtype>(nchannel, p_curr_grad_out_feat, p_curr_grad_in_feat_global,
                     p_curr_grad_in_feat_global);
    }
    break;
  case 1: // *
    std::memset(p_grad_in_feat, 0, sizeof(Dtype) * in_nrows * nchannel);
    for (int row = 0; row < in_nrows; row++) {
      // In feat global
      p_curr_in_feat = p_in_feat + in_map[0][row] * nchannel;
      p_curr_grad_in_feat = p_grad_in_feat + in_map[0][row] * nchannel;
      p_curr_grad_in_feat_global =
          p_grad_in_feat_global + glob_map[0][row] * nchannel;
      p_curr_grad_out_feat = p_grad_out_feat + in_map[0][row] * nchannel;
      p_curr_in_feat_global = p_in_feat_global + glob_map[0][row] * nchannel;

      // In feat
      cpu_mul<Dtype>(nchannel, p_curr_in_feat_global, p_curr_grad_out_feat,
                     p_curr_grad_in_feat);
      // In feat glob
      for (int j = 0; j < nchannel; j++) {
        p_curr_grad_in_feat_global[j] +=
            p_curr_grad_out_feat[j] * p_curr_in_feat[j];
      }
    }
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }
}

#endif
