#ifndef CPU_PRUNING
#define CPU_PRUNING

#include "common.hpp"

template <typename Dtype, typename Itype>
void PruningForwardKernelCPU(const Dtype *p_in_feat, Dtype *p_out_feat,
                             int nchannel,
                             const InOutMapPerKernel<Itype> &in_map,
                             const InOutMapPerKernel<Itype> &out_map) {
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
                              const InOutMapPerKernel<Itype> &in_map,
                              const InOutMapPerKernel<Itype> &out_map) {
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

#endif
