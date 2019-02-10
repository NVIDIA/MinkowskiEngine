#ifndef GPU_PRUNING
#define GPU_PRUNING

#include <array>
#include <vector>

template <typename Dtype, typename Itype>
void PruningForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                             int nchannel,
                             const std::vector<std::vector<Itype>> &in_maps,
                             const std::vector<std::vector<Itype>> &out_maps,
                             cudaStream_t stream);

template <typename Dtype, typename Itype>
void PruningBackwardKernelGPU(Dtype *d_grad_in_feat,
                              const Dtype *d_grad_out_feat, int nchannel,
                              const std::vector<std::vector<Itype>> &in_maps,
                              const std::vector<std::vector<Itype>> &out_maps,
                              cudaStream_t stream);
#endif
