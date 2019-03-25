#ifndef POOLING_MAX_CUH
#define POOLING_MAX_CUH

#include <array>
#include <vector>

#include "gpu.cuh"
#include "math_functions.hpp"

template <typename Dtype, typename Itype>
void ThrustMaxPoolingForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index, int nchannel,
                                const std::vector<std::vector<Itype>> &in_map,
                                const std::vector<std::vector<Itype>> &out_map,
                                cudaStream_t stream);

template <typename Dtype, typename Itype>
void ThrustMaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, int in_nrows,
                                 const Dtype *d_grad_out_feat, int out_nrows,
                                 const Itype *d_max_index, int nchannel,
                                 cudaStream_t stream);

template <typename Dtype, typename Itype>
void ThrustMaxPoolingForwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_out_feat, int out_nrows,
    Itype *d_max_index, int nchannel,
    const std::vector<std::vector<Itype>> &in_map,
    const std::vector<std::vector<Itype>> &out_map, cudaStream_t stream);

template <typename Dtype, typename Itype>
void ThrustMaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, int in_nrows,
                                       const Dtype *d_grad_out_feat,
                                       int out_nrows, const Itype *d_max_index,
                                       int nchannel, cudaStream_t stream);
#endif
