#ifndef POOLING_AVG_CUH
#define POOLING_AVG_CUH

#include <array>
#include <vector>

#include "gpu.cuh"
#include "math_functions.hpp"

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingForwardKernelGPU(
    const Dtype *d_in_feat, int in_nrows, Dtype *d_out_feat, int out_nrows,
    Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> &in_map,
    const std::vector<std::vector<Itype>> &out_map, bool use_avg,
    cusparseHandle_t cushandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingBackwardKernelGPU(
    Dtype *d_grad_in_feat, int in_nrows, const Dtype *d_grad_out_feat,
    int out_nrows, const Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> &in_map,
    const std::vector<std::vector<Itype>> &out_map, bool use_avg,
    cudaStream_t stream);
#endif
