#ifndef SPARSE_BROADCAST_CUH
#define SPARSE_BROADCAST_CUH

#include <array>
#include <vector>
#include <cusparse_v2.h>

#include "src/gpu.cuh"
#include "src/math_functions.hpp"


template <typename Dtype, typename Itype>
void SparseBroadcastForwardGPU(
    const Dtype *d_in_feat, int in_nrows, const Dtype *d_in_feat_global,
    int in_nrows_global, Dtype *d_out_feat, int nchannel, int op,
    const std::vector<std::vector<Itype>> sorted_in_map,
    const std::vector<std::vector<Itype>> sorted_out_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseBroadcastBackwardGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel, int op,
    const std::vector<std::vector<Itype>> sorted_in_map,
    const std::vector<std::vector<Itype>> sorted_out_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

#endif
