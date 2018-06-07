#ifndef SPARSE_POOLING_CUH
#define SPARSE_POOLING_CUH

#include <array>
#include <vector>

#include "src/gpu.cuh"
#include "src/math_functions.hpp"

template <typename Dtype, typename Itype>
void SparseMaxPoolingForwardGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index,
                                int nchannel,
                                const std::vector<std::vector<Itype>> in_map,
                                const std::vector<std::vector<Itype>> out_map,
                                cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseMaxPoolingBackwardGPU(Dtype *d_grad_in_feat, int in_nrows,
                                 const Dtype *d_grad_out_feat,
                                 int out_nrows, const Itype *d_max_index,
                                 int nchannel, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseNonzeroAvgPoolingForwardGPU(
    const Dtype *d_in_feat, int in_nrows, Dtype *d_out_feat, int out_nrows,
    Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> in_map,
    const std::vector<std::vector<Itype>> out_map, cusparseHandle_t cushandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseNonzeroAvgPoolingBackwardGPU(
    Dtype *d_grad_in_feat, int in_nrows, const Dtype *d_grad_out_feat,
    int out_nrows, const Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> in_map,
    const std::vector<std::vector<Itype>> out_map, cudaStream_t stream);
#endif
