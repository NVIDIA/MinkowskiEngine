#ifndef SPARSE_CONVOLUTION_CUH
#define SPARSE_CONVOLUTION_CUH

#include <array>
#include <vector>

#include "src/gpu.cuh"
#include "src/math_functions.hpp"

template <typename Dtype>
__global__ void copy_mapped_input(const int n, const int nchannel,
                                  const Dtype *in_feat, Dtype *out_feat,
                                  const int64_t map);

template <typename Dtype>
__global__ void add_mapped_output(const int n, const int nchannel,
                                  const Dtype *in_feat, Dtype *out_feat,
                                  const int64_t map);

template <typename Dtype>
void SparseConvolutionForwardGPU(
    const Dtype *d_in_feat, int in_nchannel, Dtype *d_out_feat,
    int out_nchannel, const Dtype *d_kernel,
    const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype>
void SparseConvolutionBackwardGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nchannel,
    const Dtype *d_grad_out_feat, int out_nchannel, const Dtype *d_kernel,
    Dtype *d_grad_kernel, const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);
#endif
