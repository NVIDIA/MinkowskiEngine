#ifndef GPU_CONVOLUTION
#define GPU_CONVOLUTION

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "src/math_functions.hpp"
#include "src/sparse_convolution.cuh"

template <typename Dtype>
__global__ void copy_mapped_input(const int n, const int nchannel,
                                  const Dtype *in_feat, Dtype *out_feat,
                                  const int64_t *map) {
  CUDA_KERNEL_LOOP(index, n) {
    const int row = index / nchannel;
    const int col = index % nchannel;
    out_feat[index] = in_feat[map[row] * nchannel + col];
  }
}

template <typename Dtype>
__global__ void add_mapped_output(const int n, const int nchannel,
                                  const Dtype *in_feat, Dtype *out_feat,
                                  const int64_t *map) {
  CUDA_KERNEL_LOOP(index, n) {
    const int row = index / nchannel;
    const int col = index % nchannel;
    atomicAdd(&out_feat[map[row] * nchannel + col], in_feat[index]);
  }
}

template <typename Dtype>
__global__ void add_mapped_output_tr(const int n, const Dtype *in_feat,
                                     const int in_nchannel, Dtype *out_feat,
                                     const int out_nchannel,
                                     const int64_t *map) {
  CUDA_KERNEL_LOOP(index, n) {
    const int row = index % in_nchannel;
    const int col = index / in_nchannel;
    atomicAdd(&out_feat[map[row] * out_nchannel + col], in_feat[index]);
  }
}

template <typename Dtype>
void SparseConvolutionForwardGPU(
    const Dtype *d_in_feat, int in_nchannel, Dtype *d_out_feat,
    int out_nchannel, const Dtype *d_kernel,
    const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  int kernel_volume, n_active_in_volume, num_kernels;
  thrust::device_vector<Dtype> d_input_buffer, d_output_buffer;
  thrust::device_vector<int64_t> d_in_map, d_out_map;

  // Copy the in_map, out_map to GPU
  // First im2col, gather all indices of in2out
  kernel_volume = in_map.size();
  // Iterate through each spatial kernel and get indices for in_map and
  // out_map
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    // Copy (*p_in_maps)[k] to GPU
    d_in_map = in_map[k];
    CUDA_POST_KERNEL_CHECK;
    d_out_map = out_map[k];
    CUDA_POST_KERNEL_CHECK;
    d_input_buffer.resize(n_active_in_volume * in_nchannel);
    CUDA_POST_KERNEL_CHECK;
    d_output_buffer.resize(n_active_in_volume * out_nchannel);
    CUDA_POST_KERNEL_CHECK;

    num_kernels = in_nchannel * n_active_in_volume;

    // Copy features to the buffer
    copy_mapped_input<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, in_nchannel, d_in_feat,
            thrust::raw_pointer_cast(d_input_buffer.data()),
            thrust::raw_pointer_cast(d_in_map.data()));
    CUDA_POST_KERNEL_CHECK;

    // GEMM
    gpu_gemm<Dtype>(cuhandle, CblasTrans, CblasTrans,
                    out_nchannel,                                      // M
                    n_active_in_volume,                                // N
                    in_nchannel,                                       // K
                    1,                                                 // alpha
                    &d_kernel[k * in_nchannel * out_nchannel],         // A
                    thrust::raw_pointer_cast(d_input_buffer.data()),   // B
                    0,                                                 // beta
                    thrust::raw_pointer_cast(d_output_buffer.data())); // C
    CUDA_POST_KERNEL_CHECK;

    // Put it back to the correct index.
    // The out_buffer is in column major order, d_out_feat in row major
    num_kernels = out_nchannel * n_active_in_volume;
    add_mapped_output_tr<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, thrust::raw_pointer_cast(d_output_buffer.data()),
            n_active_in_volume,       // In
            d_out_feat, out_nchannel, // Out
            thrust::raw_pointer_cast(d_out_map.data()));
    CUDA_POST_KERNEL_CHECK;
  }
}

template void SparseConvolutionForwardGPU<float>(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype>
void SparseConvolutionBackwardGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nchannel,
    const Dtype *d_grad_out_feat, int out_nchannel, const Dtype *d_kernel,
    Dtype *d_grad_kernel, const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  int kernel_volume, n_active_in_volume, num_kernels;
  thrust::device_vector<Dtype> d_input_buffer, d_output_buffer;
  thrust::device_vector<int64_t> d_in_map, d_out_map;

  // Copy the in_map, out_map to GPU
  // First im2col, gather all indices of in2out
  kernel_volume = in_map.size();
  // Iterate through each spatial kernel and get indices for in_map and
  // out_map
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_map[k].size();
    if (n_active_in_volume == 0)
      continue;

    // Copy (*p_in_maps)[k] to GPU
    d_in_map = in_map[k];
    CUDA_POST_KERNEL_CHECK;
    d_out_map = out_map[k];
    CUDA_POST_KERNEL_CHECK;
    d_input_buffer.resize(n_active_in_volume * in_nchannel);
    CUDA_POST_KERNEL_CHECK;
    d_output_buffer.resize(n_active_in_volume * out_nchannel);
    CUDA_POST_KERNEL_CHECK;
    num_kernels = out_nchannel * n_active_in_volume;

    // Copy gradients to the buffer
    copy_mapped_input<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, out_nchannel, d_grad_out_feat,
            thrust::raw_pointer_cast(d_output_buffer.data()),
            thrust::raw_pointer_cast(d_out_map.data()));
    CUDA_POST_KERNEL_CHECK;

    gpu_gemm<Dtype>(cuhandle, CblasNoTrans, CblasTrans,
                    in_nchannel,                                      // M
                    n_active_in_volume,                               // N
                    out_nchannel,                                     // K
                    1,                                                // alpha
                    &d_kernel[k * in_nchannel * out_nchannel],        // A
                    thrust::raw_pointer_cast(d_output_buffer.data()), // B
                    0,                                                // beta
                    thrust::raw_pointer_cast(d_input_buffer.data())   // C
                    );
    CUDA_POST_KERNEL_CHECK;

    // Accumulate gradients back to the input grad feat
    // Put it back to the correct index
    num_kernels = in_nchannel * n_active_in_volume;
    add_mapped_output_tr<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            thrust::raw_pointer_cast(d_input_buffer.data()), // In
            n_active_in_volume,                              // In channel
            d_grad_in_feat, in_nchannel,                     // Out
            thrust::raw_pointer_cast(d_in_map.data()));      // Out channel
    CUDA_POST_KERNEL_CHECK;

    // Compute gradient for kernel
    // Copy features to the buffer
    copy_mapped_input<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, in_nchannel, d_in_feat,
            thrust::raw_pointer_cast(d_input_buffer.data()),
            thrust::raw_pointer_cast(d_in_map.data()));
    CUDA_POST_KERNEL_CHECK;

    gpu_gemm<Dtype>(cuhandle, CblasTrans, CblasNoTrans,
                    in_nchannel,                                      // M
                    out_nchannel,                                     // N
                    n_active_in_volume,                               // K
                    1,                                                // alpha
                    thrust::raw_pointer_cast(d_input_buffer.data()),  // A
                    thrust::raw_pointer_cast(d_output_buffer.data()), // B
                    1,                                                // beta
                    &d_grad_kernel[k * in_nchannel * out_nchannel]    // C
                    );
    CUDA_POST_KERNEL_CHECK;
  }
}

template void SparseConvolutionBackwardGPU<float>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, const float *d_kernel,
    float *p_grad_kernel, const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

#endif
