#ifndef GPU_CONVOLUTION
#define GPU_CONVOLUTION

#include <iostream>

// Use the torch for GPU memory management. Thrust resize gives segfulat during
// debugging -g #include <torch/extension.h>

#include "convolution.cuh"

// Given a row-major matrix, use the mapping to extract a row major order matrix
template <typename Dtype, typename Itype>
__global__ void copy_mapped_input(const int n, const int nchannel,
                                  const Dtype *in_feat, Dtype *out_feat,
                                  const Itype *map) {
  CUDA_KERNEL_LOOP(index, n) {
    const int row = index / nchannel;
    const int col = index % nchannel;
    out_feat[index] = in_feat[map[row] * nchannel + col];
  }
}

template <typename Dtype, typename Itype>
__global__ void add_mapped_output_tr(const int n, const Dtype *in_feat,
                                     const int in_nchannel, Dtype *out_feat,
                                     const int out_nchannel, const Itype *map) {
  CUDA_KERNEL_LOOP(index, n) {
    const int row = index % in_nchannel;
    const int col = index / in_nchannel;
    atomicAdd(&out_feat[map[row] * out_nchannel + col], in_feat[index]);
  }
}

// Given each output, get an input feature for each corresponding kernel weight
// and add the output in place
template <typename Dtype, typename Itype>
__global__ void inplace_convolution(const int n, const Dtype *in_feat,
                                    const int in_nchannel, Dtype *out_feat,
                                    const int out_nchannel, const Dtype *kernel,
                                    const Itype *in_map, const Itype *out_map) {
  // n = out_nchannel * out_nrows
  // The kernel computes one output scalar for each output index and each output
  // channel.
  CUDA_KERNEL_LOOP(index, n) {
    const int out_ch = index % out_nchannel;
    const int out_row = index / out_nchannel;
    // Pytorch tensors in C-ordering with in_nchannels x out_nchannels
    Dtype tmp = 0.0;
    for (int in_ch = 0; in_ch < in_nchannel; in_ch++) {
      tmp += kernel[in_ch * out_nchannel + out_ch] *
             in_feat[in_map[out_row] * in_nchannel + in_ch];
    }
    // Done independently, no need for atomicAdd
    out_feat[out_map[out_row] * out_nchannel + out_ch] += tmp;
  }
}

template <typename Dtype, typename Itype>
void ConvolutionForwardKernelGPU(
    const Dtype *d_in_feat, int in_nchannel, Dtype *d_out_feat,
    int out_nchannel, const Dtype *d_kernel,
    const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  // For the in out buffer, use the pre allocated GPU memory space as thrust
  // resize gives segfault. Also initializing it with torch allows us to
  // allocate memory faster and efficiently.
  int kernel_volume, n_active_in_volume, num_kernels;
  Itype *d_in_map, *d_out_map;
  // Copy the in_map, out_map to GPU
  kernel_volume = in_maps.size();

  // Find the max_n_active fot memory allocation
  int max_n_active = -1;
  for (int k = 0; k < kernel_volume; k++)
    if (max_n_active < (int)(in_maps[k].size()))
      max_n_active = (int)(in_maps[k].size());

  // Create a large chunk of memory
  CUDA_CHECK(
      cudaMalloc((void **)&d_in_map, (2 * max_n_active) * sizeof(Itype)));
  d_out_map = d_in_map + max_n_active;

  // Iterate through each spatial kernel and get indices for in_map and out_map
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_maps[k].size();
    if (n_active_in_volume == 0)
      continue;

    num_kernels = out_nchannel * n_active_in_volume;

    // Copy (*p_in_maps)[k] to GPU
    CUDA_CHECK(cudaMemcpy(d_in_map, in_maps[k].data(),
                          sizeof(Itype) * n_active_in_volume,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_map, out_maps[k].data(),
                          sizeof(Itype) * n_active_in_volume,
                          cudaMemcpyHostToDevice));

    inplace_convolution<Dtype, Itype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, d_in_feat, in_nchannel, d_out_feat, out_nchannel,
            &d_kernel[k * in_nchannel * out_nchannel], d_in_map, d_out_map);
  }

  cudaFree(d_in_map);
}

template void ConvolutionForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

template void ConvolutionForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, int in_nchannel, double *d_out_feat,
    int out_nchannel, const double *d_kernel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void ConvolutionBackwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nchannel,
    const Dtype *d_grad_out_feat, int out_nchannel, const Dtype *d_kernel,
    Dtype *d_grad_kernel, const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream) {
  int kernel_volume, n_active_in_volume, num_kernels;
  Itype *d_in_map, *d_out_map;
  Dtype *d_in_buffer, *d_out_buffer;

  kernel_volume = in_maps.size();
  // Find the max_n_active fot memory allocation
  int max_n_active = -1;
  for (int k = 0; k < kernel_volume; k++)
    if (max_n_active < (int)(in_maps[k].size()))
      max_n_active = (int)(in_maps[k].size());

  CUDA_CHECK(cudaMalloc((void **)&d_in_map, 2 * max_n_active * sizeof(Itype)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_in_buffer,
                 (in_nchannel + out_nchannel) * max_n_active * sizeof(Dtype)));
  d_out_map = d_in_map + max_n_active;
  d_out_buffer = d_in_buffer + in_nchannel * max_n_active;
  // CUDA_CHECK(cudaMalloc((void **)&d_in_map, max_n_active * sizeof(Itype)));
  // CUDA_CHECK(cudaMalloc((void **)&d_out_map, max_n_active * sizeof(Itype)));
  // CUDA_CHECK(cudaMalloc((void **)&d_in_buffer,
  //                       in_nchannel * max_n_active * sizeof(Dtype)));
  // CUDA_CHECK(cudaMalloc((void **)&d_out_buffer,
  //                       out_nchannel * max_n_active * sizeof(Dtype)));

  // Copy the in_map, out_map to GPU
  // First im2col, gather all indices of in2out
  // Iterate through each spatial kernel and get indices for in_map and
  // out_map
  for (int k = 0; k < kernel_volume; k++) {
    n_active_in_volume = in_maps[k].size();
    if (n_active_in_volume == 0)
      continue;

    // Copy (*p_in_maps)[k] to GPU
    CUDA_CHECK(cudaMemcpy(d_in_map, in_maps[k].data(),
                          sizeof(Itype) * n_active_in_volume,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_map, out_maps[k].data(),
                          sizeof(Itype) * n_active_in_volume,
                          cudaMemcpyHostToDevice));

    // Copy (*p_in_maps)[k] to GPU
    num_kernels = out_nchannel * n_active_in_volume;

    // Copy gradients to the buffer
    copy_mapped_input<Dtype, Itype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, out_nchannel, d_grad_out_feat, d_out_buffer,
            d_out_map);

    gpu_gemm<Dtype>(cuhandle, CblasNoTrans, CblasTrans,
                    in_nchannel,                               // M
                    n_active_in_volume,                        // N
                    out_nchannel,                              // K
                    (Dtype)1.,                                 // alpha
                    &d_kernel[k * in_nchannel * out_nchannel], // A
                    d_out_buffer,                              // B
                    (Dtype)0.,                                 // beta
                    d_in_buffer);                              // C

    // Accumulate gradients back to the input grad feat
    // Put it back to the correct index
    num_kernels = in_nchannel * n_active_in_volume;
    add_mapped_output_tr<Dtype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            d_in_buffer,                 // In
            n_active_in_volume,          // In channel
            d_grad_in_feat, in_nchannel, // Out
            d_in_map);                   // Out channel

    // Compute gradient for kernel
    // Copy features to the buffer
    copy_mapped_input<Dtype, Itype>
        <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels, in_nchannel, d_in_feat, d_in_buffer, d_in_map);

    gpu_gemm<Dtype>(cuhandle, CblasTrans, CblasNoTrans,
                    in_nchannel,                                   // M
                    out_nchannel,                                  // N
                    n_active_in_volume,                            // K
                    1,                                             // alpha
                    d_in_buffer,                                   // A
                    d_out_buffer,                                  // B
                    1,                                             // beta
                    &d_grad_kernel[k * in_nchannel * out_nchannel] // C
    );
  }
  cudaFree(d_in_map);
  cudaFree(d_in_buffer);
}

template void ConvolutionBackwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, const float *d_kernel,
    float *p_grad_kernel, const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);

template void ConvolutionBackwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_grad_in_feat, int in_nchannel,
    const double *d_grad_out_feat, int out_nchannel, const double *d_kernel,
    double *p_grad_kernel, const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    cublasHandle_t cuhandle, cudaStream_t stream);
#endif
