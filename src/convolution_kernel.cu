/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef GPU_CONVOLUTION
#define GPU_CONVOLUTION

#include <iostream>

// Use the torch for GPU memory management. Thrust resize gives segfulat during
// debugging -g #include <torch/extension.h>

#include "allocators.cuh"
#include "convolution_kernel.cuh"
#include "math_functions.cuh"

#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>

namespace minkowski {

namespace detail {

bool check_direct_gemm_forward(MinkowskiAlgorithm::Mode const algo_index, //
                               ConvolutionMode::Type const &convolution_mode,
                               long const sA, long const sB, long const N) {
  if ((convolution_mode == ConvolutionMode::DIRECT_GEMM) ||
      (algo_index == MinkowskiAlgorithm::MEMORY_EFFICIENT))
    return true;
  if (convolution_mode == ConvolutionMode::COPY_GEMM)
    return false;
  if (sA == 32 && sB == 64 and N <= 490537) return true;
  if (sB <= 40) {
    if (sB <= 20) {
      return true;
    } else {
      if (N <= 295625) {
        return true;
      } else {
        return (sA <= 12);
      }
    }
  } else {
    if (sA <= 20)
      return true;
    else {
      if (N <= 74556) {
        return (sB <= 112);
      } else {
        return false;
      }
    }
  }
}

bool check_direct_gemm_backward(MinkowskiAlgorithm::Mode const algo_index, //
                                ConvolutionMode::Type const &convolution_mode,
                                long const sA, long const sB, long const N) {
  if ((convolution_mode == ConvolutionMode::DIRECT_GEMM) ||
      (algo_index == MinkowskiAlgorithm::MEMORY_EFFICIENT))
    return true;
  if (convolution_mode == ConvolutionMode::COPY_GEMM)
    return false;
  if (sA == 32 && sB == 64 and N <= 490537) return true;
  if (sB <= 40) {
    if (sA <= 20)
      return true;
    else {
      if (N <= 490540) {
        return true;
      } else {
        return (sA <= 12);
      }
    }
  } else {
    if (sA <= 20) {
      return true;
    } else {
      if (N <= 30612) {
        return (sB <= 160);
      } else {
        return false;
      }
    }
  }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul(const Dtype *__restrict__ A, const int wA, const int hA, //
       const Dtype *__restrict__ B, const int wB, const int hB, //
       Dtype *__restrict__ C,                                   //
       const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < wB)
    atomicAdd(&C[wB * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
 * wA is A's width and wB is B's width
 *
 *                +---+
 *                |B^T|
 *            +-------+
 *            |   |   |
 *            | A | C |
 *            |   |   |
 *            |   |   |
 * +------------------+
 * |    D^T   | E |
 * +----------+---+
 *
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul2(const Dtype *__restrict__ A, const int wA, const int hA, //
        const Dtype *__restrict__ B, const int wB, const int hB, //
        const Dtype *__restrict__ D, const int wD, const int hD, //
        Dtype *__restrict__ C, Dtype *__restrict__ E,
        const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. y is for rows, x is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;
  Dtype Esub = 0;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ Dtype BTs[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Ds used to
  // store the sub-matrix of D
  __shared__ Dtype DTs[BLOCK_SIZE][BLOCK_SIZE];

  // For Ds = D^T[...:..., ...:...], use the transposed grid dimension for A
  DTs[ty][tx] = (x < wD && y < hD) ? D[wD * in_row + x] : 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * out_row + s + tx] : 0;

    // Transposed kernel
    BTs[ty][tx] = ((s + ty) < wB && x < hB) ? B[wB * x + s + ty] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * BTs[k][tx];
    }

    // For Esub, reset to 0
    Esub = 0;
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Esub += DTs[k][ty] * As[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();

    // For the E matrix which requires accmulation of multiple blocks, use
    // atomic addition. This can be replaced with a more sophisticaed
    // reduction algorithm.
    if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
      atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < hB)
    atomicAdd(&C[hB * in_row + x], Csub);
}

template <typename Dtype, typename Itype>
__global__ void
add_mapped_output_tr(const size_t n, const Dtype *__restrict__ in_feat,
                     const size_t in_nchannel, Dtype *__restrict__ out_feat,
                     const size_t out_nchannel, const Itype *__restrict__ map) {
  extern __shared__ Itype map_index[];
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. y is for rows, x is for columns.
  const int x = blockDim.x * bx + tx;
  const int y = blockDim.y * by + ty;

  if (x < n && ty == 0)
    map_index[tx] = map[x];

  __syncthreads();

  if (x < n && y < out_nchannel) {
    atomicAdd(&out_feat[map_index[tx] * out_nchannel + y],
              in_feat[y * in_nchannel + x]);
  }
}

} // namespace detail

template <typename Dtype, typename Itype, typename ByteAllocator>
void ConvolutionForwardKernelGPU(
    Dtype const *d_in_feat,                      //
    default_types::size_type const in_nchannel,  //
    Dtype *d_out_feat,                           //
    default_types::size_type const out_nchannel, //
    Dtype *d_kernel, gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    default_types::size_type const in_nrows,      //
    default_types::size_type const out_nrows,     //
    ByteAllocator &allocator,                     //
    MinkowskiAlgorithm::Mode const algo_index,    //
    ConvolutionMode::Type const convolution_mode, //
    cublasHandle_t cuhandle, cudaStream_t stream) {

  size_t n_active_in_volume, shared_mem_size = -1;

  if (detail::check_direct_gemm_forward(algo_index, convolution_mode,
                                        in_nchannel, out_nchannel, in_nrows)) {
    // Define the shared memory size
    if ((in_nchannel > 16 && out_nchannel > 16 &&
         in_nchannel * out_nchannel >= 512) ||
        (in_nchannel > 24 && out_nchannel > 24))
      shared_mem_size = 32;
    else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
      shared_mem_size = 24;
    else if ((in_nchannel > 8 && out_nchannel > 8) ||
             (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
      shared_mem_size = 16;
    else
      shared_mem_size = 8;

    dim3 threads(shared_mem_size, shared_mem_size);

    // Iterate through each spatial kernel and get indices for in_map and
    // out_map
    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
      auto const k = it->first;
      n_active_in_volume = kernel_map.size(k);
      if (n_active_in_volume == 0)
        continue;

      size_t const num_grid =
          (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
      size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
      size_t const step = (n_active_in_volume + num_div - 1) / num_div;

      for (size_t s = 0; s < num_div; s++) {
        size_t const offset = step * s;
        size_t const remainder = n_active_in_volume - offset;
        size_t const curr_num_active = remainder < step ? remainder : step;
        dim3 const grid((out_nchannel + threads.x - 1) / threads.x,
                        (curr_num_active + threads.y - 1) / threads.y);

        // copy in out map
#ifdef DEBUG
        /*
        size_t map_size = curr_num_active;
        Itype *p_kernel_map = (Itype *)std::malloc(map_size * 3 *
        sizeof(Itype)); CUDA_CHECK(cudaMemcpy(p_kernel_map,
        kernel_map.kernels.begin(k), map_size * sizeof(Itype),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(p_kernel_map + 1 * map_size,
                              kernel_map.in_maps.begin(k),
                              map_size * sizeof(Itype),
        cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(p_kernel_map + 2 *
        map_size, kernel_map.out_maps.begin(k), map_size * sizeof(Itype),
        cudaMemcpyDeviceToHost));

        for (size_t i = curr_num_active - 20; i < curr_num_active; ++i) {
          std::cout << p_kernel_map[i + 0 * map_size] << ":"
                    << p_kernel_map[i + 1 * map_size] << "->"
                    << p_kernel_map[i + 2 * map_size] << "\n";
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        std::free(p_kernel_map);
        */
#endif

        switch (shared_mem_size) {
        case 32:
          detail::matmul<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
              d_in_feat, in_nchannel, curr_num_active,
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 24:
          detail::matmul<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
              d_in_feat, in_nchannel, curr_num_active,
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 16:
          detail::matmul<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
              d_in_feat, in_nchannel, curr_num_active,
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 8:
          detail::matmul<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
              d_in_feat, in_nchannel, curr_num_active,
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel, d_out_feat, kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        }
      }
#ifdef DEBUG
      LOG_DEBUG("k:", k, "num:", n_active_in_volume);
      CUDA_CHECK(cudaDeviceSynchronize());
#endif
      CUDA_CHECK(cudaGetLastError());
    }
  } else { // copy gemm
    Itype const max_numel = kernel_map.max_size();
    LOG_DEBUG("max_numel:", max_numel);
    Dtype *mapped_in_feat = reinterpret_cast<Dtype *>(
        allocator.allocate(max_numel * in_nchannel * sizeof(Dtype)));
    Dtype *mapped_out_feat = reinterpret_cast<Dtype *>(
        allocator.allocate(max_numel * out_nchannel * sizeof(Dtype)));

    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
      auto const k = it->first;
      n_active_in_volume = kernel_map.size(k);
      if (n_active_in_volume == 0)
        continue;

      LOG_DEBUG(n_active_in_volume * in_nchannel, in_nchannel);
      detail::shared_copy_kernel_map<Dtype, Itype>(
          // mapped_in_feat,
          mapped_in_feat, d_in_feat, kernel_map.in_maps.begin(k),
          n_active_in_volume * in_nchannel, in_nchannel);

#ifdef DEBUG
      /*
      size_t map_size = std::min((size_t)n_active_in_volume, (size_t)100);
      Dtype *p_tmp =
          (Dtype *)std::malloc(map_size * in_nchannel * sizeof(Dtype));
      CUDA_CHECK(cudaMemcpy(p_tmp, mapped_in_feat,
                            map_size * in_nchannel * sizeof(Dtype),
                            cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < map_size; ++i) {
        std::cout << PtrToString(&p_tmp[i * in_nchannel], in_nchannel) << "\n";
      }

      CUDA_CHECK(cudaDeviceSynchronize());
      std::free(p_tmp);
      */
#endif

      gpu_gemm<Dtype>(cuhandle, CblasNoTrans, CblasNoTrans,
                      n_active_in_volume,                        // M
                      out_nchannel,                              // N
                      in_nchannel,                               // K
                      1,                                         // alpha
                      mapped_in_feat,                            // A
                      &d_kernel[k * in_nchannel * out_nchannel], // B
                      0,                                         // beta
                      mapped_out_feat                            // C
      );

      detail::shared_accumulate_kernel_map<Dtype, Itype>(
          d_out_feat, mapped_out_feat, kernel_map.out_maps.begin(k),
          n_active_in_volume * out_nchannel, out_nchannel);
    }

    allocator.deallocate((char *)mapped_in_feat,
                         max_numel * in_nchannel * sizeof(Dtype));
    allocator.deallocate((char *)mapped_out_feat,
                         max_numel * out_nchannel * sizeof(Dtype));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// default_allocator
template void
ConvolutionForwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    float const *d_in_feat, default_types::size_type const in_nchannel,
    float *d_out_feat, default_types::size_type const out_nchannel,
    float *d_kernel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
ConvolutionForwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    double const *d_in_feat, default_types::size_type const in_nchannel,
    double *d_out_feat, default_types::size_type const out_nchannel,
    double *d_kernel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// c10_allocator
template void
ConvolutionForwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    float const *d_in_feat, default_types::size_type const in_nchannel,
    float *d_out_feat, default_types::size_type const out_nchannel,
    float *d_kernel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::c10_allocator<char> &allocator,    //
    MinkowskiAlgorithm::Mode const algo_index, //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
ConvolutionForwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    double const *d_in_feat, default_types::size_type const in_nchannel,
    double *d_out_feat, default_types::size_type const out_nchannel,
    double *d_kernel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    default_types::size_type const in_nrows, //
    default_types::size_type const out_nrows,
    detail::c10_allocator<char> &allocator,    //
    MinkowskiAlgorithm::Mode const algo_index, //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// Backward
template <typename Dtype, typename Itype, typename ByteAllocator>
void ConvolutionBackwardKernelGPU(
    Dtype const *d_in_feat,                                            //
    Dtype *d_grad_in_feat, default_types::size_type const in_nchannel, //
    Dtype const *d_grad_out_feat,
    default_types::size_type const out_nchannel,            //
    Dtype const *d_kernel, Dtype *d_grad_kernel,            //
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map, //
    default_types::size_type const in_nrows,                //
    default_types::size_type const out_nrows,               //
    ByteAllocator &allocator,                               //
    MinkowskiAlgorithm::Mode const algo_index,              //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream) {

#ifdef DEBUG
  CUDA_CHECK_ARGS(cudaDeviceSynchronize(),
                  "Error triggered from a previous kernel call.");
#endif

  size_t n_active_in_volume, shared_mem_size = -1;
  // Define the shared memory size
  if ((in_nchannel > 16 && out_nchannel > 16 &&
       in_nchannel * out_nchannel >= 512) ||
      (in_nchannel % 32 == 0 && out_nchannel % 32 == 0))
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if ((in_nchannel > 8 && out_nchannel > 8) ||
           (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
    shared_mem_size = 16;
  else
    shared_mem_size = 8;

  if (!detail::check_direct_gemm_backward(
          algo_index, convolution_mode, in_nchannel, out_nchannel, in_nrows)) {
    // find max size
    size_t max_active = kernel_map.max_size();
    size_t in_buffer_size = max_active * in_nchannel * sizeof(Dtype);
    size_t out_buffer_size = max_active * out_nchannel * sizeof(Dtype);
    Dtype *d_input_buffer = (Dtype *)allocator.allocate(in_buffer_size);
    Dtype *d_output_buffer = (Dtype *)allocator.allocate(out_buffer_size);

    dim3 threads(32, shared_mem_size);
#ifdef DEBUG
    timer t;
#endif
    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
      auto const k = it->first;
      n_active_in_volume = kernel_map.size(k);
      if (n_active_in_volume == 0)
        continue;

      // Copy (*p_in_maps)[k] to GPU
      Itype const *d_in_map = kernel_map.in_maps.begin(k);
      Itype const *d_out_map = kernel_map.out_maps.begin(k);
      // Copy gradients to the buffer
#ifdef DEBUG
      t.tic();
#endif
      detail::shared_copy_kernel_map<Dtype, Itype>(
          d_output_buffer, d_grad_out_feat, d_out_map,
          n_active_in_volume * out_nchannel, out_nchannel);
#ifdef DEBUG
      CUDA_CHECK(cudaStreamSynchronize(stream));
      LOG_DEBUG("copy input", t.toc());
      t.tic();
#endif
      gpu_gemm<Dtype>(cuhandle, CblasNoTrans, CblasTrans,
                      in_nchannel,                               // M
                      n_active_in_volume,                        // N
                      out_nchannel,                              // K
                      1,                                         // alpha
                      &d_kernel[k * in_nchannel * out_nchannel], // A
                      d_output_buffer,                           // B
                      0,                                         // beta
                      d_input_buffer                             // C
      );
#ifdef DEBUG
      CUDA_CHECK(cudaStreamSynchronize(0));
      LOG_DEBUG("input grad gemm", t.toc());
      t.tic();
#endif

      // Accumulate gradients back to the input grad feat
      // Put it back to the correct index
      dim3 const grid_tr(GET_BLOCKS(n_active_in_volume, threads.x),
                         GET_BLOCKS(in_nchannel, threads.y));
      detail::add_mapped_output_tr<Dtype, Itype>
          <<<grid_tr, threads, threads.x * sizeof(Itype), stream>>>(
              n_active_in_volume,
              d_input_buffer,              // In
              n_active_in_volume,          // In channel
              d_grad_in_feat, in_nchannel, // Out
              d_in_map);                   // Out channel
#ifdef DEBUG
      CUDA_CHECK(cudaStreamSynchronize(stream));
      LOG_DEBUG("accumulate in grad", t.toc());
      t.tic();
#endif

      // Compute gradient for kernel
      // Copy features to the buffer
      dim3 const grid_in(GET_BLOCKS(n_active_in_volume, threads.x),
                         GET_BLOCKS(in_nchannel, threads.y));
      detail::shared_copy_kernel_map<Dtype, Itype>(
          d_input_buffer, d_in_feat, d_in_map, n_active_in_volume * in_nchannel,
          in_nchannel);
#ifdef DEBUG
      LOG_DEBUG("copy in feat to buffer", t.toc());
      t.tic();
#endif
      // sync before the blas call
      CUDA_CHECK(cudaStreamSynchronize(stream));
      gpu_gemm<Dtype>(cuhandle, CblasTrans, CblasNoTrans,
                      in_nchannel,                                   // M
                      out_nchannel,                                  // N
                      n_active_in_volume,                            // K
                      1,                                             // alpha
                      d_input_buffer,                                // A
                      d_output_buffer,                               // B
                      1,                                             // beta
                      &d_grad_kernel[k * in_nchannel * out_nchannel] // C
      );
      CUDA_CHECK(cudaStreamSynchronize(0));
#ifdef DEBUG
      LOG_DEBUG("grad kernel gemm", t.toc());
      t.tic();
#endif
    }
    allocator.deallocate((char *)d_input_buffer, in_buffer_size);
    allocator.deallocate((char *)d_output_buffer, out_buffer_size);
  } else {
    dim3 threads(shared_mem_size, shared_mem_size);

    for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend(); ++it) {
      auto const k = it->first;
      n_active_in_volume = kernel_map.size(k);
      if (n_active_in_volume == 0)
        continue;

      size_t const num_grid =
          (n_active_in_volume + shared_mem_size - 1) / shared_mem_size;
      size_t const num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
      size_t const step = (n_active_in_volume + num_div - 1) / num_div;

      for (int s = 0; s < num_div; s++) {
        size_t const offset = step * s;
        size_t const remainder = n_active_in_volume - offset;
        size_t const curr_num_active = remainder < step ? remainder : step;
        dim3 const grid((in_nchannel + threads.x - 1) / threads.x,
                        (curr_num_active + threads.y - 1) / threads.y);

        switch (shared_mem_size) {
        case 32:
          detail::matmul2<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 24:
          detail::matmul2<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 16:
          detail::matmul2<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        case 8:
          detail::matmul2<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
              d_grad_out_feat, out_nchannel, curr_num_active, // A
              &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
              in_nchannel,                                    // B
              d_in_feat, in_nchannel, curr_num_active,        // D
              d_grad_in_feat,                                 // C
              &d_grad_kernel[k * in_nchannel * out_nchannel], // E
              kernel_map.in_maps.begin(k) + offset,
              kernel_map.out_maps.begin(k) + offset);
          break;
        }
      }
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

// default_allocator
template void
ConvolutionBackwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    float const *d_in_feat, float *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    float const *d_grad_out_feat,
    default_types::size_type const out_nchannel, //
    float const *d_kernel, float *p_grad_kernel, //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map,                            //
    default_types::size_type const in_nrows,    //
    default_types::size_type const out_nrows,   //
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
ConvolutionBackwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    double const *d_in_feat, double *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    double const *d_grad_out_feat,
    default_types::size_type const out_nchannel,   //
    double const *d_kernel, double *p_grad_kernel, //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map,                            //
    default_types::size_type const in_nrows,    //
    default_types::size_type const out_nrows,   //
    detail::default_allocator<char> &allocator, //
    MinkowskiAlgorithm::Mode const algo_index,  //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

// c10_allocator
template void
ConvolutionBackwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    float const *d_in_feat, float *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    float const *d_grad_out_feat,
    default_types::size_type const out_nchannel,                             //
    float const *d_kernel, float *p_grad_kernel,                             //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    default_types::size_type const in_nrows,                                 //
    default_types::size_type const out_nrows,                                //
    detail::c10_allocator<char> &allocator,                                  //
    MinkowskiAlgorithm::Mode const algo_index,                               //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

template void
ConvolutionBackwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    double const *d_in_feat, double *d_grad_in_feat,
    default_types::size_type const in_nchannel, //
    double const *d_grad_out_feat,
    default_types::size_type const out_nchannel,                             //
    double const *d_kernel, double *p_grad_kernel,                           //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    default_types::size_type const in_nrows,                                 //
    default_types::size_type const out_nrows,                                //
    detail::c10_allocator<char> &allocator,                                  //
    MinkowskiAlgorithm::Mode const algo_index,                               //
    ConvolutionMode::Type const convolution_mode, cublasHandle_t cuhandle,
    cudaStream_t stream);

} // namespace minkowski

#endif // end GPU_CONVOLUTION
