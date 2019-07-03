/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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

#include "convolution.cuh"
#include "gpu_memory_manager.hpp"

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
    const Dtype *curr_kernel = kernel + out_ch;
    const Dtype *curr_in_feat = in_feat + in_map[out_row] * in_nchannel;
    for (int in_ch = 0; in_ch < in_nchannel; in_ch++) {
      tmp += (*curr_kernel) * (*curr_in_feat);
      curr_kernel += out_nchannel;
      curr_in_feat += 1;
    }
    // Done independently, no need for atomicAdd
    out_feat[out_map[out_row] * out_nchannel + out_ch] += tmp;
  }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void matmul(const Dtype *A, const int wA, const int hA,
                       const Dtype *B, const int wB, const int hB, Dtype *C,
                       const Itype *in_map, const Itype *out_map) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.y;
  const int by = blockIdx.x;

  // Thread index
  const int tx = threadIdx.y;
  const int ty = threadIdx.x;

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
    C[wB * out_row + x] += Csub;
  // TODO: atomicAdd(&C[wB * out_row + x], Csub); // For conv transpose, it
  // might fail due to overlapping outputs
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
__global__ void matmul2(const Dtype *A, const int wA, const int hA,
                        const Dtype *B, const int wB, const int hB,
                        const Dtype *D, const int wD, const int hD, Dtype *C,
                        Dtype *E, const Itype *in_map, const Itype *out_map) {
  // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

  // Block index
  const int bx = blockIdx.y;
  const int by = blockIdx.x;

  // Thread index
  const int tx = threadIdx.y;
  const int ty = threadIdx.x;

  // Coordinate. x is for rows, y is for columns.
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
    // atomic addition. This can be replaced with a more sophisticaed reduction
    // algorithm.
    if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
      atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < hB)
    atomicAdd(&C[hB * in_row + x], Csub);
}

template <typename Dtype, typename Itype>
void ConvolutionForwardKernelGPU(
    const Dtype *d_in_feat, int in_nchannel, Dtype *d_out_feat,
    int out_nchannel, const Dtype *d_kernel,
    const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, int out_nrows,
    Itype *d_scr, cublasHandle_t cuhandle, cudaStream_t stream) {
  // For the in out buffer, use the pre allocated GPU memory space as thrust
  // resize gives segfault. Also initializing it with torch allows us to
  // allocate memory faster and efficiently.
  int kernel_volume, n_active_in_volume, num_kernels, shared_mem_size = -1;
  Itype *d_in_map, *d_out_map;
  // Copy the in_map, out_map to GPU
  kernel_volume = in_maps.size();

  // Find the max_n_active for memory allocation
  int max_n_active = -1;
  for (int k = 0; k < kernel_volume; k++)
    if (max_n_active < (int)(in_maps[k].size()))
      max_n_active = (int)(in_maps[k].size());

  d_in_map = d_scr;
  d_out_map = d_in_map + max_n_active;

  // Define the shared memory size
  if (in_nchannel % 32 == 0 && out_nchannel % 32 == 0)
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if (in_nchannel % 16 == 0 && out_nchannel % 16 == 0)
    shared_mem_size = 16;
  else if (in_nchannel % 8 == 0 && out_nchannel % 8 == 0)
    shared_mem_size = 8;
  else
    shared_mem_size = 4;

  dim3 threads(shared_mem_size, shared_mem_size);

  // Iterate through each spatial kernel and get indices for in_map and out_map
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

    dim3 grid((n_active_in_volume + threads.y - 1) / threads.y,
              (out_nchannel + threads.x - 1) / threads.x);
    switch (shared_mem_size) {
    case 32:
      matmul<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
          d_in_feat, in_nchannel, n_active_in_volume,
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel, in_nchannel,
          d_out_feat, d_in_map, d_out_map);
      break;
    case 24:
      matmul<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
          d_in_feat, in_nchannel, n_active_in_volume,
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel, in_nchannel,
          d_out_feat, d_in_map, d_out_map);
      break;
    case 16:
      matmul<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
          d_in_feat, in_nchannel, n_active_in_volume,
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel, in_nchannel,
          d_out_feat, d_in_map, d_out_map);
      break;
    case 8:
      matmul<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
          d_in_feat, in_nchannel, n_active_in_volume,
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel, in_nchannel,
          d_out_feat, d_in_map, d_out_map);
      break;
    default:
      num_kernels = out_nchannel * n_active_in_volume;
      inplace_convolution<Dtype, Itype>
          <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
              num_kernels, d_in_feat, in_nchannel, d_out_feat, out_nchannel,
              &d_kernel[k * in_nchannel * out_nchannel], d_in_map, d_out_map);
      break;
    }
  }
}

template void ConvolutionForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    int32_t *d_scr, cublasHandle_t cuhandle, cudaStream_t stream);

template void ConvolutionForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, int in_nchannel, double *d_out_feat,
    int out_nchannel, const double *d_kernel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    int32_t *d_scr, cublasHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void ConvolutionBackwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nchannel,
    const Dtype *d_grad_out_feat, int out_nchannel, const Dtype *d_kernel,
    Dtype *d_grad_kernel, const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, int out_nrows,
    Itype *d_scr, cublasHandle_t cuhandle, cudaStream_t stream) {
  int kernel_volume, n_active_in_volume, shared_mem_size = -1;
  Itype *d_in_map, *d_out_map;

  kernel_volume = in_maps.size();
  // Find the max_n_active fot memory allocation
  int max_n_active = -1;
  for (int k = 0; k < kernel_volume; k++)
    if (max_n_active < (int)(in_maps[k].size()))
      max_n_active = (int)(in_maps[k].size());

  d_in_map = d_scr;
  d_out_map = d_in_map + max_n_active;

  // Define the shared memory size
  if (in_nchannel % 32 == 0 && out_nchannel % 32 == 0)
    shared_mem_size = 32;
  else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    shared_mem_size = 24;
  else if (in_nchannel % 16 == 0 && out_nchannel % 16 == 0)
    shared_mem_size = 16;
  else if (in_nchannel % 8 == 0 && out_nchannel % 8 == 0)
    shared_mem_size = 8;
  else
    shared_mem_size = 4;

  dim3 threads(shared_mem_size, shared_mem_size);

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

    dim3 grid((n_active_in_volume + threads.y - 1) / threads.y,
              (in_nchannel + threads.x - 1) / threads.x);

    switch (shared_mem_size) {
    case 32:
      matmul2<Dtype, Itype, 32><<<grid, threads, 0, stream>>>(
          d_grad_out_feat, out_nchannel, n_active_in_volume, // A
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
          in_nchannel,                                    // B
          d_in_feat, in_nchannel, n_active_in_volume,     // D
          d_grad_in_feat,                                 // C
          &d_grad_kernel[k * in_nchannel * out_nchannel], // E
          d_in_map, d_out_map);
      break;
    case 24:
      matmul2<Dtype, Itype, 24><<<grid, threads, 0, stream>>>(
          d_grad_out_feat, out_nchannel, n_active_in_volume, // A
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
          in_nchannel,                                    // B
          d_in_feat, in_nchannel, n_active_in_volume,     // D
          d_grad_in_feat,                                 // C
          &d_grad_kernel[k * in_nchannel * out_nchannel], // E
          d_in_map, d_out_map);
      break;
    case 16:
      matmul2<Dtype, Itype, 16><<<grid, threads, 0, stream>>>(
          d_grad_out_feat, out_nchannel, n_active_in_volume, // A
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
          in_nchannel,                                    // B
          d_in_feat, in_nchannel, n_active_in_volume,     // D
          d_grad_in_feat,                                 // C
          &d_grad_kernel[k * in_nchannel * out_nchannel], // E
          d_in_map, d_out_map);
      break;
    case 8:
      matmul2<Dtype, Itype, 8><<<grid, threads, 0, stream>>>(
          d_grad_out_feat, out_nchannel, n_active_in_volume, // A
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
          in_nchannel,                                    // B
          d_in_feat, in_nchannel, n_active_in_volume,     // D
          d_grad_in_feat,                                 // C
          &d_grad_kernel[k * in_nchannel * out_nchannel], // E
          d_in_map, d_out_map);
      break;
    default:
      matmul2<Dtype, Itype, 4><<<grid, threads, 0, stream>>>(
          d_grad_out_feat, out_nchannel, n_active_in_volume, // A
          &d_kernel[k * in_nchannel * out_nchannel], out_nchannel,
          in_nchannel,                                    // B
          d_in_feat, in_nchannel, n_active_in_volume,     // D
          d_grad_in_feat,                                 // C
          &d_grad_kernel[k * in_nchannel * out_nchannel], // E
          d_in_map, d_out_map);
      break;
    }
  }
}

template void ConvolutionBackwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, const float *d_kernel,
    float *p_grad_kernel, const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    int32_t *d_scr, cublasHandle_t cuhandle, cudaStream_t stream);

template void ConvolutionBackwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_grad_in_feat, int in_nchannel,
    const double *d_grad_out_feat, int out_nchannel, const double *d_kernel,
    double *p_grad_kernel, const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, int out_nrows,
    int32_t *d_scr, cublasHandle_t cuhandle, cudaStream_t stream);
#endif
