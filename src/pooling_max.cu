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
#ifndef GPU_POOLING_MAX_KERNEL
#define GPU_POOLING_MAX_KERNEL

#include <limits>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "gpu.cuh"
#include "pooling_max.cuh"
#include "utils.hpp"

template <typename Dtype, typename Itype>
__global__ void set_gradient(const int n, const Dtype *d_grad_out,
                             Dtype *d_grad_in, const Itype *in_index) {
  CUDA_KERNEL_LOOP(index, n) { d_grad_in[in_index[index]] = d_grad_out[index]; }
}

template <typename Dtype, typename Itype>
__global__ void max_pool(const int N, const int out_nrows, const int nchannel,
                         const int nmap, const Dtype *d_in_feat,
                         Dtype *d_out_feat, Itype *d_max_index,
                         const Itype *d_in_map, const Itype *d_out_map,
                         const Itype *d_in_index_min) {
  // N == nmap * nchannel
  CUDA_KERNEL_LOOP(index, N) {
    int nrow = index / nchannel;
    int ch = index % nchannel;

    Itype out_map_row = d_out_map[nrow];
    Itype in_index = d_in_index_min[nrow];
    Itype num_in_feat;
    if (nrow == out_nrows - 1)
      num_in_feat = nmap - in_index;
    else
      num_in_feat = d_in_index_min[nrow + 1] - in_index;
    // It is guaranteed to have at least one input per output
    Itype curr_index, max_index = d_in_map[in_index] * nchannel + ch;
    Dtype curr_val, max_val = d_in_feat[max_index];
    for (int curr_iter = 0; curr_iter < num_in_feat; curr_iter++) {
      curr_index = d_in_map[in_index + curr_iter] * nchannel + ch;
      curr_val = d_in_feat[curr_index];
      if (max_val < curr_val) {
        max_val = curr_val;
        max_index = curr_index;
      }
    }
    Itype out_ind = out_map_row * nchannel + ch;
    d_out_feat[out_ind] = max_val;
    d_max_index[out_ind] = max_index;
  }
}

// Put features in to the out features according to the input index.
// The input index is sorted according to the out index so no need to take out
// index
template <typename Dtype, typename Itype>
__global__ void copy_sorted(const int n, const int nrows, const int nchannel,
                            const Dtype *in_feat, const Itype *in_index,
                            Dtype *out_feat) {
  int nrow, ch;
  CUDA_KERNEL_LOOP(index, n) {
    nrow = index / nchannel;
    ch = index % nchannel;
    out_feat[index] = in_feat[in_index[nrow] * nchannel + ch];
  }
}

template <typename Dtype, typename Itype>
void MaxPoolingForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index, int nchannel,
                                const pInOutMaps<Itype> &in_maps,
                                const pInOutMaps<Itype> &out_maps, Itype *d_scr,
                                cudaStream_t stream) {
  int nmap = 0;

  // Copy all maps to one vector
  for (const auto &map : in_maps)
    nmap += map.size();

  Itype *d_in_map = d_scr, *d_out_map = d_scr + nmap;

  CUDA_CHECK(cudaMemcpy(d_in_map,
                        in_maps[0].data(), // in_maps are contiguous of size nnz
                        nmap * sizeof(int), cudaMemcpyDeviceToDevice));

  CUDA_CHECK(
      cudaMemcpy(d_out_map,
                 out_maps[0].data(), // out_maps are contiguous of size nnz
                 nmap * sizeof(int), cudaMemcpyDeviceToDevice));

  // First, sort d_out_map and d_in_map with the d_out_map so that in_feat are
  // placed adjacent according to out_map
  thrust::sort_by_key(thrust::device, d_out_map, d_out_map + nmap, d_in_map);

  // Second, create number of in_feat per out, and starting index
  Itype *d_index, *d_in_map_min, *d_reduced_out_map;
  // CUDA_CHECK(cudaMalloc((void **)&d_index, 3 * nmap * sizeof(Itype)));
  d_index = d_scr + 2 * nmap;
  d_in_map_min = d_scr + 3 * nmap;
  d_reduced_out_map = d_scr + 4 * nmap;

  thrust::sequence(thrust::device, d_index, d_index + nmap);

  thrust::equal_to<Itype> equal_pred;
  thrust::minimum<Itype> min_op;

  auto reduction_pair =
      thrust::reduce_by_key(thrust::device,    // execution policy
                            d_out_map,         // key begin
                            d_out_map + nmap,  // key end
                            d_index,           // val begin
                            d_reduced_out_map, // key out begin
                            d_in_map_min,      // val out begin
                            equal_pred,        // binary pred
                            min_op);           // binary op

  size_t num_unique_out_map = reduction_pair.first - d_reduced_out_map;
  if (num_unique_out_map != out_nrows)
    throw std::invalid_argument(
        Formatter() << "Reduction size mismatch. out_nrows: " << out_nrows
                    << ", num_unique_out_map: " << num_unique_out_map);

  // Finally, use the max kernel to map all in_feats with the same out key to
  // out_feats Also, create out max_index for gradient
  max_pool<Dtype, Itype>
      <<<GET_BLOCKS(out_nrows * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
          nchannel * out_nrows, // N
          out_nrows, nchannel, nmap, d_in_feat, d_out_feat,
          d_max_index, // Out indices for backward
          d_in_map,    // in index
          d_reduced_out_map, d_in_map_min);

  // cudaFree(d_in_map);
  // cudaFree(d_index);
  CUDA_CHECK(cudaGetLastError());
}

template void MaxPoolingForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel, const pInOutMaps<int32_t> &in_map,
    const pInOutMaps<int32_t> &out_map, int32_t *d_scr, cudaStream_t stream);

template void MaxPoolingForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel, const pInOutMaps<int32_t> &in_map,
    const pInOutMaps<int32_t> &out_map, int32_t *d_scr, cudaStream_t stream);

template <typename Dtype, typename Itype>
void MaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, int in_nrows,
                                 const Dtype *d_grad_out_feat, int out_nrows,
                                 const Itype *d_max_index, int nchannel,
                                 cudaStream_t stream) {
  int num_kernels = out_nrows * nchannel;
  // Assume that gradients for input feature are all set to zero
  set_gradient<Dtype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, d_grad_out_feat, d_grad_in_feat, d_max_index);

  CUDA_CHECK(cudaGetLastError());
}

template void MaxPoolingBackwardKernelGPU<float, int32_t>(
    float *d_grad_in_feat, int in_nrows, const float *d_grad_out_feat,
    int out_nrows, const int32_t *d_max_index, int nchannel,
    cudaStream_t stream);

template void MaxPoolingBackwardKernelGPU<double, int32_t>(
    double *d_grad_in_feat, int in_nrows, const double *d_grad_out_feat,
    int out_nrows, const int32_t *d_max_index, int nchannel,
    cudaStream_t stream);
#endif
