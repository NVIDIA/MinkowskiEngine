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

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "allocators.cuh"
#include "gpu.cuh"
#include "pooling_max_kernel.cuh"
#include "utils.hpp"

template <typename Dtype, typename MaskItype>
__global__ void set_gradient(const size_t n, const Dtype *d_grad_out,
                             Dtype *d_grad_in, const MaskItype *in_index,
                             const Dtype unused_key) {
  CUDA_KERNEL_LOOP(index, n) {
    auto const queried_index = in_index[index];
    if (queried_index != unused_key)
      atomicAdd(&d_grad_in[in_index[index]], d_grad_out[index]);
  }
}

template <typename Dtype, typename MaskItype, typename MapItype>
__global__ void max_pool_reduced_pointer_kernel_gpu(
    const int N, const int out_nrows, const int nchannel, const int nmap,
    const Dtype *__restrict__ d_in_feat,    //
    Dtype *__restrict__ d_out_feat,         //
    MaskItype *__restrict__ d_max_index,    //
    const MapItype *__restrict__ d_in_map,  //
    const MapItype *__restrict__ d_out_map, //
    const MapItype *__restrict__ d_in_index_min) {
  // N == nmap * nchannel
  CUDA_KERNEL_LOOP(index, N) {
    int const nrow = index / nchannel;
    int const ch = index % nchannel;

    MapItype const out_map_row = d_out_map[nrow];
    MapItype const in_index = d_in_index_min[nrow];
    MapItype num_in_feat;

    if (nrow == out_nrows - 1)
      num_in_feat = nmap - in_index;
    else
      num_in_feat = d_in_index_min[nrow + 1] - in_index;
    // It is guaranteed to have at least one input per output
    MapItype curr_index, max_index = d_in_map[in_index] * nchannel + ch;
    Dtype curr_val, max_val = d_in_feat[max_index];
    for (int curr_iter = 0; curr_iter < num_in_feat; ++curr_iter) {
      curr_index = d_in_map[in_index + curr_iter] * nchannel + ch;
      curr_val = d_in_feat[curr_index];
      if (max_val < curr_val) {
        max_val = curr_val;
        max_index = curr_index;
      }
    }
    MapItype const out_ind = out_map_row * nchannel + ch;
    // TODO thrust::reduce_by_key results in erroneous results at the end for
    // very large array
    if (out_map_row < out_nrows) {
      d_out_feat[out_ind] = max_val;
      d_max_index[out_ind] = max_index;
    }
  }
}

// Put features in to the out features according to the input index.
// The input index is sorted according to the out index so no need to take out
// index
template <typename Dtype, typename Itype>
__global__ void copy_sorted(const int n, const int nrows, const int nchannel,
                            const Dtype *__restrict__ in_feat,
                            const Itype *__restrict__ in_index,
                            Dtype *__restrict__ out_feat) {
  int nrow, ch;
  CUDA_KERNEL_LOOP(index, n) {
    nrow = index / nchannel;
    ch = index % nchannel;
    out_feat[index] = in_feat[in_index[nrow] * nchannel + ch];
  }
}

namespace minkowski {

namespace detail {

template <typename Dtype>
__global__ void fill(const size_t n, Dtype *dst, Dtype const val) {
  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;
  if (x < n)
    dst[x] = val;
}

} // namespace detail

template <typename Dtype, typename MaskItype, typename MapItype>
void max_pool_forward_pointer_kernel_gpu(
    MapItype *d_in_map,     // this will be sorted
    MapItype *d_out_map,    // this will be sorted
    size_t const nmap,      // map size
    Dtype const *d_in_feat, //
    Dtype *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    MaskItype *d_max_index, //
    bool const is_sorted    //
) {

  MapItype *d_scr = (MapItype *)c10::cuda::CUDACachingAllocator::raw_alloc(
      3 * (nmap + 1) * sizeof(MapItype));

  MapItype *d_index = d_scr;                          // sequence
  MapItype *d_in_map_min = d_scr + 1 * nmap + 1;      // reduced min output maps
  MapItype *d_reduced_out_map = d_scr + 2 * nmap + 2; // reduced output maps

  // create number of in_feat per out, and starting index
  THRUST_CHECK(thrust::sequence(thrust::device, d_index, d_index + nmap));

  ////////////////////////////////
  // Reduction
  ////////////////////////////////
  // sort d_out_map and d_in_map with the d_out_map so that in_feat are
  // placed adjacent according to out_map
  if (!is_sorted)
    THRUST_CHECK(thrust::sort_by_key(thrust::device, d_out_map,
                                     d_out_map + nmap, d_in_map));

  thrust::equal_to<MapItype> equal_pred;
  thrust::minimum<MapItype> min_op;
  size_t num_unique_out_map;

  try {
    auto reduction_pair =
        thrust::reduce_by_key(thrust::device,    // execution policy
                              d_out_map,         // key begin
                              d_out_map + nmap,  // key end
                              d_index,           // val begin
                              d_reduced_out_map, // key out begin
                              d_in_map_min,      // val out begin
                              equal_pred,        // binary pred
                              min_op);           // binary op
    CUDA_CHECK(cudaStreamSynchronize(0));
    num_unique_out_map = reduction_pair.first - d_reduced_out_map;
  } THRUST_CATCH;

#ifdef DEBUG
  std::cout << "num_unique_out_map: " << num_unique_out_map << "\n";
  MapItype *p_scr = (MapItype *)std::malloc((nmap + 1) * 2 * sizeof(MapItype));
  CUDA_CHECK(cudaMemcpy(p_scr, d_in_map_min, (nmap + 1) * 2 * sizeof(MapItype),
                        cudaMemcpyDeviceToHost));
  MapItype step = std::max<MapItype>(num_unique_out_map / 100, 1);
  MapItype i = 0;
  for (; i < num_unique_out_map;) {
    std::cout << i;
    std::cout << " in_map_min: " << p_scr[i]
              << ", reduced_out_map: " << p_scr[i + 1 + nmap] << "\n";
    i += step;
  }
  i -= step;
  for (; i < num_unique_out_map; ++i) {
    std::cout << i;
    std::cout << " in_map_min: " << p_scr[i]
              << ", reduced_out_map: " << p_scr[i + 1 + nmap] << "\n";
  }
  std::free(p_scr);
  std::cout << "done printing\n";
#endif

  if (num_unique_out_map > out_nrows)
    throw std::invalid_argument(
        Formatter() << "Invalid number of out nrows: " << out_nrows
                    << ", num_unique_out_map: " << num_unique_out_map);

  // fill it with unused key
  detail::fill<<<GET_BLOCKS(out_nrows * nchannel, CUDA_NUM_THREADS),
                 CUDA_NUM_THREADS>>>(out_nrows * nchannel, d_max_index,
                                     std::numeric_limits<MaskItype>::max());

#ifdef DEBUG
  // CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "filled\n";
#endif

  // Finally, use the max kernel to map all in_feats with the same out key to
  // out_feats Also, create out max_index for gradient
  max_pool_reduced_pointer_kernel_gpu<Dtype, MaskItype, MapItype>
      <<<GET_BLOCKS(num_unique_out_map * nchannel, CUDA_NUM_THREADS),
         CUDA_NUM_THREADS>>>(nchannel * num_unique_out_map, // N
                             num_unique_out_map, nchannel, nmap, d_in_feat,
                             d_out_feat,
                             d_max_index, // Out indices for backward
                             d_in_map,    // in index
                             d_reduced_out_map, d_in_map_min);

  // cudaFree(d_in_map);
  // cudaFree(d_index);
  CUDA_CHECK(cudaGetLastError());
  // CUDA_CHECK(cudaStreamSynchronize(stream));

  c10::cuda::CUDACachingAllocator::raw_delete((void *)d_scr);
}

template <typename Dtype, typename Itype, typename ByteAllocator>
void MaxPoolingForwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_out_feat, size_t const out_nrows,
    int *d_max_index, size_t const nchannel,
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    ByteAllocator &allocator, cudaStream_t stream) {

  size_t nmap = kernel_map.size();
  size_t scratch_bytes = 2 * (nmap + 1) * sizeof(Itype);
  Itype *d_scr = (Itype *)allocator.allocate(scratch_bytes);

  Itype *d_in_map = d_scr;             // all input kernel maps
  Itype *d_out_map = d_scr + nmap + 1; // all output kernel maps

  ////////////////////////////////
  // Initialize data
  ////////////////////////////////
#ifdef DEBUG
  cudaMemset(d_scr, 0, scratch_bytes);
  std::cout << "out_nrows: " << out_nrows << ", nmap: " << nmap << "\n";
#endif
  Itype *d_curr_in_map = d_in_map;
  Itype *d_curr_out_map = d_out_map;

  for (auto k = kernel_map.key_cbegin(); k != kernel_map.key_cend(); ++k) {
    auto kernel_index = k->first;
    size_t curr_size = kernel_map.in_maps.size(kernel_index);
    CUDA_CHECK(cudaMemcpyAsync(
        d_curr_in_map, kernel_map.in_maps.begin(kernel_index),
        curr_size * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_curr_out_map, kernel_map.out_maps.begin(kernel_index),
        curr_size * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    d_curr_in_map += curr_size;
    d_curr_out_map += curr_size;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //
  max_pool_forward_pointer_kernel_gpu<Dtype, int32_t, Itype>(
      d_in_map, d_out_map, nmap, d_in_feat, d_out_feat, out_nrows, nchannel,
      d_max_index, false);

  allocator.deallocate((char *)d_scr, scratch_bytes);
}

// default_allocator
template void
MaxPoolingForwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    const float *d_in_feat, float *d_out_feat, size_t const out_nrows,
    int32_t *d_max_index, size_t const nchannel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    detail::default_allocator<char> &allocator, cudaStream_t stream);

template void
MaxPoolingForwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    const double *d_in_feat, double *d_out_feat, size_t const out_nrows,
    int32_t *d_max_index, size_t const nchannel,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    detail::default_allocator<char> &allocator, cudaStream_t stream);

// c10_allocator
template void
MaxPoolingForwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    const float *d_in_feat, float *d_out_feat, size_t const out_nrows,
    int32_t *d_max_index, size_t const nchannel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    detail::c10_allocator<char> &allocator, cudaStream_t stream);

template void
MaxPoolingForwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    const double *d_in_feat, double *d_out_feat, size_t const out_nrows,
    int32_t *d_max_index, size_t const nchannel,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    detail::c10_allocator<char> &allocator, cudaStream_t stream);

template <typename Dtype, typename MaskItype>
void MaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, size_t const in_nrows,
                                 const Dtype *d_grad_out_feat,
                                 size_t const out_nrows,
                                 const MaskItype *d_max_index,
                                 size_t const nchannel) {
  size_t const num_kernels = out_nrows * nchannel;
  // Assume that gradients for input feature are all set to zero
  LOG_DEBUG("MaxPool Backward GPU with #", num_kernels, out_nrows, nchannel);
  set_gradient<Dtype, MaskItype>
      <<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          num_kernels, d_grad_out_feat, d_grad_in_feat, d_max_index,
          std::numeric_limits<MaskItype>::max());

  // CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void max_pool_forward_pointer_kernel_gpu<float, int32_t, uint32_t>(
    uint32_t *d_in_map,     //
    uint32_t *d_out_map,    //
    size_t const nmap,      //
    float const *d_in_feat, //
    float *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    int32_t *d_max_index,   //
    bool const is_sorted    //
);

template void max_pool_forward_pointer_kernel_gpu<double, int32_t, uint32_t>(
    uint32_t *d_in_map,      //
    uint32_t *d_out_map,     //
    size_t const nmap,       //
    double const *d_in_feat, //
    double *d_out_feat,      //
    size_t const out_nrows,  //
    size_t const nchannel,   //
    int32_t *d_max_index,    //
    bool const is_sorted     //
);

template void max_pool_forward_pointer_kernel_gpu<float, int32_t, int32_t>(
    int32_t *d_in_map,      //
    int32_t *d_out_map,     //
    size_t const nmap,      //
    float const *d_in_feat, //
    float *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    int32_t *d_max_index,   //
    bool const is_sorted    //
);

template void max_pool_forward_pointer_kernel_gpu<double, int32_t, int32_t>(
    int32_t *d_in_map,       //
    int32_t *d_out_map,      //
    size_t const nmap,       //
    double const *d_in_feat, //
    double *d_out_feat,      //
    size_t const out_nrows,  //
    size_t const nchannel,   //
    int32_t *d_max_index,    //
    bool const is_sorted     //
);

template void MaxPoolingBackwardKernelGPU<float, int32_t>(
    float *d_grad_in_feat, size_t const in_nrows, const float *d_grad_out_feat,
    size_t const out_nrows, const int32_t *d_max_index, size_t const nchannel);

template void MaxPoolingBackwardKernelGPU<double, int32_t>(
    double *d_grad_in_feat, size_t const in_nrows,
    const double *d_grad_out_feat, size_t const out_nrows,
    const int32_t *d_max_index, size_t const nchannel);

// int64
template void max_pool_forward_pointer_kernel_gpu<float, int64_t, int64_t>(
    int64_t *d_in_map,      //
    int64_t *d_out_map,     //
    size_t const nmap,      //
    float const *d_in_feat, //
    float *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    int64_t *d_max_index,   //
    bool const is_sorted    //
);

template void max_pool_forward_pointer_kernel_gpu<double, int64_t, int64_t>(
    int64_t *d_in_map,       //
    int64_t *d_out_map,      //
    size_t const nmap,       //
    double const *d_in_feat, //
    double *d_out_feat,      //
    size_t const out_nrows,  //
    size_t const nchannel,   //
    int64_t *d_max_index,    //
    bool const is_sorted     //
);

template void MaxPoolingBackwardKernelGPU<float, int64_t>(
    float *d_grad_in_feat, size_t const in_nrows, const float *d_grad_out_feat,
    size_t const out_nrows, const int64_t *d_max_index, size_t const nchannel);

template void MaxPoolingBackwardKernelGPU<double, int64_t>(
    double *d_grad_in_feat, size_t const in_nrows,
    const double *d_grad_out_feat, size_t const out_nrows,
    const int64_t *d_max_index, size_t const nchannel);

} // end namespace minkowski

#endif // GPU_POOLING_MAX_KERNEL
