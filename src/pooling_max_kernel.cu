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
                             Dtype *d_grad_in, const Itype *out_index,
                             int nchannel) {
  CUDA_KERNEL_LOOP(index, n) {
    atomicAdd(&d_grad_in[out_index[index]], d_grad_out[index]);
  }
}

template <typename Dtype, typename Itype>
__global__ void
set_gradient_nonzero(const int n, const Dtype *d_grad_out, Dtype *d_grad_in,
                     int nchannel, const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    int nrow = index / nchannel;
    int ch = index % nchannel;
    atomicAdd(&d_grad_in[in_map[nrow] * nchannel + ch],
              d_grad_out[out_map[nrow] * nchannel + ch]);
  }
}

template <typename Dtype, typename Itype>
__global__ void
set_gradient_nonzero_avg(const int n, const Dtype *d_grad_out, Dtype *d_grad_in,
                         int nchannel, const Dtype *d_num_nonzero,
                         const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    int nrow = index / nchannel;
    int ch = index % nchannel;
    int curr_num_nonzero = d_num_nonzero[out_map[nrow]];
    if (curr_num_nonzero > 0)
      atomicAdd(&d_grad_in[in_map[nrow] * nchannel + ch],
                d_grad_out[out_map[nrow] * nchannel + ch] / curr_num_nonzero);
  }
}

template <typename Dtype, typename Itype>
__global__ void max_pool(const int N, const int nnz, const int nchannel,
                         const Dtype *d_in_feat, Dtype *d_out_feat,
                         Itype *d_max_index, const Itype *d_out_index,
                         const Itype *d_in_index_min) {
  // N == nnz * nchannel
  CUDA_KERNEL_LOOP(index, N) {
    int nrow = index / nchannel;
    int ch = index % nchannel;

    Itype out_index = d_out_index[nrow];
    Itype in_index = d_in_index_min[nrow];
    Itype num_in_feat;
    if (nrow == nnz)
      num_in_feat = d_in_index_min[nrow + 1] - in_index;
    else
      num_in_feat = nnz - in_index;
    Dtype curr_val, curr_max = d_in_feat[in_index * nchannel + ch];
    Itype curr_index, max_index = in_index;
    for (int curr_iter = 0; curr_iter < num_in_feat; curr_iter++) {
      curr_index = (in_index + curr_iter) * nchannel + ch;
      curr_val = d_in_feat[curr_index];
      if (curr_max < curr_val) {
        curr_max = curr_val;
        max_index = curr_index;
      }
    }
    d_out_feat[in_index * nchannel + ch] = curr_max;
    d_max_index[out_index * nchannel + ch] = max_index;
  }
}

template <typename Dtype>
__global__ void col2row_major(const int n, const int nrows, const int ncols,
                              const Dtype *colA, Dtype *rowA) {
  int i, j;
  CUDA_KERNEL_LOOP(index, n) {
    i = index % nrows;
    j = index / nrows;
    rowA[i * ncols + j] = colA[index];
  }
}

template <typename Dtype>
__global__ void col2row_major_with_div(const int n, const int nrows,
                                       const int ncols,
                                       const Dtype *num_nonzero,
                                       const Dtype *colA, Dtype *rowA) {
  int i, j;
  CUDA_KERNEL_LOOP(index, n) {
    i = index % nrows;
    j = index / nrows;
    if (num_nonzero[i]) {
      rowA[i * ncols + j] = colA[index] / num_nonzero[i];
    } else {
      rowA[i * ncols + j] = colA[index];
    }
  }
}

template <typename Dtype, typename Itype>
void MaxPoolingForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index, int nchannel,
                                const std::vector<std::vector<Itype>> &in_maps,
                                const std::vector<std::vector<Itype>> &out_maps,
                                cudaStream_t stream) {
  int nnz = 0;
  Itype *d_in_map, *d_out_map;

  // Copy all maps to one vector
  for (auto map : in_maps)
    nnz += map.size();

  CUDA_CHECK(cudaMalloc((void **)&d_in_map, 2 * nnz * sizeof(Itype)));
  d_out_map = d_in_map + nnz;

  Itype *d_in_map_iter = d_in_map, *d_out_map_iter = d_out_map;
  for (int k = 0; k < in_maps.size(); k++) {
    int curr_n = in_maps[k].size();
    if (curr_n > 0) {
      CUDA_CHECK(cudaMemcpy(d_in_map_iter, in_maps[k].data(),
                            sizeof(Itype) * curr_n, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_out_map_iter, out_maps[k].data(),
                            sizeof(Itype) * curr_n, cudaMemcpyHostToDevice));
      d_in_map_iter += curr_n;
      d_out_map_iter += curr_n;
    }
  }

  // First, sort d_out_map and d_in_map with the d_out_map so that in_feat are
  // placed adjacent according to out_map
  thrust::sort_by_key(d_out_map, d_out_map + nnz, d_in_map);

  // Second, create number of in_feat per out, and starting index
  thrust::device_vector<Itype> d_index(nnz);
  thrust::sequence(d_index.begin(), d_index.end());

  thrust::device_vector<Itype> d_in_map_min(nnz);
  thrust::device_vector<Itype> d_reduced_out_map(nnz);

  thrust::equal_to<Itype> equal_pred;
  thrust::minimum<Itype> min_op;

  auto reduction_pair =
      thrust::reduce_by_key(thrust::device,            // execution policy
                            d_out_map,                 // key begin
                            d_out_map + nnz,           // key end
                            d_index.begin(),           // val begin
                            d_reduced_out_map.begin(), // key out begin
                            d_in_map_min.begin(),      // val out begin
                            equal_pred,                // binary pred
                            min_op);                   // binary op
  size_t num_unique_out_map = reduction_pair.first - d_reduced_out_map.begin();
  if (num_unique_out_map != out_nrows)
    throw std::invalid_argument(
        Formatter() << "Reduction size mismatch. out_nrows: " << out_nrows
                    << ", num_unique_out_map: " << num_unique_out_map);

  // Finally, use the max kernel to map all in_feats with the same out key to
  // out_feats Also, create out max_index for gradient
  max_pool<Dtype, Itype>
      <<<GET_BLOCKS(nnz * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
          nchannel * nnz, nnz, nchannel, d_in_feat, d_out_feat, d_max_index,
          thrust::raw_pointer_cast(d_reduced_out_map.data()),
          thrust::raw_pointer_cast(d_in_map_min.data()));
}

template void MaxPoolingForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, cudaStream_t stream);

template void MaxPoolingForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, cudaStream_t stream);

template <typename Dtype, typename Itype>
void MaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, int in_nrows,
                                 const Dtype *d_grad_out_feat, int out_nrows,
                                 const Itype *d_max_index, int nchannel,
                                 cudaStream_t stream) {
  int num_kernels = out_nrows * nchannel;
  // Cleanup gradients
  HANDLE_ERROR(
      cudaMemset(d_grad_in_feat, 0, in_nrows * nchannel * sizeof(Dtype)));
  set_gradient<Dtype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, d_grad_out_feat, d_grad_in_feat, d_max_index, nchannel);
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
