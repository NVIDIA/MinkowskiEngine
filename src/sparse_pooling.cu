#ifndef GPU_POOLING
#define GPU_POOLING

#include <limits>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "src/sparse_pooling.cuh"

/* Sort by output key (reduce will generate output that doesn't require mapping
 * Sort in_map by out_map using sort_by_key.
 * Then, use the reduce_by_key.
 * To extract the max index, use a structure
 */
template <typename Dtype> struct ValInd {
  Dtype val;
  int ind;
};

template <typename Dtype>
__global__ void convert_to_valin(const int n, const Dtype *feat,
                                 const int start_ch, const int nchannel,
                                 const int64_t *map, ValInd<Dtype> *valind) {
  int feat_index;
  CUDA_KERNEL_LOOP(index, n) {
    feat_index = start_ch + map[index] * nchannel;
    valind[index].val = feat[feat_index];
    valind[index].ind = feat_index;
  }
}

template <typename Dtype>
__global__ void valind_to_out(const int n, const ValInd<Dtype> *valind,
                              const int64_t *out_map, Dtype *out_feat,
                              int64_t *out_index, const int start_ch,
                              const int nchannel) {
  int i;
  ValInd<Dtype> curr_valind;
  CUDA_KERNEL_LOOP(index, n) {
    i = out_map[index] * nchannel + start_ch;
    curr_valind = valind[index];
    if (out_feat[i] < curr_valind.val) {
      out_feat[i] = curr_valind.val;
      out_index[i] = curr_valind.ind;
    }
  }
}

template <typename Dtype> struct valind_comparator {
  __host__ __device__ bool operator()(ValInd<Dtype> &x, ValInd<Dtype> &y) {
    if (x.val < y.val)
      return true;
    else
      return false;
  }
};

template <typename Dtype>
struct valind_max_operator
    : public thrust::binary_function<ValInd<Dtype>, ValInd<Dtype>,
                                     ValInd<Dtype>> {
  __host__ __device__ ValInd<Dtype> operator()(ValInd<Dtype> x,
                                               ValInd<Dtype> y) {
    if (x.val > y.val)
      return x;
    else
      return y;
  }
};

template <typename Dtype>
__global__ void set_gradient(const int n, const Dtype *d_grad_out,
                             Dtype *d_grad_in, const int64_t *out_index,
                             int nchannel) {
  CUDA_KERNEL_LOOP(index, n) {
    atomicAdd(&d_grad_in[out_index[index]], d_grad_out[index]);
  }
}

void print(const thrust::device_vector<ValInd<float>> &v) {
  for (size_t i = 0; i < v.size(); i++) {
    auto tmp = static_cast<ValInd<float>>(v[i]);
    std::cout << " " << std::fixed << i << "th v: " << tmp.val
              << ", i: " << tmp.ind;
  }
  std::cout << "\n";
}

template <typename Dtype>
void SparseMaxPoolingForwardGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int64_t out_nrows, int64_t *d_max_index,
                                int64_t nchannel,
                                const std::vector<std::vector<int64_t>> in_map,
                                const std::vector<std::vector<int64_t>> out_map,
                                cudaStream_t stream) {
  int n_active = 0;
  thrust::device_vector<int64_t> d_in_map, d_out_map, d_curr_out_map,
      d_sorted_out_map, d_reduced_sorted_out_map;
  thrust::device_vector<ValInd<Dtype>> d_valind, d_reduced_valind;
  thrust::equal_to<int> equal_to;
  valind_max_operator<Dtype> valind_max;

  // Fill the output with -FLT_MAX
  thrust::fill(thrust::device, d_out_feat, d_out_feat + nchannel * out_nrows,
               -std::numeric_limits<Dtype>::max());

  // Copy all maps to one vector
  for (int k = 0; k < in_map.size(); k++)
    n_active += in_map[k].size();

  d_in_map.resize(n_active);
  d_out_map.resize(n_active);

  auto d_in_map_iter = d_in_map.begin();
  auto d_out_map_iter = d_out_map.begin();
  for (int k = 0; k < in_map.size(); k++) {
    int curr_n = in_map[k].size();
    if (curr_n > 0) {
      thrust::copy_n(in_map[k].begin(), curr_n, d_in_map_iter);
      CUDA_POST_KERNEL_CHECK;
      thrust::copy_n(out_map[k].begin(), curr_n, d_out_map_iter);
      CUDA_POST_KERNEL_CHECK;
      thrust::advance(d_in_map_iter, curr_n);
      thrust::advance(d_out_map_iter, curr_n);
    }
  }

  d_sorted_out_map = d_out_map;
  CUDA_POST_KERNEL_CHECK;
  d_reduced_sorted_out_map.resize(out_nrows);
  CUDA_POST_KERNEL_CHECK;
  d_valind.resize(n_active);
  CUDA_POST_KERNEL_CHECK;
  d_reduced_valind.resize(out_nrows);
  CUDA_POST_KERNEL_CHECK;

  const int64_t *d_in_map_ptr = thrust::raw_pointer_cast(d_in_map.data());
  ValInd<Dtype> *d_valind_ptr = thrust::raw_pointer_cast(d_valind.data());

  // Create sorted d_out_map
  thrust::sort(d_sorted_out_map.begin(), d_sorted_out_map.end());

  for (int j = 0; j < nchannel; j++) {
    d_curr_out_map = d_out_map;

    // Fill the d_valind
    convert_to_valin<Dtype>
        <<<GET_BLOCKS(n_active), CUDA_NUM_THREADS, 0, stream>>>(
            n_active, d_in_feat, j, nchannel, d_in_map_ptr, d_valind_ptr);
    CUDA_POST_KERNEL_CHECK;

    // Sort by d_out_map for reduction
    thrust::sort_by_key(d_curr_out_map.begin(), d_curr_out_map.end(),
                        d_valind.begin());
    CUDA_POST_KERNEL_CHECK;

    // reduce by key
    thrust::reduce_by_key(d_sorted_out_map.begin(), d_sorted_out_map.end(),
                          d_valind.begin(), d_reduced_sorted_out_map.begin(),
                          d_reduced_valind.begin(), equal_to, valind_max);
    CUDA_POST_KERNEL_CHECK;

    // Copy the values to the output
    valind_to_out<Dtype>
        <<<GET_BLOCKS(out_nrows), CUDA_NUM_THREADS, 0, stream>>>(
            out_nrows, thrust::raw_pointer_cast(d_reduced_valind.data()),
            thrust::raw_pointer_cast(d_reduced_sorted_out_map.data()),
            d_out_feat, d_max_index, j, nchannel);
  }
}

template void SparseMaxPoolingForwardGPU<float>(
    const float *d_in_feat, float *d_out_feat, int64_t out_nrows,
    int64_t *d_max_index, int64_t nchannel,
    const std::vector<std::vector<int64_t>> in_map,
    const std::vector<std::vector<int64_t>> out_map, cudaStream_t stream);

template <typename Dtype>
void SparseMaxPoolingBackwardGPU(Dtype *d_grad_in_feat, int64_t in_nrows,
                                 const Dtype *d_grad_out_feat,
                                 int64_t out_nrows, const int64_t *d_max_index,
                                 int64_t nchannel, cudaStream_t stream) {
  int num_kernels = out_nrows * nchannel;
  // Cleanup gradients
  HANDLE_ERROR(
      cudaMemset(d_grad_in_feat, 0, in_nrows * nchannel * sizeof(Dtype)));
  set_gradient<Dtype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, d_grad_out_feat, d_grad_in_feat, d_max_index, nchannel);
}

template void SparseMaxPoolingBackwardGPU<float>(
    float *d_grad_in_feat, int64_t in_nrows, const float *d_grad_out_feat,
    int64_t out_nrows, const int64_t *d_max_index, int64_t nchannel,
    cudaStream_t stream);

#endif
