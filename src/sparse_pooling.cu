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
template <typename Dtype, typename Itype> struct ValInd {
  Dtype val;
  Itype ind;
};

template <typename Dtype, typename Itype>
__global__ void convert_to_valin(const int n, const Dtype *feat,
                                 const int start_ch, const int nchannel,
                                 const Itype *map,
                                 ValInd<Dtype, Itype> *valind) {
  int feat_index;
  CUDA_KERNEL_LOOP(index, n) {
    feat_index = start_ch + map[index] * nchannel;
    valind[index].val = feat[feat_index];
    valind[index].ind = feat_index;
  }
}

template <typename Dtype, typename Itype>
__global__ void valind_to_out(const int n, const ValInd<Dtype, Itype> *valind,
                              const Itype *out_map, Dtype *out_feat,
                              Itype *out_index, const int start_ch,
                              const int nchannel) {
  int i;
  ValInd<Dtype, Itype> curr_valind;
  CUDA_KERNEL_LOOP(index, n) {
    i = out_map[index] * nchannel + start_ch;
    curr_valind = valind[index];
    if (out_feat[i] < curr_valind.val) {
      out_feat[i] = curr_valind.val;
      out_index[i] = curr_valind.ind;
    }
  }
}

template <typename Dtype, typename Itype> struct valind_comparator {
  __host__ __device__ bool operator()(ValInd<Dtype, Itype> &x,
                                      ValInd<Dtype, Itype> &y) {
    if (x.val < y.val)
      return true;
    else
      return false;
  }
};

template <typename Dtype, typename Itype>
struct valind_max_operator
    : public thrust::binary_function<ValInd<Dtype, Itype>, ValInd<Dtype, Itype>,
                                     ValInd<Dtype, Itype>> {
  __host__ __device__ ValInd<Dtype, Itype> operator()(ValInd<Dtype, Itype> x,
                                                      ValInd<Dtype, Itype> y) {
    if (x.val > y.val)
      return x;
    else
      return y;
  }
};

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
                     int nchannel, const Itype *d_num_nonzero,
                     const Itype *in_map, const Itype *out_map) {
  CUDA_KERNEL_LOOP(index, n) {
    int nrow = index / nchannel;
    int ch = index % nchannel;
    atomicAdd(&d_grad_in[in_map[nrow] * nchannel + ch],
              d_grad_out[out_map[nrow] * nchannel + ch] /
                  d_num_nonzero[out_map[nrow]]);
  }
}

template <typename Dtype, typename Itype>
__global__ void in_map_feat(const int n, const Dtype *in_feat,
                            const int start_ch, const int nchannel,
                            const Itype *in_map, Dtype *out_feat) {
  int feat_index;
  CUDA_KERNEL_LOOP(index, n) {
    feat_index = start_ch + in_map[index] * nchannel;
    out_feat[index] = in_feat[feat_index];
  }
}

template <typename Dtype, typename Itype>
__global__ void out_map_feat(const int n, const Dtype *in_feat,
                             const int start_ch, const int nchannel,
                             const Itype *out_map, Dtype *out_feat) {
  int i;
  CUDA_KERNEL_LOOP(index, n) {
    i = out_map[index] * nchannel + start_ch;
    out_feat[i] = in_feat[index];
  }
}

template <typename Dtype, typename Itype>
__global__ void out_map_feat_div(const int n, const Dtype *in_feat,
                                 const int start_ch, const int nchannel,
                                 const Itype *out_map, const Itype *num_nonzero,
                                 Dtype *out_feat) {
  int out_map_index;
  CUDA_KERNEL_LOOP(index, n) {
    out_map_index = out_map[index];
    out_feat[out_map_index * nchannel + start_ch] =
        in_feat[index] / num_nonzero[out_map_index];
  }
}

void print(const thrust::device_vector<ValInd<float, int32_t>> &v) {
  for (int i = 0; i < v.size(); i++) {
    auto tmp = static_cast<ValInd<float, int32_t>>(v[i]);
    std::cout << " " << std::fixed << i << "th v: " << tmp.val
              << ", i: " << tmp.ind;
  }
  std::cout << "\n";
}

template <typename Dtype, typename Itype>
void SparseMaxPoolingForwardGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index, int nchannel,
                                const std::vector<std::vector<Itype>> in_map,
                                const std::vector<std::vector<Itype>> out_map,
                                cudaStream_t stream) {
  int n_active = 0;
  thrust::device_vector<Itype> d_in_map, d_out_map, d_curr_out_map,
      d_sorted_out_map, d_reduced_sorted_out_map;
  thrust::device_vector<ValInd<Dtype, Itype>> d_valind, d_reduced_valind;
  thrust::equal_to<int> equal_to;
  valind_max_operator<Dtype, Itype> valind_max;

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

  const Itype *d_in_map_ptr = thrust::raw_pointer_cast(d_in_map.data());
  ValInd<Dtype, Itype> *d_valind_ptr = thrust::raw_pointer_cast(d_valind.data());

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

template void SparseMaxPoolingForwardGPU<float, int32_t>(
    const float *d_in_feat, float *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel,
    const std::vector<std::vector<int32_t>> in_map,
    const std::vector<std::vector<int32_t>> out_map, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseMaxPoolingBackwardGPU(Dtype *d_grad_in_feat, int in_nrows,
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

template void SparseMaxPoolingBackwardGPU<float, int32_t>(
    float *d_grad_in_feat, int in_nrows, const float *d_grad_out_feat,
    int out_nrows, const int32_t *d_max_index, int nchannel, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseNonzeroAvgPoolingForwardGPU(
    const Dtype *d_in_feat, Dtype *d_out_feat, int out_nrows,
    Itype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> in_map,
    const std::vector<std::vector<Itype>> out_map, cudaStream_t stream) {
  int n_active = 0;
  thrust::device_vector<Itype> d_in_map, d_out_map, d_curr_out_map,
      d_sorted_out_map, d_reduced_sorted_out_map;
  thrust::device_ptr<Itype> d_reduced_num_nonzero =
      thrust::device_pointer_cast(d_num_nonzero);
  thrust::device_vector<Dtype> d_in_feat_per_ch, d_reduced_in_feat_per_ch;

  thrust::fill(thrust::device, d_out_feat, d_out_feat + nchannel * out_nrows,
               0);
  thrust::fill(thrust::device, d_num_nonzero, d_num_nonzero + out_nrows, 0);

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

  // Initializing less memory when num output is known
  // https://thrust.github.io/doc/classthrust_1_1discard__iterator.html
  d_sorted_out_map = d_out_map;
  CUDA_POST_KERNEL_CHECK;
  d_reduced_sorted_out_map.resize(out_nrows);
  CUDA_POST_KERNEL_CHECK;
  d_in_feat_per_ch.resize(n_active);
  CUDA_POST_KERNEL_CHECK;
  d_reduced_in_feat_per_ch.resize(out_nrows);
  CUDA_POST_KERNEL_CHECK;

  const Itype *d_in_map_ptr = thrust::raw_pointer_cast(d_in_map.data());

  // Create sorted d_out_map
  thrust::sort(d_sorted_out_map.begin(), d_sorted_out_map.end());

  // Count num outputs
  // https://thrust.github.io/doc/classthrust_1_1discard__iterator.html
  thrust::reduce_by_key(d_sorted_out_map.begin(), d_sorted_out_map.end(),
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(), d_reduced_num_nonzero);

  for (int j = 0; j < nchannel; j++) {
    d_curr_out_map = d_out_map;

    in_map_feat<Dtype><<<GET_BLOCKS(n_active), CUDA_NUM_THREADS, 0, stream>>>(
        n_active, d_in_feat, j, nchannel, d_in_map_ptr,
        thrust::raw_pointer_cast(d_in_feat_per_ch.data()));
    CUDA_POST_KERNEL_CHECK;

    // Sort by d_out_map for reduction
    thrust::sort_by_key(d_curr_out_map.begin(), d_curr_out_map.end(),
                        d_in_feat_per_ch.begin());
    CUDA_POST_KERNEL_CHECK;

    // reduce by key
    thrust::reduce_by_key(d_sorted_out_map.begin(), d_sorted_out_map.end(),
                          d_in_feat_per_ch.begin(),
                          d_reduced_sorted_out_map.begin(),
                          d_reduced_in_feat_per_ch.begin());
    CUDA_POST_KERNEL_CHECK;

    // Copy the values to the output
    out_map_feat_div<Dtype>
        <<<GET_BLOCKS(out_nrows), CUDA_NUM_THREADS, 0, stream>>>(
            out_nrows,
            thrust::raw_pointer_cast(d_reduced_in_feat_per_ch.data()), j,
            nchannel, thrust::raw_pointer_cast(d_reduced_sorted_out_map.data()),
            d_num_nonzero, d_out_feat);
  }
}

template void SparseNonzeroAvgPoolingForwardGPU<float, int32_t>(
    const float *d_in_feat, float *d_out_feat, int out_nrows,
    int32_t *d_max_index, int nchannel,
    const std::vector<std::vector<int32_t>> in_map,
    const std::vector<std::vector<int32_t>> out_map, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseNonzeroAvgPoolingBackwardGPU(
    Dtype *d_grad_in_feat, int in_nrows, const Dtype *d_grad_out_feat,
    int out_nrows, const Itype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> in_map,
    const std::vector<std::vector<Itype>> out_map, cudaStream_t stream) {
  int curr_n, n_active = 0;
  thrust::device_vector<Itype> d_in_map, d_out_map;

  // Cleanup gradients
  HANDLE_ERROR(
      cudaMemset(d_grad_in_feat, 0, in_nrows * nchannel * sizeof(Dtype)));

  // Copy all maps to one vector
  for (int k = 0; k < in_map.size(); k++)
    n_active += in_map[k].size();

  d_in_map.resize(n_active);
  d_out_map.resize(n_active);

  auto d_in_map_iter = d_in_map.begin();
  auto d_out_map_iter = d_out_map.begin();
  for (int k = 0; k < in_map.size(); k++) {
    curr_n = in_map[k].size();
    if (curr_n > 0) {
      thrust::copy_n(in_map[k].begin(), curr_n, d_in_map_iter);
      CUDA_POST_KERNEL_CHECK;
      thrust::copy_n(out_map[k].begin(), curr_n, d_out_map_iter);
      CUDA_POST_KERNEL_CHECK;
      thrust::advance(d_in_map_iter, curr_n);
      thrust::advance(d_out_map_iter, curr_n);
    }
  }

  set_gradient_nonzero<Dtype>
      <<<GET_BLOCKS(n_active * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
          n_active * nchannel, d_grad_out_feat, d_grad_in_feat, nchannel,
          d_num_nonzero, thrust::raw_pointer_cast(d_in_map.data()),
          thrust::raw_pointer_cast(d_out_map.data()));
}

template void SparseNonzeroAvgPoolingBackwardGPU<float, int32_t>(
    float *d_grad_in_feat, int in_nrows, const float *d_grad_out_feat,
    int out_nrows, const int32_t *d_num_nonzero, int nchannel,
    const std::vector<std::vector<int32_t>> in_map,
    const std::vector<std::vector<int32_t>> out_map, cudaStream_t stream);

#endif
