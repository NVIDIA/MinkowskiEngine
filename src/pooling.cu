#ifndef GPU_POOLING
#define GPU_POOLING

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
#include "pooling.cuh"
#include "utils.hpp"

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

template <typename Dtype>
__global__ void fill(const int n, Dtype *in_feat, Dtype val) {
  CUDA_KERNEL_LOOP(index, n) { in_feat[index] = val; }
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
void MaxPoolingForwardKernelGPU(const Dtype *d_in_feat, Dtype *d_out_feat,
                                int out_nrows, Itype *d_max_index, int nchannel,
                                const std::vector<std::vector<Itype>> &in_map,
                                const std::vector<std::vector<Itype>> &out_map,
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

  THRUST_CHECK(d_in_map.resize(n_active));
  THRUST_CHECK(d_out_map.resize(n_active));

  auto d_in_map_iter = d_in_map.begin();
  auto d_out_map_iter = d_out_map.begin();
  for (int k = 0; k < in_map.size(); k++) {
    int curr_n = in_map[k].size();
    if (curr_n > 0) {
      thrust::copy_n(in_map[k].begin(), curr_n, d_in_map_iter);
      thrust::copy_n(out_map[k].begin(), curr_n, d_out_map_iter);
      thrust::advance(d_in_map_iter, curr_n);
      thrust::advance(d_out_map_iter, curr_n);
    }
  }

  THRUST_CHECK(d_sorted_out_map = d_out_map);
  THRUST_CHECK(d_reduced_sorted_out_map.resize(out_nrows));
  THRUST_CHECK(d_valind.resize(n_active));
  THRUST_CHECK(d_reduced_valind.resize(out_nrows));

  const Itype *d_in_map_ptr = thrust::raw_pointer_cast(d_in_map.data());
  ValInd<Dtype, Itype> *d_valind_ptr =
      thrust::raw_pointer_cast(d_valind.data());

  // Create sorted d_out_map
  THRUST_CHECK(thrust::sort(d_sorted_out_map.begin(), d_sorted_out_map.end()));

  for (int j = 0; j < nchannel; j++) {
    THRUST_CHECK(d_curr_out_map = d_out_map);

    // Fill the d_valind
    convert_to_valin<Dtype>
        <<<GET_BLOCKS(n_active), CUDA_NUM_THREADS, 0, stream>>>(
            n_active, d_in_feat, j, nchannel, d_in_map_ptr, d_valind_ptr);

    // Sort by d_out_map for reduction
    THRUST_CHECK(thrust::sort_by_key(d_curr_out_map.begin(),
                                     d_curr_out_map.end(), d_valind.begin()));

    // reduce by key
    THRUST_CHECK(thrust::reduce_by_key(
        d_sorted_out_map.begin(), d_sorted_out_map.end(), d_valind.begin(),
        d_reduced_sorted_out_map.begin(), d_reduced_valind.begin(), equal_to,
        valind_max));

    // Copy the values to the output
    valind_to_out<Dtype>
        <<<GET_BLOCKS(out_nrows), CUDA_NUM_THREADS, 0, stream>>>(
            out_nrows, thrust::raw_pointer_cast(d_reduced_valind.data()),
            thrust::raw_pointer_cast(d_reduced_sorted_out_map.data()),
            d_out_feat, d_max_index, j, nchannel);
  }
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

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingForwardKernelGPU(
    const Dtype *d_in_feat, int in_nrows, Dtype *d_out_feat, int out_nrows,
    Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, bool use_avg,
    cusparseHandle_t cushandle, cudaStream_t stream) {
  int nnz = 0;
  const Dtype alpha = 1;
  const Dtype beta = 0;
  cusparseMatDescr_t descr = 0;
  Itype *d_in_map, *d_out_map, *d_csr_row;
  Dtype *d_ones, *d_csr_val, *d_tmp_out_feat;

  // Copy all maps to one vector
  for (auto map : in_maps)
    nnz += map.size();

  CUDA_CHECK(cudaMalloc((void **)&d_in_map,
                        (2 * nnz + out_nrows + 1) * sizeof(Itype)));
  d_out_map = d_in_map + nnz;
  d_csr_row = d_out_map + nnz;

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

  if (use_avg) {
    CUDA_CHECK(
        cudaMalloc((void **)&d_ones,
                   (in_nrows + nnz + nchannel * out_nrows) * sizeof(Dtype)));
    d_csr_val = d_ones + in_nrows;
    d_tmp_out_feat = d_csr_val + nnz;

    fill<Dtype><<<GET_BLOCKS(in_nrows), CUDA_NUM_THREADS, 0, stream>>>(
        in_nrows, d_ones, (Dtype)1.);
  } else {
    CUDA_CHECK(cudaMalloc((void **)&d_ones,
                          (nnz + nchannel * out_nrows) * sizeof(Dtype)));
    d_csr_val = d_ones;
    d_tmp_out_feat = d_csr_val + nnz;
  }
  // CUDA_CHECK(cudaMalloc((void **)&d_ones, in_nrows * sizeof(Dtype)));
  // CUDA_CHECK(cudaMalloc((void **)&d_csr_val, nnz * sizeof(Dtype)));
  // CUDA_CHECK(cudaMalloc((void **)&d_tmp_out_feat,
  //                       nchannel * out_nrows * sizeof(Dtype)));
  fill<Dtype><<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(nnz, d_csr_val,
                                                                (Dtype)1.);

  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Sort COO first
  sort_coo_gpu(cushandle, out_nrows, in_nrows, nnz, d_out_map, d_in_map);

  // For CRS, sort row and col inds by row major.
  CUSPARSE_CHECK(cusparseXcoo2csr(cushandle, d_out_map, nnz, out_nrows,
                                  d_csr_row, CUSPARSE_INDEX_BASE_ZERO));

  CUSPARSE_CHECK(
      cusparse_csrmm<Dtype>(cushandle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
                            CUSPARSE_OPERATION_TRANSPOSE,     // op(B)
                            out_nrows,                        // M
                            nchannel,                         // N
                            in_nrows,                         // K
                            nnz, &alpha, descr,
                            d_csr_val, // val
                            d_csr_row, // row
                            d_in_map,  // col
                            d_in_feat, // B
                            nchannel,  // ldb
                            &beta,
                            d_tmp_out_feat, // C
                            out_nrows       // ldc
                            ));

  if (use_avg) {
    CUSPARSE_CHECK(
        cusparse_csrmv<Dtype>(cushandle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
                              out_nrows,                        // M
                              in_nrows,                         // K
                              nnz, &alpha, descr,
                              d_csr_val, // val
                              d_csr_row, // row
                              d_in_map,  // col
                              d_ones,    // B (in_nrows > out_nrows)
                              &beta,
                              d_num_nonzero)); // C

    col2row_major_with_div<Dtype>
        <<<GET_BLOCKS(out_nrows * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            out_nrows * nchannel, out_nrows, nchannel, d_num_nonzero,
            d_tmp_out_feat, d_out_feat);
  } else {
    col2row_major<Dtype>
        <<<GET_BLOCKS(out_nrows * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            out_nrows * nchannel, out_nrows, nchannel, d_tmp_out_feat,
            d_out_feat);
  }

  CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
  cudaFree(d_in_map);
  cudaFree(d_ones);
}

template void NonzeroAvgPoolingForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, int in_nrows, float *d_out_feat, int out_nrows,
    float *d_num_nonzero, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, bool use_avg,
    cusparseHandle_t cushandle, cudaStream_t stream);

template void NonzeroAvgPoolingForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, int in_nrows, double *d_out_feat, int out_nrows,
    double *d_num_nonzero, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, bool use_avg,
    cusparseHandle_t cushandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void NonzeroAvgPoolingBackwardKernelGPU(
    Dtype *d_grad_in_feat, int in_nrows, const Dtype *d_grad_out_feat,
    int out_nrows, const Dtype *d_num_nonzero, int nchannel,
    const std::vector<std::vector<Itype>> &in_maps,
    const std::vector<std::vector<Itype>> &out_maps, bool use_avg,
    cudaStream_t stream) {
  int nnz = 0;
  Itype *d_in_map, *d_out_map;
  // Copy all maps to one vector
  for (auto map : in_maps)
    nnz += map.size();

  CUDA_CHECK(cudaMalloc((void **)&d_in_map, 2 * nnz * sizeof(Itype)));
  d_out_map = d_in_map + nnz;

  // Cleanup gradients
  HANDLE_ERROR(
      cudaMemset(d_grad_in_feat, 0, in_nrows * nchannel * sizeof(Dtype)));

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

  if (use_avg) {
    set_gradient_nonzero_avg<Dtype>
        <<<GET_BLOCKS(nnz * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            nnz * nchannel, d_grad_out_feat, d_grad_in_feat, nchannel,
            d_num_nonzero, d_in_map, d_out_map);
  } else {
    set_gradient_nonzero<Dtype>
        <<<GET_BLOCKS(nnz * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            nnz * nchannel, d_grad_out_feat, d_grad_in_feat, nchannel, d_in_map,
            d_out_map);
  }

  cudaFree(d_in_map);
}

template void NonzeroAvgPoolingBackwardKernelGPU<float, int32_t>(
    float *d_grad_in_feat, int in_nrows, const float *d_grad_out_feat,
    int out_nrows, const float *d_num_nonzero, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, bool use_avg,
    cudaStream_t stream);

template void NonzeroAvgPoolingBackwardKernelGPU<double, int32_t>(
    double *d_grad_in_feat, int in_nrows, const double *d_grad_out_feat,
    int out_nrows, const double *d_num_nonzero, int nchannel,
    const std::vector<std::vector<int32_t>> &in_map,
    const std::vector<std::vector<int32_t>> &out_map, bool use_avg,
    cudaStream_t stream);
#endif
