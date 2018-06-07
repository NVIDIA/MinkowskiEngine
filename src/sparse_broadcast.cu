#ifndef GPU_BROADCAST
#define GPU_BROADCAST

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "src/math_functions.hpp"
#include "src/sparse_broadcast.cuh"

template <class T> struct IsIntType { static const bool value = false; };

template <> struct IsIntType<int> { static const bool value = true; };

template <typename Dtype, typename Itype>
__global__ void
channelwise_addition(const int n, const int nchannel, const Dtype *d_glob_feat,
                     const Itype *d_sorted_map, Dtype *d_out_feat) {
  int row, ch_index;
  CUDA_KERNEL_LOOP(index, n) {
    ch_index = index % nchannel;
    row = d_sorted_map[index / nchannel];
    d_out_feat[index] += d_glob_feat[row * nchannel + ch_index];
  }
}

template <typename Dtype, typename Itype>
__global__ void channelwise_multiplication(const int n, const int nchannel,
                                           const Dtype *d_glob_feat,
                                           const Itype *d_sorted_out_map,
                                           Dtype *d_out_feat) {
  int row, ch_index;
  CUDA_KERNEL_LOOP(index, n) {
    ch_index = index % nchannel;
    row = d_sorted_out_map[index / nchannel];
    d_out_feat[index] *= d_glob_feat[row * nchannel + ch_index];
  }
}

template <typename Dtype, typename Itype>
__global__ void
channelwise_division(const int n, const int nchannel, const Dtype *d_glob_feat,
                     const Itype *d_sorted_out_map, Dtype *d_out_feat) {
  int row, ch_index;
  CUDA_KERNEL_LOOP(index, n) {
    ch_index = index % nchannel;
    row = d_sorted_out_map[index / nchannel];
    d_out_feat[index] /= d_glob_feat[row * nchannel + ch_index];
  }
}

template <typename Dtype, typename Itype>
void SparseBroadcastForwardGPU(
    const Dtype *d_in_feat, int in_nrows, const Dtype *d_in_feat_global,
    int in_nrows_global, Dtype *d_out_feat, int nchannel, int op,
    const std::vector<std::vector<Itype>> sorted_in_map,
    const std::vector<std::vector<Itype>> sorted_out_map,
    cusparseHandle_t cushandle, cudaStream_t stream) {

  // Copy all in_feat to out_feat
  CUDA_CHECK(cudaMemcpy(d_out_feat, d_in_feat,
                        sizeof(Dtype) * nchannel * in_nrows,
                        cudaMemcpyDeviceToDevice));

  if (sorted_in_map.size() != 1)
    throw std::invalid_argument("InOut map must have one kernel for Broadcast");

  if (sorted_in_map[0].size() != in_nrows) {
    std::cout << "sorted_in_map[0].size(): " << sorted_in_map[0].size()
              << ", in_nrows: " << in_nrows << std::endl;
    throw std::invalid_argument("Invalid in_map");
  }

  thrust::device_vector<Itype> d_sorted_out_map = sorted_out_map[0];

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    channelwise_addition<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            nchannel * in_nrows, nchannel, d_in_feat_global,
            thrust::raw_pointer_cast(d_sorted_out_map.data()), d_out_feat);
    break;
  case 1: // *
    channelwise_multiplication<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows * nchannel), CUDA_NUM_THREADS, 0, stream>>>(
            nchannel * in_nrows, nchannel, d_in_feat_global,
            thrust::raw_pointer_cast(d_sorted_out_map.data()), d_out_feat);
    break;
  }
  CUDA_POST_KERNEL_CHECK;
}

template void SparseBroadcastForwardGPU<float, int32_t>(
    const float *d_in_feat, int in_nrows, const float *d_in_feat_global,
    int in_nrows_global, float *d_out_feat, int nchannel, int op,
    const std::vector<std::vector<int32_t>> orderd_in_map,
    const std::vector<std::vector<int32_t>> orderd_out_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void SparseBroadcastBackwardGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel, int op,
    const std::vector<std::vector<Itype>> sorted_in_map,
    const std::vector<std::vector<Itype>> sorted_out_map,
    cusparseHandle_t cushandle, cudaStream_t stream) {
  thrust::device_vector<Itype> d_sorted_in_map, d_sorted_out_map, d_csr_row;
  thrust::device_vector<Dtype> d_csr_val, d_tmp_grad_in_feat,
      d_tmp_grad_in_feat_global;
  cusparseMatDescr_t descr = 0;
  const Dtype alpha = 1;
  const Dtype beta = 0;
  int nnz = in_nrows;

  if (!IsIntType<Itype>::value)
    throw std::invalid_argument("Not implemented"); // Due to cusparseXcoo2csr

  if (sorted_in_map.size() != 1)
    throw std::invalid_argument("InOut map must have one kernel for Broadcast");

  if (sorted_in_map[0].size() != in_nrows)
    throw std::invalid_argument("Invalid in_map");

  d_sorted_in_map = sorted_in_map[0];    // COO cols
  d_sorted_out_map = sorted_out_map[0];  // COO rows
  d_csr_row.resize(in_nrows_global + 1); // CSR returns n_row + 1
  d_csr_val.resize(nnz);
  thrust::fill(d_csr_val.begin(), d_csr_val.end(), 1);

  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Sort COO first
  sort_coo_gpu(cushandle, in_nrows_global, in_nrows, nnz,
               thrust::raw_pointer_cast(d_sorted_out_map.data()),
               thrust::raw_pointer_cast(d_sorted_in_map.data()));
  // For CRS, sort row and col inds by row major.
  CUSPARSE_CHECK(cusparseXcoo2csr(
      cushandle, thrust::raw_pointer_cast(d_sorted_out_map.data()), nnz,
      in_nrows_global, thrust::raw_pointer_cast(d_csr_row.data()),
      CUSPARSE_INDEX_BASE_ZERO));

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    // For grad_in_feat, copy all grad_out_feat to grad_in_feat
    CUDA_CHECK(cudaMemcpy(d_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));

    // For grad_in_feat_glob, add all grad_out_feat
    d_tmp_grad_in_feat_global.resize(in_nrows_global * nchannel);
    CUSPARSE_CHECK(cusparse_csrmm<Dtype>(
        cushandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
        CUSPARSE_OPERATION_TRANSPOSE,     // op(B)
        in_nrows_global,                  // M
        nchannel,                         // N
        in_nrows,                         // K
        nnz, &alpha, descr,
        thrust::raw_pointer_cast(d_csr_val.data()),       // val
        thrust::raw_pointer_cast(d_csr_row.data()),       // row
        thrust::raw_pointer_cast(d_sorted_in_map.data()), // col
        d_grad_out_feat,                                  // B
        nchannel,                                         // ldb
        &beta,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()), // C
        in_nrows_global                                             // ldc
        ));

    col2row_major<Dtype>(
        in_nrows_global, nchannel,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()),
        d_grad_in_feat_global, stream);
    CUDA_POST_KERNEL_CHECK;
    break;
  case 1: // *
    // First, for grad_in_feat
    // Copy in_feat_global to tmp, then multiply the tmp with grad_out_feat
    d_tmp_grad_in_feat.resize(in_nrows * nchannel);
    d_tmp_grad_in_feat_global.resize(in_nrows_global * nchannel);
    row2col_major<Dtype>(
        in_nrows_global, nchannel, d_in_feat_global,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()), stream);
    CUSPARSE_CHECK(cusparse_csrmm<Dtype>(
        cushandle,
        CUSPARSE_OPERATION_TRANSPOSE,     // op(A)
        CUSPARSE_OPERATION_NON_TRANSPOSE, // op(B)
        in_nrows_global,                  // M
        nchannel,                         // N
        in_nrows,                         // K
        nnz, &alpha, descr,
        thrust::raw_pointer_cast(d_csr_val.data()),                 // val
        thrust::raw_pointer_cast(d_csr_row.data()),                 // row
        thrust::raw_pointer_cast(d_sorted_in_map.data()),           // col
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()), // B
        in_nrows_global,                                            // ldb
        &beta,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()), // C
        in_nrows                                             // ldc
        ));
    col2row_major<Dtype>(in_nrows, nchannel,
                         thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()),
                         d_grad_in_feat, stream);
    gpu_multiplication<Dtype>(nchannel * in_nrows, d_grad_out_feat,
                              d_grad_in_feat, d_grad_in_feat, stream);
    CUDA_POST_KERNEL_CHECK;

    // Second, for grad_in_feat_global, copy in_feat to tmp,
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()),
                          d_grad_out_feat, sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));
    gpu_multiplication<Dtype>(
        nchannel * in_nrows, d_in_feat,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()),
        thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()), stream);
    CUDA_POST_KERNEL_CHECK;
    CUSPARSE_CHECK(cusparse_csrmm<Dtype>(
        cushandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
        CUSPARSE_OPERATION_TRANSPOSE,     // op(B)
        in_nrows_global,                  // M
        nchannel,                         // N
        in_nrows,                         // K
        nnz, &alpha, descr,
        thrust::raw_pointer_cast(d_csr_val.data()),          // val
        thrust::raw_pointer_cast(d_csr_row.data()),          // row
        thrust::raw_pointer_cast(d_sorted_in_map.data()),    // col
        thrust::raw_pointer_cast(d_tmp_grad_in_feat.data()), // B
        nchannel,                                            // ldb
        &beta,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()), // C
        in_nrows_global                                             // ldc
        ));
    col2row_major<Dtype>(
        in_nrows_global, nchannel,
        thrust::raw_pointer_cast(d_tmp_grad_in_feat_global.data()),
        d_grad_in_feat_global, stream);
    CUDA_POST_KERNEL_CHECK;
    break;
  }

  CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));
}

template void SparseBroadcastBackwardGPU<float, int32_t>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nrows,
    const float *d_in_feat_global, float *d_grad_in_feat_global,
    int in_nrows_global, const float *d_grad_out_feat, int nchannel, int op,
    const std::vector<std::vector<int32_t>> sorted_in_map,
    const std::vector<std::vector<int32_t>> sorted_out_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

#endif
