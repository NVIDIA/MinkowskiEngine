/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef GPU_BROADCAST
#define GPU_BROADCAST

#include "allocators.cuh"
#include "broadcast_kernel.cuh"
#include "math_functions.cuh"

namespace minkowski {

namespace detail {

template <class T> struct IsIntType { static const bool value = false; };

template <> struct IsIntType<int> { static const bool value = true; };

template <typename Dtype>
__device__ void atomic_addition_n(Dtype *__restrict__ dst,
                                  const Dtype *__restrict__ src,
                                  const int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    atomicAdd(dst + i, src[i]);
}

/* Must be applied to collision free destinations */
template <typename Dtype>
__device__ void multiplication_n(Dtype *__restrict__ dst,
                                 const Dtype *__restrict__ src,
                                 const int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    dst[i] *= src[i];
}

template <typename Dtype, typename Itype>
__global__ void channelwise_addition(const int n, const int nchannel,
                                     const Dtype *__restrict__ d_glob_feat,
                                     const Itype *__restrict__ d_in_map,
                                     const Itype *__restrict__ d_out_map,
                                     Dtype *__restrict__ d_out_feat) {
  CUDA_KERNEL_LOOP(index, n) {
    atomic_addition_n(&d_out_feat[d_in_map[index] * nchannel],
                      &d_glob_feat[d_out_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype, typename Itype>
__global__ void channelwise_multiplication(
    const int n, const int nchannel, const Dtype *__restrict__ d_glob_feat,
    const Itype *__restrict__ d_in_map, const Itype *__restrict__ d_out_map,
    Dtype *__restrict__ d_out_feat) {
  CUDA_KERNEL_LOOP(index, n) {
    multiplication_n(&d_out_feat[d_in_map[index] * nchannel],
                     &d_glob_feat[d_out_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype>
__global__ void fill(const int n, Dtype *__restrict__ in_feat,
                     const Dtype val) {
  CUDA_KERNEL_LOOP(index, n) { in_feat[index] = val; }
}

} // namespace detail

template <typename Dtype, typename Itype, typename ByteAllocator>
void BroadcastForwardKernelGPU(
    const Dtype *d_in_feat, const int in_nrows, const Dtype *d_in_feat_global,
    const int in_nrows_global, Dtype *d_out_feat, const int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream) {

  // Sum all sizes
  size_t const num_map = kernel_map.in_maps.end() - kernel_map.in_maps.begin();

  if (num_map != in_nrows)
    throw std::invalid_argument(
        "BroadcastForwardKernelGPU: kernel_map size != in_nrows");

  // Copy all in_feat to out_feat
  CUDA_CHECK(cudaMemcpy(d_out_feat, d_in_feat,
                        sizeof(Dtype) * nchannel * in_nrows,
                        cudaMemcpyDeviceToDevice));

  // To speed up, put switch outside for loops
  switch (op) {
  case BroadcastMode::ELEMENTWISE_ADDITON: // +
    detail::channelwise_addition<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0,
           stream>>>(in_nrows, nchannel, d_in_feat_global,
                     kernel_map.in_maps.begin(), kernel_map.out_maps.begin(),
                     d_out_feat);
    break;
  case BroadcastMode::ELEMENTWISE_MULTIPLICATION: // *
    detail::channelwise_multiplication<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0,
           stream>>>(in_nrows, nchannel, d_in_feat_global,
                     kernel_map.in_maps.begin(), kernel_map.out_maps.begin(),
                     d_out_feat);
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template void
BroadcastForwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    const float *d_in_feat, int in_nrows, const float *d_in_feat_global,
    int in_nrows_global, float *d_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template void
BroadcastForwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    const double *d_in_feat, int in_nrows, const double *d_in_feat_global,
    int in_nrows_global, double *d_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template void
BroadcastForwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    const float *d_in_feat, int in_nrows, const float *d_in_feat_global,
    int in_nrows_global, float *d_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template void
BroadcastForwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    const double *d_in_feat, int in_nrows, const double *d_in_feat_global,
    int in_nrows_global, double *d_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype, typename ByteAllocator>
void BroadcastBackwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream) {
  Itype *d_scr, *d_in_map, *d_out_map; //, *d_csr_row;
  Dtype *d_dtype, *d_coo_val, *d_tmp_grad_in_feat_global, *d_tmp_grad_in_feat;
  // cusparseMatDescr_t descr = 0;
  const Dtype alpha = 1;
  const Dtype beta = 0;
  int nnz = in_nrows;

  // if (in_maps.size() != 1) {
  // All in_maps[k] are contiguous.
  // TODO. Assert contiguous.
  // }

  // Sum all sizes
  size_t const num_map = kernel_map.in_maps.end() - kernel_map.in_maps.begin();

  if (num_map != in_nrows)
    throw std::invalid_argument(
        "BroadcastBackwardKernelGPU: kernel_map size != in_nrows");

  /* In Out Map prep */
  // Malloc d_in_map, d_out_map, d_csr_row
  // CSR returns n_row + 1
  CUDA_CHECK(cudaMalloc((void **)&d_scr,
                        2 * nnz * sizeof(Itype) +                 // in out maps
                            (in_nrows_global + 1) * sizeof(Itype) // d_csr_row
                        ));

  // COO cols
  d_in_map = d_scr; // nnz
  // COO rows
  d_out_map = d_scr + nnz; // nnz
  // CSR row indices
  // d_csr_row = d_scr + 2 * nnz; // in_nrows_global + 1

  CUDA_CHECK(cudaMemcpy(
      d_in_map,
      (int *)kernel_map.in_maps.begin(), // in_maps are contiguous of size nnz
      nnz * sizeof(int), cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaMemcpy(
      d_out_map,
      (int *)kernel_map.out_maps.begin(), // out_maps are contiguous of size nnz
      nnz * sizeof(int), cudaMemcpyDeviceToDevice));

  /* tmp in out feat */
  // sparse gemm output
  CUDA_CHECK(cudaMalloc(
      (void **)&d_dtype,
      nnz * sizeof(Dtype) +                          // d_csr_val
          in_nrows * nchannel * sizeof(Dtype) +      // tmp_grad_infeat
          in_nrows_global * nchannel * sizeof(Dtype) // tmp_grad_infeat_global
      ));

  // Divide the memory space into multiple chunks
  d_tmp_grad_in_feat_global = d_dtype; // in_nrows_global * nchannel
  d_tmp_grad_in_feat = d_tmp_grad_in_feat_global +
                       in_nrows_global * nchannel; // in_nrows * nchannel
  d_coo_val = d_tmp_grad_in_feat + in_nrows * nchannel;

  // thrust::fill(d_csr_val.begin(), d_csr_val.end(), 1);
  detail::fill<Dtype>
      <<<GET_BLOCKS(nnz, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
          nnz, d_coo_val, (Dtype)1.);

  // CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  // cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  // cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Sort COO first
  THRUST_CHECK(thrust::sort_by_key(thrust::device,  //
                                   d_out_map,       // key begin
                                   d_out_map + nnz, // key end
                                   d_in_map         // value begin
                                   ));

  cusparseSpMMAlg_t mm_alg;
#if defined(CUDART_VERSION) && (CUDART_VERSION < 10010)
  TORCH_CHECK(false, "spmm sparse-dense requires CUDA 10.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010) &&                  \
    (CUDART_VERSION < 11000)
  mm_alg = CUSPARSE_MM_ALG_DEFAULT;
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
  mm_alg = CUSPARSE_SPMM_ALG_DEFAULT;
#endif

  //  +---------+ +---+
  //  | spm     | | i |
  //  +---------+ | n |
  //    in_nrows  |   |
  //              | F |
  //              |   |
  //              +---+
  //             nchannel
  size_t dim_i = in_nrows_global, dim_j = in_nrows, dim_k = nchannel;
  constexpr bool is_float32 = std::is_same<Dtype, float>::value;
  cudaDataType cuda_data_type = is_float32 ? CUDA_R_32F : CUDA_R_64F;
  cusparseSpMatDescr_t sparse_descr;
  cusparseDnMatDescr_t dense_descr;
  cusparseDnMatDescr_t result_descr;
  CUSPARSE_CHECK(cusparseCreateCoo(&sparse_descr,     //
                                   dim_i, dim_j, nnz, //
                                   d_out_map,         // rows
                                   d_in_map,          // cols
                                   d_coo_val,         // coo vals
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   cuda_data_type));

  // buffer size 0 for CUSPARSE_SPMM_COO_ALG1, CUSPARSE_SPMM_COO_ALG3,
  // CUSPARSE_SPMM_COO_ALG4, and CUSPARSE_SPMM_CSR_ALG1

  // To speed up, put switch outside for loops
  switch (op) {
  case BroadcastMode::ELEMENTWISE_ADDITON: // +
    // For grad_in_feat, copy all grad_out_feat to grad_in_feat
    CUDA_CHECK(cudaMemcpy(d_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));

    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_descr,            //
                                       dim_k, dim_j, dim_k,     //
                                       (void *)d_grad_out_feat, //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(&result_descr,                     //
                                       dim_i, dim_k, dim_i,               //
                                       (void *)d_tmp_grad_in_feat_global, //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    // Transpose the output
    // WARNING: coo sorting must have been handled in the kernel map
    // decomposition.
    CUSPARSE_CHECK(cusparseSpMM(cushandle,                        //
                                CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                CUSPARSE_OPERATION_TRANSPOSE,     //
                                (void *)&alpha,                   //
                                sparse_descr, dense_descr,        //
                                (void *)&beta, result_descr,      //
                                cuda_data_type, mm_alg, 0));

    // For grad_in_feat_glob, add all grad_out_feat
    /*
    CUSPARSE_CHECK(
        cusparse_csrmm<Dtype>(cushandle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
                              CUSPARSE_OPERATION_TRANSPOSE,     // op(B)
                              in_nrows_global,                  // M
                              nchannel,                         // N
                              in_nrows,                         // K
                              nnz, &alpha, descr,
                              d_csr_val,       // val
                              d_csr_row,       // row
                              d_in_map,        // col
                              d_grad_out_feat, // B
                              nchannel,        // ldb
                              &beta,
                              d_tmp_grad_in_feat_global, // C
                              in_nrows_global            // ldc
                              ));
    */

    col2row_major<Dtype>(in_nrows_global, nchannel, d_tmp_grad_in_feat_global,
                         d_grad_in_feat_global, stream);

    break;
  case BroadcastMode::ELEMENTWISE_MULTIPLICATION: // *
    // Second, for grad_in_feat_global, copy in_feat to tmp,

    // Forward : (A^T(sparse) x global(dense)) (*) B(feat) = C(result)
    // grad global : A(sparse) (grad C (*) B) =
    CUDA_CHECK(cudaMemcpy(d_tmp_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));
    gpu_multiplication<Dtype>(nchannel * in_nrows, d_in_feat,
                              d_tmp_grad_in_feat, d_tmp_grad_in_feat, stream);

    CUSPARSE_CHECK(cusparseCreateDnMat(&dense_descr,               //
                                       dim_k, dim_j, dim_k,        //
                                       (void *)d_tmp_grad_in_feat, //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(&result_descr,                     //
                                       dim_i, dim_k, dim_i,               //
                                       (void *)d_tmp_grad_in_feat_global, //
                                       cuda_data_type, CUSPARSE_ORDER_COL));

    // Transpose the output
    CUSPARSE_CHECK(cusparseSpMM(cushandle,                        //
                                CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                CUSPARSE_OPERATION_TRANSPOSE,     //
                                (void *)&alpha,                   //
                                sparse_descr, dense_descr,        //
                                (void *)&beta, result_descr,      //
                                cuda_data_type, mm_alg, 0));

    /*CUSPARSE_CHECK(
        cusparse_csrmm<Dtype>(cushandle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // op(A)
                              CUSPARSE_OPERATION_TRANSPOSE,     // op(B)
                              in_nrows_global,                  // M
                              nchannel,                         // N
                              in_nrows,                         // K
                              nnz, &alpha, descr,
                              d_csr_val,          // val
                              d_csr_row,          // row
                              d_in_map,           // col
                              d_tmp_grad_in_feat, // B
                              nchannel,           // ldb
                              &beta,
                              d_tmp_grad_in_feat_global, // C
                              in_nrows_global            // ldc
                              ));
     */
    col2row_major<Dtype>(in_nrows_global, nchannel, d_tmp_grad_in_feat_global,
                         d_grad_in_feat_global, stream);

    // First, for grad_in_feat
    // Copy in_feat_global to tmp, then multiply the tmp with grad_out_feat

    // Forward : (A^T(sparse) x global(dense)) (*) B(feat) = C(result)
    // grad feat : A^T(sparse) x global(dense) (*) grad C

    // Sort COO first
    // sort_coo_gpu(cushandle, in_nrows_global, in_nrows, nnz, d_out_map,
    // d_in_map);
    // cusparseSpMatDescr_t sparse_descr2;
    CUDA_CHECK(cudaMemcpy(d_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));

    detail::channelwise_multiplication<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0,
           stream>>>(in_nrows, nchannel, d_in_feat_global,
                     kernel_map.in_maps.begin(), kernel_map.out_maps.begin(),
                     d_grad_in_feat);

    /*
    row2col_major<Dtype>(in_nrows_global, nchannel, d_in_feat_global,
                         d_tmp_grad_in_feat_global, stream);
    CUSPARSE_CHECK(
        cusparse_csrmm<Dtype>(cushandle,
                              CUSPARSE_OPERATION_TRANSPOSE,     // op(A)
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // op(B)
                              in_nrows_global,                  // M
                              nchannel,                         // N
                              in_nrows,                         // K
                              nnz, &alpha, descr,
                              d_csr_val,                 // val
                              d_csr_row,                 // row
                              d_in_map,                  // col
                              d_tmp_grad_in_feat_global, // B
                              in_nrows_global,           // ldb
                              &beta,
                              d_tmp_grad_in_feat, // C
                              in_nrows            // ldc
                              ));
    col2row_major<Dtype>(in_nrows, nchannel, d_tmp_grad_in_feat, d_grad_in_feat,
                         stream);
    gpu_multiplication<Dtype>(nchannel * in_nrows, d_grad_out_feat,
                              d_grad_in_feat, d_grad_in_feat, stream);
    */

    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }

  cudaFree(d_scr);
  cudaFree(d_dtype);

  CUSPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(result_descr));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template void
BroadcastBackwardKernelGPU<float, uint32_t, detail::default_allocator<char>>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nrows,
    const float *d_in_feat_global, float *d_grad_in_feat_global,
    int in_nrows_global, const float *d_grad_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

template void
BroadcastBackwardKernelGPU<double, uint32_t, detail::default_allocator<char>>(
    const double *d_in_feat, double *d_grad_in_feat, int in_nrows,
    const double *d_in_feat_global, double *d_grad_in_feat_global,
    int in_nrows_global, const double *d_grad_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

template void
BroadcastBackwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nrows,
    const float *d_in_feat_global, float *d_grad_in_feat_global,
    int in_nrows_global, const float *d_grad_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

template void
BroadcastBackwardKernelGPU<double, uint32_t, detail::c10_allocator<char>>(
    const double *d_in_feat, double *d_grad_in_feat, int in_nrows,
    const double *d_in_feat_global, double *d_grad_in_feat_global,
    int in_nrows_global, const double *d_grad_out_feat, int nchannel,
    BroadcastMode::Type const op,
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

} // namespace minkowski

#endif // GPU_BROADCAST
