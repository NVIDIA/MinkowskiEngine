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
#ifndef GPU_POOLING_AVG
#define GPU_POOLING_AVG

#include <cusparse.h>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "allocators.cuh"
#include "pooling_avg_kernel.cuh"
#include "utils.hpp"

namespace minkowski {

template <typename Dtype>
__global__ void fill(const int n, Dtype *in_feat, Dtype val) {
  CUDA_KERNEL_LOOP(index, n) { in_feat[index] = val; }
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

template <typename Dtype, typename Itype, typename ByteAllocator>
void NonzeroAvgPoolingForwardKernelGPU(
    Dtype const *d_in_feat,                                 //
    default_types::size_type const in_nrows,                //
    Dtype *d_out_feat,                                      //
    default_types::size_type const out_nrows,               //
    Dtype *d_num_nonzero,                                   //
    default_types::size_type const nchannel,                //
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map, //
    bool const use_avg,                                     //
    ByteAllocator &allocator,                               //
    cusparseHandle_t cushandle, cudaStream_t stream) {
  const Dtype alpha = 1;
  const Dtype beta = 0;
  static_assert(sizeof(Itype) == sizeof(int),
                "cusparse requires int type index");
  Dtype *d_ones, *d_coo_val, *d_tmp_out_feat;

  constexpr bool is_int32 = sizeof(Itype) == sizeof(int32_t);
  constexpr bool is_int64 = sizeof(Itype) == sizeof(int64_t);
  constexpr bool is_float32 = std::is_same<Dtype, float>::value;
  cudaDataType cuda_data_type = is_float32 ? CUDA_R_32F : CUDA_R_64F;

  cusparseSpMMAlg_t mm_alg;
#if defined(CUDART_VERSION) && (CUDART_VERSION < 10010)
  ASSERT(false, "spmm sparse-dense requires CUDA 10.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 10010) &&                  \
    (CUDART_VERSION < 11010)
  mm_alg = CUSPARSE_COOMM_ALG1;
  static_assert(is_int32, "int64 cusparseSpMM requires CUDA 11.1 or greater");
#elif defined(CUDART_VERSION) && (CUDART_VERSION >= 11010)
  mm_alg = CUSPARSE_SPMM_COO_ALG1;
  static_assert(is_int32 || is_int64, "Invalid index type");
#endif

  /* sparse mm prep */
  size_t const sparse_nnzs =
      kernel_map.in_maps.end() - kernel_map.in_maps.begin();
  size_t one_vector_size = ((use_avg ? in_nrows : 0) // in_nrows vector
                            + sparse_nnzs            // coo vals
                            + nchannel * out_nrows   // out tmp
                            ) *
                           sizeof(Dtype);
  static_assert(is_int32, "sort_coo supports int32");
  sort_coo_gpu<ByteAllocator>(cushandle, out_nrows, in_nrows, sparse_nnzs,
                              (int *)kernel_map.out_maps.begin(),
                              (int *)kernel_map.in_maps.begin(), allocator);

  // one vector.
  d_ones = (Dtype *)allocator.allocate(one_vector_size);

  if (use_avg) {
    d_ones = d_ones;                          // in_nrows;
    d_coo_val = d_ones + in_nrows;            // sparse_nnzs
    d_tmp_out_feat = d_coo_val + sparse_nnzs; // nchannel * out_nrows
    fill<Dtype><<<GET_BLOCKS(in_nrows + sparse_nnzs, CUDA_NUM_THREADS),
                  CUDA_NUM_THREADS, 0, stream>>>(in_nrows + sparse_nnzs, d_ones,
                                                 (Dtype)1.);
  } else {
    d_coo_val = d_ones;                       // sparse_nnzs
    d_tmp_out_feat = d_coo_val + sparse_nnzs; // nchannel * out_nrows
    fill<Dtype><<<GET_BLOCKS(sparse_nnzs, CUDA_NUM_THREADS), CUDA_NUM_THREADS,
                  0, stream>>>(sparse_nnzs, d_coo_val, (Dtype)1.);
  }

#ifdef DEBUG
  std::cout << "sparse_nnzs: " << sparse_nnzs << "\n";
  Itype *p_scr = (Itype *)std::malloc((sparse_nnzs)*2 * sizeof(Itype));
  CUDA_CHECK(cudaMemcpy(p_scr, kernel_map.out_maps.begin(),
                        sparse_nnzs * sizeof(Itype), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(p_scr + sparse_nnzs, kernel_map.in_maps.begin(),
                        sparse_nnzs * sizeof(Itype), cudaMemcpyDeviceToHost));

  Itype step = std::max<Itype>(sparse_nnzs / 100, 1);
  Itype i = 0;
  for (; i < sparse_nnzs;) {
    std::cout << i;
    std::cout << " out_map: " << p_scr[i]
              << ", in_map: " << p_scr[i + sparse_nnzs] << "\n";
    i += step;
  }
  i -= step;
  for (; i < sparse_nnzs; ++i) {
    std::cout << i;
    std::cout << " out_map: " << p_scr[i]
              << ", in_map: " << p_scr[i + sparse_nnzs] << "\n";
  }
  std::free(p_scr);
  std::cout << "done printing\n";
#endif

  //  +---------+ +---+
  //  | spm     | | i |
  //  +---------+ | n |
  //    in_nrows  |   |
  //              | F |
  //              |   |
  //              +---+
  //             nchannel
  size_t dim_i = out_nrows, dim_j = in_nrows, dim_k = nchannel;
  cusparseSpMatDescr_t sparse_descr;
  cusparseDnMatDescr_t dense_descr;
  cusparseDnMatDescr_t result_descr;
  CUSPARSE_CHECK(
      cusparseCreateCoo(&sparse_descr,               //
                        dim_i, dim_j, sparse_nnzs,   //
                        kernel_map.out_maps.begin(), // rows
                        kernel_map.in_maps.begin(),  // cols
                        d_coo_val,                   // coo vals
                        is_int32 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I,
                        CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));

  CUSPARSE_CHECK(cusparseCreateDnMat(&dense_descr,        //
                                     dim_k, dim_j, dim_k, //
                                     (void *)d_in_feat,   //
                                     cuda_data_type, CUSPARSE_ORDER_COL));

  CUSPARSE_CHECK(cusparseCreateDnMat(&result_descr,          //
                                     dim_i, dim_k, dim_i,    //
                                     (void *)d_tmp_out_feat, //
                                     cuda_data_type, CUSPARSE_ORDER_COL));

  // buffer size 0 for CUSPARSE_SPMM_COO_ALG1, CUSPARSE_SPMM_COO_ALG3,
  // CUSPARSE_SPMM_COO_ALG4, and CUSPARSE_SPMM_CSR_ALG1

  // WARNING: coo sorting must have been handled in the kernel map
  // decomposition.
  CUSPARSE_CHECK(cusparseSpMM(cushandle,                        //
                              CUSPARSE_OPERATION_NON_TRANSPOSE, //
                              CUSPARSE_OPERATION_TRANSPOSE,     //
                              (void *)&alpha,                   //
                              sparse_descr, dense_descr,        //
                              (void *)&beta, result_descr,      //
                              cuda_data_type, mm_alg, 0));
#ifdef DEBUG
  CUDA_CHECK(cudaStreamSynchronize(0));
#endif
  LOG_DEBUG("SPMM");

  if (use_avg) {
    cusparseDnVecDescr_t vecX, vecY;
    // Create dense vector X
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&vecX, in_nrows, d_ones, cuda_data_type));
    // Create dense vector y
    CUSPARSE_CHECK(
        cusparseCreateDnVec(&vecY, out_nrows, d_num_nonzero, cuda_data_type));

    CUSPARSE_CHECK(cusparseSpMV(cushandle,                        //
                                CUSPARSE_OPERATION_NON_TRANSPOSE, //
                                (void *)&alpha,                   //
                                sparse_descr, vecX,               //
                                (void *)&beta, vecY,              //
                                cuda_data_type, CUSPARSE_COOMV_ALG, nullptr));
#ifdef DEBUG
    CUDA_CHECK(cudaStreamSynchronize(0));
#endif
    LOG_DEBUG("SPMV");

    col2row_major_with_div<Dtype>
        <<<GET_BLOCKS(out_nrows * nchannel, CUDA_NUM_THREADS), CUDA_NUM_THREADS,
           0, stream>>>(out_nrows * nchannel, out_nrows, nchannel,
                        d_num_nonzero, d_tmp_out_feat, d_out_feat);
#ifdef DEBUG
    CUDA_CHECK(cudaStreamSynchronize(0));
#endif
    LOG_DEBUG("col2row");

    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
  } else {
    col2row_major<Dtype><<<GET_BLOCKS(out_nrows * nchannel, CUDA_NUM_THREADS),
                           CUDA_NUM_THREADS, 0, stream>>>(
        out_nrows * nchannel, out_nrows, nchannel, d_tmp_out_feat, d_out_feat);
  }

  CUSPARSE_CHECK(cusparseDestroySpMat(sparse_descr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(dense_descr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(result_descr));

  allocator.deallocate((char *)d_ones, one_vector_size);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

// default_allocator
template void
NonzeroAvgPoolingForwardKernelGPU<float, uint32_t,
                                  detail::default_allocator<char>>(
    float const *d_in_feat,                   //
    default_types::size_type const in_nrows,  //
    float *d_out_feat,                        //
    default_types::size_type const out_nrows, //
    float *d_num_nonzero,                     //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map, //
    bool const use_avg,
    detail::default_allocator<char> &allocator, //
    cusparseHandle_t cushandle, cudaStream_t stream);

template void
NonzeroAvgPoolingForwardKernelGPU<double, uint32_t,
                                  detail::default_allocator<char>>(
    double const *d_in_feat,                  //
    default_types::size_type const in_nrows,  //
    double *d_out_feat,                       //
    default_types::size_type const out_nrows, //
    double *d_num_nonzero,                    //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const
        &kernel_map, //
    bool const use_avg,
    detail::default_allocator<char> &allocator, //
    cusparseHandle_t cushandle, cudaStream_t stream);

// c10_allocator
template void
NonzeroAvgPoolingForwardKernelGPU<float, uint32_t, detail::c10_allocator<char>>(
    float const *d_in_feat,                                                  //
    default_types::size_type const in_nrows,                                 //
    float *d_out_feat,                                                       //
    default_types::size_type const out_nrows,                                //
    float *d_num_nonzero,                                                    //
    default_types::size_type const nchannel,                                 //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    bool const use_avg,
    detail::c10_allocator<char> &allocator, //
    cusparseHandle_t cushandle, cudaStream_t stream);

template void NonzeroAvgPoolingForwardKernelGPU<double, uint32_t,
                                                detail::c10_allocator<char>>(
    double const *d_in_feat,                                                 //
    default_types::size_type const in_nrows,                                 //
    double *d_out_feat,                                                      //
    default_types::size_type const out_nrows,                                //
    double *d_num_nonzero,                                                   //
    default_types::size_type const nchannel,                                 //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map, //
    bool const use_avg,
    detail::c10_allocator<char> &allocator, //
    cusparseHandle_t cushandle, cudaStream_t stream);

// Backward
template <typename Dtype, typename Itype, typename ByteAllocator>
void NonzeroAvgPoolingBackwardKernelGPU(
    Dtype *d_grad_in_feat,                    //
    default_types::size_type const in_nrows,  //
    Dtype const *d_grad_out_feat,             //
    default_types::size_type const out_nrows, //
    Dtype const *d_num_nonzero,               //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<Itype, ByteAllocator> const &kernel_map, bool const use_avg,
    cudaStream_t stream) {
  // d_grad_in_feat must be all set to 0

  size_t sparse_nnzs = kernel_map.in_maps.end() - kernel_map.in_maps.begin();

  if (use_avg) {
    set_gradient_nonzero_avg<Dtype>
        <<<GET_BLOCKS(sparse_nnzs * nchannel, CUDA_NUM_THREADS),
           CUDA_NUM_THREADS, 0, stream>>>(
            sparse_nnzs * nchannel, d_grad_out_feat, d_grad_in_feat, nchannel,
            d_num_nonzero, kernel_map.in_maps.cdata(),
            kernel_map.out_maps.cdata());
  } else {
    set_gradient_nonzero<Dtype>
        <<<GET_BLOCKS(sparse_nnzs * nchannel, CUDA_NUM_THREADS),
           CUDA_NUM_THREADS, 0, stream>>>(
            sparse_nnzs * nchannel, d_grad_out_feat, d_grad_in_feat, nchannel,
            kernel_map.in_maps.cdata(), kernel_map.out_maps.cdata());
  }

  CUDA_CHECK(cudaDeviceSynchronize());
}

// default_allocator
template void
NonzeroAvgPoolingBackwardKernelGPU<float, uint32_t,
                                   detail::default_allocator<char>>(
    float *d_grad_in_feat,                    //
    default_types::size_type const in_nrows,  //
    float const *d_grad_out_feat,             //
    default_types::size_type const out_nrows, //
    float const *d_num_nonzero,               //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    bool const use_avg, cudaStream_t stream);

template void
NonzeroAvgPoolingBackwardKernelGPU<double, uint32_t,
                                   detail::default_allocator<char>>(
    double *d_grad_in_feat,                   //
    default_types::size_type const in_nrows,  //
    double const *d_grad_out_feat,            //
    default_types::size_type const out_nrows, //
    double const *d_num_nonzero,              //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::default_allocator<char>> const &kernel_map,
    bool const use_avg, cudaStream_t stream);

// c10_allocator
template void NonzeroAvgPoolingBackwardKernelGPU<float, uint32_t,
                                                 detail::c10_allocator<char>>(
    float *d_grad_in_feat,                    //
    default_types::size_type const in_nrows,  //
    float const *d_grad_out_feat,             //
    default_types::size_type const out_nrows, //
    float const *d_num_nonzero,               //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    bool const use_avg, cudaStream_t stream);

template void NonzeroAvgPoolingBackwardKernelGPU<double, uint32_t,
                                                 detail::c10_allocator<char>>(
    double *d_grad_in_feat,                   //
    default_types::size_type const in_nrows,  //
    double const *d_grad_out_feat,            //
    default_types::size_type const out_nrows, //
    double const *d_num_nonzero,              //
    default_types::size_type const nchannel,  //
    gpu_kernel_map<uint32_t, detail::c10_allocator<char>> const &kernel_map,
    bool const use_avg, cudaStream_t stream);

} // end namespace minkowski

#endif // end GPU_POOLING_AVG
