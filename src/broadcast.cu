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
#ifndef GPU_BROADCAST
#define GPU_BROADCAST

#include "broadcast.cuh"
#include "math_functions.hpp"

template <class T> struct IsIntType { static const bool value = false; };

template <> struct IsIntType<int> { static const bool value = true; };

template <typename Dtype>
__device__ void atomic_addition_n(Dtype *dst, const Dtype *src,
                                  int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    atomicAdd(dst + i, src[i]);
}

/* Must be applied to collision free destinations */
template <typename Dtype>
__device__ void multiplication_n(Dtype *dst, const Dtype *src,
                                 int num_elements) {
  for (int i = 0; i < num_elements; ++i)
    dst[i] *= src[i];
}

template <typename Dtype, typename Itype>
__global__ void
channelwise_addition(const int n, const int nchannel, const Dtype *d_glob_feat,
                     const Itype *d_in_map, const Itype *d_out_map,
                     Dtype *d_out_feat) {
  CUDA_KERNEL_LOOP(index, n) {
    atomic_addition_n(&d_out_feat[d_in_map[index] * nchannel],
                      &d_glob_feat[d_out_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype, typename Itype>
__global__ void
channelwise_multiplication(const int n, const int nchannel,
                           const Dtype *d_glob_feat, const Itype *d_in_map,
                           const Itype *d_out_map, Dtype *d_out_feat) {
  CUDA_KERNEL_LOOP(index, n) {
    multiplication_n(&d_out_feat[d_in_map[index] * nchannel],
                     &d_glob_feat[d_out_map[index] * nchannel], nchannel);
  }
}

template <typename Dtype>
__global__ void fill(const int n, Dtype *in_feat, Dtype val) {
  CUDA_KERNEL_LOOP(index, n) { in_feat[index] = val; }
}

template <typename Dtype, typename Itype>
void BroadcastForwardKernelGPU(
    const Dtype *d_in_feat, int in_nrows, const Dtype *d_in_feat_global,
    int in_nrows_global, Dtype *d_out_feat, int nchannel, int op,
    const pInOutMaps<Itype> &in_maps, const pInOutMaps<Itype> &out_maps,
    cusparseHandle_t cushandle, cudaStream_t stream) {

  // Sum all sizes
  int num_map = 0;
  for (const auto &in_map : in_maps)
    num_map += in_map.size();
  if (num_map != in_nrows)
    throw std::invalid_argument("Invalid in_map");

  // Copy all in_feat to out_feat
  CUDA_CHECK(cudaMemcpy(d_out_feat, d_in_feat,
                        sizeof(Dtype) * nchannel * in_nrows,
                        cudaMemcpyDeviceToDevice));

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    channelwise_addition<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows), CUDA_NUM_THREADS, 0, stream>>>(
            in_nrows, nchannel, d_in_feat_global, in_maps[0].data(),
            out_maps[0].data(), d_out_feat);
    break;
  case 1: // *
    channelwise_multiplication<Dtype, Itype>
        <<<GET_BLOCKS(in_nrows), CUDA_NUM_THREADS, 0, stream>>>(
            in_nrows, nchannel, d_in_feat_global, in_maps[0].data(),
            out_maps[0].data(), d_out_feat);
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template void BroadcastForwardKernelGPU<float, int32_t>(
    const float *d_in_feat, int in_nrows, const float *d_in_feat_global,
    int in_nrows_global, float *d_out_feat, int nchannel, int op,
    const pInOutMaps<int32_t> &in_map, const pInOutMaps<int32_t> &out_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template void BroadcastForwardKernelGPU<double, int32_t>(
    const double *d_in_feat, int in_nrows, const double *d_in_feat_global,
    int in_nrows_global, double *d_out_feat, int nchannel, int op,
    const pInOutMaps<int32_t> &in_map, const pInOutMaps<int32_t> &out_map,
    cusparseHandle_t cuhandle, cudaStream_t stream);

template <typename Dtype, typename Itype>
void BroadcastBackwardKernelGPU(
    const Dtype *d_in_feat, Dtype *d_grad_in_feat, int in_nrows,
    const Dtype *d_in_feat_global, Dtype *d_grad_in_feat_global,
    int in_nrows_global, const Dtype *d_grad_out_feat, int nchannel, int op,
    const pInOutMaps<Itype> &in_maps, const pInOutMaps<Itype> &out_maps,
    cusparseHandle_t cushandle, cudaStream_t stream) {
  Itype *d_scr, *d_in_map, *d_out_map, *d_csr_row;
  Dtype *d_dtype, *d_csr_val, *d_tmp_grad_in_feat_global, *d_tmp_grad_in_feat;
  cusparseMatDescr_t descr = 0;
  const Dtype alpha = 1;
  const Dtype beta = 0;
  int nnz = in_nrows;

  if (!IsIntType<Itype>::value)
    throw std::invalid_argument("Not implemented"); // Due to cusparseXcoo2csr

  // if (in_maps.size() != 1) {
  // All in_maps[k] are contiguous.
  // TODO. Assert contiguous.
  // }

  // Sum all sizes
  int num_map = 0;
  for (const auto &in_map : in_maps)
    num_map += in_map.size();
  if (num_map != in_nrows)
    throw std::invalid_argument("Invalid in_map");

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
  d_csr_row = d_scr + 2 * nnz; // in_nrows_global + 1

  CUDA_CHECK(cudaMemcpy(d_in_map,
                        in_maps[0].data(), // in_maps are contiguous of size nnz
                        nnz * sizeof(int), cudaMemcpyDeviceToDevice));

  CUDA_CHECK(
      cudaMemcpy(d_out_map,
                 out_maps[0].data(), // out_maps are contiguous of size nnz
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
  d_csr_val = d_tmp_grad_in_feat + in_nrows * nchannel;

  // thrust::fill(d_csr_val.begin(), d_csr_val.end(), 1);
  fill<Dtype><<<GET_BLOCKS(nnz), CUDA_NUM_THREADS, 0, stream>>>(nnz, d_csr_val,
                                                                (Dtype)1.);

  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Sort COO first
  sort_coo_gpu(cushandle, in_nrows_global, in_nrows, nnz, d_out_map, d_in_map);
  // For CSR, sort row and col inds by row major.
  CUSPARSE_CHECK(cusparseXcoo2csr(cushandle, d_out_map, nnz, in_nrows_global,
                                  d_csr_row, CUSPARSE_INDEX_BASE_ZERO));

  // To speed up, put switch outside for loops
  switch (op) {
  case 0: // +
    // For grad_in_feat, copy all grad_out_feat to grad_in_feat
    CUDA_CHECK(cudaMemcpy(d_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));
    // For grad_in_feat_glob, add all grad_out_feat
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

    col2row_major<Dtype>(in_nrows_global, nchannel, d_tmp_grad_in_feat_global,
                         d_grad_in_feat_global, stream);
    break;
  case 1: // *
    // First, for grad_in_feat
    // Copy in_feat_global to tmp, then multiply the tmp with grad_out_feat
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

    // Second, for grad_in_feat_global, copy in_feat to tmp,
    CUDA_CHECK(cudaMemcpy(d_tmp_grad_in_feat, d_grad_out_feat,
                          sizeof(Dtype) * nchannel * in_nrows,
                          cudaMemcpyDeviceToDevice));
    gpu_multiplication<Dtype>(nchannel * in_nrows, d_in_feat,
                              d_tmp_grad_in_feat, d_tmp_grad_in_feat, stream);
    CUSPARSE_CHECK(
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
    col2row_major<Dtype>(in_nrows_global, nchannel, d_tmp_grad_in_feat_global,
                         d_grad_in_feat_global, stream);
    break;
  default:
    throw std::invalid_argument(Formatter() << "Operation not supported: "
                                            << std::to_string(op));
  }

  cudaFree(d_scr);
  cudaFree(d_dtype);
  CUSPARSE_CHECK(cusparseDestroyMatDescr(descr));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template void BroadcastBackwardKernelGPU<float, int32_t>(
    const float *d_in_feat, float *d_grad_in_feat, int in_nrows,
    const float *d_in_feat_global, float *d_grad_in_feat_global,
    int in_nrows_global, const float *d_grad_out_feat, int nchannel, int op,
    const pInOutMaps<int32_t> &in_map, const pInOutMaps<int32_t> &out_map,
    cusparseHandle_t cushandle, cudaStream_t stream);

template void BroadcastBackwardKernelGPU<double, int32_t>(
    const double *d_in_feat, double *d_grad_in_feat, int in_nrows,
    const double *d_in_feat_global, double *d_grad_in_feat_global,
    int in_nrows_global, const double *d_grad_out_feat, int nchannel, int op,
    const pInOutMaps<int32_t> &in_map, const pInOutMaps<int32_t> &out_map,
    cusparseHandle_t cushandle, cudaStream_t stream);
#endif
