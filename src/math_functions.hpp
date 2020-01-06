/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#ifndef MATH_FUNCTIONS
#define MATH_FUNCTIONS

#ifndef CPU_ONLY
#include "gpu.cuh"
#endif
#include "mkl_alternate.hpp"

namespace minkowski {

template <typename Dtype>
void cpu_gemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template <typename Dtype>
void cpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void cpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void cpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

#ifndef CPU_ONLY
template <typename Dtype>
void gpu_gemm(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template <typename Dtype>
void gpu_addition(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                  cudaStream_t stream);

template <typename Dtype>
void gpu_multiplication(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                        cudaStream_t stream);

template <typename Dtype>
void col2row_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream);

template <typename Dtype>
void row2col_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream);

template <typename Dtype>
cusparseStatus_t
cusparse_csrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m,
               int n, int nnz, const Dtype *alpha,
               const cusparseMatDescr_t descrA, const Dtype *csrValA,
               const int *csrRowPtrA, const int *csrColIndA, const Dtype *x,
               const Dtype *beta, Dtype *y);

template <typename Dtype>
cusparseStatus_t
cusparse_csrmm(cusparseHandle_t handle, cusparseOperation_t transA,
               cusparseOperation_t transB, int m, int n, int k, int nnz,
               const Dtype *alpha, const cusparseMatDescr_t descrA,
               const Dtype *csrValA, const int *csrRowPtrA,
               const int *csrColIndA, const Dtype *B, int ldb,
               const Dtype *beta, Dtype *C, int ldc);

void sort_coo_gpu(cusparseHandle_t handle, const int m, const int n,
                  const int nnz, int *d_coo_row, int *d_coo_col);
#endif // not CPU_ONLY

} // end namespace minkowski

#endif // MATH_FUNCTIONS
