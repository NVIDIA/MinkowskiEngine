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
#include "math_functions.hpp"

// CUBLAS, CUSPARSE assume all dense matrices to be col major
template <>
void gpu_gemm<float>(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const float *B, const float beta, float *C) {
  // Note that cublas follows (column-major) fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, N));
}

template <>
void gpu_gemm<double>(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N,
                      const int K, const double alpha, const double *A,
                      const double *B, const double beta, double *C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, N));
}

// CUBLAS, CUSPARSE assume all dense matrices to be col major
// If op(B)=B, cusparse<t>csrmm2() is the same as cusparse<t>csrmm();
// otherwise, only op(A)=A is supported and the matrix type must be
// CUSPARSE_MATRIX_TYPE_GENERAL.
// M: # row of A
// N: # col of op(B) or C
// K: # col of A
template <>
cusparseStatus_t
cusparse_csrmv<float>(cusparseHandle_t handle, cusparseOperation_t transA,
                      int m, int n, int nnz, const float *alpha,
                      const cusparseMatDescr_t descrA, const float *csrValA,
                      const int *csrRowPtrA, const int *csrColIndA,
                      const float *x, const float *beta, float *y) {
  return cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, x, beta, y);
};

template <>
cusparseStatus_t
cusparse_csrmv<double>(cusparseHandle_t handle, cusparseOperation_t transA,
                       int m, int n, int nnz, const double *alpha,
                       const cusparseMatDescr_t descrA, const double *csrValA,
                       const int *csrRowPtrA, const int *csrColIndA,
                       const double *x, const double *beta, double *y) {
  return cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, x, beta, y);
};

template <>
cusparseStatus_t
cusparse_csrmm<float>(cusparseHandle_t handle, cusparseOperation_t transA,
                      cusparseOperation_t transB, int m, int n, int k, int nnz,
                      const float *alpha, const cusparseMatDescr_t descrA,
                      const float *csrValA, const int *csrRowPtrA,
                      const int *csrColIndA, const float *B, int ldb,
                      const float *beta, float *C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t
cusparse_csrmm<double>(cusparseHandle_t handle, cusparseOperation_t transA,
                       cusparseOperation_t transB, int m, int n, int k, int nnz,
                       const double *alpha, const cusparseMatDescr_t descrA,
                       const double *csrValA, const int *csrRowPtrA,
                       const int *csrColIndA, const double *B, int ldb,
                       const double *beta, double *C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

template <typename Dtype>
__global__ void addition_kernel(const int n, const Dtype *a, const Dtype *b,
                                Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] + b[index]; }
}

template <typename Dtype>
__global__ void multiplication_kernel(const int n, const Dtype *a,
                                      const Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] * b[index]; }
}

template <typename Dtype>
void gpu_addition(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  addition_kernel<Dtype>
      <<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
}

template void gpu_addition<float>(const int N, const float *a, const float *b,
                                  float *y, cudaStream_t stream);

template void gpu_addition<double>(const int N, const double *a,
                                   const double *b, double *y,
                                   cudaStream_t stream);

template <typename Dtype>
void gpu_multiplication(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                        cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  multiplication_kernel<Dtype>
      <<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
}

template void gpu_multiplication<float>(const int N, const float *a,
                                        const float *b, float *y,
                                        cudaStream_t stream);

template void gpu_multiplication<double>(const int N, const double *a,
                                         const double *b, double *y,
                                         cudaStream_t stream);

template <typename Dtype>
__global__ void col2row_major_kernel(const int n, const int nrows,
                                     const int ncols, const Dtype *colA,
                                     Dtype *rowA) {
  int i, j;
  CUDA_KERNEL_LOOP(index, n) {
    i = index % nrows;
    j = index / nrows;
    rowA[i * ncols + j] = colA[index];
  }
}

template <typename Dtype>
void col2row_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream) {
  col2row_major_kernel<Dtype>
      <<<GET_BLOCKS(nrows * ncols), CUDA_NUM_THREADS, 0, stream>>>(
          nrows * ncols, nrows, ncols, colA, rowA);
}

template void col2row_major<float>(const int nrows, const int ncols,
                                   const float *colA, float *rowA,
                                   cudaStream_t stream);

template void col2row_major<double>(const int nrows, const int ncols,
                                    const double *colA, double *rowA,
                                    cudaStream_t stream);

template <typename Dtype>
__global__ void row2col_major_kernel(const int n, const int nrows,
                                     const int ncols, const Dtype *rowA,
                                     Dtype *colA) {
  int i, j;
  CUDA_KERNEL_LOOP(index, n) {
    i = index / ncols;
    j = index % ncols;
    colA[i + j * nrows] = rowA[index];
  }
}

template <typename Dtype>
void row2col_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream) {
  row2col_major_kernel<Dtype>
      <<<GET_BLOCKS(nrows * ncols), CUDA_NUM_THREADS, 0, stream>>>(
          nrows * ncols, nrows, ncols, colA, rowA);
}

template void row2col_major<float>(const int nrows, const int ncols,
                                   const float *colA, float *rowA,
                                   cudaStream_t stream);

template void row2col_major<double>(const int nrows, const int ncols,
                                    const double *colA, double *rowA,
                                    cudaStream_t stream);

// Sort (row, col) pairs row-major order.
void sort_coo_gpu(cusparseHandle_t handle, const int m, const int n,
                  const int nnz, int *d_coo_row, int *d_coo_col) {
  size_t pBufferSizeInBytes = 0;
  void *pBuffer = NULL;
  int *P = NULL;

  // step 1: allocate buffer
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
      handle, m, n, nnz, d_coo_row, d_coo_col, &pBufferSizeInBytes));
  CUDA_CHECK(cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes));
  // step 2: setup permutation vector P to identity
  CUDA_CHECK(cudaMalloc((void **)&P, sizeof(int) * nnz));
  CUSPARSE_CHECK(cusparseCreateIdentityPermutation(handle, nnz, P));
  // step 3: sort COO
  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, m, n, nnz, d_coo_row, d_coo_col,
                                       P, pBuffer));
  cudaFree(pBuffer);
  cudaFree(P);
}
