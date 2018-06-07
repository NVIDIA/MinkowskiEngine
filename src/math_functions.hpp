#ifndef MATH_FUNCTIONS
#define MATH_FUNCTIONS

#include "src/gpu.cuh"
#include "src/mkl_alternate.hpp"

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
#endif
