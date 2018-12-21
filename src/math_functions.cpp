#include "math_functions.hpp"

template <>
void cpu_gemm<float>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const float *B, const float beta, float *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_gemm<double>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N,
                      const int K, const double alpha, const double *A,
                      const double *B, const double beta, double *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_add<float>(const int n, const float *a, const float *b, float *y) {
  vsAdd(n, a, b, y);
}

template <>
void cpu_add<double>(const int n, const double *a, const double *b, double *y) {
  vdAdd(n, a, b, y);
}

template <>
void cpu_mul<float>(const int n, const float *a, const float *b, float *y) {
  vsMul(n, a, b, y);
}

template <>
void cpu_mul<double>(const int n, const double *a, const double *b, double *y) {
  vdMul(n, a, b, y);
}

template <>
void cpu_div<float>(const int n, const float *a, const float *b, float *y) {
  vsDiv(n, a, b, y);
}

template <>
void cpu_div<double>(const int n, const double *a, const double *b, double *y) {
  vdMul(n, a, b, y);
}
