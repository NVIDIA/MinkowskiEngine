/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "math_functions.hpp"

namespace minkowski {

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

template <>
void cpu_axpy<float>(const int N, const float alpha, const float *X, float *Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void cpu_axpy<double>(const int N, const double alpha, const double *X,
                      double *Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

} // end namespace minkowski
