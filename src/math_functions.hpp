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
void gpu_gemm(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template <typename Dtype>
void gpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

#endif
