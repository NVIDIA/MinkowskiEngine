#ifndef GPU_H_
#define GPU_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <driver_types.h> // cuda driver types

#include <thrust/device_vector.h>

#include <exception>
#include <iostream>
#include <vector>

#include "utils.hpp"

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                  \
  /* Code block avoids redefinition of cudaError_t error */                    \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(Formatter()                                     \
                               << " " << cudaGetErrorString(error) << " at "   \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

#define CUBLAS_CHECK(condition)                                                \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(Formatter()                                     \
                               << cublasGetErrorString(status) << " at "       \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      throw std::runtime_error(Formatter()                                     \
                               << cusparseGetErrorString(err) << " at "        \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

#define CURAND_CHECK(condition)                                                \
  {                                                                            \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(Formatter()                                     \
                               << curandGetErrorString(status) << " at "       \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define CUDA_POST_KERNEL_CHECK HANDLE_ERROR(cudaPeekAtLastError())

#define THRUST_CHECK(condition)                                                \
  try {                                                                        \
    condition;                                                                 \
  } catch (thrust::system_error e) {                                           \
    throw std::runtime_error(Formatter()                                       \
                             << "Thrust error: " << e.what() << " at "         \
                             << __FILE__ << ":" << __LINE__);                  \
  }

// CUDA: library error reporting.
const char *cublasGetErrorString(cublasStatus_t error);

// CUSparse error reporting.
const char *cusparseGetErrorString(cusparseStatus_t error);

constexpr int CUDA_NUM_THREADS = 256;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v);
template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2);

void HandleError(cudaError_t err, const char *file, int line);

// AtomicAddition for double with cuda arch <= 600
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

#endif // GPU_H_
