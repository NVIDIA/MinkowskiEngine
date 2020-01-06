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

namespace minkowski {

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

#define CUDA_CHECK_ARGS(condition, ...)                                        \
  /* Code block avoids redefinition of cudaError_t error */                    \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      Formatter formatter;                                                     \
      formatter << " " << cudaGetErrorString(error) << " at ";                 \
      formatter << __FILE__ << ":" << __LINE__;                                \
      formatter.append(__VA_ARGS__);                                           \
      throw std::runtime_error(formatter.str());                               \
    }                                                                          \
  }

#define CUBLAS_CHECK(condition)                                                \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(Formatter()                                     \
                               << minkowski::cublasGetErrorString(status)      \
                               << " at " << __FILE__ << ":" << __LINE__);      \
    }                                                                          \
  }

#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      throw std::runtime_error(Formatter()                                     \
                               << minkowski::cusparseGetErrorString(err)       \
                               << " at " << __FILE__ << ":" << __LINE__);      \
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
#define CUDA_POST_KERNEL_CHECK                                                 \
  {                                                                            \
    cudaError_t status = cudaPeekAtLastError();                                \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(Formatter()                                     \
                               << " " << cudaGetErrorString(status) << " at "  \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

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

constexpr int SHARED_BLOCK_SIZE = 32;

constexpr int MAX_GRID = 65535;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v);
template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2);

} // end namespace minkowski

#endif // GPU_H_
