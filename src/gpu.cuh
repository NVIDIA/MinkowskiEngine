#ifndef GPU_H_
#define GPU_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <driver_types.h> // cuda driver types

#include <thrust/device_vector.h>

#include <iostream>
#include <vector>
#include <exception>

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
      throw std::runtime_error(Formatter() << " " << cudaGetErrorString(error) \
                << " at " << __FILE__ << ":" << __LINE__);                     \
    }                                                                          \
  }

#define CUBLAS_CHECK(condition)                                                \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(Formatter() << cublasGetErrorString(status)     \
                << " at " << __FILE__ << ":" << __LINE__);                     \
    }                                                                          \
  }

#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      throw std::runtime_error(Formatter() << cusparseGetErrorString(err)      \
              << " at " <<  __FILE__ << ":" << __LINE__);                      \
    }                                                                          \
  }

#define CURAND_CHECK(condition)                                                \
  {                                                                            \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(Formatter() << curandGetErrorString(status)     \
                << " at " << __FILE__ << ":" << __LINE__);                     \
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
    throw std::runtime_error(Formatter() << "Thrust error: " << e.what()       \
              << " at " << __FILE__ << ":" << __LINE__);                       \
  }

// CUDA: library error reporting.
const char *cublasGetErrorString(cublasStatus_t error);

// CUSparse error reporting.
const char *cusparseGetErrorString(cusparseStatus_t error);

// CUDA: use 1024 threads per block
constexpr int CUDA_NUM_THREADS = 128;

constexpr int MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
          MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v);
template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2);

void HandleError(cudaError_t err, const char *file, int line);

#endif // GPU_H_
