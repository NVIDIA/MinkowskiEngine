#ifndef GPU_H_
#define GPU_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h> // cuda driver types
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                  \
  /* Code block avoids redefinition of cudaError_t error */                    \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess)                                                  \
      std::cerr << " " << cudaGetErrorString(error);                           \
  } while (0)

#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS)                                       \
      std::cerr << " " << cublasGetErrorString(status);                        \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CURAND_CHECK(condition)                                                \
  do {                                                                         \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS)                                       \
      std::cerr << " " << curandGetErrorString(status);                        \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define CUDA_POST_KERNEL_CHECK HANDLE_ERROR(cudaPeekAtLastError())

// CUDA: library error reporting.
const char *cublasGetErrorString(cublasStatus_t error);

// CUSparse error reporting.
const char* cusparseGetErrorString(cusparseStatus_t error);

// CUDA: use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v);
template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2);

void HandleError(cudaError_t err, const char *file, int line);

#endif // GPU_H_
