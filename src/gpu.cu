#include <iomanip>
#include <iostream>
#include <cstdio>

#include "src/gpu.cuh"

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v) {
  for (size_t i = 0; i < v.size(); i++)
    std::cout << " " << std::fixed << std::setprecision(1) << v[i];
  std::cout << "\n";
}

template void print(const thrust::device_vector<float> &v);
template void print(const thrust::device_vector<int64_t> &v);

template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2) {
  for (size_t i = 0; i < v1.size(); i++)
    std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
  std::cout << "\n";
}

template void print(const thrust::device_vector<int64_t> &v1,
                    const thrust::device_vector<int64_t> &v2);

void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}
