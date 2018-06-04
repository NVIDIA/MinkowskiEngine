#include <cstdio>
#include <iomanip>
#include <iostream>

#include "src/gpu.cuh"

template <typename Dtype> void print(const thrust::device_vector<Dtype> &v) {
  for (size_t i = 0; i < v.size(); i++)
    std::cout << " " << std::fixed << std::setprecision(3) << v[i];
  std::cout << "\n";
}

template void print(const thrust::device_vector<float> &v);
template void print(const thrust::device_vector<int32_t> &v);

template <typename Dtype1, typename Dtype2>
void print(const thrust::device_vector<Dtype1> &v1,
           const thrust::device_vector<Dtype2> &v2) {
  for (size_t i = 0; i < v1.size(); i++)
    std::cout << " (" << v1[i] << "," << std::setw(2) << v2[i] << ")";
  std::cout << "\n";
}

template void print(const thrust::device_vector<int32_t> &v1,
                    const thrust::device_vector<int32_t> &v2);

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

const char* cusparseGetErrorString(cusparseStatus_t error) {
  // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "The operation completed successfully.";
  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "The cuSPARSE library was not initialized. This is usually caused "
           "by the lack of a prior call, an error in the CUDA Runtime API "
           "called by the cuSPARSE routine, or an error in the hardware "
           "setup.\n"
           "To correct: call cusparseCreate() prior to the function call; and "
           "check that the hardware, an appropriate version of the driver, and "
           "the cuSPARSE library are correctly installed.";

  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "Resource allocation failed inside the cuSPARSE library. This is "
           "usually caused by a cudaMalloc() failure.\n"
           "To correct: prior to the function call, deallocate previously "
           "allocated memory as much as possible.";

  case CUSPARSE_STATUS_INVALID_VALUE:
    return "An unsupported value or parameter was passed to the function (a "
           "negative vector size, for example).\n"
           "To correct: ensure that all the parameters being passed have valid "
           "values.";

  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "The function requires a feature absent from the device "
           "architecture; usually caused by the lack of support for atomic "
           "operations or double precision.\n"
           "To correct: compile and run the application on a device with "
           "appropriate compute capability, which is 1.1 for 32-bit atomic "
           "operations and 1.3 for double precision.";

  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "An access to GPU memory space failed, which is usually caused by a "
           "failure to bind a texture.\n"
           "To correct: prior to the function call, unbind any previously "
           "bound textures.";

  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "The GPU program failed to execute. This is often caused by a "
           "launch failure of the kernel on the GPU, which can be caused by "
           "multiple reasons.\n"
           "To correct: check that the hardware, an appropriate version of the "
           "driver, and the cuSPARSE library are correctly installed.";

  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "An internal cuSPARSE operation failed. This error is usually "
           "caused by a cudaMemcpyAsync() failure.\n"
           "To correct: check that the hardware, an appropriate version of the "
           "driver, and the cuSPARSE library are correctly installed. Also, "
           "check that the memory passed as a parameter to the routine is not "
           "being deallocated prior to the routine’s completion.";

  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "The matrix type is not supported by this function. This is usually "
           "caused by passing an invalid matrix descriptor to the function.\n"
           "To correct: check that the fields in cusparseMatDescr_t descrA "
           "were set correctly.";
  }

  return "<unknown>";
}
