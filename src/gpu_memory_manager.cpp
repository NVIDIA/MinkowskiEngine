#include "gpu_memory_manager.hpp"

template <typename Type> GPUMemoryManager<Type>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

// Explicit template instantiation for tensor dtype specification
template <> GPUMemoryManager<long>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

// Explicit template instantiation for tensor dtype specification
template <> GPUMemoryManager<float>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

// Explicit template instantiation for tensor dtype specification
template <> GPUMemoryManager<double>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

template class GPUMemoryManager<int32_t>;
