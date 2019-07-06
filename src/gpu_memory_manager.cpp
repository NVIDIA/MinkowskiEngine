#include "gpu_memory_manager.hpp"

// Explicit template instantiation for tensor dtype specification
template <> GPUMemoryManager<int8_t>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt8)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};


template <> GPUMemoryManager<int>::GPUMemoryManager() {
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
                     .dtype(torch::kFloat)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

// Explicit template instantiation for tensor dtype specification
template <> GPUMemoryManager<double>::GPUMemoryManager() {
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto options = torch::TensorOptions()
                     .dtype(torch::kDouble)
                     .device(torch::kCUDA, device_id)
                     .requires_grad(false);
  _data = torch::zeros({initial_size}, options);
};

template class GPUMemoryManager<int8_t>;
template class GPUMemoryManager<int>;
template class GPUMemoryManager<long>;
template class GPUMemoryManager<float>;
template class GPUMemoryManager<double>;
