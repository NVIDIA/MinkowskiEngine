#ifndef CPU_ONLY

#ifndef GPU_MEMORY_MANAGER
#define GPU_MEMORY_MANAGER

#include <vector>

#include "gpu.cuh"
#include "types.hpp"

#include <torch/extension.h>

using namespace std;

class GPUMemoryManager {
  int initial_size = 256;
  int device_id;
  torch::TensorOptions options;

public:
  // Scratch space, the user should keep track of the validity of the pointer
  torch::Tensor scratch_data;
  torch::Tensor scratch_data2; // cusparse_csrmm requires 2 memory spaces

  // A set of data that will be not be freed untill the class is destroyed.
  vector<torch::Tensor> vec_data;

  // Memory manager simply allocates and free memory when done.
  GPUMemoryManager() {
    CUDA_CHECK(cudaGetDevice(&device_id));
    options = torch::TensorOptions()
                  .dtype(torch::kByte)
                  .device(torch::kCUDA, device_id)
                  .requires_grad(false);
    scratch_data = torch::zeros({initial_size}, options).contiguous();
    scratch_data2 = torch::zeros({initial_size}, options).contiguous();
  }
  GPUMemoryManager(int size) : initial_size(size) { GPUMemoryManager(); }

  pInOutMaps<int> copyInOutMapToGPU(const InOutMaps<int> &map);

  void resize(int size) { scratch_data.resize_({size}).contiguous(); }
  void resize2(int size) { scratch_data2.resize_({size}).contiguous(); }

  void *data(int size) {
    if (scratch_data.numel() < size)
      resize(size);
    return (void *)scratch_data.data<unsigned char>();
  }

  void *data2(int size) {
    if (scratch_data2.numel() < size)
      resize2(size);
    return (void *)scratch_data2.data<unsigned char>();
  }

  void *gpuMalloc(int size) {
    torch::Tensor data = torch::zeros({size}, options).contiguous();
    vec_data.push_back(data);

    return (void *)data.data<unsigned char>();
  }
};

#endif // GPU_MEMORY_MANAGER
#endif // CPU_ONLY
