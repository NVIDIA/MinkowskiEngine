#ifndef GPU_MEMORY_MANAGER
#define GPU_MEMORY_MANAGER
#include "gpu.cuh"
#include <torch/extension.h>

template <typename Type> class GPUMemoryManager {
  int initial_size = 256;
  int device_id;

public:
  torch::Tensor _data;

  // Memory manager simply allocates and free memory when done.
  GPUMemoryManager();
  GPUMemoryManager(int size) : initial_size(size) { GPUMemoryManager(); }

  void reset() { _data.resize_({0}); }
  void resize(int new_size) { _data.resize_({new_size}); }
  int64_t size() { return _data.numel(); }
  Type *data() { return _data.data<Type>(); }
  Type *data(int new_size) {
    if (size() < new_size) resize(new_size);
    return data();
  }
};

#endif
