#ifndef CPU_ONLY

#ifndef GPU_MEMORY_MANAGER
#define GPU_MEMORY_MANAGER

#include <vector>

#include "gpu.cuh"
#include "types.hpp"

namespace minkowski {

using std::vector;

class GPUMemoryManager {
  int initial_size = 256;

public:
  int device_id;

  // A set of data that will be not be freed untill the class is destroyed.
  vector<void *> persist_vec_ptr;
  vector<void *> tmp_vec_ptr;

  // Memory manager simply allocates and free memory when done.
  GPUMemoryManager() { CUDA_CHECK(cudaGetDevice(&device_id)); }
  GPUMemoryManager(int size) : initial_size(size) { GPUMemoryManager(); }
  ~GPUMemoryManager() {
    for (auto p_buffer : persist_vec_ptr) {
      cudaFree(p_buffer);
    }
  }

  pInOutMaps<int> copyInOutMapToGPU(const InOutMaps<int> &map);

  void clear_tmp() {
    for (auto p_buffer : tmp_vec_ptr) {
      cudaFree(p_buffer);
    }
    tmp_vec_ptr.clear();
  }

  void set_device() {
    CUDA_CHECK(cudaSetDevice(device_id));
  }

  void *tmp_data(size_t size) {
    void *p_buffer = NULL;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&p_buffer, size));
    tmp_vec_ptr.push_back(p_buffer);
    return p_buffer;
  }

  void *gpuMalloc(size_t size) {
    void *p_buffer = NULL;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&p_buffer, size));
    persist_vec_ptr.push_back(p_buffer);
    return p_buffer;
  }
};

} // end namespace minkowski

#endif // GPU_MEMORY_MANAGER
#endif // CPU_ONLY
