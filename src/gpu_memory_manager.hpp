/* Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef CPU_ONLY

#ifndef GPU_MEMORY_MANAGER
#define GPU_MEMORY_MANAGER

#include <iostream>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "gpu.cuh"
#include "types.hpp"

namespace minkowski {

using std::vector;

// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator {
public:
  // just allocate bytes
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator() {
    // free all allocations when cached_allocator goes out of scope
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes) {
    char *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end()) {
      std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    } else {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try {
        std::cout << "cached_allocator::allocator(): no free block found; "
                     "calling cuda::malloc"
                  << std::endl;

        // allocate memory and convert cuda::pointer to raw pointer
        result = thrust::cuda::malloc<char>(num_bytes).get();
      } catch (std::runtime_error &e) {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t n) {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char *> free_blocks_type;
  typedef std::map<char *, std::ptrdiff_t> allocated_blocks_type;

  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all() {
    std::cout << "cached_allocator::free_all(): cleaning up after ourselves..."
              << std::endl;

    // deallocate all outstanding blocks in both lists
    for (free_blocks_type::iterator i = free_blocks.begin();
         i != free_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for (allocated_blocks_type::iterator i = allocated_blocks.begin();
         i != allocated_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

class GPUMemoryManager {
private:
  int initial_size = 256;
  MemoryManagerBackend backend;
  int device_id;

public:
  // A set of data that will be not be freed untill the class is destroyed.
  vector<void *> persist_vec_ptr;
  vector<void *> tmp_vec_ptr;

  // Memory manager simply allocates and free memory when done.
  GPUMemoryManager(MemoryManagerBackend backend_) : backend(backend_) {
    CUDA_CHECK(cudaGetDevice(&device_id));
    // std::cout << "GPU set to " << device_id << "\n";
  }
  GPUMemoryManager() : GPUMemoryManager(PYTORCH) {} // use pytorch by default
  ~GPUMemoryManager() {
    switch (backend) {
    case CUDA: {
      for (auto p_buffer : persist_vec_ptr) {
        cudaFree(p_buffer);
      }
      break;
    }
    case PYTORCH: {
      for (auto p_buffer : persist_vec_ptr) {
        c10::cuda::CUDACachingAllocator::raw_delete(p_buffer);
      }
      break;
    }
    }
  }

  pInOutMaps<int> copyInOutMapToGPU(const InOutMaps<int> &map);

  void clear_tmp() {
    for (auto p_buffer : tmp_vec_ptr) {
      cudaFree(p_buffer);
    }
    tmp_vec_ptr.clear();
  }

  void set_device() { CUDA_CHECK(cudaSetDevice(device_id)); }
  int get_device_id() const { return device_id; }

  void *tmp_data(size_t size) {
    void *p_buffer = NULL;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&p_buffer, size));
    tmp_vec_ptr.push_back(p_buffer);
    return p_buffer;
  }

  void *gpuMalloc(size_t size) {
    void *p_buffer = NULL;
    switch (backend) {
    case CUDA: {
      // std::cout << "Malloc CUDA: " << device_id << std::endl;
      CUDA_CHECK(cudaSetDevice(device_id));
      CUDA_CHECK(cudaMalloc(&p_buffer, size));
      persist_vec_ptr.push_back(p_buffer);
      break;
    }
    case PYTORCH: {
      // std::cout << "Malloc PYTORCH: " << device_id << std::endl;
      CUDA_CHECK(cudaSetDevice(device_id));
      p_buffer = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
          size, at::cuda::getCurrentCUDAStream());
      persist_vec_ptr.push_back(p_buffer);
      break;
    }
    }
    return p_buffer;
  }
};

} // end namespace minkowski

#endif // GPU_MEMORY_MANAGER
#endif // CPU_ONLY
