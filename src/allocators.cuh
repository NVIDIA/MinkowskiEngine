/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)
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
#ifndef ALLOCATORS_CUH
#define ALLOCATORS_CUH

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "gpu.cuh"
#include "types.hpp"

namespace minkowski {

namespace detail {

template <class T> struct default_allocator {
  typedef T value_type;
  // rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

  default_allocator() = default;

  template <class U>
  constexpr default_allocator(const default_allocator<U> &) noexcept {}

  T *allocate(std::size_t n, cudaStream_t stream = 0) const {
    T *d_tmp;
    cudaError_t error = cudaMalloc((void **)&d_tmp, n * sizeof(T));
    if (error != cudaSuccess) {
      cudaGetLastError(); // clear error
      c10::cuda::CUDACachingAllocator::emptyCache();
      LOG_DEBUG("Automatically called empty cache");
      CUDA_CHECK(cudaMalloc((void **)&d_tmp, n * sizeof(T)));
    }
    return d_tmp;
    // return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const {
    cudaFree(p);
    // mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T> struct c10_allocator {
  typedef T value_type;

  c10_allocator() = default;

  template <class U>
  constexpr c10_allocator(const c10_allocator<U> &) noexcept {}

  T *allocate(std::size_t n, cudaStream_t stream = 0) const {
    return reinterpret_cast<T *>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n * sizeof(T)));
  }

  std::shared_ptr<T[]> shared_allocate(std::size_t n,
                                       cudaStream_t stream = 0) const {
    T *d_ptr = reinterpret_cast<T *>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n * sizeof(T)));

    auto deleter = [](T *p) {
      c10::cuda::CUDACachingAllocator::raw_delete((void *)p);
    };

    return std::shared_ptr<T[]>{d_ptr,
                                std::bind(deleter, std::placeholders::_1)};
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const {
    c10::cuda::CUDACachingAllocator::raw_delete((void *)p);
  }
};

template <typename T = char> class cached_allocator {
public:
  using value_type = T;
  using free_blocks_type = std::multimap<std::ptrdiff_t, T *>;
  using allocated_blocks_type = std::map<T *, std::ptrdiff_t>;
  using iterator = typename free_blocks_type::iterator;

public:
  cached_allocator() {}
  ~cached_allocator() {
#ifndef __CUDACC__
    free_all();
#endif
  }

  T *allocate(std::ptrdiff_t num_values, cudaStream_t stream = 0) {
    T *result = 0;

    // search the cache for a free block
    auto free_block = free_blocks.find(num_values * sizeof(value_type));

    if (free_block != free_blocks.end()) {
      LOG_DEBUG("using preallocated", num_values, "of", sizeof(value_type));
      result = free_block->second;
      free_blocks.erase(free_block);
    } else {
      LOG_DEBUG("allocating", num_values, "of", sizeof(value_type));
      CUDA_CHECK(cudaMalloc((void **)&result, num_values * sizeof(value_type)));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_values));

    return result;
  }

  void deallocate(T *ptr, size_t n, cudaStream_t stream = 0) {
    // erase the allocated block from the allocated blocks map
    auto iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_values = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_values, reinterpret_cast<T *>(ptr)));
  }

private:
  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all() {
    // deallocate all outstanding blocks in both lists
    for (auto i = free_blocks.begin(); i != free_blocks.end(); i++) {
      cudaFree(i->second);
    }

    for (auto i = allocated_blocks.begin(); i != allocated_blocks.end(); i++) {
      cudaFree(i->first);
    }
  }
};

/*
 * Wrapper for the cached_allocator to share the allocated blocks.
 * disable all functions and members for __device__ functions
 * (thrust functors and kernel calls).
 */
template <typename T = char> class shared_allocator {
public:
  using self_type = shared_allocator<T>;

public:
  __host__ __device__ shared_allocator() {
    m_p_alloc = std::make_shared<cached_allocator<T>>(cached_allocator<T>());
  }
  __host__ __device__ shared_allocator(self_type const &other) {
    m_p_alloc = other.m_p_alloc;
  }
  __host__ __device__ ~shared_allocator() {}

  __host__ T *allocate(std::ptrdiff_t num_values,
                       cudaStream_t stream = 0) const {
    return m_p_alloc->allocate(num_values, stream);
  }

  __host__ void deallocate(T *ptr, size_t n, cudaStream_t stream = 0) const {
    return m_p_alloc->deallocate(ptr, n, stream);
  }

private:
  std::shared_ptr<cached_allocator<T>> m_p_alloc;
};

} // namespace detail

} // namespace minkowski

#endif // ALLOCATORS_CUH
