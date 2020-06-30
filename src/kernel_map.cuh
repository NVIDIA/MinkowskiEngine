/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef KERNEL_MAP_CUH
#define KERNEL_MAP_CUH

#include "3rdparty/hash/hash_allocator.cuh"
#include "coordinate_map_functors.cuh"
#include "types.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace minkowski {
template <typename index_type = default_types::index_type,
          typename KernelMapAllocator =
              default_allocator<default_types::index_type>>
class gpu_kernel_map {
public:
  using size_type = default_types::size_type;
  using allocator_type = KernelMapAllocator;

  class contiguous_memory {
  public:
    contiguous_memory() = delete;
    contiguous_memory(gpu_kernel_map &kernel_map) : m_kernel_map{kernel_map} {}

    inline index_type *data() { return m_data; }
    inline void data(index_type *p_data) { m_data = p_data; }

    inline index_type *begin() const { return m_data; }
    inline index_type *begin(index_type kernel_index) const {
      auto const &offset_map = m_kernel_map.m_kernel_offset_map;
      auto const iter = offset_map.find(kernel_index);
      if (iter == offset_map.end()) {
        LOG_WARN("gpu_kernel_map for kernel", kernel_index, "not found.");
        return m_data;
      } else {
        return m_data + iter->second;
      }
    }

    inline index_type *end() const { return m_data + m_kernel_map.m_capacity; }
    inline index_type *end(index_type kernel_index) const {
      auto const &offset_map = m_kernel_map.m_kernel_offset_map;
      auto const iter = offset_map.find(kernel_index);
      if (iter == offset_map.end()) {
        LOG_WARN("gpu_kernel_map for kernel", kernel_index, "not found.");
        return m_data + m_kernel_map.m_capacity;
      } else {
        return m_data + iter->second +
               m_kernel_map.m_kernel_size_map[kernel_index];
      }
    }

    size_type size(index_type kernel_index) const {
      auto const &size_map = m_kernel_map.m_kernel_size_map;
      auto const iter = size_map.find(kernel_index);
      if (iter == size_map.end())
        return 0;
      return iter->second;
    }

  private:
    index_type *m_data;
    gpu_kernel_map &m_kernel_map;
  }; // end contiguous_memory

public:
  gpu_kernel_map(size_type capacity, allocator_type alloc = allocator_type())
      : m_memory_size(3 * capacity), m_capacity{capacity}, m_memory{nullptr},
        m_allocator{alloc}, kernels{*this}, in_maps{*this}, out_maps{*this} {
    m_memory = m_allocator.allocate(m_memory_size);
    kernels.data(m_memory);
    in_maps.data(m_memory + m_capacity);
    out_maps.data(m_memory + 2 * m_capacity);
  }
  ~gpu_kernel_map() { m_allocator.deallocate(m_memory, m_memory_size); }

  // functions
  inline index_type *data() { return m_memory; }

  void decompose() {
    // the memory space must be initialized first!

    // sort
    thrust::sort_by_key(thrust::device,            //
                        kernels.begin(),           // key begin
                        kernels.end(),             // key end
                        thrust::make_zip_iterator( // value begin
                            thrust::make_tuple(    //
                                in_maps.begin(),   //
                                out_maps.begin()   //
                                )                  //
                            ));

#ifdef DEBUG
    index_type *p_kernel_map =
        (index_type *)std::malloc(m_capacity * 3 * sizeof(index_type));
    CUDA_CHECK(cudaMemcpy(p_kernel_map, data(),
                          m_memory_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));
    for (index_type i = 0; i < min(m_capacity, 100); ++i) {
      std::cout << p_kernel_map[i + 0 * m_capacity] << ":"
                << p_kernel_map[i + 1 * m_capacity] << "->"
                << p_kernel_map[i + 2 * m_capacity] << "\n";
    }
#endif

    // Need to find the start and the size of each key for the kernel map
    // generation.
    thrust::counting_iterator<index_type> min_begin{0};
    thrust::constant_iterator<index_type> size_begin{1};

    thrust::device_vector<index_type> out_key(m_capacity);
    thrust::device_vector<index_type> out_key_min(m_capacity);
    thrust::device_vector<index_type> out_key_size(m_capacity);

    auto end = thrust::reduce_by_key(
        thrust::device, // policy
        kernels.begin(), // key begin
        kernels.end(),    // key end
        thrust::make_zip_iterator(
            thrust::make_tuple(min_begin, size_begin)), // value begin
        out_key.begin(),                                // key out begin
        thrust::make_zip_iterator(thrust::make_tuple(
            out_key_min.begin(), out_key_size.begin())), // value out begin
        thrust::equal_to<index_type>(),        // key equal binary predicate
        detail::min_size_functor<index_type>() // value binary operator
    );

    size_type num_unique_keys = end.first - out_key.begin();
    LOG_DEBUG(num_unique_keys, "unique kernel map keys found");

    thrust::host_vector<index_type> cpu_out_keys(
        out_key.begin(), out_key.begin() + num_unique_keys);
    thrust::host_vector<index_type> cpu_out_offset(
        out_key_min.begin(), out_key_min.begin() + num_unique_keys);
    thrust::host_vector<index_type> cpu_out_size(
        out_key_size.begin(), out_key_size.begin() + num_unique_keys);

#ifdef DEBUG
    LOG_DEBUG("Keys:", cpu_out_keys);
    LOG_DEBUG("Mins:", cpu_out_offset);
    LOG_DEBUG("Size:", cpu_out_size);
#endif

    // create an unordered map
    for (index_type i = 0; i < num_unique_keys; ++i) {
      m_kernel_offset_map[cpu_out_keys[i]] = cpu_out_offset[i];
      m_kernel_size_map[cpu_out_keys[i]] = cpu_out_size[i];
    }

    // Initialize the decomposed begins and sizes
    m_decomposed = true;
  }

private:
  bool m_decomposed{false};
  size_type m_memory_size, m_capacity;
  index_type *m_memory;
  allocator_type m_allocator;

  std::unordered_map<index_type, index_type> m_kernel_size_map;
  std::unordered_map<index_type, index_type> m_kernel_offset_map;

public:
  contiguous_memory kernels;
  contiguous_memory in_maps;
  contiguous_memory out_maps;
}; // gpu_kernel_map

} // namespace minkowski

#endif // KERNEL_MAP_CUH
