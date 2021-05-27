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
#include "storage.cuh"
#include "types.hpp"

#include <functional>
#include <map>
#include <memory>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

namespace minkowski {

template <typename index_type, typename ByteAllocator> class gpu_kernel_map {
public:
  using size_type = default_types::size_type;
  using byte_allocator_type = ByteAllocator;
  using self_type = gpu_kernel_map<index_type, ByteAllocator>;

  class contiguous_memory {
  public:
    using self_type = contiguous_memory;

  public:
    contiguous_memory() = delete;
    contiguous_memory(gpu_kernel_map &kernel_map) : m_kernel_map{kernel_map} {}

    self_type &operator=(self_type const &other) {
      m_data = other.m_data;
      m_kernel_map = other.m_kernel_map;
      return *this;
    }

    inline typename std::map<index_type, index_type>::const_iterator
    key_cbegin() const {
      return m_kernel_map.m_kernel_offset_map.cbegin();
    }
    inline typename std::map<index_type, index_type>::const_iterator
    key_cend() const {
      return m_kernel_map.m_kernel_offset_map.cend();
    }

    inline index_type *data() { return m_data; }
    inline index_type const *cdata() const { return m_data; }
    inline void data(index_type *p_data) { m_data = p_data; }

    inline index_type *begin() const { return m_data; }
    inline index_type *begin(index_type kernel_index) const {
      auto const &offset_map = m_kernel_map.m_kernel_offset_map;
      auto const iter = offset_map.find(kernel_index);
      if (iter == offset_map.end()) {
        // LOG_WARN("gpu_kernel_map for kernel", kernel_index, "not found.");
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
        // LOG_WARN("gpu_kernel_map for kernel", kernel_index, "not found.");
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
  gpu_kernel_map()
      : m_memory_size_byte{0},
        m_capacity{0}, kernels{*this}, in_maps{*this}, out_maps{*this} {
    LOG_DEBUG("Initialized gpu_kernel_map");
  }
  gpu_kernel_map(self_type const &other)
      : m_decomposed(other.m_decomposed),                       //
        m_requires_kernel_index(other.m_requires_kernel_index), //
        m_memory_size_byte(other.m_memory_size_byte),           //
        m_capacity{other.m_capacity},                           //
        m_in_map_memory{other.m_in_map_memory},                 //
        m_out_map_memory{other.m_out_map_memory},               //
        m_allocator{other.m_allocator},                         //
        m_kernel_size_map{other.m_kernel_size_map},             //
        m_kernel_offset_map{other.m_kernel_offset_map},         //
        kernels{*this},                                         //
        in_maps{*this},                                         //
        out_maps{*this} {
    LOG_DEBUG("gpu_kernel_map copy constructor");
    in_maps.data(other.in_maps.begin());
    out_maps.data(other.out_maps.begin());
    if (m_requires_kernel_index) {
      m_kernel_index_memory = other.m_kernel_index_memory;
      kernels.data(other.kernels.begin());
    }
  }

  gpu_kernel_map(size_type capacity,
                 byte_allocator_type alloc = byte_allocator_type(),
                 bool requires_kernel_index = true)
      : m_requires_kernel_index(requires_kernel_index), m_capacity{capacity},
        m_allocator{alloc}, kernels{*this}, in_maps{*this}, out_maps{*this} {
    // kernel map without kernel index
    m_memory_size_byte = capacity * sizeof(index_type);
    index_type *ptr_in_map = reinterpret_cast<index_type *>(
        m_allocator.allocate(m_memory_size_byte));
    index_type *ptr_out_map = reinterpret_cast<index_type *>(
        m_allocator.allocate(m_memory_size_byte));
    index_type *ptr_kernel = nullptr;

    auto deleter = [](index_type *p, byte_allocator_type alloc,
                      size_type size) {
      alloc.deallocate(reinterpret_cast<char *>(p), size);
      LOG_DEBUG("Deallocate kernel map");
    };

    m_in_map_memory = std::shared_ptr<index_type[]>{
        ptr_in_map, std::bind(deleter, std::placeholders::_1, m_allocator,
                              m_memory_size_byte)};
    m_out_map_memory = std::shared_ptr<index_type[]>{
        ptr_out_map, std::bind(deleter, std::placeholders::_1, m_allocator,
                               m_memory_size_byte)};
    // kernel maps
    in_maps.data(m_in_map_memory.get());
    out_maps.data(m_out_map_memory.get());

    if (requires_kernel_index) {
      ptr_kernel = reinterpret_cast<index_type *>(
          m_allocator.allocate(m_memory_size_byte));
      m_kernel_index_memory = std::shared_ptr<index_type[]>{
          ptr_kernel, std::bind(deleter, std::placeholders::_1, m_allocator,
                                m_memory_size_byte)};
      kernels.data(m_kernel_index_memory.get());
    } else {
      m_kernel_offset_map[0] = 0;
      m_kernel_size_map[0] = capacity;
      // Initialize the decomposed begins and sizes
      m_decomposed = true;
    }
  }

  self_type swap() const {
    self_type swapped_gpu_kernel_map(*this);
    swapped_gpu_kernel_map.in_maps.data(
        swapped_gpu_kernel_map.m_out_map_memory.get());
    swapped_gpu_kernel_map.out_maps.data(
        swapped_gpu_kernel_map.m_in_map_memory.get());

#ifdef DEBUG
    size_type map_size = std::min<size_type>(in_maps.size(0), 100);

    index_type *p_kernel_map =
        (index_type *)std::malloc(map_size * 3 * sizeof(index_type));
    // CUDA_CHECK(cudaMemcpy(p_kernel_map, kernels.begin(),
    //                       map_size * sizeof(index_type),
    //                       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p_kernel_map + 1 * map_size, in_maps.begin(),
                          map_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p_kernel_map + 2 * map_size, out_maps.begin(),
                          map_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));

    for (index_type i = 0; i < map_size; ++i) {
      std::cout // << p_kernel_map[i + 0 * map_size] << ":"
          << p_kernel_map[i + 1 * map_size] << "->"
          << p_kernel_map[i + 2 * map_size] << "\n";
    }

    std::cout << "Swapped kernel map\n";

    // CUDA_CHECK(cudaMemcpy(p_kernel_map,
    // swapped_gpu_kernel_map.kernels.begin(),
    //                       map_size * sizeof(index_type),
    //                       cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        p_kernel_map + 1 * map_size, swapped_gpu_kernel_map.in_maps.begin(),
        map_size * sizeof(index_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        p_kernel_map + 2 * map_size, swapped_gpu_kernel_map.out_maps.begin(),
        map_size * sizeof(index_type), cudaMemcpyDeviceToHost));

    for (index_type i = 0; i < map_size; ++i) {
      std::cout // << p_kernel_map[i + 0 * map_size] << ":"
          << p_kernel_map[i + 1 * map_size] << "->"
          << p_kernel_map[i + 2 * map_size] << "\n";
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::free(p_kernel_map);
#endif
    return swapped_gpu_kernel_map;
  }

  self_type &operator=(self_type const &other) {
    m_decomposed = other.m_decomposed;
    m_requires_kernel_index = other.m_requires_kernel_index;

    m_memory_size_byte = other.m_memory_size_byte;
    m_capacity = other.m_capacity;

    m_kernel_index_memory = other.m_kernel_index_memory;
    m_in_map_memory = other.m_in_map_memory;
    m_out_map_memory = other.m_out_map_memory;
    m_allocator = other.m_allocator;

    m_kernel_size_map = other.m_kernel_size_map;
    m_kernel_offset_map = other.m_kernel_offset_map;

    in_maps.data(other.in_maps.begin());
    out_maps.data(other.out_maps.begin());
    kernels.data(other.kernels.begin());

    return *this;
  }

  // functions
  inline typename std::map<index_type, index_type>::const_iterator
  key_cbegin() const {
    return m_kernel_offset_map.cbegin();
  }

  inline typename std::map<index_type, index_type>::const_iterator
  key_cend() const {
    return m_kernel_offset_map.cend();
  }

  size_type size() const {
    size_type nmap = 0;
    for (auto const &k : m_kernel_size_map) {
      nmap += k.second;
    }
    return nmap;
  }

  size_type volume() const { return m_kernel_size_map.size(); }

  size_type max_size() const {
    size_type nmap = 0;
    for (auto const &k : m_kernel_size_map) {
      if (k.second > nmap)
        nmap = k.second;
    }
    return nmap;
  }

  size_type size(index_type const kernel_index) const {
    auto const iter = m_kernel_size_map.find(kernel_index);
    if (iter == m_kernel_size_map.end())
      return 0;
    return iter->second;
  }

  std::string to_string() const {
    Formatter o;
    size_type map_size = 0;
    for (auto const &kv : m_kernel_size_map) {
      map_size += kv.second;
    }
    o << "gpu_kernel_map: number of unique maps:" << m_kernel_size_map.size()
      << ", kernel map size:" << map_size;
    return o.str();
  }

  void decompose() {
    LOG_DEBUG("Decomposing", kernels.end() - kernels.begin(), "elements");
    // the memory space must be initialized first!
    // sort
    THRUST_CHECK(thrust::sort_by_key(thrust::device,            //
                                     kernels.begin(),           // key begin
                                     kernels.end(),             // key end
                                     thrust::make_zip_iterator( // value begin
                                         thrust::make_tuple(    //
                                             in_maps.begin(),   //
                                             out_maps.begin()   //
                                             )                  //
                                         )));

#ifdef DEBUG
    size_type map_size =
        std::min<size_type>(in_maps.end() - in_maps.begin(), 100);
    LOG_DEBUG("printing", map_size, "kernel maps");
    index_type *p_kernel_map =
        (index_type *)std::malloc(map_size * 3 * sizeof(index_type));
    CUDA_CHECK(cudaMemcpy(p_kernel_map, m_kernel_index_memory.get(),
                          map_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p_kernel_map + map_size, m_in_map_memory.get(),
                          map_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p_kernel_map + 2 * map_size, m_out_map_memory.get(),
                          map_size * sizeof(index_type),
                          cudaMemcpyDeviceToHost));

    for (index_type i = 0; i < map_size; ++i) {
      std::cout << p_kernel_map[i + 0 * map_size] << ":"
                << p_kernel_map[i + 1 * map_size] << "->"
                << p_kernel_map[i + 2 * map_size] << "\n";
    }
    std::free(p_kernel_map);
#endif

    // Need to find the start and the size of each key for the kernel map
    // generation.
    thrust::counting_iterator<index_type> min_begin{0};
    thrust::constant_iterator<index_type> size_begin{1};

    gpu_storage<index_type, byte_allocator_type> out_key(m_capacity);
    gpu_storage<index_type, byte_allocator_type> out_key_min(m_capacity);
    gpu_storage<index_type, byte_allocator_type> out_key_size(m_capacity);

    size_type num_unique_keys;

    try {
      auto end = thrust::reduce_by_key(
          thrust::device,  // policy
          kernels.begin(), // key begin
          kernels.end(),   // key end
          thrust::make_zip_iterator(
              thrust::make_tuple(min_begin, size_begin)), // value begin
          out_key.begin(),                                // key out begin
          thrust::make_zip_iterator(thrust::make_tuple(
              out_key_min.begin(), out_key_size.begin())), // value out begin
          thrust::equal_to<index_type>(),        // key equal binary predicate
          detail::min_size_functor<index_type>() // value binary operator
      );
      num_unique_keys = end.first - out_key.begin();
      LOG_DEBUG(num_unique_keys, "unique kernel map keys found");
    }
    THRUST_CATCH;

    auto const cpu_out_keys = out_key.to_vector(num_unique_keys);
    auto const cpu_out_offset = out_key_min.to_vector(num_unique_keys);
    auto const cpu_out_size = out_key_size.to_vector(num_unique_keys);
    // thrust::host_vector<index_type> cpu_out_keys(
    //     out_key.begin(), out_key.begin() + num_unique_keys);
    // thrust::host_vector<index_type> cpu_out_offset(
    //     out_key_min.begin(), out_key_min.begin() + num_unique_keys);
    // thrust::host_vector<index_type> cpu_out_size(
    //     out_key_size.begin(), out_key_size.begin() + num_unique_keys);

#ifdef DEBUG
    LOG_DEBUG("Printing cpu keys");
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

  friend std::ostream &operator<<(std::ostream &out,
                                  self_type const &kernel_map) {
    out << kernel_map.to_string();
    return out;
  }

private:
  bool m_decomposed{false};
  bool m_requires_kernel_index;
  size_type m_memory_size_byte, m_capacity;
  std::shared_ptr<index_type[]> m_kernel_index_memory;
  std::shared_ptr<index_type[]> m_in_map_memory;
  std::shared_ptr<index_type[]> m_out_map_memory;
  byte_allocator_type m_allocator;

  std::map<index_type, index_type> m_kernel_size_map;
  std::map<index_type, index_type> m_kernel_offset_map;

public:
  contiguous_memory kernels;
  contiguous_memory in_maps;
  contiguous_memory out_maps;
}; // gpu_kernel_map

} // namespace minkowski

#endif // KERNEL_MAP_CUH
