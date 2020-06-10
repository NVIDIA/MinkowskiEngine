/* Copyright (c) 2020 NVIDIA CORPORATION.
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
#ifndef COORDINATE_MAP_GPU_CUH
#define COORDINATE_MAP_GPU_CUH

#include "3rdparty/concurrent_unordered_map.cuh"
#include "3rdparty/hash/helper_functions.cuh"
#include "coordinate_map.hpp"
#include "coordinate_map_functors.cuh"

#include <thrust/device_vector.h>

namespace minkowski {

// clang-format off
/*
 * Inherit from the CoordinateMap for a concurrent coordinate unordered map.
 */
template <
    typename coordinate_type,
    typename MapAllocator = default_allocator<thrust::pair<coordinate<coordinate_type>, default_types::index_type>>,
    typename CoordinateAllocator = default_allocator<coordinate_type>>
class CoordinateMapGPU
    : public CoordinateMap<coordinate_type, CoordinateAllocator> {
public:
  // clang-format off
  using base_type         = CoordinateMap<coordinate_type, CoordinateAllocator>;
  using self_type         = CoordinateMapGPU<coordinate_type, MapAllocator, CoordinateAllocator>;
  using size_type         = typename base_type::size_type;
  using index_type        = typename base_type::index_type;
  using stride_type       = typename base_type::stride_type;

  // Map types
  using key_type          = coordinate<coordinate_type>;
  using mapped_type       = index_type;
  using hasher_type       = detail::coordinate_murmur3<coordinate_type>;
  using key_equal_type    = detail::coordinate_equal_to<coordinate_type>;
  using map_type          = concurrent_unordered_map<key_type,        // key
                                                     mapped_type,     // mapped_type
                                                     hasher_type,     // hasher
                                                     key_equal_type,  // equality
                                                     MapAllocator>;   // allocator
  using value_type        = typename map_type::value_type;

  // iterator
  using iterator          = typename map_type::iterator;
  using const_iterator    = typename map_type::const_iterator;

  // index vectors
  using index_vector_type = typename base_type::index_vector_type;

  // allocators
  using coordinate_allocator_type = CoordinateAllocator;
  using map_allocator_type        = MapAllocator;
  // clang-format on

  // return types
  // using the QueryResultAllocator gives segfault!
  using device_index_vector_type = thrust::device_vector<index_type>;

public:
  CoordinateMapGPU() = delete;
  CoordinateMapGPU(
      size_type const number_of_coordinates, size_type const coordinate_size,
      stride_type const stride = {1},
      coordinate_allocator_type coord_alloc = coordinate_allocator_type(),
      map_allocator_type map_alloc = map_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, coord_alloc),
        m_capacity(0), // should be updated in the reserve
        m_hasher(hasher_type{coordinate_size}),
        m_equal(key_equal_type{coordinate_size}),
        m_unused_key(coordinate<coordinate_type>{nullptr}),
        m_unused_element(std::numeric_limits<coordinate_type>::max()) {
    reserve(number_of_coordinates);
    // copy the tensor_stride
    m_device_tensor_stride = base_type::m_tensor_stride;
    LOG_DEBUG("device tensor_stride:", m_device_tensor_stride);
  }

  template <typename mapped_iterator>
  void insert(coordinate_iterator<coordinate_type> key_first,
              coordinate_iterator<coordinate_type> key_last,
              mapped_iterator value_first, mapped_iterator value_last);

  std::pair<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>
  find(coordinate_iterator<coordinate_type> key_first,
       coordinate_iterator<coordinate_type> key_last) const;

  inline void reserve(size_type size) {
    if (size > m_capacity) {
      // reserve coordinate
      base_type::reserve(size);
      // reserve map
      LOG_DEBUG("Reserve map of", size, "for concurrent_unordered_map.");
      m_map = std::move(map_type::create(compute_hash_table_size(size),
                                         m_unused_element, m_unused_key,
                                         m_hasher, m_equal, m_map_allocator));
      CUDA_TRY(cudaStreamSynchronize(0));
      m_capacity = size;
    }
  }

  // Network specific functions.

  /*
   * @brief strided coordinate map.
   */
  self_type stride(stride_type const &stride) const;

  inline size_type size() const { return m_valid_index.size(); }

  // access the coordinate data pointer
  using base_type::coordinate_data;
  using base_type::const_coordinate_data;

  // Find GPU values in the map. key_iterator must be a GPU iterator.
  // template <typename key_iterator>
  // std::pair<device_index_vector_type, device_index_vector_type>
  // find(key_iterator key_first, key_iterator key_last);

private:
  using base_type::m_coordinate_size;
  size_type m_capacity;
  hasher_type const m_hasher;
  key_equal_type const m_equal;
  key_type const m_unused_key;
  mapped_type const m_unused_element;
  device_index_vector_type m_valid_index;

  thrust::device_vector<size_type> m_device_tensor_stride;
  map_allocator_type m_map_allocator;
  coordinate_allocator_type m_coordinate_allocator;
  std::unique_ptr<map_type, std::function<void(map_type *)>> m_map;
};

} // namespace minkowski

#endif // COORDINATE_MAP_GPU_CUH
