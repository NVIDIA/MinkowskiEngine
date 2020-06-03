/* Copyright (c) 2020 NVIDIA CORPORATION.
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

#include "concurrent_coordinate_unordered_map.cuh"
#include "coordinate_map.hpp"

namespace minkowski {

/*
 * Inherit from the CoordinateMap for a concurrent coordinate unordered map.
 */
// clang-format off
template <typename coordinate_type,
          typename MapAllocator        = default_allocator<thrust::pair<coordinate<coordinate_type>, default_types::index_type>>,
          typename CoordinateAllocator = default_allocator<coordinate_type>>
class CoordinateMapGPU
    : public CoordinateMap<
          coordinate_type,
          ConcurrentCoordinateUnorderedMap<coordinate_type, MapAllocator>,
          CoordinateAllocator> {
public:
  using base_type      = CoordinateMap<coordinate_type>;
  using size_type      = typename base_type::size_type;
  using key_type       = typename base_type::key_type;
  using mapped_type    = default_types::index_type; // MapAllocator uses default_types
  using value_type     = typename base_type::value_type;
  using index_type     = default_types::index_type; // MapAllocator uses default_types
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;
  using allocator_type = CoordinateAllocator;

  // return types
  using host_index_vector_type   = thrust::host_vector<coordinate_type, allocator_type>;
  using device_index_vector_type = thrust::device_vector<coordinate_type, allocator_type>;
  // clang-format on

public:
  CoordinateMapGPU(size_type const number_of_coordinates,
                   size_type const coordinate_size)
      : base_type{number_of_coordinates, coordinate_size} {}

  // Insert GPU values to the map. key_iterator and mapped_iterator must be GPU
  // iterators.
  template <typename key_iterator, typename mapped_iterator>
  void insert(key_iterator key_first, key_iterator key_last,
              mapped_iterator value_first, mapped_iterator value_last);

  // Find GPU values in the map. key_iterator must be a GPU iterator.
  // template <typename key_iterator>
  // std::pair<device_index_vector_type, device_index_vector_type>
  // find(key_iterator key_first, key_iterator key_last);
};

} // namespace minkowski

#endif // COORDINATE_MAP_GPU_CUH
