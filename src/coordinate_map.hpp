/* Copyright (c) 2020 NVIDIA CORPORATION.
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
#ifndef COORDINATE_MAP_HPP
#define COORDINATE_MAP_HPP

#include "coordinate.hpp"
#include "coordinate_unordered_map.hpp"
#include "region.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include <robin_hood.h>

namespace minkowski {

namespace detail {

/*
template <typename Itype> struct byte_hash_vec {
  std::size_t operator()(std::vector<Itype> const &vec) const noexcept {
    return robin_hood::hash_bytes(vec.data(), sizeof(Itype) * vec.size());
  }
};
*/

template <typename Itype>
inline std::vector<Itype>
stride_copy(const vector<Itype> &src,
            const vector<Itype> &tensor_strides) noexcept {
  vector<Itype> dst{src.size()};
  const std::size_t size = tensor_strides.size();
  constexpr int COORD_START = 1;
  dst[0] = src[0];
  for (std::size_t i = 0; i < size; i++) {
    dst[i + COORD_START] =
        std::floor((float)src[i + COORD_START] / tensor_strides[i]) *
        tensor_strides[i];
  }
  return dst;
}

} // namespace detail

/*
 * @brief A wrapper for a coordinate map.
 *
 * @note
 */
// clang-format off
template <typename coordinate_type, typename CoordinateAllocator>
class CoordinateMap {

public:
  using index_type     = default_types::index_type;
  using size_type      = default_types::size_type;
  using allocator_type = CoordinateAllocator;

  // return types
  using index_vector_type = std::vector<default_types::index_type>;
  using index_set_type    = std::set<default_types::index_type>;

  // clang-format on

  // Constructors
  CoordinateMap() = delete;
  CoordinateMap(size_type const number_of_coordinates,
                size_type const coordinate_size,
                allocator_type alloc = allocator_type())
      : m_coordinate_size(coordinate_size), m_capacity(0),
        m_allocator(alloc) /* m_capacity is updated in the allocate function */
  {
    allocate(number_of_coordinates);
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   */
  template <typename key_iterator, typename mapped_iterator>
  void insert(key_iterator key_first, key_iterator key_last,
              mapped_iterator value_first, mapped_iterator value_last) {
    ASSERT(false, "Not implemented");
  }

  // /*
  //  * @brief given a key iterator begin-end pair and a value iterator
  //  begin-end
  //  * pair, insert all elements.
  //  */
  // template <key_iterator, mapped_iterator>
  // virtual std::pair(index_vector_type, index_vector_type)
  //     insert(key_iterator key_first, key_iterator key_last,
  //            mapped_iterator value_first, mapped_iterator value_last);

  /*
  // Generate strided version of the input coordinate map.
  // returns mapping: out_coord row index to in_coord row index
  CoordinateMap<MapType>
  stride(default_types::stride_type const &tensor_strides) const {
    ASSERT(false, "Not implemented");
  }
  CoordinateMap<MapType> stride_region(Region const &region) const {
    ASSERT(false, "Not implemented");
  }
  CoordinateMap<MapType> prune(bool const *p_keep,
                               size_type num_keep_coordinates) const {
    ASSERT(false, "Not implemented");
  }
  // class method
  static CoordinateMap<MapType>
  union_coords(const vector<reference_wrapper<CoordsMap<MapType>>> &maps) {
    ASSERT(false, "Not implemented");
  }

  // Generate in-out kernel maps
  InOutMapsPair<int>
  kernel_map(const CoordsMap<CoordsToIndexMap> &out_coords_map,
             const Region &region) const;
  InOutMapsPair<int>
  pruned_kernel_map(const CoordsMap<MapType> &out_coords_map) const;
  InOutMapsPair<int>
  global_reduction_map(const CoordsMap<MapType> &gout_coords_map,
                       bool return_per_batch = true) const;
  InOutMapsPair<int> stride_map(const CoordsMap &out_coords_map,
                                const vector<int> &tensor_strides) const;
  static InOutMapsPair<int>
  union_map(const vector<reference_wrapper<CoordsMap>> &in_maps,
            const CoordsMap &out_map);

  */

  coordinate_type *coordinate_data() { return m_coordinates.get(); }

  void reserve(size_type size) {
    if (m_capacity < size) {
      LOG_DEBUG("Reserve coordinates:", size, "current capacity:", m_capacity);
      allocate(size);
    }
  }

  std::string to_string() const;

  size_type capacity() const { return m_capacity; }

protected:
  // clang-format off
  void allocate(size_type const number_of_coordinates) {
    if (m_capacity < number_of_coordinates) {
      LOG_DEBUG("Allocate", number_of_coordinates, "coordinates.");
      auto const size = number_of_coordinates * m_coordinate_size;
      coordinate_type *ptr = m_allocator.allocate(size);

      auto deleter = [](coordinate_type *p, allocator_type alloc, size_type size) {
        alloc.deallocate(p, size);
      };

      m_coordinates = std::unique_ptr<coordinate_type[], std::function<void(coordinate_type *)>>{
          ptr, std::bind(deleter, std::placeholders::_1, m_allocator, size)};
      m_capacity = number_of_coordinates;
    }
  }

protected:
  // members
  size_type m_number_of_coordinates;
  size_type m_coordinate_size;
  size_type m_capacity;

  allocator_type m_allocator;
  std::unique_ptr<coordinate_type[], std::function<void(coordinate_type *)>> m_coordinates;
  // clang-format on
};

} // end namespace minkowski

#endif // COORDINATE_MAP_HPP
