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
template <typename coordinate_type,
          typename MapType = CoordinateUnorderedMap<coordinate_type>,
          typename CoordinateAllocator = std::allocator<coordinate_type>>
class CoordinateMap {

public:
  // clang-format off
  using key_type          = typename MapType::key_type;
  using mapped_type       = typename MapType::mapped_type;
  using value_type        = typename MapType::value_type;
  using map_type          = MapType;
  using allocator_type    = CoordinateAllocator;
  using index_type        = default_types::index_type;
  using size_type         = default_types::size_type;
  using self_type         = CoordinateMap<coordinate_type, map_type, allocator_type>;

  // return types
  using index_vector_type = std::vector<default_types::index_type>;
  using index_set_type    = std::set<default_types::index_type>;
  using iterator          = typename map_type::iterator;
  using const_iterator    = typename map_type::const_iterator;

  // clang-format on

  // Constructors
  CoordinateMap() = delete;
  CoordinateMap(size_type const coordinate_size)
      : m_map(MapType{coordinate_size}), m_coordinate_size(coordinate_size), m_capacity(0) {}

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
  // returns: unique_index, reverse_index, batch indices
  std::tuple<index_vector_type, index_vector_type, index_set_type>
  initialize(coordinate_type const *const p_coodinates, //
             size_type const num_coordinates,           //
             size_type const coordinate_size,           //
             bool const force_remap = false,            //
             bool const return_inverse = false);

  // Generate strided version of the input coordinate map.
  // returns mapping: out_coord row index to in_coord row index
  CoordinateMap<MapType>
  stride(default_types::stride_type const &tensor_strides) const;
  CoordinateMap<MapType> stride_region(Region const &region) const;
  CoordinateMap<MapType> prune(bool const *p_keep,
                               size_type num_keep_coordinates) const;
  // class method
  static CoordinateMap<MapType>
  union_coords(const vector<reference_wrapper<CoordsMap<MapType>>> &maps);

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
  // Iterators
  iterator begin() { return m_map.begin(); }
  const_iterator begin() const { return m_map.begin(); }

  iterator end() { return m_map.end(); }
  const_iterator end() const { return m_map.end(); }

  iterator find(key_type const &key) { return m_map.find(key); }
  const_iterator find(key_type const &key) const { return m_map.find(key); }

  size_type size() const { return m_map.size(); }

  void reserve(size_type size) {
    if (m_capacity < size) {
      allocate(size);
      m_map.reserve(size);
    }
  }

  void print() const;

  std::string to_string() const;

  size_type capacity() const { return m_capacity; }

protected:
  // clang-format off
  void allocate(size_type const number_of_coordinates) {
    auto const size = number_of_coordinates * m_coordinate_size;
    coordinate_type *ptr = m_allocator.allocate(size);

    auto deleter = [](coordinate_type *p, CoordinateMap::allocator_type alloc, CoordinateMap::size_type size) {
      alloc.deallocate(p, size);
    };

    m_coordinates = std::unique_ptr<coordinate_type[], std::function<void(coordinate_type *)>>{
        ptr, std::bind(deleter, std::placeholders::_1, m_allocator, size)};
    m_capacity = number_of_coordinates;
  }

  // members
  allocator_type m_allocator;
  map_type m_map;
  std::unique_ptr<coordinate_type[], std::function<void(coordinate_type *)>> m_coordinates;
  size_type m_number_of_coordinates;
  size_type m_coordinate_size;
  size_type m_capacity;
  // clang-format on
};

} // end namespace minkowski

#endif // COORDINATE_MAP_HPP
