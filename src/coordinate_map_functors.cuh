/*
 *  Copyright 2020 NVIDIA CORPORATION.
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
 */
#ifndef COORDINATE_FUNCTORS_CUH
#define COORDINATE_FUNCTORS_CUH

namespace minkowski {

extern template <typename coordinate_type> struct coordinate;

namespace detail {

template <typename Key, typename Element, typename Equality> struct is_used {
  using value_type = thrust::pair<Key, Element>;

  is_used(Key &&unused, Equality &&equal)
      : m_unused_key(unused), m_equal(equal) {}

  is_used(Key const &unused, Equality const &equal)
      : m_unused_key(unused), m_equal(equal) {}

  __host__ __device__ bool operator()(value_type const &x) {
    return !m_equal(x.first, m_unused_key);
  }

  Key const m_unused_key;
  Equality const m_equal;
};

template <typename coordinate_type, typename map_type>
struct insert_coordinate {
  using value_type = typename map_type::value_type;
  using mapped_type = typename map_type::mapped_type;

  insert_coordinate(
      map_type &map /* underlying map */,
      coordinate_type const *const p_coordinate /* key coordinate begin */,
      mapped_type const *const p_value /* value begin */,
      uint32_t const coordinate_size /* coordinate size */)
      : m_coordinate_size{coordinate_size},
        m_coordinate{p_coordinate}, m_value{p_value}, m_map{map} {}

  /*
   * @brief insert a <coordinate, row index> pair into the unordered_map
   *
   * @return thrust::pair<bool, uint32_t> of a success flag and the current
   * index.
   */
  __device__ thrust::pair<bool, uint32_t> operator()(uint32_t i) {
    value_type pair = thrust::make_pair(
        coordinate<coordinate_type>{&d_ptr[i * m_coordinate_size]}, p_value[i]);
    // Returns pair<iterator, (bool)insert_success>
    auto result = map.insert(pair);
    return thrust::make_pair<bool, uint32_t>(
        result.first != map.end() and result.second, i);
  }

  size_t const m_coordinate_size;
  coordinate_type const *m_coordinate;
  mapped_type const *m_value;
  map_type &m_map;
};

// clang-format off
template <typename map_type, typename pair_type, typename coordinate_type>
struct find_coordinate {
  using value_type = typename pair_type::second_type;
  find_coordinate(map_type const &_map, coordinate_type const *_d_ptr, size_t _size)
      : map{_map}, d_ptr{_d_ptr}, size{_size} {}

  __device__ value_type operator()(uint32_t i) {
    auto coord = coordinate<coordinate_type>{&d_ptr[i * size]};
    auto result = map.find(coord);
    if (result == map.end()) {
      return std::numeric_limits<value_type>::max();
    }
    return result->second;
  }

  map_type const &map;
  coordinate_type const *d_ptr;
  size_t const size;
};

} // end namespace detail

} // end namespace minkowski

#endif // COORDS_FUNCTORS_CUH
