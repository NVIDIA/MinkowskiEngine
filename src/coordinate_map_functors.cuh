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

#include "coordinate.hpp"
#include "types.hpp"

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

namespace minkowski {

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

template <typename coordinate_type, typename map_type> struct get_element {
  using value_type = thrust::pair<coordinate<coordinate_type>, uint32_t>;

  get_element(map_type &map) : m_map(map) {}

  __device__ uint32_t operator()(value_type const &x) { return x.second; }

  map_type const m_map;
};

template <typename coordinate_type, typename map_type, typename mapped_iterator>
struct insert_coordinate {
  using value_type = typename map_type::value_type;
  using mapped_type = typename map_type::mapped_type;

  /**
   * insert_coordinate functor constructor
   * @param map
   * @param p_coordinate a pointer to the start of the coordinate
   * @param value_iter a mapped_iterator that points to the begin. This could be
   * a pointer or an iterator that supports operat+(int) and operator*().
   * @param coordinate_size
   */
  insert_coordinate(map_type &map,                       // underlying map
                    coordinate_type const *p_coordinate, // key coordinate begin
                    mapped_iterator value_iter,
                    uint32_t const coordinate_size) // coordinate size
      : m_coordinate_size{coordinate_size}, m_coordinates{p_coordinate},
        m_value_iter{value_iter}, m_map{map} {}

  /*
   * @brief insert a <coordinate, row index> pair into the unordered_map
   *
   * @return thrust::pair<bool, uint32_t> of a success flag and the current
   * index.
   */
  __device__ bool operator()(uint32_t i) {
    auto coord =
        coordinate<coordinate_type>{&m_coordinates[i * m_coordinate_size]};
    value_type pair = thrust::make_pair(coord, *(m_value_iter + i));
    // m_map.insert(pair);
    // Returns pair<iterator, (bool)insert_success>
    auto result = m_map.insert(pair);
    return result.second;
  }

  size_t const m_coordinate_size;
  coordinate_type const *m_coordinates;
  mapped_iterator m_value_iter;
  map_type m_map;
};

template <typename coordinate_type, typename map_type>
struct find_coordinate
    : public thrust::unary_function<uint32_t,
                                    thrust::pair<uint32_t, uint32_t>> {
  using mapped_type = typename map_type::mapped_type;
  using return_type = thrust::pair<mapped_type, mapped_type>;

  find_coordinate(map_type const &_map, coordinate_type const *_d_ptr,
                  mapped_type const unused_element, size_t const _size)
      : m_coordinate_size{_size},
        m_unused_element(unused_element), m_coordinates{_d_ptr}, m_map{_map} {}

  __device__ mapped_type operator()(uint32_t i) {
    auto coord =
        coordinate<coordinate_type>{&m_coordinates[i * m_coordinate_size]};
    auto result = m_map.find(coord);
    if (result == m_map.end()) {
      return m_unused_element;
    }
    return result->second;
  }

  size_t const m_coordinate_size;
  mapped_type const m_unused_element;
  coordinate_type const *m_coordinates;
  map_type const m_map;
};

template <typename coordinate_type, typename map_type> struct update_value {
  using mapped_type = typename map_type::mapped_type;

  update_value(map_type &_map, coordinate_type const *coordinates,
               uint32_t const *valid_index, size_t const _size)
      : m_coordinate_size{_size}, m_coordinates{coordinates},
        m_valid_index(valid_index), m_map{_map} {}

  __device__ void operator()(uint32_t i) {
    auto coord = coordinate<coordinate_type>{
        &m_coordinates[m_valid_index[i] * m_coordinate_size]};
    auto result = m_map.find(coord);
    result->second = i;
  }

  size_t const m_coordinate_size;
  coordinate_type const *m_coordinates;
  uint32_t const *m_valid_index;
  map_type m_map;
};

template <bool value> struct is_first {
  template <typename Tuple>
  inline __device__ bool operator()(Tuple const &item) const {
    return thrust::get<0>(item) == value;
  }
};

template <typename coordinate_type, typename mapped_type>
struct is_unused_pair {

  is_unused_pair(mapped_type const unused_element)
      : m_unused_element(unused_element) {}

  template <typename Tuple>
  inline __device__ bool operator()(Tuple const &item) const {
    return thrust::get<1>(item) == m_unused_element;
  }

  mapped_type const m_unused_element;
};

template <typename T, typename pair_type, typename pair_iterator>
struct split_functor {

  split_functor(pair_iterator begin, T *firsts, T *seconds)
      : m_begin(begin), m_firsts(firsts), m_seconds(seconds) {}

  __device__ void operator()(uint32_t i) {
    pair_type const item = *(m_begin + i);
    m_firsts[i] = item.first;
    m_seconds[i] = item.second;
  }

  pair_iterator m_begin;
  T *m_firsts;
  T *m_seconds;
};

template <typename coordinate_type, typename index_type> struct stride_copy {

  stride_copy(size_t const coordinate_size, index_type const *stride,
              index_type const *valid_src_index, coordinate_type const *src,
              coordinate_type *dst)
      : m_coordinate_size(coordinate_size), m_stride(stride),
        m_valid_src_index(valid_src_index), m_src(src), m_dst(dst) {}

  __device__ void operator()(uint32_t const i) {
    const auto start = m_valid_src_index[i] * m_coordinate_size;
    m_dst[start] = m_src[start];
    for (auto j = 1; j < m_coordinate_size; j++) {
      m_dst[start + j] = ((coordinate_type)floorf(
                             __fdiv_rd(m_src[start + j], m_stride[j - 1]))) *
                         m_stride[j - 1];
    }
  }

  size_t const m_coordinate_size;
  index_type const *m_stride;
  index_type const *m_valid_src_index;
  coordinate_type const *m_src;
  coordinate_type *m_dst;
};

template <typename T> struct min_size_functor {
  using min_size_type = thrust::tuple<T, T>;

  __host__ __device__ min_size_type operator()(const min_size_type &lhs,
                                               const min_size_type &rhs) {
    return thrust::make_tuple(min(thrust::get<0>(lhs), thrust::get<0>(rhs)),
                              thrust::get<1>(lhs) + thrust::get<1>(rhs));
  }
};
} // end namespace detail

} // end namespace minkowski

#endif // COORDS_FUNCTORS_CUH
