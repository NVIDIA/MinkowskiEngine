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
#ifndef COORDINATE_UNORDERED_MAP_HPP
#define COORDINATE_UNORDERED_MAP_HPP

#include "coordinate.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <tuple>

#include <robin_hood.h>

namespace minkowski {

// unordered_map wrapper
// clang-format off
template <typename coordinate_type,
          typename Hash      = detail::coordinate_murmur3<coordinate_type>,
          typename KeyEqual  = detail::coordinate_equal_to<coordinate_type>>
class CoordinateUnorderedMap {
public:
  using size_type      = default_types::size_type;
  using key_type       = coordinate<coordinate_type>;
  using mapped_type    = default_types::index_type;
  using hasher         = Hash;
  using key_equal      = KeyEqual;
  using map_type       =
      robin_hood::unordered_flat_map<key_type,    // key
                                     mapped_type, // mapped_type
                                     hasher,      // hasher
                                     key_equal    // equality
                                     >;
  using value_type     = typename map_type::value_type;
  using iterator       = typename map_type::iterator;
  using const_iterator = typename map_type::const_iterator;
  // clang-format on

public:
  // Constructors
  CoordinateUnorderedMap() = delete;
  CoordinateUnorderedMap(size_type const coordinate_size)
      : m_map(
            map_type{0, hasher{coordinate_size}, key_equal{coordinate_size}}) {}
  CoordinateUnorderedMap(size_type const number_of_coordinates,
                         size_type const coordinate_size)
      : CoordinateUnorderedMap{coordinate_size} {
    m_map.reserve(number_of_coordinates);
  }

  // Iterators
  inline iterator begin() { return m_map.begin(); }
  inline const_iterator begin(const size_t idx) const {
    return m_map.begin(idx);
  }
  inline const_iterator begin() const { return m_map.begin(); }

  inline iterator end() { return m_map.end(); }
  inline const_iterator end() const { return m_map.end(); }

  inline iterator find(key_type const &key) { return m_map.find(key); }
  inline const_iterator find(key_type const &key) const {
    return m_map.find(key);
  }

  inline std::pair<iterator, bool> insert(value_type const &keyval) {
    return m_map.insert(keyval);
  }
  inline std::pair<iterator, bool> insert(value_type &&keyval) {
    return m_map.insert(keyval);
  }

  // mapped_type &operator[](key_type const &key) { return m_map[key]; }

  inline size_type size() const noexcept { return m_map.size(); }
  inline size_type capacity() const noexcept { return m_map.capacity(); }
  inline void reserve(size_type c) { m_map.reserve(c); }

private:
  map_type m_map;
};

} // end namespace minkowski

#endif // COORDINATE_UNORDERED_MAP_HPP
