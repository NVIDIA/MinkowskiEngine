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
#ifndef COORDINATE_MAP_KEY_HPP
#define COORDINATE_MAP_KEY_HPP

#include "types.hpp"
#include "utils.hpp"

#include <vector>

#include <pybind11/pybind11.h>

namespace minkowski {

/*
 * Will be exported to python for lazy key initialization.  For
 * instance, `sparse_tensor.coords_key` can be used for other layers
 * before feedforward
 */
class CoordinateMapKey {
public:
  // clang-format off
  using self_type     = CoordinateMapKey;
  using size_type     = default_types::size_type;
  using stride_type   = default_types::stride_type;
  using hash_key_type = default_types::coordinate_map_hash_type;
  // clang-format on

public:
  CoordinateMapKey() = delete;
  CoordinateMapKey(size_type coordinate_size)
      : m_key_set(false), m_coordinate_size{coordinate_size} {}

  CoordinateMapKey(CoordinateMapKey const &other)
      : m_key_set(other.m_key_set), m_coordinate_size{other.m_coordinate_size},
        m_key(other.m_key) {}

  CoordinateMapKey(size_type coordinate_size,
                   coordinate_map_key_type const &key)
      : m_key_set(true), m_coordinate_size{coordinate_size}, m_key(key) {
    ASSERT(coordinate_size - 1 == m_key.first.size(),
           "Invalid tensor_stride:", m_key.first,
           "coordinate_size:", m_coordinate_size);
  }

  CoordinateMapKey(stride_type tensor_stride, std::string string_id = "")
      : m_coordinate_size(tensor_stride.size() + 1), m_key{std::make_pair(
                                                         tensor_stride,
                                                         string_id)} {
    // valid tensor stride if the coordinate_size match
    m_key = std::make_pair(tensor_stride, string_id);
    m_key_set = true;
  }

  // coordinate_size functions
  size_type get_coordinate_size() const { return m_coordinate_size; }

  // key functions
  void set_key(stride_type tensor_stride, std::string string_id) {
    ASSERT(m_coordinate_size - 1 == tensor_stride.size(),
           "Invalid tensor_stride size:", tensor_stride,
           "coordinate_size:", m_coordinate_size);
    m_key = std::make_pair(tensor_stride, string_id);
    m_key_set = true;
  }

  void set_key(coordinate_map_key_type const &key) {
    ASSERT(m_coordinate_size - 1 == key.first.size(),
           "Invalid tensor_stride size:", key.first,
           "coordinate_size:", m_coordinate_size);
    LOG_DEBUG("Setting the key to ", key.first, ":", key.second);
    m_key = key;
    m_key_set = true;
  }

  coordinate_map_key_type get_key() const {
    ASSERT(is_key_set(), "Key not set");
    return m_key;
  }

  hash_key_type hash() const { return coordinate_map_key_hasher{}(m_key); }

  bool is_key_set() const noexcept { return m_key_set; }

  /*
  void stride(stride_type const &strides) {
    ASSERT(m_tensor_stride_set, "You must set the tensor strides first.");
    ASSERT(m_dimension - 1 == strides.size(), "The size of strides: ", strides,
           "does not match the dimension of the CoordinateMapKey coordinate "
           "system:",
           std::to_string(get_dimension()), ".");
    for (size_type i = 0; i < m_dimension - 1; ++i)
      m_tensor_strides[i] *= strides[i];
  }

  void up_stride(stride_type const &strides) {
    ASSERT(m_tensor_stride_set, "You must set the tensor strides first.");
    ASSERT(m_tensor_strides.size() == strides.size(),
           "The size of the strides: ", strides,
           " does not match the size of the CoordinateMapKey tensor_strides: ",
           m_tensor_strides, ".");
    for (size_type i = 0; i < m_dimension - 1; ++i) {
      ASSERT(m_tensor_strides[i] % strides[i] == 0,
             "The output tensor stride is not divisible by ",
             "up_strides. tensor stride:", m_tensor_strides,
             ", up_strides:", strides, ".");
      m_tensor_strides[i] /= strides[i];
    }
  }*/

  stride_type get_tensor_stride() const { return m_key.first; }

  bool operator==(CoordinateMapKey const &key) const {
    if (!m_key_set || !key.m_key_set)
      return false;
    return m_key == key.m_key;
  }

  // misc functions
  std::string to_string() const {
    Formatter out;
    out << "coordinate map key:" << m_key.first;
    if (m_key.second.length() > 0)
      out << ":" << m_key.second;
    return out;
  }

private:
  bool m_key_set;

  size_type m_coordinate_size;
  coordinate_map_key_type m_key;
}; // CoordinateMapKey

} // namespace minkowski

#endif // COORDINATE_MAP_KEY_HPP
