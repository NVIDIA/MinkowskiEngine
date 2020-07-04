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
  using dimension_type = default_types::tensor_order_type;
  using stride_type    = default_types::stride_type;
  using hash_key_type  = default_types::coordinate_map_hash_type;
  // clang-format on

public:
  CoordinateMapKey() { reset(); }
  CoordinateMapKey(dimension_type dim);
  CoordinateMapKey(stride_type const &tensor_strides_, dimension_type dim);

  void reset();
  void copy(py::object ohter);

  // dimension functions
  void set_dimension(dimension_type dim);
  dimension_type get_dimension() const { return m_dimension; }

  // key functions
  void set_key(coordinate_map_key_type key);
  hash_key_type get_key() const;
  bool is_key_set() const { return m_key_set; }
  bool is_tensor_stride_set() const { return m_tensor_stride_set; }

  // stride functions
  void set_tensor_stride(stride_type const &tensor_strides);
  void stride(stride_type const &strides);
  void up_stride(stride_type const &strides);
  stride_type get_tensor_stride() const { return m_tensor_strides; }

  // misc functions
  std::string to_string() const;

private:
  coordinate_map_key_type m_key; // Use the key_ for all coordshashmap query. Lazily set
  // The dimension of the current coordinate system. The order of the system is
  // D + 1.
  dimension_type m_dimension = 0;
  stride_type m_tensor_strides;
  bool m_key_set = false;
  bool m_tensor_stride_set = false;
}; // CoordinateMapKey

} // namespace minkowski

#endif // COORDINATE_MAP_KEY_HPP
