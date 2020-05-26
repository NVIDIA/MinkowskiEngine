/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#include "coordinate_map_key.hpp"
#include "utils.hpp" // vector to string

namespace py = pybind11;

namespace minkowski {

CoordinateMapKey::CoordinateMapKey(dimension_type dim) {
  reset();
  set_dimension(dim);
}

CoordinateMapKey::CoordinateMapKey(stride_type const &tensor_strides,
                                   dimension_type dim) {
  reset();
  set_dimension(dim);
  set_tensor_stride(tensor_strides);
}

void CoordinateMapKey::set_tensor_stride(stride_type const &tensor_strides) {
  dimension_type const dimension = get_dimension();
  ASSERT(dimension > 0 and dimension == tensor_strides.size(),
         "The tensor strides dimension mismatch: ", tensor_strides,
         ", dimension of the key: ", dimension);
  m_tensor_strides = tensor_strides;
  m_tensor_stride_set = true;
}

/*
 * @brief Increase the current `tensor_stride` by the input `strides`
 */
void CoordinateMapKey::stride(stride_type const &strides) {
  ASSERT(m_tensor_stride_set, "You must set the tensor strides first.");
  ASSERT(get_dimension() == strides.size(), "The size of strides: ", strides,
         " does not match the dimension of the CoordinateMapKey coordinate "
         "system: ",
         std::to_string(get_dimension()), ".");
  for (dimension_type i = 0; i < get_dimension(); i++)
    m_tensor_strides[i] *= strides[i];
}

void CoordinateMapKey::up_stride(stride_type const &strides) {
  ASSERT(m_tensor_stride_set, "You must set the tensor strides first.");
  ASSERT(get_dimension() == strides.size(), "The size of strides: ", strides,
         " does not match the dimension of the CoordinateMapKey coordinate "
         "system: ",
         std::to_string(get_dimension()), ".");
  ASSERT(m_tensor_strides.size() == strides.size(),
         "The size of the strides: ", strides,
         " does not match the size of the CoordinateMapKey tensor_strides: ",
         m_tensor_strides, ".");
  for (dimension_type i = 0; i < get_dimension(); i++) {
    ASSERT(m_tensor_strides[i] % strides[i] == 0,
           "The output tensor stride is not divisible by ",
           "up_strides. tensor stride: ", m_tensor_strides,
           ", up_strides: ", strides, ".");
    m_tensor_strides[i] /= strides[i];
  }
}

void CoordinateMapKey::copy(py::object py_other) {
  CoordinateMapKey *p_other = py_other.cast<CoordinateMapKey *>();
  set_key(p_other->get_key()); // Call first to set the key_set.
  set_dimension(p_other->get_dimension());

  m_key_set = p_other->is_key_set();
  m_tensor_stride_set = p_other->is_tensor_stride_set();

  if (p_other->is_tensor_stride_set()) {
    ASSERT(get_dimension() == p_other->get_tensor_stride().size(),
           "The size of strides: ", p_other->get_tensor_stride(),
           " does not match the dimension of the CoordinateMapKey coordinate "
           "system: ",
           std::to_string(get_dimension()), ".");
    m_tensor_strides = p_other->get_tensor_stride();
  }
}

void CoordinateMapKey::reset() {
  m_key = 0;
  m_dimension = -1;
  m_key_set = false;
  m_tensor_stride_set = false;
  m_tensor_strides.clear();
}

void CoordinateMapKey::set_key(hash_key_type key) {
  m_key = key;
  m_key_set = true;
}

void CoordinateMapKey::set_dimension(dimension_type dim) {
  ASSERT(dim > 0, "Invalid dimension: ", std::to_string(dim), ".");
  m_dimension = dim;
  m_tensor_strides.resize(m_dimension);
}

CoordinateMapKey::hash_key_type CoordinateMapKey::get_key() const {
  ASSERT(m_key_set, "CoordinateMapKey: Key Not set")
  return m_key;
}

std::string CoordinateMapKey::to_string() const {
  Formatter out;
  out << "< CoordinateMapKey, key: "
      << (m_key_set ? std::to_string(m_key) : "None") << ", tensor_stride: "
      << (m_tensor_stride_set ? ArrToString(m_tensor_strides) : "None")
      << " in dimension: " << std::to_string(m_dimension) << " >\n";
  return out;
}

} // end namespace minkowski
