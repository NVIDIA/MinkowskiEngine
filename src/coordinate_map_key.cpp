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

std::string CoordinateMapKey::to_string() const {
}

} // end namespace minkowski
