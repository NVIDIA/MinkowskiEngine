/* Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)
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

#include <torch/extension.h>
#include <vector>

namespace minkowski {

void coordinate_map_key_test() {
  // Check basic type compilation
  CoordinateMapKey key{3};
  CoordinateMapKey key2{default_types::stride_type{2, 3, 4}};
}

py::object coordinate_map_key_return(default_types::stride_type tensor_stride,
                                     std::string string_id) {
  return py::cast(new CoordinateMapKey(tensor_stride, string_id));
}

void coordinate_map_key_update(CoordinateMapKey *map_key,
                               default_types::stride_type tensor_stride,
                               std::string string_id) {
  map_key->set_key(tensor_stride, string_id);
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<minkowski::CoordinateMapKey>(m, "CoordinateMapKey")
      .def(py::init<minkowski::default_types::size_type>())
      .def(py::init<minkowski::default_types::stride_type, std::string>())
      .def("__repr__", &minkowski::CoordinateMapKey::to_string)
      .def("get_coordinate_size",
           &minkowski::CoordinateMapKey::get_coordinate_size)
      .def("get_key", &minkowski::CoordinateMapKey::get_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::default_types::stride_type, std::string)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("get_tensor_stride",
           &minkowski::CoordinateMapKey::get_tensor_stride);

  m.def("coordinate_map_key_return", &minkowski::coordinate_map_key_return,
        "Minkowski Engine coordinate map key return test");

  m.def("coordinate_map_key_test", &minkowski::coordinate_map_key_test,
        "Minkowski Engine coordinate map key test");

  m.def("coordinate_map_key_update", &minkowski::coordinate_map_key_update,
        "Minkowski Engine coordinate map key update test");
}
