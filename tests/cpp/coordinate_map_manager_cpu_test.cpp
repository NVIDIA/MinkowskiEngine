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
#include "coordinate_map_cpu.hpp"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>

namespace minkowski {

using coordinate_type = int32_t;
using index_type = default_types::index_type;
using size_type = default_types::size_type;
using stride_type = default_types::stride_type;

std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
coordinate_map_manager_test(const torch::Tensor &coordinates,
                            std::string string_id) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);

  auto const D = (index_type)coordinates.size(1);

  CoordinateMapManager<coordinate_type, std::allocator, CoordinateMapCPU>
      manager;

  stride_type tensor_stride;
  for (index_type i = 0; i < D - 1; ++i) {
    tensor_stride.push_back(1);
  }

  return manager.insert_and_map(coordinates, tensor_stride, string_id);
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<minkowski::GPUMemoryAllocatorBackend::Type>(m, "GPUMemoryAllocator")
      .value("PYTORCH", minkowski::GPUMemoryAllocatorBackend::Type::PYTORCH)
      .value("CUDA", minkowski::GPUMemoryAllocatorBackend::Type::CUDA)
      .export_values();

  py::enum_<minkowski::CoordinateMapBackend::Type>(m, "CoordinateMap")
      .value("CPU", minkowski::CoordinateMapBackend::Type::CPU)
      .value("PYTORCH", minkowski::CoordinateMapBackend::Type::CUDA)
      .export_values();

  py::class_<minkowski::CoordinateMapKey>(m, "CoordinateMapKey")
      .def(py::init<minkowski::default_types::size_type>())
      .def(py::init<minkowski::default_types::stride_type, std::string>())
      .def("__repr__", &minkowski::CoordinateMapKey::to_string)
      .def("get_dimension", &minkowski::CoordinateMapKey::get_dimension)
      .def("get_key", &minkowski::CoordinateMapKey::get_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::default_types::stride_type, std::string)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("get_tensor_stride",
           &minkowski::CoordinateMapKey::get_tensor_stride);

  py::class_<minkowski::CoordinateMapManager<int32_t, std::allocator,
                                             minkowski::CoordinateMapCPU>>(
      m, "CoordinateMapManager")
      .def(py::init<>())
      .def("insert_and_map", &minkowski::CoordinateMapManager<
                                 int32_t, std::allocator,
                                 minkowski::CoordinateMapCPU>::insert_and_map);

  m.def("coordinate_map_manager_test", &minkowski::coordinate_map_manager_test,
        "Minkowski Engine coordinate map manager test");
}
