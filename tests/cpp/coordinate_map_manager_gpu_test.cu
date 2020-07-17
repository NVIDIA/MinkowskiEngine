/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
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
#include "allocators.cuh"
#include "coordinate_map_gpu.cuh"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>

#include <algorithm>
#include <vector>

namespace minkowski {

using coordinate_type = int32_t;
using index_type = default_types::index_type;
using size_type = default_types::size_type;
using stride_type = default_types::stride_type;

namespace detail {

template <typename container_type>
std::vector<at::Tensor> to_torch(container_type const &maps) {
  LOG_DEBUG("to_torch");
  std::vector<at::Tensor> tensors;
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt)
                     .device(torch::kCUDA, 0)
                     .layout(torch::kStrided)
                     .requires_grad(false);

  for (auto it = maps.key_cbegin(); it != maps.key_cend(); ++it) {
    auto key = it->first;
    LOG_DEBUG("Copy", maps.size(key));
    at::Tensor tensor = torch::empty({maps.size(key)}, options);
    CUDA_CHECK(cudaMemcpy(tensor.data_ptr<int32_t>(), maps.begin(key),
                          maps.size(key) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    tensors.push_back(std::move(tensor));
  }
  LOG_DEBUG("Copy done");
  return tensors;
}

} // namespace detail

using manager_type =
    CoordinateMapManager<coordinate_type, detail::c10_allocator,
                         CoordinateMapGPU>;

std::tuple<py::object, py::object, std::pair<at::Tensor, at::Tensor>>
coordinate_map_manager_test(const torch::Tensor &coordinates,
                            std::string string_id) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinates, 2);

  auto const D = (index_type)coordinates.size(1);
  manager_type *p_manager = new manager_type();
  py::object py_manager = py::cast(p_manager);
  stride_type tensor_stride;
  for (index_type i = 0; i < D - 1; ++i) {
    tensor_stride.push_back(1);
  }

  auto key_and_map =
      p_manager->insert_and_map(coordinates, tensor_stride, string_id);

  return std::make_tuple(std::get<0>(key_and_map), py_manager,
                         std::get<1>(key_and_map));
}

py::object coordinate_map_manager_stride(manager_type *p_manager,
                                         CoordinateMapKey const *p_map_key,
                                         stride_type const &stride_size) {
  auto key_bool = p_manager->stride(p_map_key->get_key(), stride_size);

  auto key = CoordinateMapKey(stride_size.size() + 1, std::get<0>(key_bool));
  return py::cast(key);
}

std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
coordinate_map_manager_kernel_map(py::object manager,
                                  CoordinateMapKey const *p_in_map_key,
                                  CoordinateMapKey const *p_out_map_key,
                                  stride_type const &kernel_size) {
  manager_type *p_manager = py::cast<manager_type *>(manager);

  stride_type kernel_stride;
  stride_type kernel_dilation;
  for (index_type i = 0; i < kernel_size.size(); ++i) {
    kernel_stride.push_back(1);
    kernel_dilation.push_back(1);
  }

  auto offset = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));

  auto const &kernel_map = p_manager->kernel_map(
      p_in_map_key, p_out_map_key, kernel_size, kernel_stride, kernel_dilation,
      RegionType::HYPER_CUBE, offset, false, false);
  LOG_DEBUG("kernel_map generated");

  return std::make_pair(detail::to_torch(kernel_map.in_maps),
                        detail::to_torch(kernel_map.out_maps));
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
      .def("get_coordinate_size",
           &minkowski::CoordinateMapKey::get_coordinate_size)
      .def("get_key", &minkowski::CoordinateMapKey::get_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::default_types::stride_type, std::string)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("get_tensor_stride",
           &minkowski::CoordinateMapKey::get_tensor_stride);

  py::class_<minkowski::CoordinateMapManager<
      int32_t, minkowski::detail::c10_allocator, minkowski::CoordinateMapGPU>>(
      m, "CoordinateMapManager")
      .def(py::init<>())
      .def("insert_and_map", &minkowski::CoordinateMapManager<
                                 int32_t, minkowski::detail::c10_allocator,
                                 minkowski::CoordinateMapGPU>::insert_and_map)
      .def("kernel_map", &minkowski::CoordinateMapManager<
                             int32_t, minkowski::detail::c10_allocator,
                             minkowski::CoordinateMapGPU>::kernel_map);

  m.def("coordinate_map_manager_test", &minkowski::coordinate_map_manager_test,
        "Minkowski Engine coordinate map manager test");

  m.def("coordinate_map_manager_stride",
        &minkowski::coordinate_map_manager_stride,
        "Minkowski Engine coordinate map manager stride test");

  m.def("coordinate_map_manager_kernel_map",
        &minkowski::coordinate_map_manager_kernel_map,
        "Minkowski Engine coordinate map manager test");
}
