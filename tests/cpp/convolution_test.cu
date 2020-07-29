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
#include "coordinate_map.hpp"
#include "coordinate_map_gpu.cuh"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace minkowski {

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor ConvolutionForwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> ConvolutionBackwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

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

  py::enum_<minkowski::RegionType::Type>(m, "RegionType")
      .value("HYPER_CUBE", minkowski::RegionType::Type::HYPER_CUBE)
      .value("HYPER_CROSS", minkowski::RegionType::Type::HYPER_CROSS)
      .value("CUSTOM", minkowski::RegionType::Type::CUSTOM)
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

  py::class_<minkowski::gpu_c10_manager_type<int>>(m, "CoordinateMapManager")
      .def(py::init<>())
      .def("insert_and_map",
           &minkowski::gpu_c10_manager_type<int>::insert_and_map)
      .def("stride",
           (typename py::object //
            (minkowski::gpu_c10_manager_type<int>::*)(
                minkowski::CoordinateMapKey const *in_map_key,
                minkowski::default_types::stride_type const &kernel_stride)) &
               minkowski::gpu_c10_manager_type<int>::stride)
      .def("size", (typename minkowski::default_types::size_type //
                    (minkowski::gpu_c10_manager_type<int>::*)(
                        minkowski::CoordinateMapKey const *map_key) const) &
                       minkowski::gpu_c10_manager_type<int>::size)
      .def("kernel_map", &minkowski::gpu_c10_manager_type<int>::kernel_map);

  m.def("ConvolutionForwardGPU",
        &minkowski::ConvolutionForwardGPU<
            minkowski::default_types::dcoordinate_type,
            minkowski::detail::c10_allocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def("ConvolutionBackwardGPU",
        &minkowski::ConvolutionBackwardGPU<
            minkowski::default_types::dcoordinate_type,
            minkowski::detail::c10_allocator>,
        py::call_guard<py::gil_scoped_release>());
}
