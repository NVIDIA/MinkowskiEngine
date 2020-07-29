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
#include "kernel_region.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>

namespace minkowski {

using coordinate_type = int32_t;
using index_type = default_types::index_type;
using size_type = default_types::size_type;
using stride_type = default_types::stride_type;

std::vector<std::vector<coordinate_type>>
region_iterator_test(const torch::Tensor &coordinates,
                     const torch::Tensor &kernel_size) {
  // Create TensorArgs. These record the names and positions of each tensor as
  // parameters.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);
  torch::TensorArg arg_kernel_size(kernel_size, "kernel_size", 1);

  torch::CheckedFrom c = "region_iterator_test";
  torch::checkContiguous(c, arg_coordinates);
  torch::checkContiguous(c, arg_kernel_size);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_kernel_size, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkBackend(c, arg_kernel_size.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);
  torch::checkDim(c, arg_kernel_size, 1);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  coordinate_type *ptr = coordinates.data_ptr<coordinate_type>();
  coordinate_type *p_kernel_size = kernel_size.data_ptr<coordinate_type>();

  stride_type tensor_stride;
  stride_type s_kernel_size;
  stride_type dilation;
  for (index_type i = 0; i < D - 1; ++i) {
    tensor_stride.push_back(1);
    s_kernel_size.push_back(p_kernel_size[i]);
    dilation.push_back(1);
  }

  auto region = cpu_kernel_region<coordinate_type>(
      RegionType::HYPER_CUBE, D, tensor_stride.data(), s_kernel_size.data(),
      dilation.data());

  std::vector<coordinate_type> lb(D), ub(D);
  std::vector<coordinate_type> tmp(D);
  LOG_DEBUG(tmp.size(), tmp.capacity());
  std::vector<std::vector<coordinate_type>> all_regions;

  for (index_type i = 0; i < N; ++i) {
    region.set_bounds(&ptr[i * D], lb.data(), ub.data(), tmp.data());
    for (auto const &coordinate : region) {
      std::cout << PtrToString(coordinate.data(), D) << "\n";
      std::vector<coordinate_type> vec_coordinate(D);
      std::copy_n(coordinate.data(), D, vec_coordinate.data());
      all_regions.push_back(std::move(vec_coordinate));
    }
  }

  return all_regions;
}

std::tuple<cpu_kernel_map, size_type, double>
kernel_map_test(const torch::Tensor &in_coordinates,
                const torch::Tensor &out_coordinates,
                const torch::Tensor &kernel_size) {
  // Create TensorArgs. These record the names and positions of each tensor as
  // parameters.
  torch::TensorArg arg_in_coordinates(in_coordinates, "coordinates", 0);
  torch::TensorArg arg_out_coordinates(out_coordinates, "coordinates", 1);
  torch::TensorArg arg_kernel_size(kernel_size, "kernel_size", 2);

  torch::CheckedFrom c = "kernel_map_test";
  torch::checkContiguous(c, arg_in_coordinates);
  torch::checkContiguous(c, arg_out_coordinates);
  torch::checkContiguous(c, arg_kernel_size);
  // must match coordinate_type
  torch::checkScalarType(c, arg_in_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_out_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_kernel_size, torch::kInt);
  torch::checkBackend(c, arg_in_coordinates.tensor, torch::Backend::CPU);
  torch::checkBackend(c, arg_out_coordinates.tensor, torch::Backend::CPU);
  torch::checkBackend(c, arg_kernel_size.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_in_coordinates, 2);
  torch::checkDim(c, arg_out_coordinates, 2);
  torch::checkDim(c, arg_kernel_size, 1);

  auto const N_in = (index_type)in_coordinates.size(0);
  auto const D = (index_type)in_coordinates.size(1);

  auto const N_out = (index_type)out_coordinates.size(0);
  auto const D_out = (index_type)out_coordinates.size(1);

  ASSERT(D == D_out, "dimension mismatch");

  coordinate_type const *ptr = in_coordinates.data_ptr<coordinate_type>();
  coordinate_type const *ptr_out = out_coordinates.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> in_map{N_in, D};
  CoordinateMapCPU<coordinate_type> out_map{N_out, D};

  auto in_coordinate_range = coordinate_range<coordinate_type>(N_in, D, ptr);
  simple_range iter_in{N_in};
  in_map.insert(ptr,
                ptr + N_in * D);

  auto out_coordinate_range =
      coordinate_range<coordinate_type>(N_out, D, ptr_out);
  simple_range iter_out{N_out};
  out_map.insert(ptr_out, ptr_out + N_out * D);

  LOG_DEBUG("coordinate initialization");

  // Kernel region
  coordinate_type *p_kernel_size = kernel_size.data_ptr<coordinate_type>();
  stride_type tensor_stride;
  stride_type s_kernel_size;
  stride_type dilation;
  for (index_type i = 0; i < D - 1; ++i) {
    tensor_stride.push_back(1);
    s_kernel_size.push_back(p_kernel_size[i]);
    dilation.push_back(1);
  }

  LOG_DEBUG("kernel_region initialization");
  auto region = cpu_kernel_region<coordinate_type>(
      RegionType::HYPER_CUBE, D, tensor_stride.data(), s_kernel_size.data(),
      dilation.data());

  timer t;
  t.tic();
  auto result = in_map.kernel_map(out_map, region);

  return std::make_tuple(result, out_map.size(), t.toc());
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("region_iterator_test", &minkowski::region_iterator_test,
        "Minkowski Engine region iterator test");

  m.def("kernel_map_test", &minkowski::kernel_map_test,
        "Minkowski Engine kernel map test");
}
