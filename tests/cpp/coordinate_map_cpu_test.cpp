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
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>

namespace minkowski {

using coordinate_type = int32_t;
using index_type = default_types::index_type;
using size_type = default_types::size_type;

std::pair<size_type, double>
coordinate_map_batch_insert_test(const torch::Tensor &coordinates) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> map{N, D};

  timer t;
  t.tic();
  map.insert(ptr, ptr + N * D);
  return std::make_pair<size_type, double>(map.size(), t.toc());
}

using map_inverse_map_type =
    std::pair<std::vector<int64_t>, std::vector<int64_t>>;
std::pair<map_inverse_map_type, double>
coordinate_map_inverse_test(const torch::Tensor &coordinates) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> map{N, D};

  timer t;
  t.tic();
  std::pair<std::vector<int64_t>, std::vector<int64_t>> unique_inverse_map =
      map.insert_and_map<false>(ptr, ptr + N * D);
  return std::make_pair<std::pair<std::vector<int64_t>, std::vector<int64_t>>,
                        double>(std::move(unique_inverse_map), t.toc());
}

std::pair<std::vector<index_type>, std::vector<index_type>>
coordinate_map_batch_find_test(const torch::Tensor &coordinates,
                               const torch::Tensor &queries) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);
  torch::TensorArg arg_queries(queries, "queries", 1);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  torch::checkContiguous(c, arg_queries);

  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkScalarType(c, arg_queries, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkBackend(c, arg_queries.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);
  torch::checkDim(c, arg_queries, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  // auto const NQ = (index_type)queries.size(0);
  auto const DQ = (index_type)queries.size(1);

  ASSERT(D == DQ, "Coordinates and queries must have the same size.");
  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();
  coordinate_type const *query_ptr = queries.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> map{N, D};
  map.insert(ptr, ptr + N * D);

  auto query_coordinates = coordinate_range<coordinate_type>(N, D, query_ptr);
  auto query_results =
      map.find(query_coordinates.begin(), query_coordinates.end());

  return query_results;
}

/******************************************************************************
 * New coordinate map generation tests
 ******************************************************************************/

std::pair<size_type, std::vector<size_type>>
coordinate_map_stride_test(const torch::Tensor &coordinates,
                           const torch::Tensor &stride) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);
  torch::TensorArg arg_stride(stride, "stride", 1);

  torch::CheckedFrom c = "coordinate_map_stride_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);

  // must match coordinate_type
  torch::checkScalarType(c, arg_stride, torch::kInt);
  torch::checkBackend(c, arg_stride.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_stride, 1);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);

  auto const NS = (index_type)stride.size(0);
  ASSERT(NS == D - 1, "Invalid stride size", NS);

  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> map{N, D};

  map.insert(ptr, ptr + N * D);

  // Stride
  default_types::stride_type stride_vec(NS);
  int32_t const *stride_ptr = stride.data_ptr<int32_t>();
  for (uint32_t i = 0; i < NS; ++i) {
    stride_vec[i] = stride_ptr[i];
    ASSERT(stride_ptr[i] > 0, "Invalid stride. All strides must be positive.");
  }

  auto stride_map = map.stride(stride_vec);
  return std::make_pair(stride_map.size(), stride_map.get_tensor_stride());
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coordinate_map_batch_insert_test",
        &minkowski::coordinate_map_batch_insert_test,
        "Minkowski Engine coordinate map batch insert test");

  m.def("coordinate_map_inverse_test", &minkowski::coordinate_map_inverse_test,
        "Minkowski Engine coordinate map batch insert test");

  m.def("coordinate_map_batch_find_test",
        &minkowski::coordinate_map_batch_find_test,
        "Minkowski Engine coordinate map batch find test");

  m.def("coordinate_map_stride_test", &minkowski::coordinate_map_stride_test,
        "Minkowski Engine coordinate map stride test");
}
