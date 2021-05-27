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
#include "coordinate_map_functors.cuh"
#include "coordinate_map_gpu.cuh"
#include "types.hpp"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>

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
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinates, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  coordinate_type const *d_ptr = coordinates.data_ptr<coordinate_type>();

  LOG_DEBUG("Initialize a GPU map.");
  CoordinateMapGPU<coordinate_type> map{N, D};

  auto input_coordinates = coordinate_range<coordinate_type>(N, D, d_ptr);
  thrust::counting_iterator<uint32_t> iter{0};

  LOG_DEBUG("Insert coordinates");

  timer t;

  t.tic();
  map.insert<false>(input_coordinates.begin(), // key begin
                    input_coordinates.end());  // key end

  return std::make_pair<size_type, double>(map.size(), t.toc());
}

std::pair<at::Tensor, at::Tensor>
coordinate_map_inverse_map_test(const torch::Tensor &coordinates) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinates, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  coordinate_type const *d_ptr = coordinates.data_ptr<coordinate_type>();

  LOG_DEBUG("Initialize a GPU map.");
  CoordinateMapGPU<coordinate_type> map{N, D};

  auto input_coordinates = coordinate_range<coordinate_type>(N, D, d_ptr);
  thrust::counting_iterator<uint32_t> iter{0};

  LOG_DEBUG("Insert coordinates");

  auto mapping_inverse_mapping =
      map.insert_and_map<true>(input_coordinates.begin(), // key begin
                               input_coordinates.end());  // key end
  auto const &mapping = mapping_inverse_mapping.first;
  auto const &inverse_mapping = mapping_inverse_mapping.second;
  long const NM = mapping.size();
  long const NI = inverse_mapping.size();
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt)
                     .device(torch::kCUDA, 0)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  torch::Tensor th_mapping = torch::empty({NM}, options);
  torch::Tensor th_inverse_mapping = torch::empty({NI}, options);

  // IMPORTANT: assuming int32_t overflow does not occur.
  CUDA_CHECK(cudaMemcpy(th_mapping.data_ptr<int32_t>(),
                        mapping.cdata(),
                        NM * sizeof(int32_t), cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaMemcpy(th_inverse_mapping.data_ptr<int32_t>(),
                        inverse_mapping.cdata(),
                        NI * sizeof(int32_t), cudaMemcpyDeviceToDevice));

  return std::make_pair<at::Tensor, at::Tensor>(std::move(th_mapping),
                                                std::move(th_inverse_mapping));
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
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_queries.tensor, torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinates, 2);
  torch::checkDim(c, arg_queries, 2);

  auto const N = (index_type)coordinates.size(0);
  auto const D = (index_type)coordinates.size(1);
  auto const NQ = (index_type)queries.size(0);
  auto const DQ = (index_type)queries.size(1);

  ASSERT(D == DQ, "Coordinates and queries must have the same size.");
  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();
  coordinate_type const *query_ptr = queries.data_ptr<coordinate_type>();

  CoordinateMapGPU<coordinate_type> map{N, D};

  auto input_coordinates = coordinate_range<coordinate_type>(N, D, ptr);
  thrust::counting_iterator<uint32_t> iter{0};

  map.insert<false>(input_coordinates.begin(), // key begin
                    input_coordinates.end());  // key end

  LOG_DEBUG("Map size", map.size());
  auto query_coordinates = coordinate_range<coordinate_type>(NQ, D, query_ptr);

  LOG_DEBUG("Find coordinates.");
  auto const query_results =
      map.find(query_coordinates.begin(), query_coordinates.end());
  auto const &firsts(query_results.first);
  auto const &seconds(query_results.second);
  index_type NR = firsts.size();
  LOG_DEBUG(NR, "keys found.");

  std::vector<index_type> cpu_firsts(NR);
  std::vector<index_type> cpu_seconds(NR);

  THRUST_CHECK(thrust::copy(firsts.cbegin(), firsts.cend(), cpu_firsts.begin()));
  THRUST_CHECK(thrust::copy(seconds.cbegin(), seconds.cend(), cpu_seconds.begin()));
  return std::make_pair(cpu_firsts, cpu_seconds);
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
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CUDA);
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

  CoordinateMapGPU<coordinate_type> map{N, D};

  auto input_coordinates = coordinate_range<coordinate_type>(N, D, ptr);
  thrust::counting_iterator<uint32_t> iter{0};
  map.insert<false>(input_coordinates.begin(), // key begin
                    input_coordinates.end());  // key end

  // Stride
  default_types::stride_type stride_vec(NS);
  int32_t const *stride_ptr = stride.data_ptr<int32_t>();
  for (uint32_t i = 0; i < NS; ++i) {
    stride_vec[i] = stride_ptr[i];
    ASSERT(stride_ptr[i] > 0, "Invalid stride. All strides must be positive.");
  }

  auto const stride_map = map.stride(stride_vec);
  return std::make_pair(stride_map.size(), stride_map.get_tensor_stride());
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coordinate_map_batch_insert_test",
        &minkowski::coordinate_map_batch_insert_test,
        "Minkowski Engine coordinate map batch insert test");

  m.def("coordinate_map_inverse_map_test",
        &minkowski::coordinate_map_inverse_map_test,
        "Minkowski Engine coordinate map inverse map test");

  m.def("coordinate_map_batch_find_test",
        &minkowski::coordinate_map_batch_find_test,
        "Minkowski Engine coordinate map batch find test");

  m.def("coordinate_map_stride_test", &minkowski::coordinate_map_stride_test,
        "Minkowski Engine coordinate map stride test");
}
