/* Copyright (c) 2020 NVIDIA CORPORATION.
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

template <typename map_type, typename coordinate_type>
struct insert_coordinate {
  insert_coordinate(map_type &map, coordinate_type const *ptr,
                    size_type coordinate_size)
      : m_map(map), m_ptr(ptr), m_coordinate_size(coordinate_size) {}

  /*
   * @brief insert a <coordinate, row index> pair into the unordered_map.
   *
   * @return bool of a success flag.
   */
  bool operator()(index_type i) {
    // std::cout << ::PtrToString(m_ptr + i * m_coordinate_size,
    // m_coordinate_size) << ": " << m_ptr + i * m_coordinate_size << "\n";
    return m_map.insert(
        coordinate<coordinate_type>{m_ptr + i * m_coordinate_size}, i);
  }

  map_type &m_map;
  coordinate_type const *m_ptr;
  size_type const m_coordinate_size;
};

template <typename map_type, typename coordinate_type> struct find_coordinate {
  find_coordinate(map_type &map, coordinate_type const *ptr,
                  size_type coordinate_size)
      : m_map(map), m_ptr(ptr), m_coordinate_size(coordinate_size) {}

  /*
   * @brief insert a <coordinate, row index> pair into the unordered_map.
   *
   * @return bool of a success flag.
   */
  index_type operator()(index_type i) {
    // std::cout << ::PtrToString(m_ptr + i * m_coordinate_size,
    // m_coordinate_size) << ": " << i << "\n";
    auto const &iter =
        m_map.find(coordinate<coordinate_type>{m_ptr + i * m_coordinate_size});
    if (iter != m_map.end()) {
      return iter->second;
    } else {
      return std::numeric_limits<index_type>::max();
    }
  }

  map_type &m_map;
  coordinate_type const *m_ptr;
  size_type const m_coordinate_size;
};

size_type coordinate_map_insert_test(const torch::Tensor &coordinates) {
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

  ::simple_range iter{N};
  std::for_each(
      iter.begin(), iter.end(),
      insert_coordinate<CoordinateMapCPU<coordinate_type>, coordinate_type>{
          map, ptr, D});

  // auto coordinate_print = coordinate_print_functor<coordinate_type>{D};
  // for (auto kv : map) {
  //   std::cout << coordinate_print(kv.first) << ", " << kv.first.ptr << ": "
  //   << kv.second << "\n";
  // }
  return map.size();
}

std::vector<index_type>
coordinate_map_find_test(const torch::Tensor &coordinates,
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
  auto const NQ = (index_type)queries.size(0);
  auto const DQ = (index_type)queries.size(1);

  ASSERT(D == DQ, "Coordinates and queries must have the same size.");
  coordinate_type const *ptr = coordinates.data_ptr<coordinate_type>();
  coordinate_type const *query_ptr = queries.data_ptr<coordinate_type>();

  CoordinateMapCPU<coordinate_type> map{N, D};
  map.reserve(N);

  ::simple_range iter{N};
  std::for_each(
      iter.begin(), iter.end(),
      insert_coordinate<CoordinateMapCPU<coordinate_type>, coordinate_type>{
          map, ptr, D});

  ::simple_range iter2{NQ};
  std::vector<index_type> query_results;
  std::transform(
      iter2.begin(), iter2.end(), std::back_inserter(query_results),
      find_coordinate<CoordinateMapCPU<coordinate_type>, coordinate_type>{
          map, query_ptr, D});

  // auto coordinate_print = coordinate_print_functor<coordinate_type>{D};
  // for (auto kv : map) {
  //   std::cout << coordinate_print(kv.first) << ", " << kv.first.ptr << ": "
  //   << kv.second << "\n";
  // }
  return query_results;
}

size_type coordinate_map_insert_batch_test(const torch::Tensor &coordinates) {
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
  map.reserve(N);

  ::simple_range iter{N};
  std::for_each(
      iter.begin(), iter.end(),
      insert_coordinate<CoordinateMapCPU<coordinate_type>, coordinate_type>{
          map, ptr, D});

  // auto coordinate_print = coordinate_print_functor<coordinate_type>{D};
  // for (auto kv : map) {
  //   std::cout << coordinate_print(kv.first) << ", " << kv.first.ptr << ": "
  //   << kv.second << "\n";
  // }
  return map.size();
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coordinate_map_insert_test", &minkowski::coordinate_map_insert_test,
        "Minkowski Engine coordinate map insert test");

  m.def("coordinate_map_find_test", &minkowski::coordinate_map_find_test,
        "Minkowski Engine coordinate map find test");

  m.def("coordinate_map_insert_batch_test",
        &minkowski::coordinate_map_insert_batch_test,
        "Minkowski Engine coordinate map insert batch test");
}
