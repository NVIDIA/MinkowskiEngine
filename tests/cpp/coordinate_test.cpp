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
#include "coordinate.hpp"
#include "types.hpp"

#include <torch/extension.h>
#include <unordered_map>
#include <vector>

namespace minkowski {

using coordinate_type = int32_t;

uint32_t coordinate_test(const torch::Tensor &coordinates) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_coordinates(coordinates, "coordinates", 0);

  torch::CheckedFrom c = "coordinate_test";
  torch::checkContiguous(c, arg_coordinates);
  // must match coordinate_type
  torch::checkScalarType(c, arg_coordinates, torch::kInt);
  torch::checkBackend(c, arg_coordinates.tensor, torch::Backend::CPU);
  torch::checkDim(c, arg_coordinates, 2);

  auto const N = (uint32_t)coordinates.size(0);
  auto const D = (uint32_t)coordinates.size(1);

  using map_type =
      std::unordered_map<coordinate<coordinate_type>, default_types::index_type,
                         detail::coordinate_murmur3<coordinate_type>,
                         detail::coordinate_equal_to<coordinate_type>>;

  map_type map = map_type{N, detail::coordinate_murmur3<coordinate_type>{D},
                          detail::coordinate_equal_to<coordinate_type>{D}};
  coordinate_type const * ptr = coordinates.data_ptr<coordinate_type>();
  for (default_types::index_type i = 0; i < N; i++) {
    map[coordinate<coordinate_type>{ptr + D * i}] = i;
  }

  return map.size();
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coordinate_test", &minkowski::coordinate_test,
        "Minkowski Engine coordinate test");
}
