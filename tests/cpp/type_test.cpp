/* Copyright (c) 2020 NVIDIA Corporation.
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
#include "types.hpp"

#include <torch/extension.h>

namespace minkowski {

void type_test() {
  // Check basic type compilation
  constexpr default_types::tensor_order_type D = 3;
  constexpr default_types::dcoordinate_type a  = 3;

  const cpuInMap in_map     = std::vector<default_types::index_type>{3, 4, 5};
  const cpuOutMap out_map   = std::vector<default_types::index_type>{3, 4, 5};
  const cpuInMaps in_maps   = {in_map};
  const cpuOutMaps out_maps = {out_map};
}

} // namespace minkowski

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("type_test", &minkowski::type_test, "Minkowski Engine type test");
}
