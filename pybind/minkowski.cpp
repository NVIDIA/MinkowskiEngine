/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#include <string>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "extern.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Constant function
  m.def("is_cuda_available", &is_cuda_available);

  initialize_non_templated_classes(m);

  // Manager
  instantiate_manager<minkowski::cpu_manager_type<int32_t>>(m,
                                                            std::string("CPU"));
#ifndef CPU_ONLY
  instantiate_manager<minkowski::gpu_default_manager_type<int32_t>>(
      m, std::string("GPU_default"));
  instantiate_manager<minkowski::gpu_c10_manager_type<int32_t>>(
      m, std::string("GPU_c10"));
#endif

  // Functions
  instantiate_cpu_func<int32_t, float>(m, std::string("f"));
  instantiate_cpu_func<int32_t, double>(m, std::string("d"));

#ifndef CPU_ONLY
  instantiate_gpu_func<int32_t, float, minkowski::detail::default_allocator>(
      m, std::string("fd"));
  instantiate_gpu_func<int32_t, double, minkowski::detail::default_allocator>(
      m, std::string("dd"));

  instantiate_gpu_func<int32_t, float, minkowski::detail::c10_allocator>(
      m, std::string("fc"));
  instantiate_gpu_func<int32_t, double, minkowski::detail::c10_allocator>(
      m, std::string("dc"));
#endif
}
