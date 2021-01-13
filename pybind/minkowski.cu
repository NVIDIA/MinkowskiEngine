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
#include "extern.hpp"

#include <string>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Constant function
  m.def("is_cuda_available", &is_cuda_available);
  m.def("cuda_version", &cuda_version);
  m.def("cudart_version", &cudart_version);
  m.def("get_gpu_memory_info", &get_gpu_memory_info);

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
  non_templated_cpu_func(m);
  instantiate_cpu_func<int32_t>(m, "");

#ifndef CPU_ONLY
  instantiate_gpu_func<int32_t, minkowski::detail::default_allocator>(
      m, std::string(""));

  instantiate_gpu_func<int32_t, minkowski::detail::c10_allocator>(
      m, std::string(""));

  non_templated_gpu_func(m);
#endif
}
