/* Copyright (c) Pan He (pan.he@ufl.edu).
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
 */
#include "common.hpp"

#include "pooling_max.hpp"
#ifndef CPU_ONLY
#include "pooling_max.cuh"
#endif

#include <pybind11/pybind11.h>

template <typename Dtype, typename Itype>
void GlobalMaxPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor max_index,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnOriginInOutPerKernel(
      py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  max_index.resize_({out_nrows, nchannel});
  max_index.zero_();

  MaxPoolingForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), max_index.data<Itype>(),
      nchannel, std::get<0>(in_out), std::get<1>(in_out), out_nrows);
}

template <typename Dtype, typename Itype>
void GlobalMaxPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat, at::Tensor max_index,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelCPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), max_index.data<Itype>(), in_feat.size(1),
      p_coords_manager->_in_maps[map_key],
      p_coords_manager->_out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void GlobalMaxPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnOriginInOutPerKernel(
      py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows, nchannel});
  num_nonzero.zero_();

  // Compute the scratch space
  const auto &maps = std::get<0>(in_out);
  int nnz = 0;
  for (auto &map : maps)
    nnz += map.size();
  Itype *d_scr = p_coords_manager->getScratchGPUMemory(5 * nnz);

  MaxPoolingForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), out_nrows,
      num_nonzero.data<Itype>(), nchannel, std::get<0>(in_out),
      std::get<1>(in_out), d_scr, at::cuda::getCurrentCUDAStream());
}

template <typename Dtype, typename Itype>
void GlobalMaxPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 at::Tensor num_nonzero,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelGPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), num_nonzero.data<Itype>(), in_feat.size(1),
      at::cuda::getCurrentCUDAStream());
}
#endif

template void GlobalMaxPoolingForwardCPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void GlobalMaxPoolingForwardCPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void GlobalMaxPoolingBackwardCPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void GlobalMaxPoolingBackwardCPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template void GlobalMaxPoolingForwardGPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void GlobalMaxPoolingForwardGPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void GlobalMaxPoolingBackwardGPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void GlobalMaxPoolingBackwardGPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif
