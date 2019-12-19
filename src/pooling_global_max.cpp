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

template <typename Dtype>
void GlobalMaxPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor max_index,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out =
      p_coords_manager->getOriginInOutMaps(py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);

  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  max_index.resize_({out_nrows, nchannel});
  max_index.zero_();

  MaxPoolingForwardKernelCPU<Dtype, int>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), max_index.data<int>(),
      nchannel, get<0>(in_out), get<1>(in_out), out_nrows);
}

template <typename Dtype>
void GlobalMaxPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat, at::Tensor max_index,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelCPU<Dtype, int>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), max_index.data<int>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename Dtype>
void GlobalMaxPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto& in_out =
      p_coords_manager->getOriginInOutMapsGPU(py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows, nchannel});
  num_nonzero.zero_();

  // Compute the scratch space
  const int nmap = getInOutMapsSize(in_out.first);
  int *d_scr =
      (int *)p_coords_manager->getScratchGPUMemory(5 * nmap * sizeof(int));

  MaxPoolingForwardKernelGPU<Dtype, int>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), out_nrows,
      num_nonzero.data<int>(), nchannel, get<0>(in_out), get<1>(in_out), d_scr,
      at::cuda::getCurrentCUDAStream());

  p_coords_manager->clearScratchGPUMemory();
}

template <typename Dtype>
void GlobalMaxPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 at::Tensor num_nonzero,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelGPU<Dtype, int>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), num_nonzero.data<int>(), in_feat.size(1),
      at::cuda::getCurrentCUDAStream());
}
#endif

template void GlobalMaxPoolingForwardCPU<float>(at::Tensor in_feat,
                                                at::Tensor out_feat,
                                                at::Tensor num_nonzero,
                                                py::object py_in_coords_key,
                                                py::object py_out_coords_key,
                                                py::object py_coords_manager);

template void GlobalMaxPoolingForwardCPU<double>(at::Tensor in_feat,
                                                 at::Tensor out_feat,
                                                 at::Tensor num_nonzero,
                                                 py::object py_in_coords_key,
                                                 py::object py_out_coords_key,
                                                 py::object py_coords_manager);

template void GlobalMaxPoolingBackwardCPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void GlobalMaxPoolingBackwardCPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template void GlobalMaxPoolingForwardGPU<float>(at::Tensor in_feat,
                                                at::Tensor out_feat,
                                                at::Tensor num_nonzero,
                                                py::object py_in_coords_key,
                                                py::object py_out_coords_key,
                                                py::object py_coords_manager);

template void GlobalMaxPoolingForwardGPU<double>(at::Tensor in_feat,
                                                 at::Tensor out_feat,
                                                 at::Tensor num_nonzero,
                                                 py::object py_in_coords_key,
                                                 py::object py_out_coords_key,
                                                 py::object py_coords_manager);

template void GlobalMaxPoolingBackwardGPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void GlobalMaxPoolingBackwardGPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif
