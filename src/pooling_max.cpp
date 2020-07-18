/* Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#include "common.hpp"

#include "pooling_max.hpp"
#ifndef CPU_ONLY
#include "pooling_max.cuh"
#endif

#include <pybind11/pybind11.h>

namespace minkowski {

template <typename MapType, typename Dtype>
void MaxPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor max_index, vector<int> tensor_strides,
                          vector<int> strides, vector<int> kernel_sizes,
                          vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMaps(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  max_index.resize_({out_nrows, nchannel});
  max_index.zero_();

  MaxPoolingForwardKernelCPU<Dtype, int>(
      in_feat.template data<Dtype>(), out_feat.template data<Dtype>(),
      max_index.data<int>(), nchannel, in_out.first, in_out.second, out_nrows);
}

template <typename MapType, typename Dtype>
void MaxPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                           at::Tensor grad_out_feat, at::Tensor max_index,
                           vector<int> tensor_strides, vector<int> strides,
                           vector<int> kernel_sizes, vector<int> dilations,
                           int region_type, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false, true);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelCPU<Dtype, int>(
      grad_in_feat.template data<Dtype>(), in_feat.size(0),
      grad_out_feat.template data<Dtype>(), grad_out_feat.size(0),
      max_index.data<int>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename MapType, typename Dtype>
void MaxPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor num_nonzero, vector<int> tensor_strides,
                          vector<int> strides, vector<int> kernel_sizes,
                          vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMapsGPU(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows, nchannel});
  num_nonzero.zero_();

  // Compute the scratch space
  int nmap = getInOutMapsSize(in_out.first);

  int *d_scr =
      (int *)p_coords_manager->getScratchGPUMemory(5 * nmap * sizeof(int));

  MaxPoolingForwardKernelGPU<Dtype, int>(
      in_feat.template data<Dtype>(), out_feat.template data<Dtype>(),
      out_nrows, num_nonzero.data<int>(), nchannel, in_out.first, in_out.second,
      d_scr, at::cuda::getCurrentCUDAStream());

  p_coords_manager->clearScratchGPUMemory();
}

template <typename MapType, typename Dtype>
void MaxPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                           at::Tensor grad_out_feat, at::Tensor num_nonzero,
                           vector<int> tensor_strides, vector<int> strides,
                           vector<int> kernel_sizes, vector<int> dilations,
                           int region_type, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelGPU<Dtype, int>(
      grad_in_feat.template data<Dtype>(), in_feat.size(0),
      grad_out_feat.template data<Dtype>(), grad_out_feat.size(0),
      num_nonzero.data<int>(), in_feat.size(1),
      at::cuda::getCurrentCUDAStream());
}
#endif

template void MaxPoolingForwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingForwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingBackwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingBackwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY

template void MaxPoolingForwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingForwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingBackwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void MaxPoolingBackwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif // CPU_ONLY

} // end namespace minkowski
