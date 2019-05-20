/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "common.hpp"

#include "pooling_max.hpp"
#ifndef CPU_ONLY
#include "pooling_max.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void MaxPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor max_index, std::vector<int> tensor_strides,
                          std::vector<int> strides,
                          std::vector<int> kernel_sizes,
                          std::vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

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

template <uint8_t D, typename Dtype, typename Itype>
void MaxPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor max_index, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false);

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelCPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), max_index.data<Itype>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void MaxPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  const int nchannel = in_feat.size(1);
  out_feat.resize_({out_nrows, nchannel});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows, nchannel});
  num_nonzero.zero_();

  MaxPoolingForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), out_nrows,
      num_nonzero.data<Itype>(), nchannel, std::get<0>(in_out),
      std::get<1>(in_out), at::cuda::getCurrentCUDAStream());
}

template <uint8_t D, typename Dtype, typename Itype>
void MaxPoolingBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  MaxPoolingBackwardKernelGPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), num_nonzero.data<Itype>(), in_feat.size(1),
      at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  SWITCH_DIM_TYPES(MaxPoolingForwardCPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, tensor_strides, strides, kernel_sizes,
                   dilations, region_type, offsets, py_in_coords_key,
                   py_out_coords_key, py_coords_manager);
}

template void DimSwitchMaxPoolingForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void DimSwitchMaxPoolingForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(MaxPoolingBackwardCPU, Dtype, Itype, in_feat, grad_in_feat,
                   grad_out_feat, num_nonzero, tensor_strides, strides,
                   kernel_sizes, dilations, region_type, py_in_coords_key,
                   py_out_coords_key, py_coords_manager);
}

template void DimSwitchMaxPoolingBackwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchMaxPoolingBackwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  SWITCH_DIM_TYPES(MaxPoolingForwardGPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, tensor_strides, strides, kernel_sizes,
                   dilations, region_type, offsets, py_in_coords_key,
                   py_out_coords_key, py_coords_manager);
}

template void DimSwitchMaxPoolingForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void DimSwitchMaxPoolingForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(MaxPoolingBackwardGPU, Dtype, Itype, in_feat, grad_in_feat,
                   grad_out_feat, num_nonzero, tensor_strides, strides,
                   kernel_sizes, dilations, region_type, py_in_coords_key,
                   py_out_coords_key, py_coords_manager);
}

template void DimSwitchMaxPoolingBackwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchMaxPoolingBackwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
