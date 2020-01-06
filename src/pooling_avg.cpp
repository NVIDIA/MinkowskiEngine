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

#include "pooling_avg.hpp"
#ifndef CPU_ONLY
#include "pooling_avg.cuh"
#endif

#include <pybind11/pybind11.h>

namespace minkowski {

template <typename Dtype>
void AvgPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor num_nonzero, vector<int> tensor_strides,
                          vector<int> strides, vector<int> kernel_sizes,
                          vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getInOutMaps(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();

  Dtype *num_nonzero_data = NULL;
  if (use_avg) {
    num_nonzero.resize_({out_nrows});
    num_nonzero.zero_();
    num_nonzero_data = num_nonzero.data<Dtype>();
  }

  NonzeroAvgPoolingForwardKernelCPU<Dtype, int>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), num_nonzero_data,
      in_feat.size(1), in_out.first, in_out.second, out_nrows, use_avg);
}

template <typename Dtype>
void AvgPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                           at::Tensor grad_out_feat, at::Tensor num_nonzero,
                           vector<int> tensor_strides, vector<int> strides,
                           vector<int> kernel_sizes, vector<int> dilations,
                           int region_type, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false, true);

  ASSERT(
      p_coords_manager->existsInOutMapKey(map_key),
      "The in-out map doesn't exist for backward. Did you run forward pass?");

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  NonzeroAvgPoolingBackwardKernelCPU<Dtype, int>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      num_nonzero.data<Dtype>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
      use_avg);
}

#ifndef CPU_ONLY
template <typename Dtype>
void AvgPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor num_nonzero, vector<int> tensor_strides,
                          vector<int> strides, vector<int> kernel_sizes,
                          vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getInOutMapsGPU(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();

  Dtype *num_nonzero_data = NULL;
  if (use_avg) {
    num_nonzero.resize_({out_nrows});
    num_nonzero.zero_();
    num_nonzero_data = num_nonzero.data<Dtype>();
  }

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

  NonzeroAvgPoolingForwardKernelGPU<Dtype, int>(
      in_feat.data<Dtype>(), in_feat.size(0), out_feat.data<Dtype>(), out_nrows,
      num_nonzero_data, in_feat.size(1), in_out.first, in_out.second, use_avg,
      handle, at::cuda::getCurrentCUDAStream());
}

template <typename Dtype>
void AvgPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                           at::Tensor grad_out_feat, at::Tensor num_nonzero,
                           vector<int> tensor_strides, vector<int> strides,
                           vector<int> kernel_sizes, vector<int> dilations,
                           int region_type, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false, true);

  ASSERT(
      p_coords_manager->existsInOutMapKey(map_key),
      "The in-out map doesn't exist for backward. Did you run forward pass?");

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  NonzeroAvgPoolingBackwardKernelGPU<Dtype, int>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), num_nonzero.data<Dtype>(), in_feat.size(1),
      p_coords_manager->d_in_maps[map_key],
      p_coords_manager->d_out_maps[map_key], use_avg,
      at::cuda::getCurrentCUDAStream());
}
#endif

template void AvgPoolingForwardCPU<float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingForwardCPU<double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingBackwardCPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingBackwardCPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template void AvgPoolingForwardGPU<float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingForwardGPU<double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingBackwardGPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void AvgPoolingBackwardGPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif // end CPU_ONLY

} //end namespace minkowski
