/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "common.hpp"

#include "pooling_avg.hpp"
#ifndef CPU_ONLY
#include "pooling_avg.cuh"
#endif

#include <pybind11/pybind11.h>

namespace minkowski {

template <typename MapType, typename Dtype>
void PoolingTransposeForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                vector<int> tensor_strides, vector<int> strides,
                                vector<int> kernel_sizes, vector<int> dilations,
                                int region_type, at::Tensor offsets,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMaps(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, true, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  NonzeroAvgPoolingForwardKernelCPU<Dtype, int>(
      in_feat.template data<Dtype>(), out_feat.template data<Dtype>(),
      num_nonzero.template data<Dtype>(), in_feat.size(1), in_out.first,
      in_out.second, out_nrows, false);
}

template <typename MapType, typename Dtype>
void PoolingTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  bool reverse_map = false;
  const InOutMapKey rev_map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_out_coords_key, py_in_coords_key, false, true);
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, true, true);

  // Check if the reverse map exists first
  if (p_coords_manager->in_maps.find(rev_map_key) !=
      p_coords_manager->in_maps.end())
    reverse_map = true;

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  if (!reverse_map) {
    ASSERT(
        p_coords_manager->in_maps.find(map_key) !=
            p_coords_manager->in_maps.end(),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    NonzeroAvgPoolingBackwardKernelCPU<Dtype, int>(
        grad_in_feat.template data<Dtype>(), in_feat.size(0),
        grad_out_feat.template data<Dtype>(),
        num_nonzero.template data<Dtype>(), in_feat.size(1),
        p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
        false);
  } else {
    ASSERT(
        p_coords_manager->in_maps.find(rev_map_key) !=
            p_coords_manager->in_maps.end(),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    NonzeroAvgPoolingBackwardKernelCPU<Dtype, int>(
        grad_in_feat.template data<Dtype>(), in_feat.size(0),
        grad_out_feat.template data<Dtype>(),
        num_nonzero.template data<Dtype>(), in_feat.size(1),
        p_coords_manager->out_maps[rev_map_key],
        p_coords_manager->in_maps[rev_map_key], false);
  }
}

#ifndef CPU_ONLY
template <typename MapType, typename Dtype>
void PoolingTransposeForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                vector<int> tensor_strides, vector<int> strides,
                                vector<int> kernel_sizes, vector<int> dilations,
                                int region_type, at::Tensor offsets,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMapsGPU(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, true, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

  NonzeroAvgPoolingForwardKernelGPU<Dtype, int>(
      in_feat.template data<Dtype>(), in_feat.size(0),
      out_feat.template data<Dtype>(), out_nrows,
      num_nonzero.template data<Dtype>(), in_feat.size(1), get<0>(in_out),
      get<1>(in_out), false, handle, at::cuda::getCurrentCUDAStream());
}

template <typename MapType, typename Dtype>
void PoolingTransposeBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  bool reverse_map = false;
  const InOutMapKey rev_map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_out_coords_key, py_in_coords_key, false, true);
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, true, true);

  // Check if the reverse map exists first
  if (p_coords_manager->in_maps.find(rev_map_key) !=
      p_coords_manager->in_maps.end())
    reverse_map = true;

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  if (!reverse_map) {
    ASSERT(
        p_coords_manager->d_in_maps.find(map_key) !=
            p_coords_manager->d_in_maps.end(),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    NonzeroAvgPoolingBackwardKernelGPU<Dtype, int>(
        grad_in_feat.template data<Dtype>(), in_feat.size(0),
        grad_out_feat.template data<Dtype>(), grad_out_feat.size(0),
        num_nonzero.template data<Dtype>(), in_feat.size(1),
        p_coords_manager->d_in_maps[map_key],
        p_coords_manager->d_out_maps[map_key], false,
        at::cuda::getCurrentCUDAStream());
  } else {
    ASSERT(
        p_coords_manager->d_in_maps.find(rev_map_key) !=
            p_coords_manager->d_in_maps.end(),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    NonzeroAvgPoolingBackwardKernelGPU<Dtype, int>(
        grad_in_feat.template data<Dtype>(), in_feat.size(0),
        grad_out_feat.template data<Dtype>(), grad_out_feat.size(0),
        num_nonzero.template data<Dtype>(), in_feat.size(1),
        p_coords_manager->d_out_maps[rev_map_key],
        p_coords_manager->d_in_maps[rev_map_key], false,
        at::cuda::getCurrentCUDAStream());
  }
}
#endif

template void PoolingTransposeForwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeForwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeBackwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeBackwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY

template void PoolingTransposeForwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeForwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeBackwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void PoolingTransposeBackwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, vector<int> tensor_strides, vector<int> strides,
    vector<int> kernel_sizes, vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif // CPU_ONLY

} // end namespace minkowski
