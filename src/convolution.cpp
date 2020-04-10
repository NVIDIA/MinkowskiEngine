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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "common.hpp"

#include "convolution.hpp"
#ifndef CPU_ONLY
#include "convolution.cuh"
#endif

#include <pybind11/pybind11.h>

namespace minkowski {

template <typename MapType, typename Dtype>
void ConvolutionForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, vector<int> tensor_strides,
                           vector<int> strides, vector<int> kernel_sizes,
                           vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMaps(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();

  ConvolutionForwardKernelCPU<Dtype, int>(
      in_feat.template data<Dtype>(), in_feat.size(1),
      out_feat.template data<Dtype>(), out_feat.size(1),
      kernel.template data<Dtype>(), in_out.first, in_out.second);
}

template <typename MapType, typename Dtype>
void ConvolutionBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                            at::Tensor grad_out_feat, at::Tensor kernel,
                            at::Tensor grad_kernel, vector<int> tensor_strides,
                            vector<int> strides, vector<int> kernel_sizes,
                            vector<int> dilations, int region_type,
                            py::object py_in_coords_key,
                            py::object py_out_coords_key,
                            py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false, false);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();

  ConvolutionBackwardKernelCPU<Dtype, int>(
      in_feat.template data<Dtype>(), grad_in_feat.template data<Dtype>(),
      in_feat.size(1), grad_out_feat.template data<Dtype>(),
      grad_out_feat.size(1), kernel.template data<Dtype>(),
      grad_kernel.template data<Dtype>(), p_coords_manager->in_maps[map_key],
      p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename MapType, typename Dtype>
void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, vector<int> tensor_strides,
                           vector<int> strides, vector<int> kernel_sizes,
                           vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const auto &in_out = p_coords_manager->getInOutMapsGPU(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

  ConvolutionForwardKernelGPU<Dtype, int>(
      in_feat.template data<Dtype>(), in_feat.size(1),
      out_feat.template data<Dtype>(), out_feat.size(1),
      kernel.template data<Dtype>(), in_out.first, in_out.second, out_nrows,
      handle, at::cuda::getCurrentCUDAStream());
}

template <typename MapType, typename Dtype>
void ConvolutionBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                            at::Tensor grad_out_feat, at::Tensor kernel,
                            at::Tensor grad_kernel, vector<int> tensor_strides,
                            vector<int> strides, vector<int> kernel_sizes,
                            vector<int> dilations, int region_type,
                            py::object py_in_coords_key,
                            py::object py_out_coords_key,
                            py::object py_coords_manager) {
  CoordsManager<MapType> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<MapType> *>();
  const InOutMapKey map_key = p_coords_manager->getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false, false);

  ASSERT(p_coords_manager->d_in_maps.find(map_key) !=
             p_coords_manager->d_in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

  ConvolutionBackwardKernelGPU<Dtype, int>(
      in_feat.template data<Dtype>(), grad_in_feat.template data<Dtype>(),
      in_feat.size(1), grad_out_feat.template data<Dtype>(),
      grad_out_feat.size(1), kernel.template data<Dtype>(),
      grad_kernel.template data<Dtype>(), p_coords_manager->d_in_maps[map_key],
      p_coords_manager->d_out_maps[map_key], grad_out_feat.size(0), handle,
      at::cuda::getCurrentCUDAStream());
}
#endif // end CPU_ONLY

template void ConvolutionForwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void ConvolutionForwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void ConvolutionBackwardCPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, vector<int> tensor_strides,
    vector<int> strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void ConvolutionBackwardCPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, vector<int> tensor_strides,
    vector<int> strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template void ConvolutionBackwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, vector<int> tensor_strides,
    vector<int> strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void ConvolutionBackwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, vector<int> tensor_strides,
    vector<int> strides, vector<int> kernel_sizes, vector<int> dilations,
    int region_type, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
template void ConvolutionForwardGPU<CoordsToIndexMap, float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void ConvolutionForwardGPU<CoordsToIndexMap, double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif // end CPU_ONLY

} // end namespace minkowski
