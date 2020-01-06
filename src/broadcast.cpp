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

#include "broadcast.hpp"
#ifndef CPU_ONLY
#include "broadcast.cuh"
#endif

#include <pybind11/pybind11.h>

namespace minkowski {

template <typename Dtype>
at::Tensor BroadcastForwardCPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                               int op, py::object py_in_coords_key,
                               py::object py_glob_coords_key,
                               py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getOriginInOutMaps(py_in_coords_key,
                                                            py_glob_coords_key);

  auto out_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());

  BroadcastForwardKernelCPU<Dtype, int>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      in_out.first, in_out.second);

  return out_feat;
}

template <typename Dtype>
void BroadcastBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_glob_coords_key,
                          py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_glob_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  BroadcastBackwardKernelCPU<Dtype, int>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename Dtype>
at::Tensor BroadcastForwardGPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                               int op, py::object py_in_coords_key,
                               py::object py_glob_coords_key,
                               py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  // Both coords must exist
  // Use the global pooling mapping
  const auto &in_out = p_coords_manager->getOriginInOutMapsGPU(
      py_in_coords_key, py_glob_coords_key);

  auto out_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

  BroadcastForwardKernelGPU<Dtype, int>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      in_out.first, in_out.second, handle, at::cuda::getCurrentCUDAStream());

  return out_feat;
}

template <typename Dtype>
void BroadcastBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_glob_coords_key,
                          py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_glob_coords_key);

  ASSERT(p_coords_manager->d_in_maps.find(map_key) !=
             p_coords_manager->d_in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

  BroadcastBackwardKernelGPU<Dtype, int>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->d_in_maps[map_key],
      p_coords_manager->d_out_maps[map_key], handle,
      at::cuda::getCurrentCUDAStream());
}
#endif

template at::Tensor BroadcastForwardCPU<float>(at::Tensor in_feat,
                                               at::Tensor in_feat_glob, int op,
                                               py::object py_in_coords_key,
                                               py::object py_out_coords_key,
                                               py::object py_coords_manager);

template at::Tensor BroadcastForwardCPU<double>(at::Tensor in_feat,
                                                at::Tensor in_feat_glob, int op,
                                                py::object py_in_coords_key,
                                                py::object py_out_coords_key,
                                                py::object py_coords_manager);

template void BroadcastBackwardCPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardCPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template at::Tensor BroadcastForwardGPU<float>(at::Tensor in_feat,
                                               at::Tensor in_feat_glob, int op,
                                               py::object py_in_coords_key,
                                               py::object py_out_coords_key,
                                               py::object py_coords_manager);

template at::Tensor BroadcastForwardGPU<double>(at::Tensor in_feat,
                                                at::Tensor in_feat_glob, int op,
                                                py::object py_in_coords_key,
                                                py::object py_out_coords_key,
                                                py::object py_coords_manager);

template void BroadcastBackwardGPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardGPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif // CPU_ONLY

} // namespace me
