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

template <typename Dtype, typename Itype>
void BroadcastForwardCPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->_in_maps.find(map_key) ==
      p_coords_manager->_in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  out_feat.resize_as_(in_feat);
  out_feat.zero_();

  BroadcastForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->_in_maps[map_key],
      p_coords_manager->_out_maps[map_key]);
}

template <typename Dtype, typename Itype>
void BroadcastBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->_in_maps.find(map_key) ==
      p_coords_manager->_in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  BroadcastBackwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->_in_maps[map_key],
      p_coords_manager->_out_maps[map_key]);
}

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void BroadcastForwardGPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->_in_maps.find(map_key) ==
      p_coords_manager->_in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  out_feat.resize_as_(in_feat);
  out_feat.zero_();

  Itype *d_scr = p_coords_manager->getScratchGPUMemory(
      p_coords_manager->_out_maps[map_key][0].size());

  cusparseHandle_t handle =
      THCState_getCurrentSparseHandle(at::globalContext().getTHCState());

  BroadcastForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->_in_maps[map_key], p_coords_manager->_out_maps[map_key],
      d_scr, handle, at::cuda::getCurrentCUDAStream());
}

template <typename Dtype, typename Itype>
void BroadcastBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<Itype> *>();

  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->_in_maps.find(map_key) ==
      p_coords_manager->_in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  Itype *d_scr = p_coords_manager->getScratchGPUMemory(
      2 * p_coords_manager->_out_maps[map_key][0].size() + // in_map + out_map
      in_feat_glob.size(0) + 1                             // d_csr_row
  );

  Dtype *d_dscr = (Dtype *)p_coords_manager->getDScratchGPUMemory(
      (in_feat.size(0) + // d_csr_val
       (in_feat.size(0) + in_feat_glob.size(0)) * in_feat.size(1)) *
      sizeof(Dtype) // tmp_grad_infeat(_global)
  );

  cusparseHandle_t handle =
      THCState_getCurrentSparseHandle(at::globalContext().getTHCState());

  BroadcastBackwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->_in_maps[map_key], p_coords_manager->_out_maps[map_key],
      d_scr, d_dscr, handle, at::cuda::getCurrentCUDAStream());

  // p_coords_manager->gpu_memory_manager.reset();
  // p_coords_manager->dgpu_memory_manager.reset();
}
#endif

template void BroadcastForwardCPU<float, int32_t>(at::Tensor in_feat,
                                                  at::Tensor in_feat_glob,
                                                  at::Tensor out_feat, int op,
                                                  py::object py_in_coords_key,
                                                  py::object py_out_coords_key,
                                                  py::object py_coords_manager);

template void BroadcastForwardCPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardCPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardCPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template void BroadcastForwardGPU<float, int32_t>(at::Tensor in_feat,
                                                  at::Tensor in_feat_glob,
                                                  at::Tensor out_feat, int op,
                                                  py::object py_in_coords_key,
                                                  py::object py_out_coords_key,
                                                  py::object py_coords_manager);

template void BroadcastForwardGPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardGPU<float, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void BroadcastBackwardGPU<double, int32_t>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
