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
#include "pruning.hpp"
#include "common.hpp"
#ifndef CPU_ONLY
#include "pruning.cuh"
#endif

template <uint8_t D, typename Dtype, typename Itype>
void PruningForwardCPU(at::Tensor in_feat,  // CPU feat
                       at::Tensor out_feat, // CPU out feat
                       at::Tensor use_feat, // uint8 CPU data
                       py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();

  // Get the total number of coords
  at::Tensor sum = use_feat.sum();
  int64_t tot_n = sum.item<int64_t>();
  if (tot_n < 1)
    throw std::invalid_argument(Formatter()
                                << "Invalid total number of features"
                                << std::to_string(tot_n));
  out_feat.resize_({tot_n, in_feat.size(1)});
  out_feat.zero_();

  auto in_out = p_coords_manager->setupAndReturnPruningInOutPerKernel(
      use_feat, py_in_coords_key, py_out_coords_key);

  PruningForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), in_feat.size(1),
      std::get<0>(in_out), std::get<1>(in_out));
}

template <uint8_t D, typename Dtype, typename Itype>
void PruningBackwardCPU(at::Tensor grad_in_feat,  // CPU feat
                        at::Tensor grad_out_feat, // CPU out feat
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();

  InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);
  int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
  int nchannel = grad_out_feat.size(1);
  grad_in_feat.resize_({in_nrows, nchannel});
  grad_in_feat.zero_();

  PruningBackwardKernelCPU<Dtype, Itype>(grad_in_feat.data<Dtype>(),
                                         grad_out_feat.data<Dtype>(), nchannel,
                                         p_coords_manager->_in_maps[map_key],
                                         p_coords_manager->_out_maps[map_key]);
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void PruningForwardGPU(at::Tensor in_feat,  // GPU feat
                       at::Tensor out_feat, // GPU out feat
                       at::Tensor use_feat, // uint8 CPU data
                       py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();

  // Get the total number of coords
  at::Tensor sum = use_feat.sum();
  int64_t tot_n = sum.item<int64_t>();
  if (tot_n < 1)
    throw std::invalid_argument(Formatter()
                                << "Invalid total number of features"
                                << std::to_string(tot_n));
  out_feat.resize_({tot_n, in_feat.size(1)});
  out_feat.zero_();

  auto in_out = p_coords_manager->setupAndReturnPruningInOutPerKernel(
      use_feat, py_in_coords_key, py_out_coords_key);

  PruningForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), in_feat.size(1),
      std::get<0>(in_out), std::get<1>(in_out),
      at::cuda::getCurrentCUDAStream());
}

template <uint8_t D, typename Dtype, typename Itype>
void PruningBackwardGPU(at::Tensor grad_in_feat,  // GPU feat
                        at::Tensor grad_out_feat, // GPU out feat
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();

  InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);
  int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
  int nchannel = grad_out_feat.size(1);
  grad_in_feat.resize_({in_nrows, nchannel});
  grad_in_feat.zero_();

  PruningBackwardKernelGPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), grad_out_feat.data<Dtype>(), nchannel,
      p_coords_manager->_in_maps[map_key], p_coords_manager->_out_maps[map_key],
      at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchPruningForwardCPU(int D, at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor use_feat,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PruningForwardCPU, Dtype, Itype, in_feat, out_feat, use_feat,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchPruningForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor use_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPruningForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor use_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPruningBackwardCPU(int D, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PruningBackwardCPU, Dtype, Itype, grad_in_feat,
                   grad_out_feat, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchPruningBackwardCPU<float, int32_t>(
    int D, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPruningBackwardCPU<double, int32_t>(
    int D, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchPruningForwardGPU(int D, at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor use_feat,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PruningForwardGPU, Dtype, Itype, in_feat, out_feat, use_feat,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchPruningForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor use_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPruningForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor use_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPruningBackwardGPU(int D, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PruningBackwardGPU, Dtype, Itype, grad_in_feat,
                   grad_out_feat, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchPruningBackwardGPU<float, int32_t>(
    int D, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPruningBackwardGPU<double, int32_t>(
    int D, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
