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
#include "pruning.hpp"
#include "common.hpp"
#ifndef CPU_ONLY
#include "pruning.cuh"
#endif

namespace minkowski {

template <typename Dtype>
void PruningForwardCPU(at::Tensor in_feat,  // CPU feat
                       at::Tensor out_feat, // CPU out feat
                       at::Tensor use_feat, // uint8 CPU data
                       py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getPruningInOutMaps(
      use_feat, py_in_coords_key, py_out_coords_key);

  // Get the total number of coords
  at::Tensor sum = use_feat.sum();
  const int64_t tot_n = sum.item<int64_t>();
  if (tot_n == 0) {
    WARNING(true, "MinkowskiPruning: Generating an empty SparseTensor");
    out_feat.resize_({0, in_feat.size(1)});
  } else {
    out_feat.resize_({tot_n, in_feat.size(1)});
    out_feat.zero_();

    PruningForwardKernelCPU<Dtype, int>(in_feat.data<Dtype>(),
                                        out_feat.data<Dtype>(), in_feat.size(1),
                                        get<0>(in_out), get<1>(in_out));
  }
}

template <typename Dtype>
void PruningBackwardCPU(at::Tensor grad_in_feat,  // CPU feat
                        at::Tensor grad_out_feat, // CPU out feat
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();

  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  const int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
  const int nchannel = grad_out_feat.size(1);

  grad_in_feat.resize_({in_nrows, nchannel});
  grad_in_feat.zero_();

  if (grad_out_feat.size(0) > 0)
    PruningBackwardKernelCPU<Dtype, int>(
        grad_in_feat.data<Dtype>(), grad_out_feat.data<Dtype>(), nchannel,
        p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
  else
    WARNING(true, "MinkowskiPruning: Backprop from a size-0 sparse tensor.");
}

#ifndef CPU_ONLY
template <typename Dtype>
void PruningForwardGPU(at::Tensor in_feat,  // GPU feat
                       at::Tensor out_feat, // GPU out feat
                       at::Tensor use_feat, // uint8 CPU data
                       py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getPruningInOutMapsGPU(
      use_feat, py_in_coords_key, py_out_coords_key);

  // Get the total number of coords
  at::Tensor sum = use_feat.sum();
  const int64_t tot_n = sum.item<int64_t>();
  if (tot_n == 0) {
    WARNING(true, "MinkowskiPruning: Generating an empty SparseTensor");
    out_feat.resize_({0, in_feat.size(1)});
  } else {
    out_feat.resize_({tot_n, in_feat.size(1)});
    out_feat.zero_();

    PruningForwardKernelGPU<Dtype, int>(
        in_feat.data<Dtype>(), out_feat.data<Dtype>(), in_feat.size(1),
        get<0>(in_out), get<1>(in_out), at::cuda::getCurrentCUDAStream());
  }
}

template <typename Dtype>
void PruningBackwardGPU(at::Tensor grad_in_feat,  // GPU feat
                        at::Tensor grad_out_feat, // GPU out feat
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();

  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  ASSERT(p_coords_manager->d_in_maps.find(map_key) !=
             p_coords_manager->d_in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  const int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
  const int nchannel = grad_out_feat.size(1);
  grad_in_feat.resize_({in_nrows, nchannel});
  grad_in_feat.zero_();

  if (grad_out_feat.size(0) > 0)
    PruningBackwardKernelGPU<Dtype, int>(
        grad_in_feat.data<Dtype>(), grad_out_feat.data<Dtype>(), nchannel,
        p_coords_manager->d_in_maps[map_key],
        p_coords_manager->d_out_maps[map_key], at::cuda::getCurrentCUDAStream());
  else
    WARNING(true, "MinkowskiPruning: Backprop from a size-0 sparse tensor.");
}
#endif

template void PruningForwardCPU<float>(at::Tensor in_feat, at::Tensor out_feat,
                                       at::Tensor use_feat,
                                       py::object py_in_coords_key,
                                       py::object py_out_coords_key,
                                       py::object py_coords_manager);

template void PruningForwardCPU<double>(at::Tensor in_feat, at::Tensor out_feat,
                                        at::Tensor use_feat,
                                        py::object py_in_coords_key,
                                        py::object py_out_coords_key,
                                        py::object py_coords_manager);

template void PruningBackwardCPU<float>(at::Tensor grad_in_feat,
                                        at::Tensor grad_out_feat,
                                        py::object py_in_coords_key,
                                        py::object py_out_coords_key,
                                        py::object py_coords_manager);

template void PruningBackwardCPU<double>(at::Tensor grad_in_feat,
                                         at::Tensor grad_out_feat,
                                         py::object py_in_coords_key,
                                         py::object py_out_coords_key,
                                         py::object py_coords_manager);

#ifndef CPU_ONLY
template void PruningForwardGPU<float>(at::Tensor in_feat, at::Tensor out_feat,
                                       at::Tensor use_feat,
                                       py::object py_in_coords_key,
                                       py::object py_out_coords_key,
                                       py::object py_coords_manager);

template void PruningForwardGPU<double>(at::Tensor in_feat, at::Tensor out_feat,
                                        at::Tensor use_feat,
                                        py::object py_in_coords_key,
                                        py::object py_out_coords_key,
                                        py::object py_coords_manager);

template void PruningBackwardGPU<float>(at::Tensor grad_in_feat,
                                        at::Tensor grad_out_feat,
                                        py::object py_in_coords_key,
                                        py::object py_out_coords_key,
                                        py::object py_coords_manager);

template void PruningBackwardGPU<double>(at::Tensor grad_in_feat,
                                         at::Tensor grad_out_feat,
                                         py::object py_in_coords_key,
                                         py::object py_out_coords_key,
                                         py::object py_coords_manager);
#endif // not CPU_ONLY

} // end namespace minkowski
