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
vector<at::Tensor> GlobalPoolingForwardCPU(at::Tensor in_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager,
                                           bool use_avg, int pooling_mode) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto batch_size = p_coords_manager->getBatchSize();

  if (batch_size == 1) {

    p_coords_manager->setOriginCoordsKey(py_out_coords_key);
    auto out_feat = in_feat.sum(0, true);
    if (use_avg)
      out_feat /= in_feat.size(0);
    auto num_nonzero = torch::zeros({batch_size}, in_feat.options());
    num_nonzero[0] = in_feat.size(0);
    return {out_feat, num_nonzero};

  } else {

    if (pooling_mode == 0)
      pooling_mode = in_feat.size(0) / batch_size > 100 ? 1 : 2;

    auto out_feat =
        torch::zeros({batch_size, in_feat.size(1)}, in_feat.options());
    auto num_nonzero = torch::zeros({batch_size}, in_feat.options());

    // If the policy is GlobalPoolingMode.INDEX_SELECT
    switch (pooling_mode) {
    case 1: {
      const auto vec_maps = p_coords_manager->getRowIndicesPerBatch(
          py_in_coords_key, py_out_coords_key);
      for (int b = 0; b < batch_size; ++b) {
        if (use_avg)
          out_feat[b] = in_feat.index_select(0, vec_maps[b]).mean(0);
        else
          out_feat[b] = in_feat.index_select(0, vec_maps[b]).sum(0);
        num_nonzero[b] = vec_maps[b].numel();
      }
    } break;
    case 2: {
      const auto &in_outs = p_coords_manager->getOriginInOutMaps(
          py_in_coords_key, py_out_coords_key);

      NonzeroAvgPoolingForwardKernelCPU<Dtype, int>(
          in_feat.data<Dtype>(), out_feat.data<Dtype>(),
          num_nonzero.data<Dtype>(), in_feat.size(1), in_outs.first,
          in_outs.second, batch_size, use_avg);
    } break;
    default:
      ASSERT(false, "Invalid pooling mode", pooling_mode);
    }
    return {out_feat, num_nonzero};
  }
}

template <typename Dtype>
at::Tensor
GlobalPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_out_feat,
                         at::Tensor num_nonzero, py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto batch_size = p_coords_manager->getBatchSize();

  auto grad_in_feat = torch::empty_like(in_feat);

  if (batch_size == 1) {
    if (use_avg)
      grad_in_feat.copy_(grad_out_feat / in_feat.size(0));
    else
      grad_in_feat.copy_(grad_out_feat);
  } else {
    const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
        py_in_coords_key, py_out_coords_key);

    ASSERT(
        p_coords_manager->existsInOutMapKey(map_key),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    grad_in_feat.zero_();

    NonzeroAvgPoolingBackwardKernelCPU<Dtype, int>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
        use_avg);
  }
  return grad_in_feat;
}

#ifndef CPU_ONLY
template <typename Dtype>
vector<at::Tensor> GlobalPoolingForwardGPU(at::Tensor in_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager,
                                           bool use_avg, int pooling_mode) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto batch_size = p_coords_manager->getBatchSize();

  if (batch_size == 1) {

    p_coords_manager->setOriginCoordsKey(py_out_coords_key);
    auto out_feat = in_feat.sum(0, true);
    if (use_avg)
      out_feat /= in_feat.size(0);
    auto num_nonzero = torch::zeros({batch_size}, in_feat.options());
    num_nonzero[0] = in_feat.size(0);
    return {out_feat, num_nonzero};

  } else {

    if (pooling_mode == 0)
      pooling_mode = in_feat.size(0) / batch_size > 100 ? 1 : 2;

    auto out_feat =
        torch::zeros({batch_size, in_feat.size(1)}, in_feat.options());
    auto num_nonzero = torch::zeros({batch_size}, in_feat.options());

    // If the policy is GlobalPoolingMode.INDEX_SELECT
    switch (pooling_mode) {
    case 1: {
      const auto vec_maps = p_coords_manager->getRowIndicesPerBatch(
          py_in_coords_key, py_out_coords_key);
      for (int b = 0; b < batch_size; ++b) {
        if (use_avg)
          out_feat[b] =
              in_feat.index_select(0, vec_maps[b].to(in_feat.device())).mean(0);
        else
          out_feat[b] =
              in_feat.index_select(0, vec_maps[b].to(in_feat.device())).sum(0);
        num_nonzero[b] = vec_maps[b].numel();
      }
    } break;
    case 2: {
      const auto &in_outs = p_coords_manager->getOriginInOutMapsGPU(
          py_in_coords_key, py_out_coords_key);

      cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
      cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

      NonzeroAvgPoolingForwardKernelGPU<Dtype, int>(
          in_feat.data<Dtype>(), in_feat.size(0), out_feat.data<Dtype>(),
          batch_size, num_nonzero.data<Dtype>(), in_feat.size(1), in_outs.first,
          in_outs.second, use_avg, handle, at::cuda::getCurrentCUDAStream());

    } break;
    default:
      ASSERT(false, "Invalid pooling mode", pooling_mode);
    }
    return {out_feat, num_nonzero};
  }
}

template <typename Dtype>
at::Tensor
GlobalPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_out_feat,
                         at::Tensor num_nonzero, py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_man = py_coords_manager.cast<CoordsManager *>();
  const auto batch_size = p_coords_man->getBatchSize();

  auto grad_in_feat = torch::empty_like(in_feat);

  if (batch_size == 1) {
    if (use_avg)
      grad_in_feat.copy_(grad_out_feat / in_feat.size(0));
    else
      grad_in_feat.copy_(grad_out_feat);
  } else {
    const InOutMapKey map_key =
        p_coords_man->getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

    ASSERT(
        p_coords_man->existsInOutMapKey(map_key),
        "The in-out map doesn't exist for backward. Did you run forward pass?");

    p_coords_man->copyInOutMapsToGPU(map_key);

    grad_in_feat.zero_();

    NonzeroAvgPoolingBackwardKernelGPU<Dtype, int>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), grad_out_feat.size(0),
        num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_man->d_in_maps[map_key], p_coords_man->d_out_maps[map_key],
        use_avg, at::cuda::getCurrentCUDAStream());
  }
  return grad_in_feat;
}
#endif // CPU_ONLY

template vector<at::Tensor>
GlobalPoolingForwardCPU<float>(at::Tensor in_feat, py::object py_in_coords_key,
                               py::object py_out_coords_key,
                               py::object py_coords_manager, bool use_avg,
                               int pooling_mode);

template vector<at::Tensor>
GlobalPoolingForwardCPU<double>(at::Tensor in_feat, py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager, bool use_avg,
                                int pooling_mode);

template at::Tensor GlobalPoolingBackwardCPU<float>(
    at::Tensor in_feat, at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template at::Tensor GlobalPoolingBackwardCPU<double>(
    at::Tensor in_feat, at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template vector<at::Tensor>
GlobalPoolingForwardGPU<float>(at::Tensor in_feat, py::object py_in_coords_key,
                               py::object py_out_coords_key,
                               py::object py_coords_manager, bool use_avg,
                               int pooling_mode);

template vector<at::Tensor>
GlobalPoolingForwardGPU<double>(at::Tensor in_feat, py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager, bool use_avg,
                                int pooling_mode);

template at::Tensor GlobalPoolingBackwardGPU<float>(
    at::Tensor in_feat, at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template at::Tensor GlobalPoolingBackwardGPU<double>(
    at::Tensor in_feat, at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif // end CPU_ONLY

} // end namespace minkowski
