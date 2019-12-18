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
#include "union.hpp"
#include "common.hpp"
#ifndef CPU_ONLY
#include "union.cuh"
#endif

template <typename Dtype>
at::Tensor UnionForwardCPU(vector<at::Tensor> in_feats,
                           vector<py::object> py_in_coords_keys,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  // Basic assertions
  ASSERT(in_feats.size() > 1, "The number of input tensors must be > 1.");
  const size_t n_in = in_feats.size();
  for (size_t i = 1; i < n_in; i++) {
    ASSERT(in_feats[0].dtype() == in_feats[i].dtype(),
           "Datatype mismatch: ", in_feats[0].dtype(),
           " != ", in_feats[i].dtype());

    ASSERT(in_feats[0].device() == in_feats[i].device(),
           "Device mismatch: ", in_feats[0].device(),
           " != ", in_feats[i].device());

    ASSERT(in_feats[0].size(1) == in_feats[i].size(1),
           "Feature size mismatch: ", in_feats[0].size(1),
           " != ", in_feats[i].size(1));
  }

  // Create new out map and get the in-out map
  const auto &in_out =
      p_coords_manager->getUnionInOutMaps(py_in_coords_keys, py_out_coords_key);

  // Out feat memory alloc
  const long out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  auto out_feat =
      torch::zeros({out_nrows, in_feats[0].size(1)}, in_feats[0].options());

  // In feat pointers
  vector<Dtype *> p_in_feats;
  p_in_feats.reserve(n_in);
  for (auto &in_feat : in_feats)
    p_in_feats.push_back(in_feat.data<Dtype>());

  UnionForwardKernelCPU<Dtype, int>(p_in_feats, out_feat.data<Dtype>(),
                                    in_feats[0].size(1), get<0>(in_out),
                                    get<1>(in_out));

  return out_feat;
}

template <typename Dtype>
vector<at::Tensor>
UnionBackwardCPU(at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
                 py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const int nchannel = grad_out_feat.size(1);
  const size_t n_in = py_in_coords_keys.size();

  const InOutMapKey map_key = p_coords_manager->getUnionMapHashKey(
      py_in_coords_keys, py_out_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  vector<at::Tensor> grad_in_feats;
  vector<Dtype *> p_grad_in_feats;

  grad_in_feats.reserve(n_in);
  p_grad_in_feats.reserve(n_in);

  for (auto &py_in_coords_key : py_in_coords_keys) {
    const int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
    auto grad_in_feat =
        torch::zeros({in_nrows, nchannel}, grad_out_feat.options());
    grad_in_feats.push_back(grad_in_feat);
    p_grad_in_feats.push_back(grad_in_feat.data<Dtype>());
  }

  UnionBackwardKernelCPU<Dtype, int>(
      p_grad_in_feats, grad_out_feat.data<Dtype>(), nchannel,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);

  return grad_in_feats;
}

#ifndef CPU_ONLY
template <typename Dtype>
at::Tensor UnionForwardGPU(vector<at::Tensor> in_feats,
                           vector<py::object> py_in_coords_keys,
                           py::object py_out_coords_key,
                           py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  // Basic assertions
  ASSERT(in_feats.size() > 1, "The number of input tensors must be > 1.");
  const size_t n_in = in_feats.size();
  for (size_t i = 1; i < n_in; i++) {
    ASSERT(in_feats[0].dtype() == in_feats[i].dtype(),
           "Datatype mismatch: ", in_feats[0].dtype(),
           " != ", in_feats[i].dtype());

    ASSERT(in_feats[0].device() == in_feats[i].device(),
           "Device mismatch: ", in_feats[0].device(),
           " != ", in_feats[i].device());

    ASSERT(in_feats[0].size(1) == in_feats[i].size(1),
           "Feature size mismatch: ", in_feats[0].size(1),
           " != ", in_feats[i].size(1));
  }

  // Create new out map and get the in-out map
  const auto &in_out = p_coords_manager->getUnionInOutMapsGPU(
      py_in_coords_keys, py_out_coords_key);

  // Out feat memory alloc
  const long out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  auto out_feat =
      torch::zeros({out_nrows, in_feats[0].size(1)}, in_feats[0].options());

  // In feat pointers
  vector<Dtype *> p_in_feats;
  p_in_feats.reserve(n_in);
  for (auto &in_feat : in_feats)
    p_in_feats.push_back(in_feat.data<Dtype>());

  UnionForwardKernelGPU<Dtype, int>(
      p_in_feats, out_feat.data<Dtype>(), in_feats[0].size(1), in_out.first,
      in_out.second, at::cuda::getCurrentCUDAStream());

  return out_feat;
}

template <typename Dtype>
vector<at::Tensor>
UnionBackwardGPU(at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
                 py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const int nchannel = grad_out_feat.size(1);
  const size_t n_in = py_in_coords_keys.size();

  const InOutMapKey map_key = p_coords_manager->getUnionMapHashKey(
      py_in_coords_keys, py_out_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

  vector<at::Tensor> grad_in_feats;
  vector<Dtype *> p_grad_in_feats;

  grad_in_feats.reserve(n_in);
  p_grad_in_feats.reserve(n_in);

  for (auto &py_in_coords_key : py_in_coords_keys) {
    const int in_nrows = p_coords_manager->getCoordsSize(py_in_coords_key);
    auto grad_in_feat =
        torch::zeros({in_nrows, nchannel}, grad_out_feat.options());
    grad_in_feats.push_back(grad_in_feat);
    p_grad_in_feats.push_back(grad_in_feat.data<Dtype>());
  }

  UnionBackwardKernelGPU<Dtype, int>(
      p_grad_in_feats, grad_out_feat.data<Dtype>(), nchannel,
      p_coords_manager->d_in_maps[map_key],
      p_coords_manager->d_out_maps[map_key], at::cuda::getCurrentCUDAStream());

  return grad_in_feats;
}
#endif

template at::Tensor UnionForwardCPU<float>(vector<at::Tensor> in_feats,
                                           vector<py::object> py_in_coords_keys,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager);

template at::Tensor UnionForwardCPU<double>(
    vector<at::Tensor> in_feats, vector<py::object> py_in_coords_keys,
    py::object py_out_coords_key, py::object py_coords_manager);

template vector<at::Tensor> UnionBackwardCPU<float>(
    at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
    py::object py_out_coords_key, py::object py_coords_manager);

template vector<at::Tensor> UnionBackwardCPU<double>(
    at::Tensor grad_out_feat, vector<py::object> py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template at::Tensor UnionForwardGPU<float>(vector<at::Tensor> in_feats,
                                           vector<py::object> py_in_coords_keys,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager);

template at::Tensor UnionForwardGPU<double>(
    vector<at::Tensor> in_feats, vector<py::object> py_in_coords_keys,
    py::object py_out_coords_key, py::object py_coords_manager);

template vector<at::Tensor> UnionBackwardGPU<float>(
    at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
    py::object py_out_coords_key, py::object py_coords_manager);

template vector<at::Tensor> UnionBackwardGPU<double>(
    at::Tensor grad_out_feat, vector<py::object> py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#endif
