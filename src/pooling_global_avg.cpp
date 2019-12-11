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

template <typename Dtype>
void GlobalPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out =
      p_coords_manager->getOriginInOutMaps(py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  NonzeroAvgPoolingForwardKernelCPU<Dtype, int>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), num_nonzero.data<Dtype>(),
      in_feat.size(1), get<0>(in_out), get<1>(in_out), out_nrows, use_avg);
}

template <typename Dtype>
void GlobalPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  ASSERT(p_coords_manager->in_maps.find(map_key) !=
             p_coords_manager->in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

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
void GlobalPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const auto &in_out = p_coords_manager->getOriginInOutMapsGPU(
      py_in_coords_key, py_out_coords_key);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  // int dtype_mult = dtypeMultiplier<Dtype, int>(), nmap = 0;
  const int nmap = getInOutMapsSize(in_out.first);

  int *d_scr = (int *)p_coords_manager->getScratchGPUMemory(
      2 * nmap * sizeof(int) +      // in out maps to sort
      (out_nrows + 1) * sizeof(int) // csr_row
  );

  Dtype *d_dscr = (Dtype *)p_coords_manager->getScratchGPUMemory2(
      ((use_avg ? in_feat.size(0) : 0) + // d_ones
       nmap +                            // d_csr_val
       in_feat.size(1) * out_nrows       // d_tmp_out_feat
       ) *
      sizeof(Dtype));

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());

  NonzeroAvgPoolingForwardKernelGPU<Dtype, int>(
      in_feat.data<Dtype>(), in_feat.size(0), out_feat.data<Dtype>(), out_nrows,
      num_nonzero.data<Dtype>(), in_feat.size(1), get<0>(in_out),
      get<1>(in_out), use_avg, d_scr, d_dscr, handle,
      at::cuda::getCurrentCUDAStream());
}

template <typename Dtype>
void GlobalPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg) {
  CoordsManager *p_coords_manager = py_coords_manager.cast<CoordsManager *>();
  const InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

  ASSERT(p_coords_manager->d_in_maps.find(map_key) !=
             p_coords_manager->d_in_maps.end(),
         "The in-out map doesn't exist for backward. Did you run forward pass?")

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

template void GlobalPoolingForwardCPU<float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void GlobalPoolingForwardCPU<double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void GlobalPoolingBackwardCPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template void GlobalPoolingBackwardCPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template void GlobalPoolingForwardGPU<float>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void GlobalPoolingForwardGPU<double>(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void GlobalPoolingBackwardGPU<float>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template void GlobalPoolingBackwardGPU<double>(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);
#endif
