/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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

#include "coordinate_map.hpp"
#include "coordinate_map_cpu.hpp"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "broadcast_cpu.cpp"
#include "broadcast_kernel.cuh"

#include <ATen/cuda/CUDAUtils.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor BroadcastForwardGPU(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    BroadcastMode::Type const broadcast_mode,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(in_feat.is_cuda(), "in_feat must be on GPU");
  ASSERT(in_feat.dim() == 2, "Invalid in_feat.dim():", in_feat.dim());

  ASSERT(in_feat_glob.is_contiguous(), "in_feat_glob must be contiguous");
  ASSERT(in_feat_glob.is_cuda(), "in_feat_glob must be on GPU");
  ASSERT(in_feat_glob.dim() == 2,
         "Invalid in_feat_glob.dim():", in_feat_glob.dim());

  ASSERT(at::cuda::check_device({in_feat, in_feat_glob}),
         "all inputs must be on the same device");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  coordinate_map_key_type glob_key = p_glob_map_key->get_key();

  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  ASSERT(p_map_manager->exists(glob_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  int64_t const batch_size = p_map_manager->origin_map_size();

  ASSERT(in_feat_glob.size(0) == batch_size, "Invalid in_feat_glob size",
         in_feat_glob.size(0), "!=", batch_size);
  ASSERT(in_feat.size(1) == in_feat_glob.size(1), "Invalid feature sizes",
         in_feat.size(1), "!=", in_feat_glob.size(1));
  ASSERT(in_feat.scalar_type() == in_feat_glob.scalar_type(),
         "Incompatible scalar_type. Use the same float type for both in_feat "
         "and in_feat_glob.")

  const auto &in_outs = p_map_manager->origin_map(p_in_map_key);

  auto out_feat =
      torch::empty({in_feat.size(0), in_feat.size(1)}, in_feat.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  cusparseHandle_t handle = getCurrentCUDASparseHandle();
  cusparseSetStream(handle, stream);

  AT_DISPATCH_FLOATING_TYPES(
      in_feat.scalar_type(), "broadcast_forward_gpu", [&] {
        BroadcastForwardKernelGPU<scalar_t, uint32_t>(
            in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
            in_feat_glob.template data_ptr<scalar_t>(), in_feat_glob.size(0),
            out_feat.template data_ptr<scalar_t>(), in_feat.size(1),
            broadcast_mode, in_outs, handle, stream);
      });

  return out_feat;
}

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> BroadcastBackwardGPU(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    at::Tensor const &grad_out_feat, BroadcastMode::Type const op,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(in_feat.is_cuda(), "in_feat must be on GPU");
  ASSERT(in_feat.dim() == 2, "Invalid in_feat.dim():", in_feat.dim());

  ASSERT(in_feat_glob.is_contiguous(), "in_feat_glob must be contiguous");
  ASSERT(in_feat_glob.is_cuda(), "in_feat_glob must be on GPU");
  ASSERT(in_feat_glob.dim() == 2,
         "Invalid in_feat_glob.dim():", in_feat_glob.dim());

  ASSERT(grad_out_feat.is_contiguous(), "grad_out_feat must be contiguous");
  ASSERT(grad_out_feat.is_cuda(), "grad_out_feat must be on GPU");
  ASSERT(grad_out_feat.dim() == 2,
         "Invalid grad_out_feat.dim():", grad_out_feat.dim());

  ASSERT(at::cuda::check_device({in_feat, in_feat_glob, grad_out_feat}),
         "all inputs must be on the same device");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  coordinate_map_key_type glob_key = p_glob_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  ASSERT(p_map_manager->exists(glob_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  int64_t const batch_size = p_map_manager->origin_map_size();

  ASSERT(in_feat_glob.size(0) == batch_size, "Invalid in_feat_glob size",
         in_feat_glob.size(0), "!=", batch_size);
  ASSERT(in_feat.size(1) == in_feat_glob.size(1), "Invalid feature sizes",
         in_feat.size(1), "!=", in_feat_glob.size(1));
  ASSERT(in_feat.scalar_type() == in_feat_glob.scalar_type(),
         "Incompatible scalar_type. Use the same float type for both in_feat "
         "and in_feat_glob.")
  ASSERT(in_feat.scalar_type() == grad_out_feat.scalar_type(),
         "Incompatible scalar_type. Use the same float type for both in_feat "
         "and grad_out_feat.")

  auto grad_in_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());
  auto grad_glob_feat = torch::zeros(
      {in_feat_glob.size(0), in_feat_glob.size(1)}, in_feat_glob.options());

  const auto &in_outs = p_map_manager->origin_map(p_in_map_key);

  auto stream = at::cuda::getCurrentCUDAStream();
  cusparseHandle_t handle = getCurrentCUDASparseHandle();
  cusparseSetStream(handle, stream);

  AT_DISPATCH_FLOATING_TYPES(
      in_feat.scalar_type(), "broadcast_backward_gpu", [&] {
        BroadcastBackwardKernelGPU<scalar_t, uint32_t>(
            in_feat.template data_ptr<scalar_t>(),
            grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
            in_feat_glob.template data_ptr<scalar_t>(),
            grad_glob_feat.template data_ptr<scalar_t>(), in_feat_glob.size(0),
            grad_out_feat.template data_ptr<scalar_t>(), in_feat.size(1), op,
            in_outs, handle, stream);
      });

  return {grad_in_feat, grad_glob_feat};
}

// default allocators
template at::Tensor
BroadcastForwardGPU<default_types::dcoordinate_type, detail::default_allocator>(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    BroadcastMode::Type const broadcast_mode,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
BroadcastBackwardGPU<default_types::dcoordinate_type,
                     detail::default_allocator>(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    at::Tensor const &grad_out_feat, BroadcastMode::Type const op,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10 allocators
template at::Tensor
BroadcastForwardGPU<default_types::dcoordinate_type, detail::c10_allocator>(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    BroadcastMode::Type const broadcast_mode,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
BroadcastBackwardGPU<default_types::dcoordinate_type, detail::c10_allocator>(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    at::Tensor const &grad_out_feat, BroadcastMode::Type const op,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

} // namespace minkowski
