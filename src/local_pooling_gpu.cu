/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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

#include "pooling_avg_kernel.cuh"
#include "pooling_max_kernel.cuh"

// Ninja
#include "local_pooling_cpu.cpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> LocalPoolingForwardGPU(
    at::Tensor const &in_feat,
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(in_feat.is_cuda(), "in_feat must be on CUDA");
  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  // create an output coordinate map
  if (!p_out_map_key->is_key_set()) {
    coordinate_map_key_type out_key =
        std::get<0>(p_map_manager->stride(in_key, kernel_stride));
    p_out_map_key->set_key(out_key);
  }

  auto const &in_out = p_map_manager->kernel_map(
      p_in_map_key,    //
      p_out_map_key,   //
      kernel_size,     //
      kernel_stride,   //
      kernel_dilation, //
      region_type,     //
      offset, false /* is_transpose */, true /* is_pool */);

  auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
  at::Tensor out_feat =
      torch::zeros({out_nrows, in_feat.size(1)}, in_feat.options());
  LOG_DEBUG("Allocated", out_nrows, "x", in_feat.size(1), "features.");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (pooling_mode == PoolingMode::LOCAL_MAX_POOLING) {
    at::Tensor max_index = torch::empty({0}, torch::TensorOptions()
                                                 .device(in_feat.device())
                                                 .dtype(torch::kInt)
                                                 .requires_grad(false));
    max_index.resize_({out_nrows, in_feat.size(1)});
    max_index.zero_();
    TemplatedAllocator<char> byte_allocator;
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "local_pooling_forward_gpu", [&] {
          MaxPoolingForwardKernelGPU<scalar_t, default_types::index_type,
                                     TemplatedAllocator<char>>(
              in_feat.template data_ptr<scalar_t>(),
              out_feat.template data_ptr<scalar_t>(), out_nrows,
              max_index.data_ptr<int>(), in_feat.size(1), in_out,
              byte_allocator, stream);
        });
    return std::make_pair(out_feat, max_index);

  } else {
    at::Tensor num_nonzero =
        torch::empty({0}, in_feat.options().requires_grad(false));

    if (pooling_mode == PoolingMode::LOCAL_AVG_POOLING) {
      num_nonzero.resize_({out_nrows});
      num_nonzero.zero_();
    }
    cusparseHandle_t handle = getCurrentCUDASparseHandle();
    cusparseSetStream(handle, stream);

    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "local_pooling_forward_gpu", [&] {
          TemplatedAllocator<char> byte_allocator;
          NonzeroAvgPoolingForwardKernelGPU<scalar_t, default_types::index_type,
                                            TemplatedAllocator<char>>(
              in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
              out_feat.template data_ptr<scalar_t>(), out_nrows,
              num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
              in_out, pooling_mode == PoolingMode::LOCAL_AVG_POOLING,
              byte_allocator, handle, stream);
        });

    return std::make_pair(out_feat, num_nonzero);
  }
}

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor LocalPoolingBackwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &num_nonzero,                     //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {
  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(grad_out_feat.is_contiguous(), "grad_out_feata must be contiguous");

  ASSERT(in_feat.is_cuda(), "in_feat must be on CUDA");
  ASSERT(grad_out_feat.is_cuda(), "in_feat must be on CUDA");

  ASSERT(in_feat.scalar_type() == grad_out_feat.scalar_type(), "type mismatch");

  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
  ASSERT(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  coordinate_map_key_type out_key = p_out_map_key->get_key();
  ASSERT(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

  auto const &in_out = p_map_manager->kernel_map(
      p_in_map_key,    //
      p_out_map_key,   //
      kernel_size,     //
      kernel_stride,   //
      kernel_dilation, //
      region_type,     //
      offset, false /* is_transpose */, true /* is_pool */);

  at::Tensor grad_in_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (pooling_mode == PoolingMode::LOCAL_MAX_POOLING) {
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "local_pooling_backward_gpu", [&] {
          MaxPoolingBackwardKernelGPU<scalar_t>(
              grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
              grad_out_feat.template data_ptr<scalar_t>(),
              grad_out_feat.size(0), num_nonzero.data_ptr<int>(),
              in_feat.size(1));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "local_pooling_backward_gpu", [&] {
          NonzeroAvgPoolingBackwardKernelGPU<
              scalar_t, default_types::index_type, TemplatedAllocator<char>>(
              grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
              grad_out_feat.template data_ptr<scalar_t>(),
              grad_out_feat.size(0), num_nonzero.template data_ptr<scalar_t>(),
              in_feat.size(1), in_out,
              pooling_mode == PoolingMode::LOCAL_AVG_POOLING, stream);
        });
  }

  return grad_in_feat;
}

// Forward
template std::pair<at::Tensor, at::Tensor>
LocalPoolingForwardGPU<default_types::dcoordinate_type,
                       detail::default_allocator>(
    at::Tensor const &in_feat,
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
LocalPoolingForwardGPU<default_types::dcoordinate_type, detail::c10_allocator>(
    at::Tensor const &in_feat,
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

// Backward
template at::Tensor LocalPoolingBackwardGPU<default_types::dcoordinate_type,
                                            detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &num_nonzero,                     //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template at::Tensor
LocalPoolingBackwardGPU<default_types::dcoordinate_type, detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &num_nonzero,                     //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

} // end namespace minkowski
