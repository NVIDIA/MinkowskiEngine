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

#include "global_pooling_cpu.cpp"
#include "pooling_avg_kernel.cuh"
#include "pooling_max_kernel.cuh"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::tuple<at::Tensor, at::Tensor> GlobalPoolingForwardGPU(
    at::Tensor const &in_feat,
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(in_feat.is_cuda(), "in_feat must be on GPU");
  ASSERT(in_feat.dim() == 2, "Invalid in_feat.dim():", in_feat.dim());

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key) || p_map_manager->exists_field(in_key),
         ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  ASSERT(pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX,
         "Invalid pooling mode");
  bool const is_field = p_map_manager->exists_field(in_key);

  if (!p_out_map_key->is_key_set()) {
    if (is_field) {
      coordinate_map_key_type out_key =
          std::get<0>(p_map_manager->origin_field());
      p_out_map_key->set_key(out_key);
    } else {
      coordinate_map_key_type out_key = std::get<0>(p_map_manager->origin());
      p_out_map_key->set_key(out_key);
    }
  }

  int64_t const batch_size = p_map_manager->origin_map_size();
  bool const use_avg =
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX;

  if (batch_size == 1) {
    // Simple reduction
    if (pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_DEFAULT ||
        pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_KERNEL ||
        pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX) {
      auto pair = in_feat.max(0, true);
      return {std::get<0>(pair), std::get<1>(pair).to(torch::kInt)};
    } else {
      auto out_feat = in_feat.sum(0, true);
      auto num_nonzero = torch::zeros({batch_size}, in_feat.options());
      if (use_avg)
        out_feat /= in_feat.size(0);
      num_nonzero[0] = in_feat.size(0);
      return {out_feat, num_nonzero};
    }

  } else {
    // batch_size > 1
    // TODO Default to specific pooling mode conversion.
    // Regular case
    // if (pooling_mode == 0)
    //   pooling_mode = in_feat.size(0) / batch_size > 100 ? 1 : 2;

    // origin kernel map
    if (pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_DEFAULT ||
        pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
        pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_KERNEL ||
        pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
        pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX ||
        pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX) {
      auto out_feat =
          torch::zeros({batch_size, in_feat.size(1)}, in_feat.options());
      auto num_nonzero = torch::zeros({batch_size}, in_feat.options());

      // If the policy is GlobalPoolingMode.INDEX_SELECT
      switch (pooling_mode) {
      case PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX:
      case PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX: {
        std::vector<at::Tensor> const vec_maps =
            p_map_manager->origin_map_th(p_in_map_key).second;
        for (int b = 0; b < batch_size; ++b) {
          if (use_avg)
            out_feat[b] = in_feat.index_select(0, vec_maps[b]).mean(0);
          else
            out_feat[b] = in_feat.index_select(0, vec_maps[b]).sum(0);
          num_nonzero[b] = vec_maps[b].numel();
        }
      } break;
      case PoolingMode::GLOBAL_SUM_POOLING_DEFAULT:
      case PoolingMode::GLOBAL_AVG_POOLING_DEFAULT:
      case PoolingMode::GLOBAL_SUM_POOLING_KERNEL:
      case PoolingMode::GLOBAL_AVG_POOLING_KERNEL: {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cusparseHandle_t handle = getCurrentCUDASparseHandle();
        cusparseSetStream(handle, stream);

        TemplatedAllocator<char> byte_allocator;

        if (is_field) {
          const auto &in_outs = p_map_manager->origin_field_map(p_in_map_key);
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "global_pooling_forward_gpu", [&] {
                NonzeroAvgPoolingForwardKernelGPU<scalar_t,
                                                  default_types::index_type,
                                                  TemplatedAllocator<char>>(
                    in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                    out_feat.template data_ptr<scalar_t>(), batch_size,
                    num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                    in_outs, use_avg, byte_allocator, handle, stream);
              });
        } else {
          const auto &in_outs = p_map_manager->origin_map(p_in_map_key);
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "global_pooling_forward_gpu", [&] {
                NonzeroAvgPoolingForwardKernelGPU<scalar_t,
                                                  default_types::index_type,
                                                  TemplatedAllocator<char>>(
                    in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                    out_feat.template data_ptr<scalar_t>(), batch_size,
                    num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                    in_outs, use_avg, byte_allocator, handle, stream);
              });
        }
      } break;
      }
      return {out_feat, num_nonzero};
    } else {
      // Max pool
      auto out_feat =
          torch::zeros({batch_size, in_feat.size(1)}, in_feat.options());
      at::Tensor max_index = torch::empty({batch_size, in_feat.size(1)},
                                          torch::TensorOptions()
                                              .device(in_feat.device())
                                              .dtype(torch::kInt)
                                              .requires_grad(false));

      switch (pooling_mode) {
      case PoolingMode::GLOBAL_MAX_POOLING_KERNEL:
        // TODO
      case PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX: {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        TemplatedAllocator<char> byte_allocator;
        if (is_field) {
          const auto &in_outs = p_map_manager->origin_field_map(p_in_map_key);
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "global_pooling_forward_gpu", [&] {
                MaxPoolingForwardKernelGPU<scalar_t, default_types::index_type,
                                           TemplatedAllocator<char>>(
                    in_feat.template data_ptr<scalar_t>(),
                    out_feat.template data_ptr<scalar_t>(), batch_size,
                    max_index.data_ptr<int>(), in_feat.size(1), in_outs,
                    byte_allocator, stream);
              });
        } else {
          const auto &in_outs = p_map_manager->origin_map(p_in_map_key);
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "global_pooling_forward_gpu", [&] {
                MaxPoolingForwardKernelGPU<scalar_t, default_types::index_type,
                                           TemplatedAllocator<char>>(
                    in_feat.template data_ptr<scalar_t>(),
                    out_feat.template data_ptr<scalar_t>(), batch_size,
                    max_index.data_ptr<int>(), in_feat.size(1), in_outs,
                    byte_allocator, stream);
              });
        }
      } break;
      default:
        ASSERT(false, "Invalid pooling mode");
      }
      return {out_feat, max_index};
    }
  }
}

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor GlobalPoolingBackwardGPU(
    at::Tensor const &in_feat,            //
    at::Tensor &grad_out_feat,            //
    at::Tensor const &num_nonzero,        //
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  ASSERT(in_feat.is_cuda(), "in_feat must be on CUDA");
  ASSERT(grad_out_feat.is_cuda(), "grad_out_feat must be on CUDA");
  ASSERT(num_nonzero.is_cuda(), "num_nonzero must be on CUDA");
  ASSERT(grad_out_feat.dim() == 2,
         "Invalid grad_out_feat.dim():", grad_out_feat.dim());
  if (!grad_out_feat.is_contiguous())
    grad_out_feat = grad_out_feat.contiguous();

  ASSERT(in_feat.scalar_type() == grad_out_feat.scalar_type(), "type mismatch");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key) || p_map_manager->exists_field(in_key),
         ERROR_MAP_NOT_FOUND);
  coordinate_map_key_type out_key = p_out_map_key->get_key();
  ASSERT(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

  ASSERT(grad_out_feat.size(0) == p_map_manager->size(out_key),
         "Invalid grad_out size", grad_out_feat.size(0),
         "!=", p_map_manager->size(out_key));

  ASSERT(in_feat.size(1) == grad_out_feat.size(1),
         "Input feature size and kernel size mismatch");

  ASSERT(pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_DEFAULT ||
             pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_KERNEL ||
             pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX ||
             pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX ||
             pooling_mode == PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX,
         "Invalid pooling mode");
  bool const is_field = p_map_manager->exists_field(in_key);
  int64_t const batch_size = p_map_manager->size(out_key);
  bool const use_avg =
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX;

  auto grad_in_feat = torch::empty_like(in_feat);
  // TODO Default to specific pooling mode conversion.
  // Regular case
  // if (pooling_mode == 0)
  //   pooling_mode = in_feat.size(0) / batch_size > 100 ? 1 : 2;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_DEFAULT ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_DEFAULT ||
      pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_KERNEL ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_KERNEL ||
      pooling_mode == PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX ||
      pooling_mode == PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX) {

    LOG_DEBUG("GLOBAL_POOLING");
    if (batch_size == 1) {
      if (use_avg) {
        LOG_DEBUG("Copying grad_out_feat. size:", in_feat.size(0));
        grad_in_feat.copy_(grad_out_feat / in_feat.size(0));
      } else
        grad_in_feat.copy_(grad_out_feat);
    } else {
      if (is_field) {
        const auto &in_outs = p_map_manager->origin_field_map(p_in_map_key);
        grad_in_feat.zero_();
        AT_DISPATCH_FLOATING_TYPES(
            in_feat.scalar_type(), "global_pooling_backward_gpu", [&] {
              NonzeroAvgPoolingBackwardKernelGPU<scalar_t,
                                                 default_types::index_type,
                                                 TemplatedAllocator<char>>(
                  grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                  grad_out_feat.template data_ptr<scalar_t>(),
                  grad_out_feat.size(0),
                  num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                  in_outs, use_avg, stream);
            });
      } else {
        const auto &in_outs = p_map_manager->origin_map(p_in_map_key);
        grad_in_feat.zero_();
        AT_DISPATCH_FLOATING_TYPES(
            in_feat.scalar_type(), "global_pooling_backward_gpu", [&] {
              NonzeroAvgPoolingBackwardKernelGPU<scalar_t,
                                                 default_types::index_type,
                                                 TemplatedAllocator<char>>(
                  grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                  grad_out_feat.template data_ptr<scalar_t>(),
                  grad_out_feat.size(0),
                  num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                  in_outs, use_avg, stream);
            });
      }
    }
  } else {
    // MAX Pooling
    grad_in_feat.zero_();
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "global_pooling_backward_gpu", [&] {
          MaxPoolingBackwardKernelGPU<scalar_t>(
              grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
              grad_out_feat.template data_ptr<scalar_t>(),
              grad_out_feat.size(0), num_nonzero.template data_ptr<int>(),
              in_feat.size(1));
        });
  }
  return grad_in_feat;
}

// default allocator
template std::tuple<at::Tensor, at::Tensor> GlobalPoolingForwardGPU(
    at::Tensor const &in_feat,
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template at::Tensor GlobalPoolingBackwardGPU(
    at::Tensor const &in_feat,            //
    at::Tensor &grad_out_feat,            //
    at::Tensor const &num_nonzero,        //
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10
template std::tuple<at::Tensor, at::Tensor> GlobalPoolingForwardGPU(
    at::Tensor const &in_feat,
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

template at::Tensor GlobalPoolingBackwardGPU(
    at::Tensor const &in_feat,            //
    at::Tensor &grad_out_feat,            //
    at::Tensor const &num_nonzero,        //
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

} // end namespace minkowski
