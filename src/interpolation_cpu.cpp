/*
 * Copyright (c) 2020 NVIDIA Corporation.
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

#include "interpolation_kernel.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type>
std::vector<at::Tensor>
InterpolationForwardCPU(at::Tensor const &in_feat,      //
                        at::Tensor const &tfield,       //
                        CoordinateMapKey *p_in_map_key, //
                        cpu_manager_type<coordinate_type> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(!in_feat.is_cuda(), "in_feat must be CPU");
  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());

  ASSERT(tfield.is_contiguous(), "tfield must be contiguous");
  ASSERT(!tfield.is_cuda(), "tfield must be CPU");
  ASSERT(tfield.dim() == 2, "tfield.dim():", tfield.dim());

  ASSERT(tfield.dtype() == in_feat.dtype(),
         "tfield and in_feat must have the same dtype");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  auto map_weight =
      p_map_manager->interpolation_map_weight(tfield, p_in_map_key);

  LOG_DEBUG("out_feat with size", tfield.size(0), in_feat.size(1));
  auto out_feat =
      torch::zeros({tfield.size(0), in_feat.size(1)}, tfield.options());

  AT_DISPATCH_FLOATING_TYPES(
      in_feat.scalar_type(), "interpolation_forward_cpu", [&] {
        LOG_DEBUG("InterpolationForwardKernelCPU");
        InterpolationForwardKernelCPU<scalar_t, scalar_t, int>(
            in_feat.template data_ptr<scalar_t>(),
            out_feat.template data_ptr<scalar_t>(), in_feat.size(1),
            map_weight[0].template data_ptr<int>(),      // in
            map_weight[1].template data_ptr<int>(),      // out
            map_weight[2].template data_ptr<scalar_t>(), // weight
            map_weight[0].numel());
      });

  // to out_feats
  map_weight.insert(map_weight.begin(), out_feat);
  return map_weight;
}

template <typename coordinate_type>
at::Tensor
InterpolationBackwardCPU(at::Tensor &grad_out_feat,      //
                         at::Tensor const &in_map,       //
                         at::Tensor const &out_map,      //
                         at::Tensor const &weight,       //
                         CoordinateMapKey *p_in_map_key, //
                         cpu_manager_type<coordinate_type> *p_map_manager) {

  if (!grad_out_feat.is_contiguous())
    grad_out_feat = grad_out_feat.contiguous();
  ASSERT(!grad_out_feat.is_cuda(), "grad_out_feat must be CPU");
  ASSERT(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  uint32_t const in_nrows = p_map_manager->size(in_key);
  uint32_t const nchannel = grad_out_feat.size(1);

  LOG_DEBUG("grad_in_feat with size", in_nrows, nchannel);
  auto grad_in_feat =
      torch::zeros({in_nrows, nchannel}, grad_out_feat.options());

  AT_DISPATCH_FLOATING_TYPES(
      grad_out_feat.scalar_type(), "interpolation_backward_cpu", [&] {
        LOG_DEBUG("InterpolationBackwardKernelCPU");
        InterpolationBackwardKernelCPU<scalar_t, scalar_t, int>(
            grad_in_feat.template data_ptr<scalar_t>(), in_nrows, nchannel,
            grad_out_feat.template data_ptr<scalar_t>(),
            in_map.template data_ptr<int>(),      // in
            out_map.template data_ptr<int>(),     // out
            weight.template data_ptr<scalar_t>(), // weight
            in_map.numel());
      });

  // to out_feats
  return grad_in_feat;
}

template std::vector<at::Tensor>
InterpolationForwardCPU<int32_t>(at::Tensor const &in_feat,      //
                                 at::Tensor const &tfield,       //
                                 CoordinateMapKey *p_in_map_key, //
                                 cpu_manager_type<int32_t> *p_map_manager);

template at::Tensor
InterpolationBackwardCPU<int32_t>(at::Tensor &grad_out_feat,      //
                                  at::Tensor const &in_map,       //
                                  at::Tensor const &out_map,      //
                                  at::Tensor const &weight,       //
                                  CoordinateMapKey *p_in_map_key, //
                                  cpu_manager_type<int32_t> *p_map_manager);
} // end namespace minkowski
