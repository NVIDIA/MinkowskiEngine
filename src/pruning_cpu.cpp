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

#include "pruning.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type>
at::Tensor
PruningForwardCPU(at::Tensor const &in_feat, // CPU feat
                  at::Tensor const &keep,    // uint8 / bool / byte CPU data
                  CoordinateMapKey *p_in_map_key,  //
                  CoordinateMapKey *p_out_map_key, //
                  cpu_manager_type<coordinate_type> *p_map_manager) {
  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(keep.is_contiguous(), "keep must be contiguous");

  ASSERT(!in_feat.is_cuda(), "in_feat must be CPU");
  ASSERT(!keep.is_cuda(), "keep must be CPU");

  ASSERT(keep.dtype() == torch::kBool || keep.dtype() == torch::kByte,
         "keep must be a boolean tensor");

  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
  ASSERT(keep.dim() == 1, "keep.dim():", keep.dim());

  auto const N = in_feat.size(0);
  ASSERT(N == keep.size(0), "Input feature size and keep size mismatch");

  // create out coordinate map
  coordinate_map_key_type const &in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  ASSERT(N == p_map_manager->size(in_key), "Invalid in_feat size", N,
         "!=", p_map_manager->size(in_key));

  bool const *keep_begin = keep.template data_ptr<bool>();

  if (!p_out_map_key->is_key_set()) {
    coordinate_map_key_type out_key =
        p_map_manager->prune(in_key, keep_begin, keep_begin + N);
    p_out_map_key->set_key(out_key);
  }

  const auto &in_out = p_map_manager->kernel_map(p_in_map_key, p_out_map_key);
  LOG_DEBUG("Generated kernel map");

  // Get the total number of coords
  const int64_t tot_n = p_map_manager->size(p_out_map_key->get_key());
  const auto nchannel = in_feat.size(1);
  at::Tensor out_feat =
      torch::empty({tot_n, nchannel}, in_feat.options());
  LOG_DEBUG("out_feat", tot_n, "x", nchannel);

  if (tot_n == 0) {
    WARNING(true, "MinkowskiPruning: Generating an empty SparseTensor");
  } else {
    out_feat.zero_();
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "pruning_forward_cpu", [&] {
          PruningForwardKernelCPU<scalar_t>(
              in_feat.template data_ptr<scalar_t>(),
              out_feat.template data_ptr<scalar_t>(), nchannel,
              std::get<0>(in_out), std::get<1>(in_out));
        });
  }

  return out_feat;
}

template <typename coordinate_type>
at::Tensor
PruningBackwardCPU(at::Tensor &grad_out_feat,       // CPU out feat
                   CoordinateMapKey *p_in_map_key,  //
                   CoordinateMapKey *p_out_map_key, //
                   cpu_manager_type<coordinate_type> *p_map_manager) {
  if (!grad_out_feat.is_contiguous())
    grad_out_feat = grad_out_feat.contiguous();

  ASSERT(!grad_out_feat.is_cuda(), "grad_out_feat must be CPU");

  ASSERT(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());

  coordinate_map_key_type const &in_key = p_in_map_key->get_key();
  coordinate_map_key_type const &out_key = p_out_map_key->get_key();
  const int64_t N_in = p_map_manager->size(in_key);
  const int64_t N_out = p_map_manager->size(out_key);

  ASSERT(grad_out_feat.size(0) == N_out, "Invalid grad_out_feat size",
         grad_out_feat.size(0), "!=", N_out);

  const auto &in_out = p_map_manager->kernel_map(p_in_map_key, p_out_map_key);
  const int nchannel = grad_out_feat.size(1);
  at::Tensor grad_in_feat =
      torch::zeros({N_in, nchannel}, grad_out_feat.options());

  if (grad_out_feat.size(0) > 0)
    AT_DISPATCH_FLOATING_TYPES(
        grad_out_feat.scalar_type(), "pruning_backward_cpu", [&] {
          PruningBackwardKernelCPU<scalar_t>(
              grad_in_feat.template data_ptr<scalar_t>(),
              grad_out_feat.template data_ptr<scalar_t>(), nchannel,
              std::get<0>(in_out), std::get<1>(in_out));
        });
  else
    WARNING(true, "MinkowskiPruning: Backprop from a size-0 sparse tensor.");

  return grad_in_feat;
}

template at::Tensor PruningForwardCPU<default_types::dcoordinate_type>(
    at::Tensor const &in_feat, at::Tensor const &use_feat,
    CoordinateMapKey *p_in_map_key,  //
    CoordinateMapKey *p_out_map_key, //
    cpu_manager_type<default_types::dcoordinate_type> *p_map_manager);

template at::Tensor PruningBackwardCPU<default_types::dcoordinate_type>(
    at::Tensor &grad_out_feat,
    CoordinateMapKey *p_in_map_key,  //
    CoordinateMapKey *p_out_map_key, //
    cpu_manager_type<default_types::dcoordinate_type> *p_map_manager);

} // end namespace minkowski
