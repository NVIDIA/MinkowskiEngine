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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "dispatcher.hpp"
#include "types.hpp"

#include <torch/extension.h>
#include <torch/script.h>

#ifndef CPU_ONLY
#include <ATen/cuda/CUDAUtils.h>
#endif

namespace minkowski {

template <typename Dtype, typename MaskItype, typename MapItype>
void max_pooling_forward_pointer_kernel_cpu(Dtype const *p_in_feat,
                                            Dtype *p_out_feat,
                                            MaskItype *p_mask_index,
                                            size_t const nchannel,
                                            MapItype const *const p_in_maps,  //
                                            MapItype const *const p_out_maps, //
                                            size_t const map_size);

template <typename Dtype, typename MaskItype>
void MaxPoolingBackwardKernelCPU(Dtype *p_grad_in_feat, size_t const in_nrows,
                                 Dtype const *p_grad_out_feat,
                                 size_t const out_nrows,
                                 MaskItype const *p_mask_index,
                                 size_t const nchannel);

#ifndef CPU_ONLY
template <typename Dtype, typename MaskItype, typename MapItype>
void max_pool_forward_pointer_kernel_gpu(
    MapItype *d_in_map,     // this will be sorted
    MapItype *d_out_map,    // this will be sorted
    size_t const nmap,      // map size
    Dtype const *d_in_feat, //
    Dtype *d_out_feat,      //
    size_t const out_nrows, //
    size_t const nchannel,  //
    MaskItype *d_max_index, //
    bool const is_sorted    //
);

template <typename Dtype, typename MaskItype>
void MaxPoolingBackwardKernelGPU(Dtype *d_grad_in_feat, size_t const in_nrows,
                                 const Dtype *d_grad_out_feat,
                                 size_t const out_nrows,
                                 const MaskItype *d_max_index,
                                 size_t const nchannel);
#endif

std::pair<torch::Tensor, torch::Tensor>
max_pool_fw(torch::Tensor const &in_map,  //
            torch::Tensor const &out_map, //
            torch::Tensor const &in_feat, //
            int const out_nrows, bool const is_sorted) {
  // Out feat
  at::Tensor out_feat =
      at::zeros({out_nrows, in_feat.size(1)}, in_feat.options());
  at::Tensor max_index = torch::zeros({out_nrows, in_feat.size(1)},
                                      in_map.options().requires_grad(false));

  if (in_feat.device().is_cuda()) {
#ifdef CPU_ONLY
    AT_ERROR("Please compile again with CUDA support");
#else
    ASSERT(in_map.is_cuda(), "in_map must be a CUDA tensor.");
    ASSERT(out_map.is_cuda(), "kernel must be a CUDA tensor.");
    ASSERT(at::cuda::check_device({in_map, out_map, in_feat}),
           "all inputs must be on the same device");

    MINK_DISPATCH_INTEGER_TYPES(
        in_map.scalar_type(), integer_t, "max_pool_forward_gpu", [&] {
          LOG_DEBUG("Integer size", sizeof(integer_t));
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "max_pool_forward_gpu", [&] {
                max_pool_forward_pointer_kernel_gpu<scalar_t, integer_t,
                                                    integer_t>(
                    in_map.data_ptr<integer_t>(), out_map.data_ptr<integer_t>(),
                    in_map.numel(), in_feat.data_ptr<scalar_t>(),
                    out_feat.data_ptr<scalar_t>(), out_nrows, in_feat.size(1),
                    max_index.data_ptr<integer_t>(), is_sorted);
              });
        });
#endif
  } else {
    MINK_DISPATCH_INTEGER_TYPES(
        in_map.scalar_type(), integer_t, "max_pool_forward_cpu", [&] {
          LOG_DEBUG("Integer size", sizeof(integer_t));
          AT_DISPATCH_FLOATING_TYPES(
              in_feat.scalar_type(), "max_pool_forward_cpu", [&] {
                // Dtype, MaskItype, MapType
                max_pooling_forward_pointer_kernel_cpu<scalar_t, integer_t,
                                                       integer_t>(
                    in_feat.data_ptr<scalar_t>(), out_feat.data_ptr<scalar_t>(),
                    max_index.data_ptr<integer_t>(), in_feat.size(1),
                    in_map.data_ptr<integer_t>(), out_map.data_ptr<integer_t>(),
                    in_map.numel());
              });
        });
  }
  return {out_feat, max_index};
}

torch::Tensor max_pool_bw(torch::Tensor const &grad_out_feat, //
                          torch::Tensor const &mask_index,    //
                          int const in_nrows) {
  int const out_nrows = grad_out_feat.size(0);
  at::Tensor grad_in_feat =
      at::zeros({in_nrows, grad_out_feat.size(1)}, grad_out_feat.options());

  if (grad_out_feat.device().is_cuda()) {
#ifdef CPU_ONLY
    AT_ERROR("Please compile again with CUDA support");
#else
    ASSERT(mask_index.is_cuda(), "kernel must be a CUDA tensor.");
    ASSERT(at::cuda::check_device({mask_index, grad_out_feat}),
           "all inputs must be on the same device");
    MINK_DISPATCH_INTEGER_TYPES(
        mask_index.scalar_type(), integer_t, "max_pool_backward_gpu", [&] {
          AT_DISPATCH_FLOATING_TYPES(
              grad_out_feat.scalar_type(), "max_pool_backward_gpu", [&] {
                MaxPoolingBackwardKernelGPU<scalar_t, integer_t>(
                    grad_in_feat.data_ptr<scalar_t>(), in_nrows,
                    grad_out_feat.data_ptr<scalar_t>(), out_nrows,
                    mask_index.data_ptr<integer_t>(), grad_out_feat.size(1));
              });
        });
#endif
  } else {
    MINK_DISPATCH_INTEGER_TYPES(
        mask_index.scalar_type(), integer_t, "max_pool_backward_cpu", [&] {
          AT_DISPATCH_FLOATING_TYPES(
              grad_out_feat.scalar_type(), "max_pool_backward_cpu", [&] {
                MaxPoolingBackwardKernelCPU<scalar_t, integer_t>(
                    grad_in_feat.data_ptr<scalar_t>(), in_nrows,   //
                    grad_out_feat.data_ptr<scalar_t>(), out_nrows, //
                    mask_index.data_ptr<integer_t>(), grad_out_feat.size(1));
              });
        });
  }

  return grad_in_feat;
}

/*
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DirectMaxPool : public torch::autograd::Function<DirectMaxPool> {
public:
  static variable_list forward(AutogradContext *ctx, Variable in_map,
                               Variable out_map, Variable in_feat,
                               int64_t num_out, bool is_sorted) {
    auto out_pair = max_pool_fw(in_map, out_map, in_feat, num_out, is_sorted);
    ctx->saved_data["in_nrows"] = in_feat.size(0);
    ctx->save_for_backward({std::get<1>(out_pair)});
    return {std::get<0>(out_pair)};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto mask_index = saved[0];
    int const in_nrows = ctx->saved_data["in_nrows"].toInt();

    auto grad = max_pool_bw(grad_outs[0], mask_index, in_nrows);
    return {Variable(), Variable(), grad, Variable(), Variable()};
  }
};

torch::Tensor direct_max_pool(torch::Tensor &in_map, torch::Tensor &out_map,
                              torch::Tensor &in_feat, int64_t num_out,
                              bool is_sorted) {
  return DirectMaxPool::apply(in_map, out_map, in_feat, num_out, is_sorted)[0];
}

static auto registry = torch::RegisterOperators().op(
    "MinkowskiEngineBackend::direct_max_pool", &minkowski::direct_max_pool);
*/

} // namespace minkowski
