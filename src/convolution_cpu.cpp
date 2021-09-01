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
#include "coordinate_map.hpp"
#include "coordinate_map_cpu.hpp"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "convolution_kernel.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type>
at::Tensor
ConvolutionForwardCPU(at::Tensor const &in_feat,                         //
                      at::Tensor const &kernel,                          //
                      default_types::stride_type const &kernel_size,     //
                      default_types::stride_type const &kernel_stride,   //
                      default_types::stride_type const &kernel_dilation, //
                      RegionType::Type const region_type,                //
                      at::Tensor const &offset,                          //
                      bool const expand_coordinates,                     //
                      ConvolutionMode::Type const convolution_mode,      //
                      CoordinateMapKey *p_in_map_key,                    //
                      CoordinateMapKey *p_out_map_key,                   //
                      cpu_manager_type<coordinate_type> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  ASSERT(kernel.is_contiguous(), "kernel must be contiguous");

  ASSERT(!in_feat.is_cuda(), "in_feat must be CPU");
  ASSERT(!kernel.is_cuda(), "kernel must be CPU");

  ASSERT(in_feat.scalar_type() == kernel.scalar_type(), "type mismatch");

  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
  ASSERT(kernel.dim() == 3, "kernel.dim():", kernel.dim());

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  // create out coordinate map
  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  if (!p_out_map_key->is_key_set()) {
    if (expand_coordinates) {
      auto map_it = p_map_manager->find(p_in_map_key->get_key());
      ASSERT(map_it != p_map_manager->map_end(), ERROR_MAP_NOT_FOUND);
      auto const &in_map = (*map_it).second;

      auto out_tensor_stride = detail::stride_tensor_stride(
          in_map.get_tensor_stride(), kernel_stride, false /* is_transpose */);

      LOG_DEBUG("Coordinate Expansion");
      auto kernel_region = cpu_kernel_region<coordinate_type>(
          region_type,                       //
          in_map.coordinate_size(),          //
          in_map.get_tensor_stride().data(), //
          kernel_size.data(),                //
          kernel_dilation.data(),            //
          0, // volume. Will be initialized automatically
          offset.data_ptr<coordinate_type>(), offset.size(0),
          false // is_transpose
      );

      coordinate_map_key_type out_key =
          std::get<0>(p_map_manager->stride_region(
              in_key, kernel_region, out_tensor_stride, expand_coordinates));
      p_out_map_key->set_key(out_key);
    } else {
      coordinate_map_key_type out_key =
          std::get<0>(p_map_manager->stride(in_key, kernel_stride));
      p_out_map_key->set_key(out_key);
    }
  }

  cpu_kernel_map const &in_out = p_map_manager->kernel_map(
      p_in_map_key,    //
      p_out_map_key,   //
      kernel_size,     //
      kernel_stride,   //
      kernel_dilation, //
      region_type,     //
      offset, false /* is_transpose */, false /* is_pool */);

  auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
  at::Tensor out_feat =
      torch::zeros({out_nrows, kernel.size(2)}, in_feat.options());
  LOG_DEBUG("Allocated", out_nrows, "x", kernel.size(2), "out_features.");

  if (out_nrows > 0)
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "convolution_forward_cpu", [&] {
          ConvolutionForwardKernelCPU<scalar_t, coordinate_type>(
              in_feat.template data_ptr<scalar_t>(), in_feat.size(1),
              out_feat.template data_ptr<scalar_t>(), out_feat.size(1),
              kernel.template data_ptr<scalar_t>(), in_out.first,
              in_out.second);
        });

  return out_feat;
}

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardCPU(at::Tensor const &in_feat,                         //
                       at::Tensor &grad_out_feat,                         //
                       at::Tensor const &kernel,                          //
                       default_types::stride_type const &kernel_size,     //
                       default_types::stride_type const &kernel_stride,   //
                       default_types::stride_type const &kernel_dilation, //
                       RegionType::Type const region_type,                //
                       at::Tensor const &offset,                          //
                       ConvolutionMode::Type const convolution_mode,      //
                       CoordinateMapKey *p_in_map_key,                    //
                       CoordinateMapKey *p_out_map_key,                   //
                       cpu_manager_type<coordinate_type> *p_map_manager) {

  ASSERT(in_feat.is_contiguous(), "in_feat must be contiguous");
  // ASSERT(grad_out_feat.is_contiguous(), "grad_out_feata must be contiguous");
  grad_out_feat = grad_out_feat.contiguous();
  ASSERT(kernel.is_contiguous(), "kernel must be contiguous");

  ASSERT(!in_feat.is_cuda(), "in_feat must be CPU");
  ASSERT(!grad_out_feat.is_cuda(), "in_feat must be CPU");
  ASSERT(!kernel.is_cuda(), "kernel must be CPU");

  ASSERT(in_feat.scalar_type() == kernel.scalar_type(), "type mismatch");
  ASSERT(in_feat.scalar_type() == grad_out_feat.scalar_type(), "type mismatch");

  ASSERT(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
  ASSERT(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());
  ASSERT(kernel.dim() == 3, "kernel.dim():", kernel.dim());

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  coordinate_map_key_type out_key = p_out_map_key->get_key();
  ASSERT(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

  cpu_kernel_map const &in_out = p_map_manager->kernel_map(
      p_in_map_key,    //
      p_out_map_key,   //
      kernel_size,     //
      kernel_stride,   //
      kernel_dilation, //
      region_type,     //
      offset, false /* is_transpose */, false /* is_pool */);

  at::Tensor grad_in_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());
  at::Tensor grad_kernel = torch::zeros(
      {kernel.size(0), kernel.size(1), kernel.size(2)}, kernel.options());

  if (in_feat.size(0) > 0)
    AT_DISPATCH_FLOATING_TYPES(
        in_feat.scalar_type(), "convolution_backward_cpu", [&] {
          ConvolutionBackwardKernelCPU<scalar_t, coordinate_type>(
              in_feat.template data_ptr<scalar_t>(),
              grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(1),
              grad_out_feat.template data_ptr<scalar_t>(),
              grad_out_feat.size(1), kernel.template data_ptr<scalar_t>(),
              grad_kernel.template data_ptr<scalar_t>(), in_out.first,
              in_out.second);
        });

  return std::make_pair(grad_in_feat, grad_kernel);
}

template at::Tensor ConvolutionForwardCPU<default_types::dcoordinate_type>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    bool const expand_coordinates,                     //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    cpu_manager_type<default_types::dcoordinate_type> *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardCPU<default_types::dcoordinate_type>(
    at::Tensor const &in_feat,                         //
    at::Tensor &grad_out_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offsets,                         //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    cpu_manager_type<default_types::dcoordinate_type> *p_map_manager);

} // end namespace minkowski
