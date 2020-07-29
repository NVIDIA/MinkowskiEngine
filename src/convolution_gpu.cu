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
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "convolution_kernel.cuh"
#include "kernel_map.cuh"

// Ninja
#include "convolution_cpu.cpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski {

template <typename coordinate_type, typename feature_type,
          template <typename C> class TemplatedAllocator>
at::Tensor ConvolutionForwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  torch::TensorArg arg_in_feat(in_feat, "in_feat", 0);
  torch::TensorArg arg_kernel(kernel, "kernel", 1);
  torch::TensorArg arg_offset(offset, "offset", 2);

  torch::CheckedFrom c = "ConvolutionForwardGPU";
  torch::checkContiguous(c, arg_in_feat);
  torch::checkContiguous(c, arg_kernel);
  torch::checkContiguous(c, arg_offset);

  torch::checkBackend(c, arg_in_feat.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_kernel.tensor, torch::Backend::CUDA);

  torch::checkScalarType(c, arg_in_feat,
                         std::is_same<feature_type, float>::value
                             ? torch::kFloat
                             : torch::kFloat64);
  torch::checkScalarType(c, arg_kernel,
                         std::is_same<feature_type, float>::value
                             ? torch::kFloat
                             : torch::kFloat64);

  torch::checkDim(c, arg_in_feat, 2);

  ASSERT(in_feat.size(1) == kernel.size(1),
         "Input feature size and kernel size mismatch");

  // create out coordinate map
  // TODO: custom upsampling
  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

  ASSERT(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
         in_feat.size(0), "!=", p_map_manager->size(in_key));

  if (!p_out_map_key->is_key_set()) {
    coordinate_map_key_type out_key =
        std::get<0>(p_map_manager->stride(in_key, kernel_stride));
    p_out_map_key->set_key(out_key);
  }

  auto const &in_out = p_map_manager->kernel_map(p_in_map_key,    //
                                                 p_out_map_key,   //
                                                 kernel_size,     //
                                                 kernel_stride,   //
                                                 kernel_dilation, //
                                                 region_type,     //
                                                 offset, false, false);

  auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
  at::Tensor out_feat =
      torch::zeros({out_nrows, kernel.size(2)}, in_feat.options());
  LOG_DEBUG("Allocated", out_nrows, "x", kernel.size(2), "out_features.");

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

  ConvolutionForwardKernelGPU<feature_type, default_types::index_type,
                              TemplatedAllocator<char>>(
      in_feat.template data_ptr<feature_type>(),  //
      in_feat.size(1),                            //
      out_feat.template data_ptr<feature_type>(), //
      out_feat.size(1),                           //
      kernel.template data_ptr<feature_type>(),   //
      in_out,                                     //
      out_nrows,
      handle, at::cuda::getCurrentCUDAStream());

  return out_feat;
}

template <typename coordinate_type, typename feature_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> ConvolutionBackwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager) {

  torch::TensorArg arg_in_feat(in_feat, "in_feat", 0);
  torch::TensorArg arg_grad_out_feat(grad_out_feat, "grad_out_feat", 1);
  torch::TensorArg arg_kernel(kernel, "kernel", 2);
  torch::TensorArg arg_offset(offset, "offset", 3);

  torch::CheckedFrom c = "ConvolutionBackwardGPU";
  torch::checkContiguous(c, arg_in_feat);
  torch::checkContiguous(c, arg_grad_out_feat);
  torch::checkContiguous(c, arg_kernel);
  torch::checkContiguous(c, arg_offset);

  torch::checkBackend(c, arg_in_feat.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_grad_out_feat.tensor, torch::Backend::CUDA);
  torch::checkBackend(c, arg_kernel.tensor, torch::Backend::CUDA);

  torch::checkScalarType(c, arg_in_feat,
                         std::is_same<feature_type, float>::value
                             ? torch::kFloat
                             : torch::kFloat64);
  torch::checkScalarType(c, arg_grad_out_feat,
                         std::is_same<feature_type, float>::value
                             ? torch::kFloat
                             : torch::kFloat64);
  torch::checkScalarType(c, arg_kernel,
                         std::is_same<feature_type, float>::value
                             ? torch::kFloat
                             : torch::kFloat64);

  torch::checkDim(c, arg_in_feat, 2);
  torch::checkDim(c, arg_grad_out_feat, 2);

  coordinate_map_key_type in_key = p_in_map_key->get_key();
  ASSERT(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
  coordinate_map_key_type out_key = p_out_map_key->get_key();
  ASSERT(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

  auto const &in_out = p_map_manager->kernel_map(p_in_map_key,    //
                                                 p_out_map_key,   //
                                                 kernel_size,     //
                                                 kernel_stride,   //
                                                 kernel_dilation, //
                                                 region_type,     //
                                                 offset, false, false);

  at::Tensor grad_in_feat =
      torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());
  at::Tensor grad_kernel = torch::zeros(
      {kernel.size(0), kernel.size(1), kernel.size(2)}, kernel.options());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

  ConvolutionBackwardKernelGPU<feature_type, default_types::index_type,
                               TemplatedAllocator<char>>(
      in_feat.template data_ptr<feature_type>(),       //
      grad_in_feat.template data_ptr<feature_type>(),  //
      in_feat.size(1),                                 //
      grad_out_feat.template data_ptr<feature_type>(), //
      grad_out_feat.size(1),                           //
      kernel.template data_ptr<feature_type>(),        //
      grad_kernel.template data_ptr<feature_type>(),   //
      in_out,                                          //
      grad_out_feat.size(0),                           //
      handle, at::cuda::getCurrentCUDAStream());

  return std::make_pair(grad_in_feat, grad_kernel);
}

// Forward
// default_allocator
template at::Tensor ConvolutionForwardGPU<default_types::dcoordinate_type,
                                          float, detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template at::Tensor ConvolutionForwardGPU<default_types::dcoordinate_type,
                                          double, detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10_allocator
template at::Tensor ConvolutionForwardGPU<default_types::dcoordinate_type,
                                          float, detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

template at::Tensor ConvolutionForwardGPU<default_types::dcoordinate_type,
                                          double, detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

// Backward
// default_allocator
template std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardGPU<default_types::dcoordinate_type, float,
                       detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardGPU<default_types::dcoordinate_type, double,
                       detail::default_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::default_allocator>
        *p_map_manager);

// c10_allocator
template std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardGPU<default_types::dcoordinate_type, float,
                       detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

template std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardGPU<default_types::dcoordinate_type, double,
                       detail::c10_allocator>(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<default_types::dcoordinate_type, detail::c10_allocator>
        *p_map_manager);

} // end namespace minkowski
