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
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "src/common.hpp"

/*************************************
 * Convolution
 *************************************/
template <typename Dtype, typename Itype>
void ConvolutionForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, std::vector<int> tensor_strides,
                           std::vector<int> strides,
                           std::vector<int> kernel_sizes,
                           std::vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype, typename Itype>
void ConvolutionBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, std::vector<int> tensor_strides,
                           std::vector<int> strides,
                           std::vector<int> kernel_sizes,
                           std::vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype, typename Itype>
void ConvolutionBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif

/*************************************
 * Convolution Transpose
 *************************************/
template <typename Dtype, typename Itype>
void ConvolutionTransposeForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager,
    bool generate_new_coords);

template <typename Dtype, typename Itype>
void ConvolutionTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void ConvolutionTransposeForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager,
    bool generate_new_coords);

template <typename Dtype, typename Itype>
void ConvolutionTransposeBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif

/*************************************
 * AvgPooling
 *************************************/
template <typename Dtype, typename Itype>
void AvgPoolingForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void AvgPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void AvgPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void AvgPoolingBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);
#endif

/*************************************
 * MaxPooling
 *************************************/
template <typename Dtype, typename Itype>
void MaxPoolingForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void MaxPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void MaxPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void MaxPoolingBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif

/*************************************
 * PoolingTranspose
 *************************************/
template <typename Dtype, typename Itype>
void PoolingTransposeForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void PoolingTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void PoolingTransposeForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void PoolingTransposeBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);
#endif

/*************************************
 * GlobalPooling
 *************************************/
template <typename Dtype, typename Itype>
void GlobalPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void GlobalPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void GlobalPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void GlobalPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg);
#endif

/*************************************
 * GlobalMaxPooling
 *************************************/
template <typename Dtype, typename Itype>
void GlobalMaxPoolingForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void GlobalMaxPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void GlobalMaxPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void GlobalMaxPoolingBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * Broadcast
 *************************************/
template <typename Dtype, typename Itype>
void BroadcastForwardCPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager);

template <typename Dtype, typename Itype>
void BroadcastBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void BroadcastForwardGPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager);

template <typename Dtype, typename Itype>
void BroadcastBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager);
#endif

/*************************************
 * Pruning
 *************************************/
template <typename Dtype, typename Itype>
void PruningForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                       at::Tensor use_feat, py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager);

template <typename Dtype, typename Itype>
void PruningBackwardCPU(at::Tensor grad_in_feat, at::Tensor grad_out_feat,
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void PruningForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                       at::Tensor use_feat, py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager);

template <typename Dtype, typename Itype>
void PruningBackwardGPU(at::Tensor grad_in_feat, at::Tensor grad_out_feat,
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager);
#endif

/*************************************
 * Voxelization
 *************************************/
#ifndef CPU_ONLY
#include <pybind11/numpy.h>
std::vector<py::array_t<int>>
SparseVoxelization(py::array_t<uint64_t, py::array::c_style> keys,
                   py::array_t<int, py::array::c_style> labels,
                   int ignore_label, bool has_label);
#endif
