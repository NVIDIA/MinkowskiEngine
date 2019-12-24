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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "src/common.hpp"

/*************************************
 * Convolution
 *************************************/
template <typename Dtype>
void ConvolutionForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, std::vector<int> tensor_strides,
                           std::vector<int> strides,
                           std::vector<int> kernel_sizes,
                           std::vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype>
void ConvolutionBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, std::vector<int> tensor_strides,
                           std::vector<int> strides,
                           std::vector<int> kernel_sizes,
                           std::vector<int> dilations, int region_type,
                           at::Tensor offsets, py::object py_in_coords_key,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype>
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
template <typename Dtype>
void ConvolutionTransposeForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager,
    bool generate_new_coords);

template <typename Dtype>
void ConvolutionTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void ConvolutionTransposeForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager,
    bool generate_new_coords);

template <typename Dtype>
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
template <typename Dtype>
void AvgPoolingForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype>
void AvgPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype>
void AvgPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype>
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
template <typename Dtype>
void MaxPoolingForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype>
void MaxPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void MaxPoolingForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype>
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
template <typename Dtype>
void PoolingTransposeForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype>
void PoolingTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> tensor_strides,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void PoolingTransposeForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype>
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
template <typename Dtype>
vector<at::Tensor> GlobalPoolingForwardCPU(at::Tensor in_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager,
                                           bool use_avg, int pooling_mode);

template <typename Dtype>
at::Tensor
GlobalPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_out_feat,
                         at::Tensor num_nonzero, py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype>
vector<at::Tensor> GlobalPoolingForwardGPU(at::Tensor in_feat,
                                           py::object py_in_coords_key,
                                           py::object py_out_coords_key,
                                           py::object py_coords_manager,
                                           bool use_avg, int pooling_mode);

template <typename Dtype>
at::Tensor
GlobalPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_out_feat,
                         at::Tensor num_nonzero, py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager, bool use_avg);
#endif

/*************************************
 * GlobalMaxPooling
 *************************************/
template <typename Dtype>
void GlobalMaxPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager);

template <typename Dtype>
void GlobalMaxPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 at::Tensor num_nonzero,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void GlobalMaxPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor num_nonzero,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager);

template <typename Dtype>
void GlobalMaxPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 at::Tensor num_nonzero,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager);
#endif

/*************************************
 * Broadcast
 *************************************/
template <typename Dtype>
at::Tensor BroadcastForwardCPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                               int op, py::object py_in_coords_key,
                               py::object py_out_coords_key,
                               py::object py_coords_manager);

template <typename Dtype>
void BroadcastBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
at::Tensor BroadcastForwardGPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                               int op, py::object py_in_coords_key,
                               py::object py_out_coords_key,
                               py::object py_coords_manager);

template <typename Dtype>
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
template <typename Dtype>
void PruningForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                       at::Tensor use_feat, py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager);

template <typename Dtype>
void PruningBackwardCPU(at::Tensor grad_in_feat, at::Tensor grad_out_feat,
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
void PruningForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                       at::Tensor use_feat, py::object py_in_coords_key,
                       py::object py_out_coords_key,
                       py::object py_coords_manager);

template <typename Dtype>
void PruningBackwardGPU(at::Tensor grad_in_feat, at::Tensor grad_out_feat,
                        py::object py_in_coords_key,
                        py::object py_out_coords_key,
                        py::object py_coords_manager);
#endif

/*************************************
 * Union
 *************************************/
template <typename Dtype>
at::Tensor UnionForwardCPU(vector<at::Tensor> in_feats,
                           vector<py::object> py_in_coords_keys,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype>
vector<at::Tensor>
UnionBackwardCPU(at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
                 py::object py_out_coords_key, py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype>
at::Tensor UnionForwardGPU(vector<at::Tensor> in_feat,
                           vector<py::object> py_in_coords_keys,
                           py::object py_out_coords_key,
                           py::object py_coords_manager);

template <typename Dtype>
vector<at::Tensor>
UnionBackwardGPU(at::Tensor grad_out_feat, vector<py::object> py_in_coords_keys,
                 py::object py_out_coords_key, py::object py_coords_manager);
#endif
/*************************************
 * Quantization
 *************************************/
vector<int>
quantize_np(py::array_t<int, py::array::c_style | py::array::forcecast> coords);

vector<py::array> quantize_label_np(
    py::array_t<int, py::array::c_style | py::array::forcecast> coords,
    py::array_t<int, py::array::c_style | py::array::forcecast> labels,
    int invalid_label);

at::Tensor quantize_th(at::Tensor coords);

vector<at::Tensor> quantize_label_th(at::Tensor coords, at::Tensor labels,
                                     int invalid_label);
