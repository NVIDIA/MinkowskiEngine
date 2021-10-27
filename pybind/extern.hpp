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

#ifndef CPU_ONLY
#include "allocators.cuh"
#include "coordinate_map_gpu.cuh"
#include <cuda.h>
#endif

#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace minkowski {

/*************************************
 * Convolution
 *************************************/
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
                      cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardCPU(at::Tensor const &in_feat,                         //
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
                       cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor ConvolutionForwardGPU(
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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> ConvolutionBackwardGPU(
    at::Tensor const &in_feat,                         //
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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Convolution Transpose
 *************************************/
template <typename coordinate_type>
at::Tensor ConvolutionTransposeForwardCPU(
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
    cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor> ConvolutionTransposeBackwardCPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offsets,                         //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor ConvolutionTransposeForwardGPU(
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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> ConvolutionTransposeBackwardGPU(
    at::Tensor const &in_feat,                         //
    at::Tensor const &grad_out_feat,                   //
    at::Tensor const &kernel,                          //
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    ConvolutionMode::Type const convolution_mode,      //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Local Pooling
 *************************************/
template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
LocalPoolingForwardCPU(at::Tensor const &in_feat,
                       default_types::stride_type const &kernel_size,     //
                       default_types::stride_type const &kernel_stride,   //
                       default_types::stride_type const &kernel_dilation, //
                       RegionType::Type const region_type,                //
                       at::Tensor const &offset,                          //
                       PoolingMode::Type pooling_mode,                    //
                       CoordinateMapKey *p_in_map_key,                    //
                       CoordinateMapKey *p_out_map_key,                   //
                       cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
LocalPoolingBackwardCPU(at::Tensor const &in_feat,                         //
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
                        cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Local Pooling Transpose
 *************************************/
template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor> LocalPoolingTransposeForwardCPU(
    at::Tensor const &in_feat,
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    bool generate_new_coordinates,                     //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor LocalPoolingTransposeBackwardCPU(
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
    cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> LocalPoolingTransposeForwardGPU(
    at::Tensor const &in_feat,
    default_types::stride_type const &kernel_size,     //
    default_types::stride_type const &kernel_stride,   //
    default_types::stride_type const &kernel_dilation, //
    RegionType::Type const region_type,                //
    at::Tensor const &offset,                          //
    bool generate_new_coordinates,                     //
    PoolingMode::Type pooling_mode,                    //
    CoordinateMapKey *p_in_map_key,                    //
    CoordinateMapKey *p_out_map_key,                   //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor LocalPoolingTransposeBackwardGPU(
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
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Global Pooling
 *************************************/
template <typename coordinate_type>
std::tuple<at::Tensor, at::Tensor>
GlobalPoolingForwardCPU(at::Tensor const &in_feat,
                        PoolingMode::Type const pooling_mode, //
                        CoordinateMapKey *p_in_map_key,       //
                        CoordinateMapKey *p_out_map_key,      //
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
GlobalPoolingBackwardCPU(at::Tensor const &in_feat, at::Tensor &grad_out_feat,
                         at::Tensor const &num_nonzero,
                         PoolingMode::Type const pooling_mode, //
                         CoordinateMapKey *p_in_map_key,       //
                         CoordinateMapKey *p_out_map_key,      //
                         cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::tuple<at::Tensor, at::Tensor> GlobalPoolingForwardGPU(
    at::Tensor const &in_feat,
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor GlobalPoolingBackwardGPU(
    at::Tensor const &in_feat,            //
    at::Tensor &grad_out_feat,            //
    at::Tensor const &num_nonzero,        //
    PoolingMode::Type const pooling_mode, //
    CoordinateMapKey *p_in_map_key,       //
    CoordinateMapKey *p_out_map_key,      //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Broadcast
 *************************************/
template <typename coordinate_type>
at::Tensor
BroadcastForwardCPU(at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
                    BroadcastMode::Type const broadcast_mode,
                    CoordinateMapKey *p_in_map_key,   //
                    CoordinateMapKey *p_glob_map_key, //
                    cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
BroadcastBackwardCPU(at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
                     at::Tensor const &grad_out_feat,
                     BroadcastMode::Type const op,
                     CoordinateMapKey *p_in_map_key,   //
                     CoordinateMapKey *p_glob_map_key, //
                     cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor BroadcastForwardGPU(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    BroadcastMode::Type const broadcast_mode,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor> BroadcastBackwardGPU(
    at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
    at::Tensor const &grad_out_feat, BroadcastMode::Type const op,
    CoordinateMapKey *p_in_map_key,   //
    CoordinateMapKey *p_glob_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Pruning
 *************************************/
template <typename coordinate_type>
at::Tensor
PruningForwardCPU(at::Tensor const &in_feat, // CPU feat
                  at::Tensor const &keep,    // uint8 / bool / byte CPU data
                  CoordinateMapKey *p_in_map_key,  //
                  CoordinateMapKey *p_out_map_key, //
                  cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor PruningBackwardCPU(at::Tensor &grad_out_feat,       // CPU out feat
                              CoordinateMapKey *p_in_map_key,  //
                              CoordinateMapKey *p_out_map_key, //
                              cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor PruningForwardGPU(
    at::Tensor const &in_feat,       // GPU feat
    at::Tensor const &keep,          // uint8 CPU data
    CoordinateMapKey *p_in_map_key,  //
    CoordinateMapKey *p_out_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor PruningBackwardGPU(
    at::Tensor &grad_out_feat,       // GPU out feat
    CoordinateMapKey *p_in_map_key,  //
    CoordinateMapKey *p_out_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Interpolation
 *************************************/
template <typename coordinate_type>
std::vector<at::Tensor>
InterpolationForwardCPU(at::Tensor const &in_feat,      //
                        at::Tensor const &tfield,       //
                        CoordinateMapKey *p_in_map_key, //
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
InterpolationBackwardCPU(at::Tensor &grad_out_feat,      //
                         at::Tensor const &in_map,       //
                         at::Tensor const &out_map,      //
                         at::Tensor const &weight,       //
                         CoordinateMapKey *p_in_map_key, //
                         cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
std::vector<at::Tensor> InterpolationForwardGPU(
    at::Tensor const &in_feat,      //
    at::Tensor const &tfield,       //
    CoordinateMapKey *p_in_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
at::Tensor InterpolationBackwardGPU(
    at::Tensor &grad_out_feat,      //
    at::Tensor const &in_maps,      //
    at::Tensor const &out_maps,     //
    at::Tensor const &weights,      //
    CoordinateMapKey *p_in_map_key, //
    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

/*************************************
 * Quantization
 *************************************/
std::vector<py::array> quantize_np(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> coords);

std::vector<at::Tensor> quantize_th(at::Tensor &coords);

std::vector<py::array> quantize_label_np(
    py::array_t<int, py::array::c_style | py::array::forcecast> coords,
    py::array_t<int, py::array::c_style | py::array::forcecast> labels,
    int invalid_label);

std::vector<at::Tensor> quantize_label_th(at::Tensor coords, at::Tensor labels,
                                          int invalid_label);

std::pair<torch::Tensor, torch::Tensor>
max_pool_fw(torch::Tensor const &in_map,  //
            torch::Tensor const &out_map, //
            torch::Tensor const &in_feat, //
            int const out_nrows, bool const is_sorted);

torch::Tensor max_pool_bw(torch::Tensor const &grad_out_feat, //
                          torch::Tensor const &mask_index,    //
                          int const in_nrows);

#ifndef CPU_ONLY
template <typename th_int_type>
torch::Tensor coo_spmm(torch::Tensor const &rows, torch::Tensor const &cols,
                       torch::Tensor const &vals, int64_t const dim_i,
                       int64_t const dim_j, torch::Tensor const &mat2,
                       int64_t const spmm_algorithm_id, bool const is_sorted);

template <typename th_int_type>
std::vector<torch::Tensor> // output, sorted rows, sorted cols, sorted vals.
coo_spmm_average(torch::Tensor const &rows, torch::Tensor const &cols,
                 int64_t const dim_i, int64_t const dim_j,
                 torch::Tensor const &mat2, int64_t const spmm_algorithm_id);

std::pair<size_t, size_t> get_memory_info();
#endif

} // end namespace minkowski

namespace py = pybind11;

template <typename coordinate_type>
void instantiate_cpu_func(py::module &m, const std::string &dtypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &minkowski::ConvolutionForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &minkowski::ConvolutionBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &minkowski::ConvolutionTransposeForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &minkowski::ConvolutionTransposeBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("LocalPoolingForwardCPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("LocalPoolingBackwardCPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("LocalPoolingTransposeForwardCPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingTransposeForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("LocalPoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingTransposeBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &minkowski::GlobalPoolingForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &minkowski::GlobalPoolingBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &minkowski::PruningForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &minkowski::PruningBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &minkowski::BroadcastForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &minkowski::BroadcastBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("InterpolationForwardCPU") + dtypestr).c_str(),
        &minkowski::InterpolationForwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("InterpolationBackwardCPU") + dtypestr).c_str(),
        &minkowski::InterpolationBackwardCPU<coordinate_type>,
        py::call_guard<py::gil_scoped_release>());
}

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
void instantiate_gpu_func(py::module &m, const std::string &dtypestr) {
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &minkowski::ConvolutionForwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &minkowski::ConvolutionBackwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &minkowski::ConvolutionTransposeForwardGPU<coordinate_type,
                                                   TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &minkowski::ConvolutionTransposeBackwardGPU<coordinate_type,
                                                    TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("LocalPoolingForwardGPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingForwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());
  m.def(
      (std::string("LocalPoolingBackwardGPU") + dtypestr).c_str(),
      &minkowski::LocalPoolingBackwardGPU<coordinate_type, TemplatedAllocator>,
      py::call_guard<py::gil_scoped_release>());

  m.def((std::string("LocalPoolingTransposeForwardGPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingTransposeForwardGPU<coordinate_type,
                                                    TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("LocalPoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &minkowski::LocalPoolingTransposeBackwardGPU<coordinate_type,
                                                     TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      (std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
      &minkowski::GlobalPoolingForwardGPU<coordinate_type, TemplatedAllocator>,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      (std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
      &minkowski::GlobalPoolingBackwardGPU<coordinate_type, TemplatedAllocator>,
      py::call_guard<py::gil_scoped_release>());

  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &minkowski::PruningForwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &minkowski::PruningBackwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &minkowski::BroadcastForwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &minkowski::BroadcastBackwardGPU<coordinate_type, TemplatedAllocator>,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      (std::string("InterpolationForwardGPU") + dtypestr).c_str(),
      &minkowski::InterpolationForwardGPU<coordinate_type, TemplatedAllocator>,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      (std::string("InterpolationBackwardGPU") + dtypestr).c_str(),
      &minkowski::InterpolationBackwardGPU<coordinate_type, TemplatedAllocator>,
      py::call_guard<py::gil_scoped_release>());
}
#endif

void non_templated_cpu_func(py::module &m) {
  m.def("quantize_np", &minkowski::quantize_np);
  m.def("quantize_th", &minkowski::quantize_th);
  m.def("quantize_label_np", &minkowski::quantize_label_np);
  m.def("quantize_label_th", &minkowski::quantize_label_th);
  m.def("direct_max_pool_fw", &minkowski::max_pool_fw,
        py::call_guard<py::gil_scoped_release>());
  m.def("direct_max_pool_bw", &minkowski::max_pool_bw,
        py::call_guard<py::gil_scoped_release>());
}

#ifndef CPU_ONLY
void non_templated_gpu_func(py::module &m) {
  m.def("coo_spmm_int32", &minkowski::coo_spmm<int32_t>,
        py::call_guard<py::gil_scoped_release>());
  m.def("coo_spmm_average_int32", &minkowski::coo_spmm_average<int32_t>,
        py::call_guard<py::gil_scoped_release>());
}
#endif

void initialize_non_templated_classes(py::module &m) {
  // Enums
  py::enum_<minkowski::GPUMemoryAllocatorBackend::Type>(
      m, "GPUMemoryAllocatorType")
      .value("PYTORCH", minkowski::GPUMemoryAllocatorBackend::Type::PYTORCH)
      .value("CUDA", minkowski::GPUMemoryAllocatorBackend::Type::CUDA)
      .export_values();

  py::enum_<minkowski::CUDAKernelMapMode::Mode>(m, "CUDAKernelMapMode")
      .value("MEMORY_EFFICIENT",
             minkowski::CUDAKernelMapMode::Mode::MEMORY_EFFICIENT)
      .value("SPEED_OPTIMIZED",
             minkowski::CUDAKernelMapMode::Mode::SPEED_OPTIMIZED)
      .export_values();

  py::enum_<minkowski::MinkowskiAlgorithm::Mode>(m, "MinkowskiAlgorithm")
      .value("DEFAULT", minkowski::MinkowskiAlgorithm::Mode::DEFAULT)
      .value("MEMORY_EFFICIENT",
             minkowski::MinkowskiAlgorithm::Mode::MEMORY_EFFICIENT)
      .value("SPEED_OPTIMIZED",
             minkowski::MinkowskiAlgorithm::Mode::SPEED_OPTIMIZED)
      .export_values();

  py::enum_<minkowski::CoordinateMapBackend::Type>(m, "CoordinateMapType")
      .value("CPU", minkowski::CoordinateMapBackend::Type::CPU)
      .value("CUDA", minkowski::CoordinateMapBackend::Type::CUDA)
      .export_values();

  py::enum_<minkowski::RegionType::Type>(m, "RegionType")
      .value("HYPER_CUBE", minkowski::RegionType::Type::HYPER_CUBE)
      .value("HYPER_CROSS", minkowski::RegionType::Type::HYPER_CROSS)
      .value("CUSTOM", minkowski::RegionType::Type::CUSTOM)
      .export_values();

  py::enum_<minkowski::PoolingMode::Type>(m, "PoolingMode")
      .value("LOCAL_SUM_POOLING",
             minkowski::PoolingMode::Type::LOCAL_SUM_POOLING)
      .value("LOCAL_AVG_POOLING",
             minkowski::PoolingMode::Type::LOCAL_AVG_POOLING)
      .value("LOCAL_MAX_POOLING",
             minkowski::PoolingMode::Type::LOCAL_MAX_POOLING)
      .value("GLOBAL_SUM_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_DEFAULT)
      .value("GLOBAL_AVG_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_DEFAULT)
      .value("GLOBAL_MAX_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_DEFAULT)
      .value("GLOBAL_SUM_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_KERNEL)
      .value("GLOBAL_AVG_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_KERNEL)
      .value("GLOBAL_MAX_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_KERNEL)
      .value("GLOBAL_SUM_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_AVG_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_MAX_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_PYTORCH_INDEX)
      .export_values();

  py::enum_<minkowski::BroadcastMode::Type>(m, "BroadcastMode")
      .value("ELEMENTWISE_ADDITON",
             minkowski::BroadcastMode::Type::ELEMENTWISE_ADDITON)
      .value("ELEMENTWISE_MULTIPLICATION",
             minkowski::BroadcastMode::Type::ELEMENTWISE_MULTIPLICATION)
      .export_values();

  py::enum_<minkowski::ConvolutionMode::Type>(m, "ConvolutionMode")
      .value("DEFAULT", minkowski::ConvolutionMode::Type::DEFAULT)
      .value("DIRECT_GEMM", minkowski::ConvolutionMode::Type::DIRECT_GEMM)
      .value("COPY_GEMM", minkowski::ConvolutionMode::Type::COPY_GEMM)
      .export_values();

  // Classes
  py::class_<minkowski::CoordinateMapKey>(m, "CoordinateMapKey")
      .def(py::init<minkowski::default_types::size_type>())
      .def(py::init<minkowski::default_types::stride_type, std::string>())
      .def("__repr__", &minkowski::CoordinateMapKey::to_string)
      .def("__hash__", &minkowski::CoordinateMapKey::hash)
      .def("is_key_set", &minkowski::CoordinateMapKey::is_key_set)
      .def("get_coordinate_size",
           &minkowski::CoordinateMapKey::get_coordinate_size)
      .def("get_key", &minkowski::CoordinateMapKey::get_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::default_types::stride_type, std::string)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::coordinate_map_key_type const &)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("get_tensor_stride", &minkowski::CoordinateMapKey::get_tensor_stride)
      .def("__eq__", [](const minkowski::CoordinateMapKey &self, const minkowski::CoordinateMapKey &other)
                     {
                       return self == other;
                     });
      //.def(py::self == py::self);
}

template <typename manager_type>
void instantiate_manager(py::module &m, const std::string &dtypestr) {
  py::class_<manager_type>(
      m, (std::string("CoordinateMapManager") + dtypestr).c_str())
      .def(py::init<>())
      .def(py::init<minkowski::MinkowskiAlgorithm::Mode,
                    minkowski::default_types::size_type>())
      .def("__repr__",
           py::overload_cast<>(&manager_type::to_string, py::const_))
      .def("print_coordinate_map",
           py::overload_cast<minkowski::CoordinateMapKey const *>(
               &manager_type::to_string, py::const_))
      .def("insert_and_map", &manager_type::insert_and_map)
      .def("insert_field", &manager_type::insert_field)
      .def("field_to_sparse_map", &manager_type::field_to_sparse_map)
      .def("field_to_sparse_insert_and_map",
           &manager_type::field_to_sparse_insert_and_map)
      .def("exists_field_to_sparse",
           py::overload_cast<minkowski::CoordinateMapKey const *,
                             minkowski::CoordinateMapKey const *>(
               &manager_type::exists_field_to_sparse, py::const_))
      .def("get_field_to_sparse_map", &manager_type::get_field_to_sparse_map)
      .def("stride", &manager_type::py_stride)
      .def("origin", &manager_type::py_origin)
      .def("origin_field", &manager_type::py_origin_field)
      .def("get_coordinates", &manager_type::get_coordinates)
      .def("get_coordinate_field", &manager_type::get_coordinate_field)
      .def("get_coordinate_map_keys", &manager_type::get_coordinate_map_keys)
      .def("field_to_sparse_keys", &manager_type::field_to_sparse_keys)
      .def("size", py::overload_cast<minkowski::CoordinateMapKey const *>(
                       &manager_type::size, py::const_))
      .def("get_random_string_id", &manager_type::get_random_string_id)
      .def("origin_map_size", &manager_type::origin_map_size)
      .def("origin_map", &manager_type::origin_map_th)
      .def("origin_field_map", &manager_type::origin_field_map_th)
      .def("union_map", &manager_type::union_map_th)
      .def("stride_map", &manager_type::stride_map_th)
      .def("kernel_map", &manager_type::kernel_map_th)
      .def("interpolation_map_weight", &manager_type::interpolation_map_weight);
}

bool is_cuda_available() {
#ifndef CPU_ONLY
  return true;
#else
  return false;
#endif
}

int cuda_version() {
#if defined(CUDA_VERSION)
  return CUDA_VERSION;
#else
  return -1;
#endif
}

int cudart_version() {
#if defined(CUDART_VERSION)
  return CUDART_VERSION;
#else
  return -1;
#endif
}

std::pair<size_t, size_t> get_gpu_memory_info() {
#ifndef CPU_ONLY
  return minkowski::get_memory_info();
#else
  return std::make_pair(0, 0);
#endif
}
