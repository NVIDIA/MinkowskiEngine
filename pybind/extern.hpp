#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "src/common.hpp"

/*************************************
 * Convolution
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchConvolutionForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchConvolutionBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor kernel, at::Tensor grad_kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchConvolutionForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchConvolutionBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor kernel, at::Tensor grad_kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * Convolution Transpose
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchConvolutionTransposeForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchConvolutionTransposeBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor kernel, at::Tensor grad_kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchConvolutionTransposeForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchConvolutionTransposeBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor kernel, at::Tensor grad_kernel,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * AvgPooling
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif

/*************************************
 * MaxPooling
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchMaxPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * PoolingTranspose
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> tensor_strides, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * GlobalPooling
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif

/*************************************
 * Broadcast
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchBroadcastForwardCPU(int D, at::Tensor in_feat,
                                  at::Tensor in_feat_glob, at::Tensor out_feat,
                                  int op, py::object py_in_coords_key,
                                  py::object py_out_coords_key,
                                  py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchBroadcastBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchBroadcastForwardGPU(int D, at::Tensor in_feat,
                                  at::Tensor in_feat_glob, at::Tensor out_feat,
                                  int op, py::object py_in_coords_key,
                                  py::object py_out_coords_key,
                                  py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchBroadcastBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif

/*************************************
 * Pruning
 *************************************/
template <typename Dtype, typename Itype>
void DimSwitchPruningForwardCPU(int D, at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor use_feat,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPruningBackwardCPU(int D, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
                                 py::object py_in_coords_key,
                                 py::object py_out_coords_key,
                                 py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchPruningForwardGPU(int D, at::Tensor in_feat, at::Tensor out_feat,
                                at::Tensor use_feat,
                                py::object py_in_coords_key,
                                py::object py_out_coords_key,
                                py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPruningBackwardGPU(int D, at::Tensor grad_in_feat,
                                 at::Tensor grad_out_feat,
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

void cuda_thread_exit(void);
#endif
