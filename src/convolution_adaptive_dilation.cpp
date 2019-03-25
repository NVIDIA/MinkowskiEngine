#include "common.hpp"

#include "convolution.hpp"
#ifndef CPU_ONLY
#include "convolution.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void ConvolutionAdaptiveDilationForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernelAdaptiveDilation(
      dilations, pixel_dists, strides, kernel_sizes, dilations_key, region_type,
      offsets, py_in_coords_key, py_out_coords_key, false);

  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument(
        Formatter() << "Input feature size and kernel size mismatch");
  }

  int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();

  ConvolutionForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(1), out_feat.data<Dtype>(),
      out_feat.size(1), kernel.data<Dtype>(), std::get<0>(in_out),
      std::get<1>(in_out));
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void ConvolutionAdaptiveDilationForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernelAdaptiveDilation(
      dilations, pixel_dists, strides, kernel_sizes, dilations_key, region_type,
      offsets, py_in_coords_key, py_out_coords_key, false);

  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument(
        Formatter() << "Input feature size and kernel size mismatch");
  }

  int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();

  cublasHandle_t handle =
      THCState_getCurrentBlasHandle(at::globalContext().getTHCState());

  ConvolutionForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(1), out_feat.data<Dtype>(),
      out_feat.size(1), kernel.data<Dtype>(), std::get<0>(in_out),
      std::get<1>(in_out), out_nrows, handle, at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchConvolutionAdaptiveDilationForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(ConvolutionAdaptiveDilationForwardCPU, Dtype, Itype, in_feat,
                   out_feat, kernel, dilations, pixel_dists, strides,
                   kernel_sizes, dilations_key, region_type, offsets,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchConvolutionAdaptiveDilationForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchConvolutionAdaptiveDilationForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchConvolutionAdaptiveDilationForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(ConvolutionAdaptiveDilationForwardGPU, Dtype, Itype, in_feat,
                   out_feat, kernel, dilations, pixel_dists, strides,
                   kernel_sizes, dilations_key, region_type, offsets,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchConvolutionAdaptiveDilationForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchConvolutionAdaptiveDilationForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor kernel,
    at::Tensor dilations, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations_key, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
