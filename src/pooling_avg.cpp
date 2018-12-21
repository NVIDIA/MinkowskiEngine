#include "common.hpp"

#include "pooling.hpp"
#ifndef CPU_ONLY
#include "pooling.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void AvgPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor num_nonzero, std::vector<int> pixel_dists,
                          std::vector<int> strides,
                          std::vector<int> kernel_sizes,
                          std::vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      pixel_dists, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  NonzeroAvgPoolingForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), num_nonzero.data<Dtype>(),
      in_feat.size(1), std::get<0>(in_out), std::get<1>(in_out), out_nrows,
      use_avg);
}

template <uint8_t D, typename Dtype, typename Itype>
void AvgPoolingBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  InOutMapKey map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false);

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  NonzeroAvgPoolingBackwardKernelCPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      num_nonzero.data<Dtype>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
      use_avg);
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void AvgPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                          at::Tensor num_nonzero, std::vector<int> pixel_dists,
                          std::vector<int> strides,
                          std::vector<int> kernel_sizes,
                          std::vector<int> dilations, int region_type,
                          at::Tensor offsets, py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      pixel_dists, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  cusparseHandle_t handle =
      THCState_getCurrentSparseHandle(at::globalContext().getTHCState());

  NonzeroAvgPoolingForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(0), out_feat.data<Dtype>(), out_nrows,
      num_nonzero.data<Dtype>(), in_feat.size(1), std::get<0>(in_out),
      std::get<1>(in_out), use_avg, handle, at::cuda::getCurrentCUDAStream());
}

template <uint8_t D, typename Dtype, typename Itype>
void AvgPoolingBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  InOutMapKey map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, false);

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  NonzeroAvgPoolingBackwardKernelGPU<Dtype, Itype>(
      grad_in_feat.data<Dtype>(), in_feat.size(0), grad_out_feat.data<Dtype>(),
      grad_out_feat.size(0), num_nonzero.data<Dtype>(), in_feat.size(1),
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
      use_avg, at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(AvgPoolingForwardCPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, pixel_dists, strides, kernel_sizes, dilations,
                   region_type, offsets, py_in_coords_key, py_out_coords_key,
                   py_coords_manager, use_avg);
}

template void DimSwitchAvgPoolingForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(AvgPoolingBackwardCPU, Dtype, Itype, in_feat, grad_in_feat,
                   grad_out_feat, num_nonzero, pixel_dists, strides,
                   kernel_sizes, dilations, region_type, py_in_coords_key,
                   py_out_coords_key, py_coords_manager, use_avg);
}

template void DimSwitchAvgPoolingBackwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(AvgPoolingForwardGPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, pixel_dists, strides, kernel_sizes, dilations,
                   region_type, offsets, py_in_coords_key, py_out_coords_key,
                   py_coords_manager, use_avg);
}

template void DimSwitchAvgPoolingForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchAvgPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(AvgPoolingBackwardGPU, Dtype, Itype, in_feat, grad_in_feat,
                   grad_out_feat, num_nonzero, pixel_dists, strides,
                   kernel_sizes, dilations, region_type, py_in_coords_key,
                   py_out_coords_key, py_coords_manager, use_avg);
}

template void DimSwitchAvgPoolingBackwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif
