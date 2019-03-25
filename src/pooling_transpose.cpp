#include "common.hpp"

#include "pooling_avg.hpp"
#ifndef CPU_ONLY
#include "pooling_avg.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void PoolingTransposeForwardCPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      pixel_dists, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, true);

  const int out_nrows = p_coords_manager->getCoordsSize(py_out_coords_key);
  out_feat.resize_({out_nrows, in_feat.size(1)});
  out_feat.zero_();
  num_nonzero.resize_({out_nrows});
  num_nonzero.zero_();

  NonzeroAvgPoolingForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), out_feat.data<Dtype>(), num_nonzero.data<Dtype>(),
      in_feat.size(1), std::get<0>(in_out), std::get<1>(in_out), out_nrows,
      false);
}

template <uint8_t D, typename Dtype, typename Itype>
void PoolingTransposeBackwardCPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  bool reverse_map = false;
  InOutMapKey rev_map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_out_coords_key, py_in_coords_key, false);
  InOutMapKey map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, true);

  // Check if the reverse map exists first
  if (p_coords_manager->in_maps.find(rev_map_key) !=
      p_coords_manager->in_maps.end())
    reverse_map = true;

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  if (!reverse_map)
    NonzeroAvgPoolingBackwardKernelCPU<Dtype, Itype>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
        false);
  else
    NonzeroAvgPoolingBackwardKernelCPU<Dtype, Itype>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_manager->out_maps[rev_map_key],
        p_coords_manager->in_maps[rev_map_key], false);
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void PoolingTransposeForwardGPU(
    at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnInOutPerKernel(
      pixel_dists, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, true);

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
      std::get<1>(in_out), false, handle, at::cuda::getCurrentCUDAStream());
}

template <uint8_t D, typename Dtype, typename Itype>
void PoolingTransposeBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor num_nonzero, std::vector<int> pixel_dists,
    std::vector<int> strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  bool reverse_map = false;
  InOutMapKey rev_map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_out_coords_key, py_in_coords_key, false);
  InOutMapKey map_key = p_coords_manager->getMapHashKey(
      pixel_dists, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, true);

  // Check if the reverse map exists first
  if (p_coords_manager->in_maps.find(rev_map_key) !=
      p_coords_manager->in_maps.end())
    reverse_map = true;

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();

  if (!reverse_map)
    NonzeroAvgPoolingBackwardKernelGPU<Dtype, Itype>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), grad_out_feat.size(0),
        num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
        false, at::cuda::getCurrentCUDAStream());
  else
    NonzeroAvgPoolingBackwardKernelGPU<Dtype, Itype>(
        grad_in_feat.data<Dtype>(), in_feat.size(0),
        grad_out_feat.data<Dtype>(), grad_out_feat.size(0),
        num_nonzero.data<Dtype>(), in_feat.size(1),
        p_coords_manager->out_maps[rev_map_key],
        p_coords_manager->in_maps[rev_map_key], false,
        at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PoolingTransposeForwardCPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, pixel_dists, strides, kernel_sizes, dilations,
                   region_type, offsets, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchPoolingTransposeForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void DimSwitchPoolingTransposeForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PoolingTransposeBackwardCPU, Dtype, Itype, in_feat,
                   grad_in_feat, grad_out_feat, num_nonzero, pixel_dists,
                   strides, kernel_sizes, dilations, region_type,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchPoolingTransposeBackwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPoolingTransposeBackwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PoolingTransposeForwardGPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, pixel_dists, strides, kernel_sizes, dilations,
                   region_type, offsets, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchPoolingTransposeForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template void DimSwitchPoolingTransposeForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchPoolingTransposeBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(PoolingTransposeBackwardGPU, Dtype, Itype, in_feat,
                   grad_in_feat, grad_out_feat, num_nonzero, pixel_dists,
                   strides, kernel_sizes, dilations, region_type,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchPoolingTransposeBackwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchPoolingTransposeBackwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
