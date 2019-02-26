#include "common.hpp"

#include "pooling.hpp"
#ifndef CPU_ONLY
#include "pooling.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void GlobalPoolingForwardCPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, int batch_size,
                             bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnOriginInOutPerKernel(
      batch_size, py_in_coords_key, py_out_coords_key);

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
void GlobalPoolingBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

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
void GlobalPoolingForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                             at::Tensor num_nonzero,
                             py::object py_in_coords_key,
                             py::object py_out_coords_key,
                             py::object py_coords_manager, int batch_size,
                             bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  auto in_out = p_coords_manager->setupAndReturnOriginInOutPerKernel(
      batch_size, py_in_coords_key, py_out_coords_key);

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
void GlobalPoolingBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                              at::Tensor grad_out_feat, at::Tensor num_nonzero,
                              py::object py_in_coords_key,
                              py::object py_out_coords_key,
                              py::object py_coords_manager, bool use_avg) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKey(
      py_in_coords_key, py_out_coords_key);

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
void DimSwitchGlobalPoolingForwardCPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg) {
  SWITCH_DIM_TYPES(GlobalPoolingForwardCPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, py_in_coords_key, py_out_coords_key,
                   py_coords_manager, batch_size, use_avg);
}

template void DimSwitchGlobalPoolingForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template void DimSwitchGlobalPoolingForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(GlobalPoolingBackwardCPU, Dtype, Itype, in_feat,
                   grad_in_feat, grad_out_feat, num_nonzero, py_in_coords_key,
                   py_out_coords_key, py_coords_manager, use_avg);
}

template void DimSwitchGlobalPoolingBackwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void DimSwitchGlobalPoolingBackwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

#ifndef CPU_ONLY
template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingForwardGPU(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg) {
  SWITCH_DIM_TYPES(GlobalPoolingForwardGPU, Dtype, Itype, in_feat, out_feat,
                   num_nonzero, py_in_coords_key, py_out_coords_key,
                   py_coords_manager, batch_size, use_avg);
}

template void DimSwitchGlobalPoolingForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template void DimSwitchGlobalPoolingForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, int batch_size, bool use_avg);

template <typename Dtype, typename Itype>
void DimSwitchGlobalPoolingBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg) {
  SWITCH_DIM_TYPES(GlobalPoolingBackwardGPU, Dtype, Itype, in_feat,
                   grad_in_feat, grad_out_feat, num_nonzero, py_in_coords_key,
                   py_out_coords_key, py_coords_manager, use_avg);
}

template void DimSwitchGlobalPoolingBackwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);

template void DimSwitchGlobalPoolingBackwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat,
    at::Tensor grad_out_feat, at::Tensor num_nonzero,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager, bool use_avg);
#endif
