#include "common.hpp"

#include "broadcast.hpp"
#ifndef CPU_ONLY
#include "broadcast.cuh"
#endif

#include <pybind11/pybind11.h>

template <uint8_t D, typename Dtype, typename Itype>
void BroadcastForwardCPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->in_maps.find(map_key) ==
      p_coords_manager->in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  out_feat.resize_as_(in_feat);
  out_feat.zero_();

  BroadcastForwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

template <uint8_t D, typename Dtype, typename Itype>
void BroadcastBackwardCPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->in_maps.find(map_key) ==
      p_coords_manager->in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  BroadcastBackwardKernelCPU<Dtype, Itype>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key]);
}

#ifndef CPU_ONLY
template <uint8_t D, typename Dtype, typename Itype>
void BroadcastForwardGPU(at::Tensor in_feat, at::Tensor in_feat_glob,
                         at::Tensor out_feat, int op,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key,
                         py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();
  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->in_maps.find(map_key) ==
      p_coords_manager->in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  out_feat.resize_as_(in_feat);
  out_feat.zero_();

  cusparseHandle_t handle =
      THCState_getCurrentSparseHandle(at::globalContext().getTHCState());

  BroadcastForwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), in_feat.size(0), in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
      handle, at::cuda::getCurrentCUDAStream());
}

template <uint8_t D, typename Dtype, typename Itype>
void BroadcastBackwardGPU(at::Tensor in_feat, at::Tensor grad_in_feat,
                          at::Tensor in_feat_glob, at::Tensor grad_in_feat_glob,
                          at::Tensor grad_out_feat, int op,
                          py::object py_in_coords_key,
                          py::object py_out_coords_key,
                          py::object py_coords_manager) {
  CoordsManager<D, Itype> *p_coords_manager =
      py_coords_manager.cast<CoordsManager<D, Itype> *>();

  // Both coords must exist
  // Use the global pooling mapping
  InOutMapKey map_key = p_coords_manager->getOriginMapHashKeyCheck(
      py_in_coords_key, py_out_coords_key);

  if (p_coords_manager->in_maps.find(map_key) ==
      p_coords_manager->in_maps.end())
    throw std::invalid_argument(
        Formatter() << "Input Output map not found: "
                    << std::to_string(hash_vec<InOutMapKey>(map_key)));

  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_in_feat_glob.resize_as_(in_feat_glob);
  grad_in_feat_glob.zero_();

  cusparseHandle_t handle =
      THCState_getCurrentSparseHandle(at::globalContext().getTHCState());

  BroadcastBackwardKernelGPU<Dtype, Itype>(
      in_feat.data<Dtype>(), grad_in_feat.data<Dtype>(), in_feat.size(0),
      in_feat_glob.data<Dtype>(), grad_in_feat_glob.data<Dtype>(),
      in_feat_glob.size(0), grad_out_feat.data<Dtype>(), in_feat.size(1), op,
      p_coords_manager->in_maps[map_key], p_coords_manager->out_maps[map_key],
      handle, at::cuda::getCurrentCUDAStream());
}
#endif

template <typename Dtype, typename Itype>
void DimSwitchBroadcastForwardCPU(int D, at::Tensor in_feat,
                                  at::Tensor in_feat_glob, at::Tensor out_feat,
                                  int op, py::object py_in_coords_key,
                                  py::object py_out_coords_key,
                                  py::object py_coords_manager) {
  SWITCH_DIM_TYPES(BroadcastForwardCPU, Dtype, Itype, in_feat, in_feat_glob,
                   out_feat, op, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchBroadcastForwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat,
    int op, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchBroadcastForwardCPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat,
    int op, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchBroadcastBackwardCPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(BroadcastBackwardCPU, Dtype, Itype, in_feat, grad_in_feat,
                   in_feat_glob, grad_in_feat_glob, grad_out_feat, op,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchBroadcastBackwardCPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchBroadcastBackwardCPU<double, int32_t>(
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
                                  py::object py_coords_manager) {
  SWITCH_DIM_TYPES(BroadcastForwardGPU, Dtype, Itype, in_feat, in_feat_glob,
                   out_feat, op, py_in_coords_key, py_out_coords_key,
                   py_coords_manager);
}

template void DimSwitchBroadcastForwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat,
    int op, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchBroadcastForwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor in_feat_glob, at::Tensor out_feat,
    int op, py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template <typename Dtype, typename Itype>
void DimSwitchBroadcastBackwardGPU(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager) {
  SWITCH_DIM_TYPES(BroadcastBackwardGPU, Dtype, Itype, in_feat, grad_in_feat,
                   in_feat_glob, grad_in_feat_glob, grad_out_feat, op,
                   py_in_coords_key, py_out_coords_key, py_coords_manager);
}

template void DimSwitchBroadcastBackwardGPU<float, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);

template void DimSwitchBroadcastBackwardGPU<double, int32_t>(
    int D, at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor in_feat_glob,
    at::Tensor grad_in_feat_glob, at::Tensor grad_out_feat, int op,
    py::object py_in_coords_key, py::object py_out_coords_key,
    py::object py_coords_manager);
#endif
