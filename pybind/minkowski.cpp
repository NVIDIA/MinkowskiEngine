#include <string>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "extern.hpp"
#include "src/common.hpp"

namespace py = pybind11;

template <typename Dtype, typename Itype>
void instantiate_func(py::module &m, const std::string &dtypestr,
                      const std::string &itypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &DimSwitchConvolutionForwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &DimSwitchConvolutionBackwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &DimSwitchConvolutionForwardGPU<Dtype, Itype>);
  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &DimSwitchConvolutionBackwardGPU<Dtype, Itype>);

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &DimSwitchConvolutionTransposeForwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &DimSwitchConvolutionTransposeBackwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &DimSwitchConvolutionTransposeForwardGPU<Dtype, Itype>);
  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &DimSwitchConvolutionTransposeBackwardGPU<Dtype, Itype>);

  m.def((std::string("ConvolutionAdaptiveDilationForwardCPU") + dtypestr).c_str(),
        &DimSwitchConvolutionAdaptiveDilationForwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionAdaptiveDilationForwardGPU") + dtypestr).c_str(),
        &DimSwitchConvolutionAdaptiveDilationForwardGPU<Dtype, Itype>);

  m.def((std::string("AvgPoolingForwardCPU") + dtypestr).c_str(),
        &DimSwitchAvgPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("AvgPoolingBackwardCPU") + dtypestr).c_str(),
        &DimSwitchAvgPoolingBackwardCPU<Dtype, Itype>);
  m.def((std::string("AvgPoolingForwardGPU") + dtypestr).c_str(),
        &DimSwitchAvgPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("AvgPoolingBackwardGPU") + dtypestr).c_str(),
        &DimSwitchAvgPoolingBackwardGPU<Dtype, Itype>);

  m.def((std::string("MaxPoolingForwardCPU") + dtypestr).c_str(),
        &DimSwitchMaxPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("MaxPoolingBackwardCPU") + dtypestr).c_str(),
        &DimSwitchMaxPoolingBackwardCPU<Dtype, Itype>);
  m.def((std::string("MaxPoolingForwardGPU") + dtypestr).c_str(),
        &DimSwitchMaxPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("MaxPoolingBackwardGPU") + dtypestr).c_str(),
        &DimSwitchMaxPoolingBackwardGPU<Dtype, Itype>);

  m.def((std::string("PoolingTransposeForwardCPU") + dtypestr).c_str(),
        &DimSwitchPoolingTransposeForwardCPU<Dtype, Itype>);
  m.def((std::string("PoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &DimSwitchPoolingTransposeBackwardCPU<Dtype, Itype>);
  m.def((std::string("PoolingTransposeForwardGPU") + dtypestr).c_str(),
        &DimSwitchPoolingTransposeForwardGPU<Dtype, Itype>);
  m.def((std::string("PoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &DimSwitchPoolingTransposeBackwardGPU<Dtype, Itype>);

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &DimSwitchGlobalPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &DimSwitchGlobalPoolingBackwardCPU<Dtype, Itype>);
  m.def((std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
        &DimSwitchGlobalPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
        &DimSwitchGlobalPoolingBackwardGPU<Dtype, Itype>);

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &DimSwitchBroadcastForwardCPU<Dtype, Itype>);
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &DimSwitchBroadcastBackwardCPU<Dtype, Itype>);
  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &DimSwitchBroadcastForwardGPU<Dtype, Itype>);
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &DimSwitchBroadcastBackwardGPU<Dtype, Itype>);

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &DimSwitchPruningForwardCPU<Dtype, Itype>);
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &DimSwitchPruningBackwardCPU<Dtype, Itype>);
  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &DimSwitchPruningForwardGPU<Dtype, Itype>);
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &DimSwitchPruningBackwardGPU<Dtype, Itype>);
}

template <uint8_t D, typename Itype>
void instantiate_dim_itype(py::module &m, const std::string &dim,
                           const std::string &itypestr) {
  std::string coords_name = std::string("PyCoordsManager") + dim + itypestr;
  py::class_<CoordsManager<D, Itype>>(m, coords_name.c_str())
      .def(py::init<>())
      .def(py::init<int>())
      .def("existsCoordsKey", (bool (CoordsManager<D, Itype>::*)(py::object)) &
                                  CoordsManager<D, Itype>::existsCoordsKey)
      .def("getCoordsKey", &CoordsManager<D, Itype>::getCoordsKey)
      .def("getKernelMap", &CoordsManager<D, Itype>::getKernelMap)
      .def("getCoordsSize", (int (CoordsManager<D, Itype>::*)(py::object)) &
                                CoordsManager<D, Itype>::getCoordsSize)
      .def("getCoords", &CoordsManager<D, Itype>::getCoords)
      .def("initializeCoords", (uint64_t(CoordsManager<D, Itype>::*)(
                                   at::Tensor, py::object, bool)) &
                                   CoordsManager<D, Itype>::initializeCoords)
      .def("getCoordsMapping", &CoordsManager<D, Itype>::getCoordsMapping)
      .def("__repr__",
           [](const CoordsManager<D, Itype> &a) { return a.toString(); });
}

template <uint8_t D>
void instantiate_dim(py::module &m, const std::string &dim) {
  std::string name = std::string("PyCoordsKey") + dim;
  py::class_<PyCoordsKey<D>>(m, name.c_str())
      .def(py::init<>())
      .def("copy", &PyCoordsKey<D>::copy)
      .def("setKey", &PyCoordsKey<D>::setKey)
      .def("getKey", &PyCoordsKey<D>::getKey)
      .def("setPixelDist", &PyCoordsKey<D>::setPixelDist)
      .def("getPixelDist", &PyCoordsKey<D>::getPixelDist)
      .def("__repr__", [](const PyCoordsKey<D> &a) { return a.toString(); });

  // Instantiate Itypes
  instantiate_dim_itype<D, int32_t>(m, dim, std::string("int32"));
  instantiate_func<float, int32_t>(m, std::string("f"),
                                   std::string("int32"));
  instantiate_func<double, int32_t>(m, std::string("d"),
                                   std::string("int32"));
}

void bind_native(py::module &m) {
  m.def("SparseVoxelization", &SparseVoxelization);
  m.def("CUDAThreadExit", &cuda_thread_exit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_native(m);
  instantiate_dim<1>(m, std::to_string(1));
  instantiate_dim<2>(m, std::to_string(2));
  instantiate_dim<3>(m, std::to_string(3));
  instantiate_dim<4>(m, std::to_string(4));
  instantiate_dim<5>(m, std::to_string(5));
  instantiate_dim<6>(m, std::to_string(6));
  instantiate_dim<7>(m, std::to_string(7));
}
