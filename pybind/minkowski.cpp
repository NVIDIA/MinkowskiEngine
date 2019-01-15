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
  m.def("ConvolutionForwardCPU", &DimSwitchConvolutionForwardCPU<Dtype, Itype>);
  m.def("ConvolutionBackwardCPU",
        &DimSwitchConvolutionBackwardCPU<Dtype, Itype>);
  m.def("ConvolutionForwardGPU", &DimSwitchConvolutionForwardGPU<Dtype, Itype>);
  m.def("ConvolutionBackwardGPU",
        &DimSwitchConvolutionBackwardGPU<Dtype, Itype>);

  m.def("ConvolutionTransposeForwardCPU",
        &DimSwitchConvolutionTransposeForwardCPU<Dtype, Itype>);
  m.def("ConvolutionTransposeBackwardCPU",
        &DimSwitchConvolutionTransposeBackwardCPU<Dtype, Itype>);
  m.def("ConvolutionTransposeForwardGPU",
        &DimSwitchConvolutionTransposeForwardGPU<Dtype, Itype>);
  m.def("ConvolutionTransposeBackwardGPU",
        &DimSwitchConvolutionTransposeBackwardGPU<Dtype, Itype>);

  m.def("AvgPoolingForwardCPU", &DimSwitchAvgPoolingForwardCPU<Dtype, Itype>);
  m.def("AvgPoolingBackwardCPU", &DimSwitchAvgPoolingBackwardCPU<Dtype, Itype>);
  m.def("AvgPoolingForwardGPU", &DimSwitchAvgPoolingForwardGPU<Dtype, Itype>);
  m.def("AvgPoolingBackwardGPU", &DimSwitchAvgPoolingBackwardGPU<Dtype, Itype>);

  m.def("PoolingTransposeForwardCPU",
        &DimSwitchPoolingTransposeForwardCPU<Dtype, Itype>);
  m.def("PoolingTransposeBackwardCPU",
        &DimSwitchPoolingTransposeBackwardCPU<Dtype, Itype>);
  m.def("PoolingTransposeForwardGPU",
        &DimSwitchPoolingTransposeForwardGPU<Dtype, Itype>);
  m.def("PoolingTransposeBackwardGPU",
        &DimSwitchPoolingTransposeBackwardGPU<Dtype, Itype>);

  m.def("GlobalPoolingForwardCPU",
        &DimSwitchGlobalPoolingForwardCPU<Dtype, Itype>);
  m.def("GlobalPoolingBackwardCPU",
        &DimSwitchGlobalPoolingBackwardCPU<Dtype, Itype>);
  m.def("GlobalPoolingForwardGPU",
        &DimSwitchGlobalPoolingForwardGPU<Dtype, Itype>);
  m.def("GlobalPoolingBackwardGPU",
        &DimSwitchGlobalPoolingBackwardGPU<Dtype, Itype>);

  m.def("BroadcastForwardCPU", &DimSwitchBroadcastForwardCPU<Dtype, Itype>);
  m.def("BroadcastBackwardCPU", &DimSwitchBroadcastBackwardCPU<Dtype, Itype>);
  m.def("BroadcastForwardGPU", &DimSwitchBroadcastForwardGPU<Dtype, Itype>);
  m.def("BroadcastBackwardGPU", &DimSwitchBroadcastBackwardGPU<Dtype, Itype>);
}

template <uint8_t D, typename Itype>
void instantiate_dim_itype(py::module &m, const std::string &dim,
                           const std::string &itypestr) {
  std::string coords_name = std::string("PyCoordsManager") + dim + itypestr;
  py::class_<CoordsManager<D, Itype>>(m, coords_name.c_str())
      .def(py::init<>())
      .def("existsCoordsKey", (bool (CoordsManager<D, Itype>::*)(py::object)) &
                                  CoordsManager<D, Itype>::existsCoordsKey)
      .def("getCoordsKey", &CoordsManager<D, Itype>::getCoordsKey)
      .def("getCoordsSize", (int (CoordsManager<D, Itype>::*)(py::object)) &
                                CoordsManager<D, Itype>::getCoordsSize)
      .def("getCoords", &CoordsManager<D, Itype>::getCoords)
      .def("initializeCoords",
           (uint64_t(CoordsManager<D, Itype>::*)(at::Tensor, py::object)) &
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
  instantiate_func<float, int32_t>(m, std::string("float"),
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
