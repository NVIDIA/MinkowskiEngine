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
#include <string>

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "extern.hpp"
#include "src/common.hpp"

namespace py = pybind11;

namespace mink = minkowski;

template <typename MapType, typename Dtype>
void instantiate_func(py::module &m, const std::string &dtypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("AvgPoolingForwardCPU") + dtypestr).c_str(),
        &mink::AvgPoolingForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::AvgPoolingBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("AvgPoolingForwardGPU") + dtypestr).c_str(),
        &mink::AvgPoolingForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::AvgPoolingBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("MaxPoolingForwardCPU") + dtypestr).c_str(),
        &mink::MaxPoolingForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::MaxPoolingBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("MaxPoolingForwardGPU") + dtypestr).c_str(),
        &mink::MaxPoolingForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::MaxPoolingBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PoolingTransposeForwardCPU") + dtypestr).c_str(),
        &mink::PoolingTransposeForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &mink::PoolingTransposeBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PoolingTransposeForwardGPU") + dtypestr).c_str(),
        &mink::PoolingTransposeForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &mink::PoolingTransposeBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &mink::GlobalPoolingForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::GlobalPoolingBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
        &mink::GlobalPoolingForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::GlobalPoolingBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalMaxPoolingForwardCPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalMaxPoolingForwardGPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &mink::BroadcastForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &mink::BroadcastBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &mink::BroadcastForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &mink::BroadcastBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &mink::PruningForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &mink::PruningBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &mink::PruningForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &mink::PruningBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("UnionForwardCPU") + dtypestr).c_str(),
        &mink::UnionForwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardCPU") + dtypestr).c_str(),
        &mink::UnionBackwardCPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("UnionForwardGPU") + dtypestr).c_str(),
        &mink::UnionForwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardGPU") + dtypestr).c_str(),
        &mink::UnionBackwardGPU<MapType, Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif
}

template <typename MapType> void instantiate_coordsman(py::module &m) {
  std::string coords_name = std::string("CoordsManager");
  py::class_<mink::CoordsManager<MapType>>(m, coords_name.c_str())
      .def(py::init<int>())
      .def(py::init<int, mink::MemoryManagerBackend>())
      .def("existsCoordsKey",
           (bool (mink::CoordsManager<MapType>::*)(py::object) const) &
               mink::CoordsManager<MapType>::existsCoordsKey)
      .def("getCoordsKey", &mink::CoordsManager<MapType>::getCoordsKey)
      .def("getKernelMap", &mink::CoordsManager<MapType>::getKernelMap)
#ifndef CPU_ONLY
      .def("getKernelMapGPU", &mink::CoordsManager<MapType>::getKernelMapGPU)
#endif
      .def("getCoordsMap", &mink::CoordsManager<MapType>::getCoordsMap)
      .def("getUnionMap", &mink::CoordsManager<MapType>::getUnionMap)
      .def("getCoordsSize",
           (int (mink::CoordsManager<MapType>::*)(py::object) const) &
               mink::CoordsManager<MapType>::getCoordsSize)
      .def("getCoords", &mink::CoordsManager<MapType>::getCoords)
      .def("getBatchSize", &mink::CoordsManager<MapType>::getBatchSize)
      .def("getBatchIndices", &mink::CoordsManager<MapType>::getBatchIndices)
      .def("getRowIndicesAtBatchIndex",
           &mink::CoordsManager<MapType>::getRowIndicesAtBatchIndex)
      .def("getRowIndicesPerBatch",
           &mink::CoordsManager<MapType>::getRowIndicesPerBatch)
      .def("setOriginCoordsKey",
           &mink::CoordsManager<MapType>::setOriginCoordsKey)
      .def("initializeCoords",
           (uint64_t(mink::CoordsManager<MapType>::*)(
               at::Tensor, at::Tensor, at::Tensor, py::object, const bool,
               const bool, const bool, const bool)) &
               mink::CoordsManager<MapType>::initializeCoords,
           py::call_guard<py::gil_scoped_release>())
      .def("createStridedCoords",
           &mink::CoordsManager<MapType>::createStridedCoords)
      .def("createTransposedStridedRegionCoords",
           &mink::CoordsManager<MapType>::createTransposedStridedRegionCoords)
      .def("createPrunedCoords",
           &mink::CoordsManager<MapType>::createPrunedCoords)
      .def("createOriginCoords",
           &mink::CoordsManager<MapType>::createOriginCoords)
      .def("printDiagnostics", &mink::CoordsManager<MapType>::printDiagnostics)
      .def("__repr__",
           [](const mink::CoordsManager<MapType> &a) { return a.toString(); });
}

template <typename MapType> void instantiate(py::module &m) {
  instantiate_coordsman<MapType>(m);
  instantiate_func<MapType, float>(m, std::string("f"));
  instantiate_func<MapType, double>(m, std::string("d"));
}

template <typename MapType> void bind_native(py::module &m) {
  std::string name = std::string("CoordsKey");
  py::class_<mink::CoordsKey>(m, name.c_str())
      .def(py::init<>())
      .def("copy", &mink::CoordsKey::copy)
      .def("isKeySet", &mink::CoordsKey::isKeySet)
      .def("setKey", &mink::CoordsKey::setKey)
      .def("getKey", &mink::CoordsKey::getKey)
      .def("setDimension", &mink::CoordsKey::setDimension)
      .def("getDimension", &mink::CoordsKey::getDimension)
      .def("setTensorStride", &mink::CoordsKey::setTensorStride)
      .def("getTensorStride", &mink::CoordsKey::getTensorStride)
      .def("__repr__", [](const mink::CoordsKey &a) { return a.toString(); });

  // Quantization
  m.def("quantize_np", &mink::quantize_np<MapType>);
  m.def("quantize_th", &mink::quantize_th<MapType>);
  m.def("quantize_label_np", &mink::quantize_label_np);
  m.def("quantize_label_th", &mink::quantize_label_th);
  m.def("quantization_average_features", &mink::quantization_average_features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<mink::MemoryManagerBackend>(m, "MemoryManagerBackend")
      .value("CUDA", mink::MemoryManagerBackend::CUDA)
      .value("PYTORCH", mink::MemoryManagerBackend::PYTORCH)
      .export_values();

  bind_native<mink::CoordsToIndexMap>(m);
  instantiate<mink::CoordsToIndexMap>(m);
}
