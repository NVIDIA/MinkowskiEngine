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

template <typename Dtype>
void instantiate_func(py::module &m, const std::string &dtypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &mink::ConvolutionTransposeBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("AvgPoolingForwardCPU") + dtypestr).c_str(),
        &mink::AvgPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::AvgPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("AvgPoolingForwardGPU") + dtypestr).c_str(),
        &mink::AvgPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::AvgPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("MaxPoolingForwardCPU") + dtypestr).c_str(),
        &mink::MaxPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::MaxPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("MaxPoolingForwardGPU") + dtypestr).c_str(),
        &mink::MaxPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::MaxPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PoolingTransposeForwardCPU") + dtypestr).c_str(),
        &mink::PoolingTransposeForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &mink::PoolingTransposeBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PoolingTransposeForwardGPU") + dtypestr).c_str(),
        &mink::PoolingTransposeForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &mink::PoolingTransposeBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &mink::GlobalPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::GlobalPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
        &mink::GlobalPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::GlobalPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalMaxPoolingForwardCPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardCPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalMaxPoolingForwardGPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardGPU") + dtypestr).c_str(),
        &mink::GlobalMaxPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &mink::BroadcastForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &mink::BroadcastBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &mink::BroadcastForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &mink::BroadcastBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &mink::PruningForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &mink::PruningBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &mink::PruningForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &mink::PruningBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("UnionForwardCPU") + dtypestr).c_str(),
        &mink::UnionForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardCPU") + dtypestr).c_str(),
        &mink::UnionBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("UnionForwardGPU") + dtypestr).c_str(),
        &mink::UnionForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardGPU") + dtypestr).c_str(),
        &mink::UnionBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif
}

void instantiate_coordsman(py::module &m) {
  std::string coords_name = std::string("CoordsManager");
  py::class_<mink::CoordsManager>(m, coords_name.c_str())
      .def(py::init<int>())
      .def("existsCoordsKey",
           (bool (mink::CoordsManager::*)(py::object) const) &
               mink::CoordsManager::existsCoordsKey)
      .def("getCoordsKey", &mink::CoordsManager::getCoordsKey)
      .def("getKernelMap", &mink::CoordsManager::getKernelMap)
#ifndef CPU_ONLY
      .def("getKernelMapGPU", &mink::CoordsManager::getKernelMapGPU)
#endif
      .def("getCoordsMap", &mink::CoordsManager::getCoordsMap)
      .def("getUnionMap", &mink::CoordsManager::getUnionMap)
      .def("getCoordsSize", (int (mink::CoordsManager::*)(py::object) const) &
                                mink::CoordsManager::getCoordsSize)
      .def("getCoords", &mink::CoordsManager::getCoords)
      .def("getBatchSize", &mink::CoordsManager::getBatchSize)
      .def("getBatchIndices", &mink::CoordsManager::getBatchIndices)
      .def("getRowIndicesAtBatchIndex",
           &mink::CoordsManager::getRowIndicesAtBatchIndex)
      .def("getRowIndicesPerBatch", &mink::CoordsManager::getRowIndicesPerBatch)
      .def("setOriginCoordsKey", &mink::CoordsManager::setOriginCoordsKey)
      .def("initializeCoords",
           (uint64_t(mink::CoordsManager::*)(at::Tensor, at::Tensor, py::object,
                                             bool, bool, bool)) &
               mink::CoordsManager::initializeCoords,
           py::call_guard<py::gil_scoped_release>())
      .def("createStridedCoords", &mink::CoordsManager::createStridedCoords)
      .def("createTransposedStridedRegionCoords",
           &mink::CoordsManager::createTransposedStridedRegionCoords)
      .def("createPrunedCoords", &mink::CoordsManager::createPrunedCoords)
      .def("createOriginCoords", &mink::CoordsManager::createOriginCoords)
      .def("printDiagnostics", &mink::CoordsManager::printDiagnostics)
      .def("__repr__",
           [](const mink::CoordsManager &a) { return a.toString(); });
}

void instantiate(py::module &m) {
  instantiate_coordsman(m);
  instantiate_func<float>(m, std::string("f"));
  instantiate_func<double>(m, std::string("d"));
}

void bind_native(py::module &m) {
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
  m.def("quantize_np", &mink::quantize_np);
  m.def("quantize_th", &mink::quantize_th);
  m.def("quantize_label_np", &mink::quantize_label_np);
  m.def("quantize_label_th", &mink::quantize_label_th);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_native(m);
  instantiate(m);
}
