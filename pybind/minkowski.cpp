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

template <typename Dtype>
void instantiate_func(py::module &m, const std::string &dtypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &ConvolutionForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &ConvolutionBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &ConvolutionForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &ConvolutionBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &ConvolutionTransposeForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &ConvolutionTransposeBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &ConvolutionTransposeForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &ConvolutionTransposeBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("AvgPoolingForwardCPU") + dtypestr).c_str(),
        &AvgPoolingForwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardCPU") + dtypestr).c_str(),
        &AvgPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("AvgPoolingForwardGPU") + dtypestr).c_str(),
        &AvgPoolingForwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("AvgPoolingBackwardGPU") + dtypestr).c_str(),
        &AvgPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("MaxPoolingForwardCPU") + dtypestr).c_str(),
        &MaxPoolingForwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardCPU") + dtypestr).c_str(),
        &MaxPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("MaxPoolingForwardGPU") + dtypestr).c_str(),
        &MaxPoolingForwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("MaxPoolingBackwardGPU") + dtypestr).c_str(),
        &MaxPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PoolingTransposeForwardCPU") + dtypestr).c_str(),
        &PoolingTransposeForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &PoolingTransposeBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PoolingTransposeForwardGPU") + dtypestr).c_str(),
        &PoolingTransposeForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &PoolingTransposeBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &GlobalPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &GlobalPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
        &GlobalPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
        &GlobalPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("GlobalMaxPoolingForwardCPU") + dtypestr).c_str(),
        &GlobalMaxPoolingForwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardCPU") + dtypestr).c_str(),
        &GlobalMaxPoolingBackwardCPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("GlobalMaxPoolingForwardGPU") + dtypestr).c_str(),
        &GlobalMaxPoolingForwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
  m.def((std::string("GlobalMaxPoolingBackwardGPU") + dtypestr).c_str(),
        &GlobalMaxPoolingBackwardGPU<Dtype>,
        py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &BroadcastForwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &BroadcastBackwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &BroadcastForwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &BroadcastBackwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &PruningForwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &PruningBackwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &PruningForwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &PruningBackwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#endif

  m.def((std::string("UnionForwardCPU") + dtypestr).c_str(),
        &UnionForwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardCPU") + dtypestr).c_str(),
        &UnionBackwardCPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#ifndef CPU_ONLY
  m.def((std::string("UnionForwardGPU") + dtypestr).c_str(),
        &UnionForwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
  m.def((std::string("UnionBackwardGPU") + dtypestr).c_str(),
        &UnionBackwardGPU<Dtype>, py::call_guard<py::gil_scoped_release>());
#endif
}

void instantiate_coordsman(py::module &m) {
  std::string coords_name = std::string("CoordsManager");
  py::class_<CoordsManager>(m, coords_name.c_str())
      .def(py::init<int>())
      .def("existsCoordsKey", (bool (CoordsManager::*)(py::object) const) &
                                  CoordsManager::existsCoordsKey)
      .def("getCoordsKey", &CoordsManager::getCoordsKey)
      .def("getKernelMap", &CoordsManager::getKernelMap)
      .def("getCoordsSize", (int (CoordsManager::*)(py::object) const) &
                                CoordsManager::getCoordsSize)
      .def("getCoords", &CoordsManager::getCoords)
      .def("getBatchSize", &CoordsManager::getBatchSize)
      .def("getBatchIndices", &CoordsManager::getBatchIndices)
      .def("getRowIndicesPerBatch", &CoordsManager::getRowIndicesPerBatch)
      .def("setOriginCoordsKey", &CoordsManager::setOriginCoordsKey)
      .def("initializeCoords",
           (uint64_t(CoordsManager::*)(at::Tensor, at::Tensor, py::object, bool,
                                       bool, bool)) &
               CoordsManager::initializeCoords,
           py::call_guard<py::gil_scoped_release>())
      .def("createStridedCoords", &CoordsManager::createStridedCoords)
      .def("createTransposedStridedRegionCoords",
           &CoordsManager::createTransposedStridedRegionCoords)
      .def("createPrunedCoords", &CoordsManager::createPrunedCoords)
      .def("createOriginCoords", &CoordsManager::createOriginCoords)
      .def("printDiagnostics", &CoordsManager::printDiagnostics)
      .def("__repr__", [](const CoordsManager &a) { return a.toString(); });
}

void instantiate(py::module &m) {
  instantiate_coordsman(m);
  instantiate_func<float>(m, std::string("f"));
  instantiate_func<double>(m, std::string("d"));
}

void bind_native(py::module &m) {
  std::string name = std::string("CoordsKey");
  py::class_<CoordsKey>(m, name.c_str())
      .def(py::init<>())
      .def("copy", &CoordsKey::copy)
      .def("isKeySet", &CoordsKey::isKeySet)
      .def("setKey", &CoordsKey::setKey)
      .def("getKey", &CoordsKey::getKey)
      .def("setDimension", &CoordsKey::setDimension)
      .def("getDimension", &CoordsKey::getDimension)
      .def("setTensorStride", &CoordsKey::setTensorStride)
      .def("getTensorStride", &CoordsKey::getTensorStride)
      .def("__repr__", [](const CoordsKey &a) { return a.toString(); });

  // Quantization
  m.def("quantize_np", &quantize_np);
  m.def("quantize_th", &quantize_th);
  m.def("quantize_label_np", &quantize_label_np);
  m.def("quantize_label_th", &quantize_label_th);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_native(m);
  instantiate(m);
}
