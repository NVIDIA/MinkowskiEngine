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

template <typename Dtype, typename Itype>
void instantiate_func(py::module &m, const std::string &dtypestr,
                      const std::string &itypestr) {
  m.def((std::string("ConvolutionForwardCPU") + dtypestr).c_str(),
        &ConvolutionForwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionBackwardCPU") + dtypestr).c_str(),
        &ConvolutionBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionForwardGPU") + dtypestr).c_str(),
        &ConvolutionForwardGPU<Dtype, Itype>);
  m.def((std::string("ConvolutionBackwardGPU") + dtypestr).c_str(),
        &ConvolutionBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("ConvolutionTransposeForwardCPU") + dtypestr).c_str(),
        &ConvolutionTransposeForwardCPU<Dtype, Itype>);
  m.def((std::string("ConvolutionTransposeBackwardCPU") + dtypestr).c_str(),
        &ConvolutionTransposeBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("ConvolutionTransposeForwardGPU") + dtypestr).c_str(),
        &ConvolutionTransposeForwardGPU<Dtype, Itype>);
  m.def((std::string("ConvolutionTransposeBackwardGPU") + dtypestr).c_str(),
        &ConvolutionTransposeBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("AvgPoolingForwardCPU") + dtypestr).c_str(),
        &AvgPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("AvgPoolingBackwardCPU") + dtypestr).c_str(),
        &AvgPoolingBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("AvgPoolingForwardGPU") + dtypestr).c_str(),
        &AvgPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("AvgPoolingBackwardGPU") + dtypestr).c_str(),
        &AvgPoolingBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("MaxPoolingForwardCPU") + dtypestr).c_str(),
        &MaxPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("MaxPoolingBackwardCPU") + dtypestr).c_str(),
        &MaxPoolingBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("MaxPoolingForwardGPU") + dtypestr).c_str(),
        &MaxPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("MaxPoolingBackwardGPU") + dtypestr).c_str(),
        &MaxPoolingBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("PoolingTransposeForwardCPU") + dtypestr).c_str(),
        &PoolingTransposeForwardCPU<Dtype, Itype>);
  m.def((std::string("PoolingTransposeBackwardCPU") + dtypestr).c_str(),
        &PoolingTransposeBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("PoolingTransposeForwardGPU") + dtypestr).c_str(),
        &PoolingTransposeForwardGPU<Dtype, Itype>);
  m.def((std::string("PoolingTransposeBackwardGPU") + dtypestr).c_str(),
        &PoolingTransposeBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("GlobalPoolingForwardCPU") + dtypestr).c_str(),
        &GlobalPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("GlobalPoolingBackwardCPU") + dtypestr).c_str(),
        &GlobalPoolingBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("GlobalPoolingForwardGPU") + dtypestr).c_str(),
        &GlobalPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("GlobalPoolingBackwardGPU") + dtypestr).c_str(),
        &GlobalPoolingBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("GlobalMaxPoolingForwardCPU") + dtypestr).c_str(),
        &GlobalMaxPoolingForwardCPU<Dtype, Itype>);
  m.def((std::string("GlobalMaxPoolingBackwardCPU") + dtypestr).c_str(),
        &GlobalMaxPoolingBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("GlobalMaxPoolingForwardGPU") + dtypestr).c_str(),
        &GlobalMaxPoolingForwardGPU<Dtype, Itype>);
  m.def((std::string("GlobalMaxPoolingBackwardGPU") + dtypestr).c_str(),
        &GlobalMaxPoolingBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("BroadcastForwardCPU") + dtypestr).c_str(),
        &BroadcastForwardCPU<Dtype, Itype>);
  m.def((std::string("BroadcastBackwardCPU") + dtypestr).c_str(),
        &BroadcastBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("BroadcastForwardGPU") + dtypestr).c_str(),
        &BroadcastForwardGPU<Dtype, Itype>);
  m.def((std::string("BroadcastBackwardGPU") + dtypestr).c_str(),
        &BroadcastBackwardGPU<Dtype, Itype>);
#endif

  m.def((std::string("PruningForwardCPU") + dtypestr).c_str(),
        &PruningForwardCPU<Dtype, Itype>);
  m.def((std::string("PruningBackwardCPU") + dtypestr).c_str(),
        &PruningBackwardCPU<Dtype, Itype>);
#ifndef CPU_ONLY
  m.def((std::string("PruningForwardGPU") + dtypestr).c_str(),
        &PruningForwardGPU<Dtype, Itype>);
  m.def((std::string("PruningBackwardGPU") + dtypestr).c_str(),
        &PruningBackwardGPU<Dtype, Itype>);
#endif
}

template <typename Itype>
void instantiate_itype(py::module &m, const std::string &itypestr) {
  std::string coords_name = std::string("PyCoordsManager") + itypestr;
  py::class_<CoordsManager<Itype>>(m, coords_name.c_str())
      .def(py::init<>())
      .def("existsCoordsKey", (bool (CoordsManager<Itype>::*)(py::object)) &
                                  CoordsManager<Itype>::existsCoordsKey)
      .def("getCoordsKey", &CoordsManager<Itype>::getCoordsKey)
      .def("getKernelMap", &CoordsManager<Itype>::getKernelMap)
      .def("getCoordsSize", (int (CoordsManager<Itype>::*)(py::object)) &
                                CoordsManager<Itype>::getCoordsSize)
      .def("getCoords", &CoordsManager<Itype>::getCoords)
      .def("getRowIndicesPerBatch",
           &CoordsManager<Itype>::getRowIndicesPerBatch)
      .def("initializeCoords",
           (uint64_t(CoordsManager<Itype>::*)(at::Tensor, py::object, bool)) &
               CoordsManager<Itype>::initializeCoords)
      .def("__repr__",
           [](const CoordsManager<Itype> &a) { return a.toString(); });
}

void instantiate(py::module &m) {
  // Instantiate Itypes
  instantiate_itype<int32_t>(m, std::string("int32"));
  instantiate_func<float, int32_t>(m, std::string("f"), std::string("int32"));
  instantiate_func<double, int32_t>(m, std::string("d"), std::string("int32"));
}

void bind_native(py::module &m) {
#ifndef CPU_ONLY
  m.def("SparseVoxelization", &SparseVoxelization);
  py::class_<GPUMemoryManager<int32_t>>(m, "MemoryManager")
      .def(py::init<>())
      .def("size", &GPUMemoryManager<int32_t>::size)
      .def("resize", &GPUMemoryManager<int32_t>::resize);
#endif

  std::string name = std::string("PyCoordsKey");
  py::class_<PyCoordsKey>(m, name.c_str())
      .def(py::init<>())
      .def("copy", &PyCoordsKey::copy)
      .def("setKey", &PyCoordsKey::setKey)
      .def("getKey", &PyCoordsKey::getKey)
      .def("setDimension", &PyCoordsKey::setDimension)
      .def("getDimension", &PyCoordsKey::getDimension)
      .def("setTensorStride", &PyCoordsKey::setTensorStride)
      .def("getTensorStride", &PyCoordsKey::getTensorStride)
      .def("__repr__", [](const PyCoordsKey &a) { return a.toString(); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_native(m);
  instantiate(m);
}
