/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "common.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace minkowski {

CoordsKey::CoordsKey(int dim) {
  reset();
  setDimension(dim);
}

void CoordsKey::setTensorStride(const std::vector<int> &tensor_strides) {
  int D = getDimension();
  ASSERT(D < 0 or (D > 0 and D == tensor_strides.size()),
         "The tensor strides dimension mismatch: ", ArrToString(tensor_strides),
         ", dimension of the key: ", D);
  tensor_strides_ = tensor_strides;
  tensor_stride_set = true;
}

void CoordsKey::stride(const std::vector<int> &strides) {
  ASSERT(tensor_stride_set, "You must set the tensor strides first.");
  ASSERT(getDimension() == strides.size(),
         "The size of strides: ", ArrToString(strides),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  for (int i = 0; i < getDimension(); i++)
    tensor_strides_[i] *= strides[i];
}

void CoordsKey::up_stride(const std::vector<int> &strides) {
  ASSERT(tensor_stride_set, "You must set the tensor strides first.");
  ASSERT(getDimension() == strides.size(),
         "The size of strides: ", ArrToString(strides),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  ASSERT(tensor_strides_.size() == strides.size(),
         "The size of the strides: ", ArrToString(strides),
         " does not match the size of the PyCoordKey tensor_strides_: ",
         ArrToString(tensor_strides_), ".");
  for (int i = 0; i < getDimension(); i++) {
    ASSERT(tensor_strides_[i] % strides[i] == 0,
           "The output tensor stride is not divisible by ",
           "up_strides. tensor stride: ", ArrToString(tensor_strides_),
           ", up_strides: ", ArrToString(strides), ".");
    tensor_strides_[i] /= strides[i];
  }
}

void CoordsKey::copy(py::object py_other) {
  CoordsKey *p_other = py_other.cast<CoordsKey *>();
  setKey(p_other->key_); // Call first to set the key_set.

  setDimension(p_other->D_);
  ASSERT(getDimension() == p_other->tensor_strides_.size(),
         "The size of strides: ", ArrToString(p_other->tensor_strides_),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  tensor_strides_ = p_other->tensor_strides_;
  tensor_stride_set = p_other->tensor_stride_set;
}

void CoordsKey::reset() {
  key_ = 0;
  D_ = -1;
  key_set = false;
  tensor_stride_set = false;
  tensor_strides_.clear();
}

void CoordsKey::setKey(uint64_t key) {
  key_ = key;
  key_set = true;
}

void CoordsKey::setDimension(int dim) {
  ASSERT(dim > 0, "The dimension should be a positive integer, you put: ",
         std::to_string(dim), ".");
  D_ = dim;
  tensor_strides_.resize(D_);
}

uint64_t CoordsKey::getKey() const {
  ASSERT(key_set, "CoordsKey: Key Not set")
  return key_;
}

std::string CoordsKey::toString() const {
  Formatter out;
  out << "< CoordsKey, key: " << (key_set ? std::to_string(key_) : "None")
      << ", tensor_stride: "
      << (tensor_stride_set ? ArrToString(tensor_strides_) : "None")
      << " in dimension: " << std::to_string(D_) << " >\n";
  return out;
}

} // end namespace minkowski
