#include "common.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <uint8_t D>
void PyCoordsKey<D>::setTensorStride(const Arr<D, int> &tensor_strides) {
  for (int i = 0; i < D; i++)
    tensor_strides_[i] = tensor_strides[i];
}

template <uint8_t D> void PyCoordsKey<D>::stride(const Arr<D, int> &strides) {
  for (int i = 0; i < D; i++)
    tensor_strides_[i] *= strides[i];
}

template <uint8_t D>
void PyCoordsKey<D>::up_stride(const Arr<D, int> &strides) {
  for (int i = 0; i < D; i++) {
    if (tensor_strides_[i] % strides[i] > 0)
      throw std::invalid_argument(
          Formatter() << "The output tensor stride is not divisible by "
                         "up_strides. tensor stride: "
                      << ArrToString(tensor_strides_)
                      << ", up_strides: " << ArrToString(strides));
    tensor_strides_[i] /= strides[i];
  }
}

template <uint8_t D> void PyCoordsKey<D>::copy(py::object py_other) {
  PyCoordsKey<D> *p_other = py_other.cast<PyCoordsKey<D> *>();
  tensor_strides_ = p_other->tensor_strides_;
  setKey(p_other->key_);
}

template <uint8_t D> void PyCoordsKey<D>::reset() {
  key_ = 0;
  key_set = false;
  for (int i = 0; i < D; i++)
    tensor_strides_[i] = 0;
}

template <uint8_t D> void PyCoordsKey<D>::setKey(uint64_t key) {
  key_ = key;
  key_set = true;
}

template <uint8_t D> uint64_t PyCoordsKey<D>::getKey() {
  if (key_set)
    return key_;
  else
    throw std::invalid_argument(Formatter() << "PyCoordsKey: Key Not set");
}

template <uint8_t D> std::string PyCoordsKey<D>::toString() const {
  return "< CoordsKey, key: " + std::to_string(key_) +
         ", tensor_stride: " + ArrToString(tensor_strides_) + " > ";
}

INSTANTIATE_CLASS_DIM(PyCoordsKey);
