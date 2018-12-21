#include "common.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <uint8_t D>
void PyCoordsKey<D>::setPixelDist(const Arr<D, int> &pixel_dists) {
  for (int i = 0; i < D; i++)
    pixel_dists_[i] = pixel_dists[i];
}

template <uint8_t D> void PyCoordsKey<D>::stride(const Arr<D, int> &strides) {
  for (int i = 0; i < D; i++)
    pixel_dists_[i] *= strides[i];
}

template <uint8_t D>
void PyCoordsKey<D>::up_stride(const Arr<D, int> &strides) {
  for (int i = 0; i < D; i++) {
    if (pixel_dists_[i] % strides[i] > 0)
      throw std::invalid_argument(
          Formatter() << "The output pixel dist is not divisible by "
                         "up_strides. pixel dists: "
                      << ArrToString(pixel_dists_)
                      << ", up_strides: " << ArrToString(strides));
    pixel_dists_[i] /= strides[i];
  }
}

template <uint8_t D> void PyCoordsKey<D>::copy(py::object py_other) {
  PyCoordsKey<D> *p_other = py_other.cast<PyCoordsKey<D> *>();
  pixel_dists_ = p_other->pixel_dists_;
  setKey(p_other->key_);
}

template <uint8_t D> void PyCoordsKey<D>::reset() {
  key_ = 0;
  key_set = false;
  for (int i = 0; i < D; i++)
    pixel_dists_[i] = 0;
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
         ", pixel_dist: " + ArrToString(pixel_dists_) + " > ";
}

template class PyCoordsKey<1>;
template class PyCoordsKey<2>;
