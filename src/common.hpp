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
#ifndef COMMON
#define COMMON
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>

#include "coords_manager.hpp"
#include "instantiation.hpp"
#include "thread_pool.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <driver_types.h> // cuda driver types

#include <THC/THCBlas.h>
#include <thrust/device_vector.h>

#include "gpu.cuh"
#include "gpu_memory_manager.hpp"
#endif

template <typename T> std::string ArrToString(T arr) {
  std::string buf = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    buf += (i ? ", " : "") + std::to_string(arr[i]);
  }
  buf += "]";
  return buf;
}

template <typename Dtype, typename Itype> int dtypeMultiplier() {
  // if larger, return ceil of sizeof(Dtype) / sizeof(Itype)
  return sizeof(Dtype) > sizeof(Itype)
             ? (sizeof(Dtype) + sizeof(Itype) - 1) / sizeof(Itype)
             : 1;
}

template <typename T> void PyPrintArr(T arr) { py::print(ArrToString(arr)); }

template <uint8_t D>
std::vector<int> computeOutTensorStride(const Arr<D, int> &tensor_strides,
                                        const Arr<D, int> &strides,
                                        bool is_transpose) {
  std::vector<int> out_tensor_strides;
  for (int i = 0; i < D; i++) {
    if (is_transpose) {
      if (tensor_strides[i] % strides[i] > 0)
        throw std::invalid_argument(
            Formatter() << "The output tensor stride is not divisible by "
                           "up_strides. tensor stride: "
                        << ArrToString(tensor_strides)
                        << ", up_strides: " << ArrToString(strides));
      out_tensor_strides.push_back(tensor_strides[i] / strides[i]);
    } else
      out_tensor_strides.push_back(tensor_strides[i] * strides[i]);
  }
  return out_tensor_strides;
}

template <uint8_t D>
long ComputeKernelVolume(int region_type, const Arr<D, int> &kernel_size,
                         int n_offset) {
  int kernel_volume;
  if (region_type == 0) { // Hypercube
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume *= k;
  } else if (region_type == 1) { // Hypercross
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume += k - 1;
  } else if (region_type == 2) {
    kernel_volume = n_offset;
  } else {
    throw std::invalid_argument("Invalid region type");
  }
  return kernel_volume;
}

// Will be exported to python for lazy key initialization.
// For instance, ConvModule.out_coords_key can be used for other layers before
// feedforward
template <uint8_t D> class PyCoordsKey {
private:
  uint64_t key_; // Use the key_ for all coordshashmap query. Lazily set

public:
  bool key_set = false;
  Arr<D, int> tensor_strides_;
  PyCoordsKey() { reset(); }
  void reset();
  void copy(py::object ohter);
  void setKey(uint64_t key);
  uint64_t getKey();
  void setTensorStride(const Arr<D, int> &tensor_strides);
  void stride(const Arr<D, int> &strides);
  void up_stride(const Arr<D, int> &strides);
  Arr<D, int> getTensorStride() { return tensor_strides_; };
  std::string toString() const;
};

#endif
