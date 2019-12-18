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
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <driver_types.h> // cuda driver types

#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>

#include "gpu.cuh"
#include "gpu_memory_manager.hpp"
#endif

// Will be exported to python for lazy key initialization.
// For instance, ConvModule.out_coords_key can be used for other layers before
// feedforward
class CoordsKey {
private:
  uint64_t key_; // Use the key_ for all coordshashmap query. Lazily set
  int D_;        // dimension of the current coordinate system

public:
  bool key_set = false;
  std::vector<int> tensor_strides_;

  // Functions
  CoordsKey() { reset(); }
  CoordsKey(int dim);
  void reset();
  void copy(py::object ohter);
  void setKey(uint64_t key);
  void setDimension(int dim);
  bool isKeySet() const { return key_set; }
  uint64_t getKey() const;
  uint64_t getDimension() const { return D_; }
  void setTensorStride(const std::vector<int> &tensor_strides);
  void stride(const std::vector<int> &strides);
  void up_stride(const std::vector<int> &strides);
  std::vector<int> getTensorStride() const { return tensor_strides_; }
  std::string toString() const;
};

#endif // COMMON
