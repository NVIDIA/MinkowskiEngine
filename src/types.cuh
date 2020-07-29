/*
 * Copyright 2020 NVIDIA Corporation.
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
 */
#ifndef TYPES_CUH
#define TYPES_CUH

#include "types.hpp"

#include "3rdparty/concurrent_unordered_map.cuh"
#include "coords_functors.cuh"

#include <cstddef>
#include <thrust/device_vector.h>

namespace minkowski {

// The size of a coordinate is fixed globally and is embedded in the hash
// function and the equality function.
template <typename coordinate_type> struct coordinate {
  coordinate() {}
  __host__ __device__ coordinate(coordinate_type *_ptr) : ptr{_ptr} {}

  coordinate_type const *data() { return ptr; }
  __host__ __device__ coordinate_type operator[](size_t i) const {
    return ptr[i];
  }

  // members
  coordinate_type const *ptr;
};

/**
 * Wrapper for a long vector. Used for InOutMap and to save a device pointer.
 *
 * TODO: replace it with thrust::device_vector<Itype, Alloc>
 * https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
 **/
template <typename Itype> struct pVector {
  pVector(Itype *ptr, size_t size) : ptr_(ptr), size_(size) {}

  size_t size() const { return size_; };

  Itype const *data() const { return ptr_; };
  Itype *data() { return ptr_; };

  // members
  Itype *ptr_;
  size_t size_;
};

// all coordinate-map type definitions
template <typename coordinate_type, typename V> struct ckey_value_types {
  using ctype = coordinate_type;
  using key_type = coordinate<coordinate_type>;
  using value_type = V;
  using mapped_type = V;
  using pair_type = thrust::pair<key_type, V>;
  using map_type =
      concurrent_unordered_map<key_type, value_type,
                               coordinate_murmur3<coordinate_type>,
                               coordinate_equal_to<coordinate_type>>;
};

} // namespace minkowski

#endif // TYPES_CUH
