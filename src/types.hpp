/*
 *  Copyright (c) 2020 NVIDIA Corporation.
 *  Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef TYPES
#define TYPES

#include <array>
#include <functional>
#include <vector>

#ifndef CPU_ONLY
#include <thrust/host_vector.h>
#endif

namespace minkowski {

// clang-format off
template <typename uint_type, typename int_type, typename float_type>
struct type_wrapper {
  using tensor_order_type = uint_type;
  using index_type        = uint_type;
  using stride_type       = uint_type;
  using size_type         = uint_type;
  using dcoordinate_type  = int_type;
  using ccoordinate_type  = float_type;
#ifndef CPU_ONLY
  using index_vector_type = thrust::host_vector<index_type>;
#else
  using index_vector_type = std::vector<index_type>;
#endif // CPU_ONLY
};

using default_types = type_wrapper<uint32_t, int32_t, float>;

// Vector backend

// D-Dimensional coordinate + batch dimension = D + 1
template <typename int_type = default_types::stride_type>
using strides = std::vector<int_type>;

// For hashing kernel sizes, strides, nd dilations.
template <default_types::tensor_order_type D,
          typename int_type = default_types::index_type>
using dim_array = std::array<int_type, D>;

template <typename data_type, typename size_type = default_types::size_type>
struct ptr_vector {

  ptr_vector(data_type *ptr, size_type size) : ptr_(ptr), size_(size) {}
  size_type size() const { return size_; };
  data_type *data() { return ptr_; };
  const data_type *data() const { return ptr_; };

  // members
  data_type *ptr_;
  default_types::size_type size_;
};

// Key for InOutMap
// (in_coords_key, out_coords_key, stride hash, kernel size, dilation,
// is_transpose, is_pool)
using InOutMapKey = std::array<uint64_t, 8>;

/*
 * Kernel map specific types
 */
using cpuInMap  = default_types::index_vector_type;
using cpuOutMap = default_types::index_vector_type;

// Input index to output index mapping for each spatial kernel
using cpuInMaps  = std::vector<cpuInMap>;
using cpuOutMaps = std::vector<cpuOutMap>;
// clang-format on

using cpuInOutMapsPair    = std::pair<cpuInMaps, cpuOutMaps>;
using cpuInOutMapsRefPair = std::pair<cpuInMaps &, cpuOutMaps &>;

// GPU memory manager backend. No effect with CPU_ONLY build
enum GPUMemoryManagerBackend { CUDA = 0, PYTORCH = 1 };

// FNV64-1a
// uint64_t for unsigned long, must use CXX -m64
template <typename T> uint64_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (auto x : p) {
    hash ^= x;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

struct InOutMapKeyHash {
  uint64_t operator()(InOutMapKey const &p) const {
    return hash_vec<InOutMapKey>(p);
  }
};

template <uint32_t D, typename Itype> struct ArrHash {
  uint64_t operator()(dim_array<D, Itype> const &p) const {
    return hash_vec<dim_array<D, Itype>>(p);
  }
};

} // end namespace minkowski

#endif // TYPES
