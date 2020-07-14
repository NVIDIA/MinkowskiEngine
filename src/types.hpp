/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <tuple>
#include <vector>

namespace minkowski {

namespace py = pybind11;

// clang-format off
template <typename uint_type, typename int_type, typename float_type>
struct type_wrapper {
  using tensor_order_type        = uint_type;
  using index_type               = uint_type;
  using stride_type              = std::vector<uint_type>;
  using size_type                = uint_type;
  using dcoordinate_type         = int_type;
  using coordinate_map_hash_type = uint64_t;
  using index_vector_type        = std::vector<index_type>;
};
// clang-format on

using default_types = type_wrapper<uint32_t, int32_t, float>;

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

// clang-format off
/*
 * Kernel map specific types
 */
using cpu_in_map  = default_types::index_vector_type;
using cpu_out_map = default_types::index_vector_type;

// Input index to output index mapping for each spatial kernel
using cpu_in_maps  = std::vector<cpu_in_map>;
using cpu_out_maps = std::vector<cpu_out_map>;

using cpu_kernel_map           = std::pair<cpu_in_maps, cpu_out_maps>;
using cpu_reference_kernel_map = std::pair<cpu_in_maps &, cpu_out_maps &>;
// clang-format on

using coordinate_map_key_type =
    std::pair<default_types::stride_type, std::string>;

template <typename vector_type>
std::vector<vector_type>
initialize_maps(default_types::size_type number_of_vectors,
                default_types::size_type vector_size) {
  auto vectors = std::vector<vector_type>();
  for (default_types::size_type i = 0; i < number_of_vectors; ++i) {
    auto vector = vector_type(vector_size);
    vectors.push_back(std::move(vector));
  }
  return vectors;
}

// GPU memory manager backend. No effect with CPU_ONLY build
namespace GPUMemoryAllocatorBackend {
enum Type { PYTORCH = 0, CUDA = 1 };
}

namespace MapManagerType {
enum Type { CPU = 0, CUDA = 1, C10 = 2 };
}

namespace CUDAKernelMapMode {
enum Mode { MEMORY_EFFICIENT = 0, SPEED_OPTIMIZED = 1 };
}

namespace CoordinateMapBackend {
enum Type { CPU = 0, CUDA = 1 };
}

namespace RegionType {
enum region_type { HYPER_CUBE, HYPER_CROSS, CUSTOM };
}

/* Key for KernelMap
 *
 * A tuple of (CoordinateMapKey (input),
 *             CoordinateMapKey (output),
 *             kernel stride,
 *             kernel size,
 *             kernel dilation,
 *             kernel region type,
 *             is_transpose,
 *             is_pool)
 */
using kernel_map_key_type =
    std::tuple<coordinate_map_key_type,    // in
               coordinate_map_key_type,    // out
               default_types::stride_type, // kernel size
               default_types::stride_type, // kernel stride
               default_types::stride_type, // kernel dilation
               RegionType::region_type,    // kernel region type
               bool,                       // is transpose
               bool                        // is pool
               >;

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

/*
struct KernelMapKeyHash {
  uint64_t operator()(KernelMapKey const &key) const {
  }
};

template <uint32_t D, typename Itype> struct ArrHash {
  uint64_t operator()(dim_array<D, Itype> const &p) const {
    return hash_vec<dim_array<D, Itype>>(p);
  }
};
*/

} // end namespace minkowski

#endif // TYPES_HPP
