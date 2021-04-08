/* Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)
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
#ifndef COORDINATE_MAP_GPU_CUH
#define COORDINATE_MAP_GPU_CUH

#include "3rdparty/concurrent_unordered_map.cuh"
#include "3rdparty/hash/helper_functions.cuh"
#include "allocators.cuh"
#include "coordinate_map.hpp"
#include "coordinate_map_functors.cuh"
#include "kernel_map.cuh"
#include "storage.cuh"

#include <torch/extension.h>

namespace minkowski {

/*
 * Inherit from the CoordinateMap for a concurrent coordinate unordered map.
 */
template <typename coordinate_type, template <typename T>
                                    class TemplatedAllocator =
                                        detail::c10_allocator>
class CoordinateMapGPU
    : public CoordinateMap<coordinate_type, TemplatedAllocator> {
public:
  // clang-format off
  using base_type           = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using self_type           = CoordinateMapGPU<coordinate_type, TemplatedAllocator>;
  using size_type           = typename base_type::size_type;
  using index_type          = typename base_type::index_type;
  using stride_type         = typename base_type::stride_type;

  using map_allocator_type  = TemplatedAllocator<thrust::pair<coordinate<coordinate_type>, index_type>>;
  using byte_allocator_type = TemplatedAllocator<char>;

  // Map types
  using key_type            = coordinate<coordinate_type>;
  using mapped_type         = index_type;
  using hasher_type         = detail::coordinate_murmur3<coordinate_type>;
  using key_equal_type      = detail::coordinate_equal_to<coordinate_type>;
  using map_type            = concurrent_unordered_map<key_type,          // key
                                                       mapped_type,        // mapped_type
                                                       hasher_type,        // hasher
                                                       key_equal_type,     // equality
                                                       map_allocator_type>;// allocator
  using value_type          = typename map_type::value_type;

  // return types
  using kernel_map_type     = gpu_kernel_map<index_type, byte_allocator_type>;

  // iterator
  using iterator            = typename map_type::iterator;
  using const_iterator      = typename map_type::const_iterator;

  // index vectors
  using index_storage_type     = gpu_storage<default_types::index_type, byte_allocator_type>;
  using coordnate_storage_type = gpu_storage<coordinate_type, byte_allocator_type>;
  // clang-format on

  // return types
  // using the QueryResultAllocator gives segfault!

public:
  CoordinateMapGPU() = delete;
  CoordinateMapGPU(size_type const number_of_coordinates,
                   size_type const coordinate_size,
                   size_type const hashtable_occupancy = 50,
                   stride_type const stride = {1},
                   map_allocator_type map_alloc = map_allocator_type(),
                   byte_allocator_type byte_alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, byte_alloc),
        m_hashtable_occupancy{hashtable_occupancy},
        m_capacity(0), // should be updated in the reserve
        m_hasher(hasher_type{coordinate_size}),
        m_equal(key_equal_type{coordinate_size}),
        m_unused_key(coordinate<coordinate_type>{nullptr}),
        m_unused_element(std::numeric_limits<coordinate_type>::max()),
        m_map_allocator(map_alloc) {
    // reserve coordinates
    reserve(number_of_coordinates);
    // copy the tensor_stride
    LOG_DEBUG("tensor_stride", base_type::m_tensor_stride);
    m_device_tensor_stride.from_vector(base_type::m_tensor_stride);
    // m_device_tensor_stride = base_type::m_tensor_stride;
    LOG_DEBUG("device tensor_stride set");
    static_assert(
        sizeof(index_type) == sizeof(size_type),
        "kernel_map shared memory requires the type sizes to be the same");
    static_assert(
        sizeof(coordinate_type) == sizeof(size_type),
        "kernel_map shared memory requires the type sizes to be the same");
  }

  template <bool remap>
  void insert(coordinate_iterator<coordinate_type> key_first,
              coordinate_iterator<coordinate_type> key_last);

  template <bool remap>
  std::pair<index_storage_type, index_storage_type>
  insert_and_map(coordinate_iterator<coordinate_type> key_first,
                 coordinate_iterator<coordinate_type> key_last);

  std::pair<index_storage_type, index_storage_type>
  find(coordinate_iterator<coordinate_type> key_first,
       coordinate_iterator<coordinate_type> key_last) const;

  inline void reserve(size_type size) {
    if (size > m_capacity) {
      // reserve coordinate
      base_type::reserve(size);
      // reserve map
      LOG_DEBUG("Reserve map of",
                compute_hash_table_size(size, m_hashtable_occupancy),
                "for concurrent_unordered_map of size", size);
      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      LOG_DEBUG("Done concurrent_unordered_map creation");
      CUDA_TRY(cudaStreamSynchronize(0));
      m_capacity = size;
      LOG_DEBUG("Reserved concurrent_unordered_map");
    }
  }

  // Network specific functions.

  /*
   * @brief strided coordinate map.
   */
  self_type stride(stride_type const &stride) const;
  self_type stride_region(cpu_kernel_region<coordinate_type> &kernel,
                          stride_type const &out_tensor_stride) const;
  self_type origin() const;
  self_type prune(bool const *keep_begin, bool const *keep_end) const;
  self_type
  merge(std::vector<std::reference_wrapper<self_type>> const &maps) const;

  kernel_map_type kernel_map(self_type const &out_coordinate_map,
                             gpu_kernel_region<coordinate_type> const &kernel,
                             CUDAKernelMapMode::Mode kernel_map_mode,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  kernel_map_type stride_map(self_type const &out_coordinate_map,
                             stride_type const &out_tensor_stride,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  kernel_map_type origin_map(self_type const &origin_coordinate_map,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield) const;

  template <typename coordinate_field_type>
  std::pair<at::Tensor, at::Tensor>
  field_map(coordinate_field_type const *p_tfield,
            size_type const num_tfield) const;

  std::vector<at::Tensor>
  union_map(std::vector<std::reference_wrapper<self_type>> const &maps,
            uint32_t thread_dim = CUDA_NUM_THREADS) const;

  // Returns the number of elements in the coordinate map
  inline size_type size() const { return m_size; }
  void copy_coordinates(coordinate_type *dst_coordinate) const;
  std::string to_string() const {
    Formatter o;
    o << "CoordinateMapGPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }
  inline map_type const const_hash_map() const { return *m_map.get(); };

  // Insert indices given initialized coordinates
  void initialize_valid_indices(size_t const N_unique);

  // access the coordinate data pointer
  using base_type::const_coordinate_data;
  using base_type::coordinate_data;
  using base_type::coordinate_size;

  // Find GPU values in the map. key_iterator must be a GPU iterator.
  // template <typename key_iterator>
  // std::pair<device_index_vector_type, device_index_vector_type>
  // find(key_iterator key_first, key_iterator key_last);

private:
  using base_type::m_byte_allocator;
  using base_type::m_coordinate_size;

  size_type m_hashtable_occupancy;
  size_type m_capacity;
  size_type m_size;
  hasher_type const m_hasher;
  key_equal_type const m_equal;
  key_type const m_unused_key;
  mapped_type const m_unused_element;
  index_storage_type m_valid_row_index;
  index_storage_type m_valid_map_index;
  index_storage_type m_inverse_row_index;
  index_storage_type m_device_tensor_stride;
  map_allocator_type m_map_allocator;
  std::shared_ptr<map_type> m_map;
};

template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator =
              detail::c10_allocator>
class CoordinateFieldMapGPU
    : public CoordinateMap<coordinate_field_type, TemplatedAllocator> {
  // Coordinate wrapper
public:
  using base_type = CoordinateMap<coordinate_field_type, TemplatedAllocator>;
  using coordinate_map_type =
      CoordinateMapGPU<coordinate_int_type, TemplatedAllocator>;
  using self_type =
      CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                            TemplatedAllocator>;
  using size_type = typename base_type::size_type;
  using index_type = typename base_type::index_type;
  using stride_type = typename base_type::stride_type;
  using byte_allocator_type = TemplatedAllocator<char>;
  using map_allocator_type = TemplatedAllocator<
      thrust::pair<coordinate<coordinate_int_type>, index_type>>;
  using kernel_map_type = gpu_kernel_map<index_type, byte_allocator_type>;

  // Map types
  using key_type = coordinate<coordinate_int_type>;
  using mapped_type = index_type;
  using hasher_type = detail::coordinate_murmur3<coordinate_int_type>;
  using key_equal_type = detail::coordinate_equal_to<coordinate_int_type>;
  using int_hash_map_type =
      concurrent_unordered_map<key_type,            // key
                               mapped_type,         // mapped_type
                               hasher_type,         // hasher
                               key_equal_type,      // equality
                               map_allocator_type>; // allocator
public:
  CoordinateFieldMapGPU() = delete;
  CoordinateFieldMapGPU(size_type const number_of_coordinates,
                        size_type const coordinate_size,
                        stride_type const &stride = {1},
                        byte_allocator_type alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, alloc),
        m_size(number_of_coordinates) {
    base_type::reserve(number_of_coordinates);
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   *
   * @return none
   */
  void insert(coordinate_field_type const *coordinate_begin,
              coordinate_field_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;
    base_type::allocate(N);
    // copy data directly to the ptr
    CUDA_CHECK(cudaMemcpy(base_type::coordinate_data(), coordinate_begin,
                          N * m_coordinate_size * sizeof(coordinate_field_type),
                          cudaMemcpyDeviceToDevice));
  }

  void copy_coordinates(coordinate_field_type *dst_coordinate) const {
    CUDA_CHECK(
        cudaMemcpy(dst_coordinate, base_type::const_coordinate_data(),
                   size() * m_coordinate_size * sizeof(coordinate_field_type),
                   cudaMemcpyDeviceToDevice));
  }

  void quantize_coordinates(coordinate_int_type *p_dst_coordinates,
                            stride_type const &tensor_stride) const;

  using base_type::const_coordinate_data;
  using base_type::coordinate_data;

  coordinate_map_type origin() const;

  kernel_map_type origin_map(coordinate_map_type const &origin_coordinate_map,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;

  inline size_type size() const noexcept { return m_size; }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateFieldMapGPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

private:
  using base_type::m_byte_allocator;
  using base_type::m_coordinate_size;
  size_type m_size;
};

} // namespace minkowski

#endif // COORDINATE_MAP_GPU_CUH
