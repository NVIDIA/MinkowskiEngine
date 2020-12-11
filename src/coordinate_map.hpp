/* Copyright (c) 2020 NVIDIA CORPORATION.
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
#ifndef COORDINATE_MAP_HPP
#define COORDINATE_MAP_HPP

#include "coordinate.hpp"
#include "kernel_region.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include <robin_hood.h>

namespace minkowski {

namespace detail {

/*
template <typename Itype> struct byte_hash_vec {
  std::size_t operator()(std::vector<Itype> const &vec) const noexcept {
    return robin_hood::hash_bytes(vec.data(), sizeof(Itype) * vec.size());
  }
};
*/

/*
 * @note assume that `src`, `dst`, and `stride` are initialized correctly.
 */
template <typename Itype>
inline void
stride_coordinate(const coordinate<Itype> &src, std::vector<Itype> &dst,
                  const default_types::stride_type &stride) noexcept {
  dst[0] = src[0];
  for (default_types::index_type i = 0; i < stride.size(); ++i) {
    dst[i + 1] = std::floor((float)src[i + 1] / stride[i]) * stride[i];
  }
}

template <typename Itype, typename stride_type>
inline void stride_coordinate(const coordinate<Itype> &src,
                              std::vector<Itype> &dst,
                              const stride_type stride) noexcept {
  dst[0] = src[0];
  for (default_types::index_type i = 0; i < dst.size() - 1; ++i) {
    dst[i + 1] = std::floor((float)src[i + 1] / stride[i]) * stride[i];
  }
}

inline default_types::stride_type
stride_tensor_stride(const default_types::stride_type &tensor_stride,
                     const default_types::stride_type &stride,
                     bool is_transpose = false) {
  ASSERT(tensor_stride.size() == stride.size(), "stride size mismatch.");
  default_types::stride_type strided_tensor_stride{tensor_stride};
  if (is_transpose) {
    for (default_types::size_type i = 0; i < tensor_stride.size(); ++i) {
      ASSERT(strided_tensor_stride[i] % stride[i] == 0,
             "Invalid up stride on tensor stride:", tensor_stride,
             "kernel stride:", stride);
      strided_tensor_stride[i] /= stride[i];
    }
  } else {
    for (default_types::size_type i = 0; i < tensor_stride.size(); ++i)
      strided_tensor_stride[i] *= stride[i];
  }
  return strided_tensor_stride;
}

} // namespace detail

/*
 * @brief A wrapper for a coordinate map.
 *
 * @note
 */
// clang-format off
template <typename coordinate_type, template <typename T> class TemplatedAllocator>
class CoordinateMap {

public:
  using self_type   = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using index_type  = default_types::index_type;
  using size_type   = default_types::size_type;
  using stride_type = default_types::stride_type;

  // return types
  using index_vector_type   = std::vector<default_types::index_type>;
  using index_set_type      = std::set<default_types::index_type>;

  using byte_allocator_type = TemplatedAllocator<char>;


  // Constructors
  CoordinateMap() = delete;
  CoordinateMap(size_type const number_of_coordinates,
                size_type const coordinate_size,
                stride_type const &stride = {1},
                byte_allocator_type alloc = byte_allocator_type())
      : m_coordinate_size(coordinate_size),
        m_capacity(0), /* m_capacity is updated in the allocate function */
        m_tensor_stride(stride), m_byte_allocator(alloc) {
    allocate(number_of_coordinates);
    expand_tensor_stride();
    LOG_DEBUG("tensor stride:", m_tensor_stride);
  }

  /*
   * @brief given a key iterator begin-end pair and a value iterator begin-end
   * pair, insert all elements.
   */
  template <typename key_iterator, typename mapped_iterator>
  void insert(key_iterator key_first, key_iterator key_last,
              mapped_iterator value_first, mapped_iterator value_last) {
    ASSERT(false, "Not implemented"); // no virtual members for a templated class
  }

  /*
   * @brief Generate a new set of coordinates with the provided strides.
   *
   * @return a coordinate map with specified tensor strides * current tensor
   * stride.
   */
  self_type stride(stride_type const &tensor_strides) const {
    ASSERT(false, "Not implemented"); // no virtual members for a templated class
  }

  // clang-format on

  coordinate_type *coordinate_data() { return m_coordinates.get(); }
  coordinate_type const *const_coordinate_data() const {
    return m_coordinates.get();
  }

  void reserve(size_type size) {
    if (m_capacity < size) {
      LOG_DEBUG("Reserve coordinates:", size, "current capacity:", m_capacity);
      allocate(size);
    }
  }

  std::string to_string() const;

  stride_type const &get_tensor_stride() const noexcept {
    return m_tensor_stride;
  }

  inline size_type capacity() const noexcept { return m_capacity; }

  inline size_type coordinate_size() const noexcept {
    return m_coordinate_size;
  }

protected:
  // clang-format off
  void allocate(size_type const number_of_coordinates) {
    if (m_capacity < number_of_coordinates) {
      LOG_DEBUG("Allocate", number_of_coordinates, "coordinates.");
      auto const size = number_of_coordinates * m_coordinate_size;
      m_coordinates = allocate_ptr(size);
      m_capacity = number_of_coordinates;
    }
  }

  // clang-format on
  std::shared_ptr<coordinate_type[]> allocate_ptr(size_type const size) {
    coordinate_type *ptr = reinterpret_cast<coordinate_type *>(
        m_byte_allocator.allocate(size * sizeof(coordinate_type)));

    auto deleter = [](coordinate_type *p, byte_allocator_type alloc,
                      size_type size) {
      alloc.deallocate(reinterpret_cast<char *>(p), size);
    };

    return std::shared_ptr<coordinate_type[]>{
        ptr, std::bind(deleter, std::placeholders::_1, m_byte_allocator,
                       size * sizeof(coordinate_type))};
  }

private:
  /*
   * @brief expand the m_tensor_stride to m_coordinate_size - 1 if it has 1.
   */
  void expand_tensor_stride() {
    if (m_tensor_stride.size() == 1) {
      for (size_type i = 0; i < m_coordinate_size - 2; ++i) {
        m_tensor_stride.push_back(m_tensor_stride[0]);
      }
    }
    ASSERT(m_tensor_stride.size() == m_coordinate_size - 1,
           "Invalid tensor stride", m_tensor_stride);
  }

protected:
  // members
  size_type m_number_of_coordinates;
  size_type m_coordinate_size;
  size_type m_capacity;
  stride_type m_tensor_stride;

  byte_allocator_type m_byte_allocator;
  std::shared_ptr<coordinate_type[]> m_coordinates;
};

} // end namespace minkowski

#endif // COORDINATE_MAP_HPP
