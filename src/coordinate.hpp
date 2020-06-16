/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
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
#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#include "utils.hpp"

namespace minkowski {

// The size of a coordinate is defined in the equality functor, and the hash
// functor.
template <typename coordinate_type> class coordinate {
  using self_type = coordinate<coordinate_type>;

public:
  // Constructors
  coordinate() = delete;

  MINK_CUDA_HOST_DEVICE inline coordinate(self_type &other)
      : m_ptr(other.m_ptr) {}
  MINK_CUDA_HOST_DEVICE inline coordinate(self_type const &other)
      : m_ptr(other.m_ptr) {}
  MINK_CUDA_HOST_DEVICE inline coordinate(coordinate_type const *ptr)
      : m_ptr{ptr} {}

  // helper functions
  MINK_CUDA_HOST_DEVICE inline coordinate_type const *data() const {
    return m_ptr;
  }
  MINK_CUDA_HOST_DEVICE inline coordinate_type operator[](uint32_t i) const {
    return m_ptr[i];
  }
  MINK_CUDA_HOST_DEVICE inline void data(coordinate_type *ptr) { m_ptr = ptr; }

private:
  coordinate_type const *m_ptr {nullptr};
};

template <typename T> struct coordinate_print_functor {
  inline coordinate_print_functor(size_t _coordinate_size)
      : m_coordinate_size(_coordinate_size) {}

  std::string operator()(coordinate<T> const &v) {
    Formatter out;
    auto actual_delim = ", ";
    auto delim = "";
    out << '[';
    for (auto i = 0; i < m_coordinate_size; ++i) {
      out << delim << v[i];
      delim = actual_delim;
    }
    out << "]";
    return out;
  }

  size_t const m_coordinate_size;
};

/*
 * @brief iterator wrapper for a coordinate pointer
 */
template <typename coordinate_type> class coordinate_iterator {
public:
  // clang-format off
  using self_type         = coordinate_iterator<coordinate_type>;
  using size_type         = int32_t;

  // iterator traits
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = coordinate<coordinate_type>;
  using difference_type   = int32_t;
  using pointer           = coordinate<coordinate_type>*;
  using reference         = coordinate<coordinate_type>&;
  // clang-format on

public:
  coordinate_iterator() = delete;
  MINK_CUDA_HOST_DEVICE coordinate_iterator(coordinate_type const *ptr,
                                            size_type const coordinate_size,
                                            difference_type const steps = 0)
      : m_ptr{ptr}, m_coordinate_size{coordinate_size},
        m_coordinate{m_ptr}, m_steps{steps} {}

  // reference operator*();
  // pointer   operator->();
  MINK_CUDA_HOST_DEVICE inline reference operator*() noexcept {
    m_coordinate = value_type{m_ptr + m_coordinate_size * m_steps};
    return m_coordinate;
  }
  MINK_CUDA_HOST_DEVICE inline pointer operator->() noexcept {
    m_coordinate = value_type{m_ptr + m_coordinate_size * m_steps};
    return &m_coordinate;
  }

  // this_type& operator++();
  // this_type  operator++(int);
  MINK_CUDA_HOST_DEVICE inline self_type &operator++() noexcept {
    ++m_steps;
    return *this;
  }
  MINK_CUDA_HOST_DEVICE inline self_type operator++(int) noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps + 1};
  }

  // this_type& operator--();
  // this_type  operator--(int);
  MINK_CUDA_HOST_DEVICE inline self_type &operator--() noexcept {
    --m_steps;
    return *this;
  }
  MINK_CUDA_HOST_DEVICE inline self_type operator--(int) noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps - 1};
  }

  // this_type &operator+=(difference_type n);
  // this_type &operator-=(difference_type n);
  MINK_CUDA_HOST_DEVICE inline self_type &
  operator+=(difference_type n) noexcept {
    m_steps += n;
    return *this;
  }
  MINK_CUDA_HOST_DEVICE inline self_type &
  operator-=(difference_type n) noexcept {
    m_steps -= n;
    return *this;
  }

  // this_type operator+(difference_type n) const;
  // this_type operator-(difference_type n) const;
  MINK_CUDA_HOST_DEVICE inline self_type
  operator+(difference_type n) const noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps + n};
  }
  MINK_CUDA_HOST_DEVICE inline self_type
  operator-(difference_type n) const noexcept {
    return self_type{m_ptr, m_coordinate_size, m_steps - n};
  }

  MINK_CUDA_HOST_DEVICE inline size_type coordinate_size() const noexcept {
    return m_coordinate_size;
  }

  MINK_CUDA_HOST_DEVICE inline difference_type
  operator-(self_type const &other) const noexcept {
    return m_steps - other.m_steps;
  }

  MINK_CUDA_HOST_DEVICE inline bool
  operator==(self_type const &other) const noexcept {
    return current_position() == other.current_position();
  }

  MINK_CUDA_HOST_DEVICE inline bool
  operator!=(self_type const &other) const noexcept {
    return current_position() != other.current_position();
  }

private:
  inline coordinate_type const *current_position() const noexcept {
    return m_ptr + m_coordinate_size * m_steps;
  }

private:
  coordinate_type const *m_ptr{nullptr};
  size_type const m_coordinate_size = 0;
  value_type m_coordinate;
  difference_type m_steps = 0;
};

/*
 * @brief range wrapper for coordinates
 */
template <typename coordinate_type> class coordinate_range {
  using iterator = coordinate_iterator<coordinate_type>;
  using size_type = typename coordinate_iterator<coordinate_type>::size_type;

public:
  coordinate_range() = delete;
  coordinate_range(size_type const number_of_coordinates,
                   size_type const coordinate_size, coordinate_type const *ptr)
      : m_ptr(ptr), m_coordinate_size(coordinate_size),
        m_number_of_coordinates(number_of_coordinates) {}

  iterator begin() const { return iterator(m_ptr, m_coordinate_size); }
  iterator end() const {
    return iterator(m_ptr, m_coordinate_size, m_number_of_coordinates);
  }
  size_type size() { return m_number_of_coordinates; }

private:
  coordinate_type const *m_ptr{nullptr};
  size_type const m_coordinate_size = 0;
  size_type const m_number_of_coordinates = 0;
  size_type m_steps = 0;
};

namespace detail {

template <typename coordinate_type> struct coordinate_equal_to {
  MINK_CUDA_HOST_DEVICE inline coordinate_equal_to(size_t _coordinate_size)
      : coordinate_size(_coordinate_size) {}
  MINK_CUDA_HOST_DEVICE inline bool
  operator()(coordinate<coordinate_type> const &lhs,
             coordinate<coordinate_type> const &rhs) const {
    if ((lhs.data() == nullptr) and (rhs.data() == nullptr))
      return true;
    if ((lhs.data() == nullptr) xor (rhs.data() == nullptr))
      return false;
    for (size_t i = 0; i < coordinate_size; i++) {
      if (lhs[i] != rhs[i])
        return false;
    }
    return true;
  }

  size_t coordinate_size;
};

/*******************************************************************************
 * The following section uses a different license.
 ******************************************************************************/
/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MurmurHash3_32 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.

/**
 * @brief Specialization of MurmurHash3_32 operator for bytes.
 */
// clang-format off
template <typename coordinate_type> struct coordinate_murmur3 {
  using result_type = uint32_t;

  MINK_CUDA_HOST_DEVICE inline coordinate_murmur3(uint32_t _coordinate_size)
      : m_seed(0), coordinate_size(_coordinate_size),
        len(_coordinate_size * sizeof(coordinate_type)) {}

  MINK_CUDA_HOST_DEVICE inline uint32_t rotl32(uint32_t x, int8_t r) const {
    return (x << r) | (x >> (32 - r));
  }

  MINK_CUDA_HOST_DEVICE inline uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  MINK_CUDA_HOST_DEVICE result_type
  operator()(coordinate<coordinate_type> const &key) const {
    uint8_t const *data   = reinterpret_cast<uint8_t const *>(key.data());
    size_t const nblocks  = len / 4;
    result_type h1        = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    auto getblock32 = [] MINK_CUDA_HOST_DEVICE (const uint32_t *p, int i) -> uint32_t {
    // Individual byte reads for unaligned accesses (very likely)
      auto q = (uint8_t const *)(p + i);
      return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
    };

    //----------
    // body
    uint32_t const *const blocks = (uint32_t const *)(data + nblocks * 4);
    for (size_t i = -nblocks; i; i++) {
      uint32_t k1 = getblock32(blocks, i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    uint8_t const *tail = (uint8_t const *)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

private:
  uint32_t m_seed;
  uint32_t coordinate_size;
  int len;
};

} // end namespace detail

} // end namespace minkowski

#endif // end COORDINATE_HPP
