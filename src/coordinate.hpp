/* Copyright (c) NVIDIA CORPORATION.
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
template <typename coordinate_type> struct coordinate {
  MINK_CUDA_HOST_DEVICE inline coordinate() {}
  MINK_CUDA_HOST_DEVICE inline coordinate(coordinate_type const *_ptr)
      : ptr{_ptr} {}

  coordinate_type const *data() { return ptr; }
  MINK_CUDA_HOST_DEVICE inline coordinate_type operator[](uint32_t i) const {
    return ptr[i];
  }

  coordinate_type const *ptr;
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

namespace detail {

template <typename coordinate_type> struct coordinate_equal_to {
  MINK_CUDA_HOST_DEVICE inline coordinate_equal_to(size_t _coordinate_size)
      : coordinate_size(_coordinate_size) {}
  MINK_CUDA_HOST_DEVICE inline bool
  operator()(coordinate<coordinate_type> const &lhs,
             coordinate<coordinate_type> const &rhs) const {
    if ((lhs.ptr == nullptr) and (rhs.ptr == nullptr))
      return true;
    if ((lhs.ptr == nullptr) xor (rhs.ptr == nullptr))
      return false;
    for (size_t i = 0; i < coordinate_size; i++) {
      if (lhs[i] != rhs[i])
        return false;
    }
    return true;
  }

  size_t const coordinate_size;
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
    uint8_t const *data   = reinterpret_cast<uint8_t const *>(key.ptr);
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
