/*
 *  Copyright 2020 NVIDIA Corporation.
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
#ifndef COORDS_FUNCTORS_CUH
#define COORDS_FUNCTORS_CUH

namespace minkowski {

extern template <typename coordinate_type> struct coordinate;

namespace detail {

template <typename coordinate_type> struct coordinate_equal_to {
  coordinate_equal_to(size_t _coordinate_size)
      : coordinate_size(_coordinate_size) {}
  __host__ __device__ bool
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

// clang-format off
template <typename map_type, typename pair_type, typename coordinate_type>
struct insert_coordinate {
  insert_coordinate(map_type &_map, coordinate_type *_d_ptr, size_t _size)
      : map{_map}, d_ptr{_d_ptr}, size{_size} {}

  insert_coordinate(map_type &_map,
                    thrust::device_vector<coordinate_type> &coords_vec,
                    size_t _size)
      : map{_map}, d_ptr{thrust::raw_pointer_cast(coords_vec.data())}, size{_size} {}

  /*
   * @brief insert a <coordinate, row index> pair into the unordered_map
   *
   * @return thrust::pair<bool, uint32_t> of a success flag and the current
   * index.
   */
  __device__ thrust::pair<bool, uint32_t> operator()(uint32_t i) {
    auto coord = coordinate<coordinate_type>{&d_ptr[i * size]};
    pair_type pair = thrust::make_pair(coord, i);
    // Returns pair<iterator, (bool)insert_success>
    auto result = map.insert(pair);
    return thrust::make_pair<bool, uint32_t>(
        result.first != map.end() and result.second, i);
  }

  map_type &map;
  coordinate_type const *d_ptr;
  size_t const size;
};

template <typename map_type, typename pair_type, typename coordinate_type>
struct find_coordinate {
  using value_type = typename pair_type::second_type;
  find_coordinate(map_type const &_map, coordinate_type const *_d_ptr, size_t _size)
      : map{_map}, d_ptr{_d_ptr}, size{_size} {}

  __device__ value_type operator()(uint32_t i) {
    auto coord = coordinate<coordinate_type>{&d_ptr[i * size]};
    auto result = map.find(coord);
    if (result == map.end()) {
      return std::numeric_limits<value_type>::max();
    }
    return result->second;
  }

  map_type const &map;
  coordinate_type const *d_ptr;
  size_t const size;
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

  __host__ __device__ coordinate_murmur3(uint32_t _coordinate_size)
      : m_seed(0), coordinate_size(_coordinate_size),
        len(_coordinate_size * sizeof(coordinate_type)) {}

  __host__ __device__ uint32_t rotl32(uint32_t x, int8_t r) const {
    return (x << r) | (x >> (32 - r));
  }

  __host__ __device__ uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  __device__ result_type
  operator()(coordinate<coordinate_type> const &key) const {
    uint8_t const *data   = reinterpret_cast<uint8_t const *>(key.ptr);
    size_t const nblocks  = len / 4;
    result_type h1        = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    auto getblock32 = [] __host__ __device__(const uint32_t *p, int i) -> uint32_t {
    // Individual byte reads for unaligned accesses (very likely)
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Hashing in host code is not supported.");
#else
      auto q = (uint8_t const *)(p + i);
      return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
#endif
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

#endif // COORDS_FUNCTORS_CUH
