/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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

namespace minkowski {

using std::array;
using std::pair;
using std::vector;

// D-Dimensional coordinate + batch dimension = D + 1
template <typename Itype> using Stride = vector<Itype>;

// For hashing kernel sizes, strides, and dilations.
template <uint8_t D, typename Itype> using Arr = array<Itype, D>;

// unordered map key type
template <typename Itype> struct Coord {
  Itype *ptr;
  int size;

  Coord(){};
  Coord(Itype *ptr_, int size_) : ptr(ptr_), size(size_){};

  bool operator==(const Coord &other) const {
    bool equal = size == other.size;
    int i = 0;
    while (equal && i < size) {
      equal &= ptr[i] == other.ptr[i];
      i++;
    }
    return equal;
  };

  Itype *data() { return ptr; }
  Itype operator[](const int index) const { return ptr[index]; }
};

template <typename Itype> struct pVector {
  Itype *ptr_;
  int size_;

  pVector(Itype *ptr, int size) : ptr_(ptr), size_(size) {}
  int size() const { return size_; };
  Itype *data() { return ptr_; };
  const Itype *data() const { return ptr_; };
};

// Key for InOutMap
// (in_coords_key, out_coords_key, stride hash, kernel size, dilation,
// is_transpose, is_pool)
using InOutMapKey = array<uint64_t, 8>;

// Input index to output index mapping for each spatial kernel
template <typename Itype> using InOutMaps = vector<vector<Itype>>;

// Input index to output index mapping in ptr, sise pair
// Used for device pointer and size
template <typename Itype> using pInOutMaps = vector<pVector<Itype>>;

template <typename Itype>
using InOutMapsPair = pair<InOutMaps<Itype>, InOutMaps<Itype>>;

template <typename Itype>
using pInOutMapsPair = pair<pInOutMaps<Itype>, pInOutMaps<Itype>>;

template <typename Itype>
using InOutMapsRefPair = pair<InOutMaps<Itype> &, InOutMaps<Itype> &>;

template <typename Itype>
using pInOutMapsRefPair = pair<pInOutMaps<Itype> &, pInOutMaps<Itype> &>;

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

template <uint8_t D, typename Itype> struct ArrHash {
  uint64_t operator()(Arr<D, Itype> const &p) const {
    return hash_vec<Arr<D, Itype>>(p);
  }
};

} // end namespace minkowski

#endif // TYPES
