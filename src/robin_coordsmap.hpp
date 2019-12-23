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
#ifndef ROBIN_COORDSMAP
#define ROBIN_COORDSMAP

#include <memory>
#include <set>

#include "robin_hood.h"

#include "region.hpp"
#include "types.hpp"

template <typename Itype> struct byte_hash_vec {
  size_t operator()(std::vector<Itype> const &vec) const noexcept {
    return robin_hood::hash_bytes(vec.data(), sizeof(Itype) * vec.size());
  }
};

template <typename Itype>
vector<int> stride_copy(const vector<Itype> &src,
                        const vector<Itype> &tensor_strides) {
  vector<Itype> dst(src.size());
#ifdef BATCH_FIRST
  int b = 1;
  dst[0] = src[0];
#else
  int b = 0;
  dst[tensor_strides.size()] = src[tensor_strides.size()];
#endif

  for (int i = 0; i < tensor_strides.size(); i++) {
    if (src[i + b] > 0)
      dst[i + b] = (src[i + b] / tensor_strides[i]) * tensor_strides[i];
    else
      dst[i + b] = ((src[i + b] - tensor_strides[i] + 1) / tensor_strides[i]) *
                   tensor_strides[i];
  }
  return dst;
}

using CoordsInnerMap =
    robin_hood::unordered_flat_map<std::vector<int>, int, byte_hash_vec<int>>;

class CoordsMap {
private:
  CoordsInnerMap map;

public:
  int nrows, ncols;

  // Constructors
  CoordsMap() {}
  // batch indices must be arranged in ascending order for coordsmap.map
  CoordsMap(int ncols_, const set<int> &batch_indices);

  // Initializations
  pair<vector<int>, set<int>> initialize(int *p_coords_, int nrows_, int ncols_,
                                         bool force_remap = false);

  // Generate strided version of the input coordinate map.
  // returns mapping: out_coord row index to in_coord row index
  CoordsMap stride(const vector<int> &tensor_strides) const;
  CoordsMap stride_region(const Region &region) const;
  CoordsMap prune(bool *p_keep, int n) const;

  // class method
  static CoordsMap union_coords(const vector<shared_ptr<CoordsMap>> &p_maps);

  // Generate in-out kernel maps
  InOutMapsPair<int> kernel_map(const CoordsMap &out_coords_map,
                                const Region &region) const;
  InOutMapsPair<int> pruned_kernel_map(const CoordsMap &out_coords_map) const;
  InOutMapsPair<int> global_reduction_map(const CoordsMap &gout_coords_map,
                                          bool return_per_batch = true) const;
  InOutMapsPair<int> stride_map(const CoordsMap &out_coords_map,
                                const vector<int> &tensor_strides) const;
  static InOutMapsPair<int>
  union_map(const vector<shared_ptr<CoordsMap>> &p_in_maps,
            const CoordsMap &out_map);

  // Iterators
  CoordsInnerMap::iterator begin() { return map.begin(); }
  CoordsInnerMap::const_iterator begin() const { return map.begin(); }
  CoordsInnerMap::iterator end() { return map.end(); }
  CoordsInnerMap::const_iterator end() const { return map.end(); }
  CoordsInnerMap::iterator find(const vector<int> &key) {
    return map.find(key);
  }
  CoordsInnerMap::const_iterator find(const vector<int> &key) const {
    return map.find(key);
  }

  size_t size() const { return map.size(); }
  void reserve(size_t size) { map.reserve(size); }

  int &operator[](const vector<int> &coord) { return map[coord]; }

  void print() const;
};

#endif
