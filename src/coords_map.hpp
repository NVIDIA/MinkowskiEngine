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

#include <cmath>
#include <memory>
#include <set>
#include <tuple>

#include <robin_hood.h>

#include "primitives/small_vector.hpp"

#include "region.hpp"
#include "types.hpp"

namespace minkowski {

using std::reference_wrapper;
using std::set;
using std::tuple;
using std::vector;

template <typename Itype> struct byte_hash_vec {
  std::size_t operator()(vector<Itype> const &vec) const noexcept {
    return robin_hood::hash_bytes(vec.data(), sizeof(Itype) * vec.size());
  }
};

template <typename Itype>
inline vector<Itype> stride_copy(const vector<Itype> &src,
                                 const vector<Itype> &tensor_strides) noexcept {
  vector<Itype> dst{src.size()};
  const std::size_t size = tensor_strides.size();
#ifdef BATCH_FIRST
  constexpr int COORD_START = 1;
  dst[0] = src[0];
#else
  constexpr int COORD_START = 0;
  dst[size] = src[size];
#endif
  for (std::size_t i = 0; i < size; i++) {
    dst[i + COORD_START] =
        std::floor((float)src[i + COORD_START] / tensor_strides[i]) *
        tensor_strides[i];
  }
  return dst;
}

// clang-format off

// Coord specific types
using coordinate_type      = int32_t;
using CoordsMapVectorVType = small_vector<coordinate_type, 4>;
using CoordsToIndexMap     = robin_hood::unordered_flat_map<vector<coordinate_type>,
                                                            int32_t,
                                                            byte_hash_vec<coordinate_type>>;
using CoordsToVectorMap    = robin_hood::unordered_flat_map<vector<coordinate_type>,
                                                            CoordsMapVectorVType,
                                                            byte_hash_vec<coordinate_type>>;
/*
 * A wrapper for an unordered_map for coordinate management
 *
 * @note
 */
template <typename MapType = CoordsToIndexMap>
struct CoordsMap {
  using key_type   = typename MapType::key_type;
  using value_type = typename MapType::mapped_type;
  using map_type   = typename MapType;

  // Empty Constructors
  CoordsMap() {}
  CoordsMap(std::size_t ncols_, set<int> const &batch_indices);

  // Initializations
  vector<int> initialize(coordinate_type const *p_coords_,
                         std::size_t nrows_,
                         std::size_t ncols_,
                         bool force_remap = false);

  tuple<vector<int>, vector<int>, set<int>>
  initialize_batch(const int *p_coords_,
                   const int nrows_,
                   const int ncols_,
                   const bool force_remap = false,
                   const bool return_inverse = false);

  // Generate strided version of the input coordinate map.
  // returns mapping: out_coord row index to in_coord row index
  CoordsMap<MapType> stride(const vector<int> &tensor_strides) const;
  CoordsMap<MapType> stride_region(const Region &region) const;
  CoordsMap<MapType> prune(const bool *p_keep, int n) const;

  // class method
  static CoordsMap<MapType>
  union_coords(const vector<reference_wrapper<CoordsMap<MapType>>> &maps);

  // Generate in-out kernel maps
  InOutMapsPair<int>
  kernel_map(const CoordsMap<CoordsToIndexMap> &out_coords_map,
             const Region &region) const;
  InOutMapsPair<int>
  pruned_kernel_map(const CoordsMap<MapType> &out_coords_map) const;
  InOutMapsPair<int>
  global_reduction_map(const CoordsMap<MapType> &gout_coords_map,
                       bool return_per_batch = true) const;
  InOutMapsPair<int> stride_map(const CoordsMap &out_coords_map,
                                const vector<int> &tensor_strides) const;
  static InOutMapsPair<int>
  union_map(const vector<reference_wrapper<CoordsMap>> &in_maps,
            const CoordsMap &out_map);

  // Iterators
  typename map_type::iterator begin() { return map.begin(); }
  typename map_type::const_iterator begin() const { return map.begin(); }

  typename map_type::iterator end() { return map.end(); }
  typename map_type::const_iterator end() const { return map.end(); }

  typename map_type::iterator find(key_type const &key) { return map.find(key); }
  typename map_type::const_iterator find(key_type const &key) const { return map.find(key); }

  std::size_t size() const { return map.size(); }
  void reserve(std::size_t size) { map.reserve(size); }

  value_type &operator[](const vector<int> &coord) { return map[coord]; }

  void print() const;

  // members
  map_type map;
  int nrows, ncols;
};
// clang-format on

} // end namespace minkowski

#endif // robin coordsmap
