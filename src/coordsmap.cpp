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
#include <iostream>
#include <numeric>
#include <omp.h>

#include "coordsmap.hpp"

namespace minkowski {

namespace detail {
template <typename MapType>
inline void Assign(MapType &map, const vector<int> &coord, int value);
template <>
inline void Assign(CoordsToIndexMap &map, const vector<int> &coord, int value) {
  map[coord] = value;
}
template <>
inline void Assign(CoordsToVectorMap &map, const vector<int> &coord,
                   int value) {
  map[coord].push_back(value);
}
inline void Assign(CoordsMap<CoordsToIndexMap> &map, const vector<int> &coord,
                   int value) {
  map[coord] = value;
}
template <>
inline void Assign(CoordsMap<CoordsToVectorMap> &map, const vector<int> &coord,
                   int value) {
  map[coord].push_back(value);
}

template <typename MapType>
inline void MoveAssign(MapType &map, vector<int> &coord, int value);
template <>
inline void MoveAssign(CoordsToIndexMap &map, vector<int> &coord, int value) {
  map[move(coord)] = value;
}
template <>
inline void MoveAssign(CoordsToVectorMap &map, vector<int> &coord, int value) {
  map[move(coord)].push_back(value);
}
template <>
inline void MoveAssign(CoordsMap<CoordsToIndexMap> &map, vector<int> &coord,
                       int value) {
  map[move(coord)] = value;
}
template <>
inline void MoveAssign(CoordsMap<CoordsToVectorMap> &map, vector<int> &coord,
                       int value) {
  map[move(coord)].push_back(value);
}

template <typename MapType, typename VType>
inline void PrunedAssign(CoordsMap<MapType> &map, const vector<int> &coord,
                         const bool *p_keep, const VType &keep_index,
                         int &value);
template <>
inline void PrunedAssign<CoordsToIndexMap, int>(
    CoordsMap<CoordsToIndexMap> &map, const vector<int> &coord,
    const bool *p_keep, const int &keep_index, int &value) {
  if (p_keep[keep_index])
    map[coord] = value++;
}
template <>
inline void PrunedAssign<CoordsToVectorMap, CoordsMapVectorVType>(
    CoordsMap<CoordsToVectorMap> &map, const vector<int> &coord,
    const bool *p_keep, const CoordsMapVectorVType &keep_index, int &value) {
  for (const auto &k : keep_index) {
    if (p_keep[k])
      map[coord].push_back(value++);
  }
}

template <typename VType> inline int GetFirst(const VType &value);
template <> inline int GetFirst<int>(const int &value) { return value; }
template <>
inline int GetFirst<CoordsMapVectorVType>(const CoordsMapVectorVType &value) {
  return value[0];
}

template <typename VTypeIn, typename VTypeOut>
inline int sub_kernel_size(const VTypeIn &in_val,
                                  const VTypeOut &out_val);
template <>
inline int sub_kernel_size<int, int>(const int &in_val,
                                            const int &out_val) {
  return 1;
}
template <>
inline int
sub_kernel_size<CoordsMapVectorVType, int>(const CoordsMapVectorVType &in_val,
                                           const int &out_val) {
  return in_val.size();
}
template <>
inline int sub_kernel_size<int, CoordsMapVectorVType>(
    const int &in_val, const CoordsMapVectorVType &out_val) {
  return out_val.size();
}
template <>
inline int sub_kernel_size<CoordsMapVectorVType, CoordsMapVectorVType>(
    const CoordsMapVectorVType &in_val, const CoordsMapVectorVType &out_val) {
  return in_val.size() * out_val.size();
}

template <typename VTypeIn, typename VTypeOut>
inline void PopulateSubKernelMap(vector<int> &in_map, const VTypeIn &in_index,
                                 vector<int> &out_map,
                                 const VTypeOut &out_index);
template <>
inline void
PopulateSubKernelMap<int, int>(vector<int> &in_map, const int &in_index,
                               vector<int> &out_map, const int &out_index) {
  in_map.push_back(in_index);
  out_map.push_back(out_index);
}

template <>
inline void PopulateSubKernelMap<CoordsMapVectorVType, int>(
    vector<int> &in_map, const CoordsMapVectorVType &in_indices,
    vector<int> &out_map, const int &out_index) {
  for (auto &in_index : in_indices) {
    in_map.push_back(in_index);
    out_map.push_back(out_index);
  }
}

template <>
inline void PopulateSubKernelMap<int, CoordsMapVectorVType>(
    vector<int> &in_map, const int &in_index, vector<int> &out_map,
    const CoordsMapVectorVType &out_indices) {
  for (auto &out_index : out_indices) {
    in_map.push_back(in_index);
    out_map.push_back(out_index);
  }
}

template <>
inline void PopulateSubKernelMap<CoordsMapVectorVType, CoordsMapVectorVType>(
    vector<int> &in_map, const CoordsMapVectorVType &in_indices,
    vector<int> &out_map, const CoordsMapVectorVType &out_indices) {
  for (auto &in_index : in_indices) {
    for (auto &out_index : out_indices) {
      in_map.push_back(in_index);
      out_map.push_back(out_index);
    }
  }
}

/**
 * Kernel mpa update function.
 *
 * in_map, out_map must be resized accordingly to accomodate the indices.
 */
template <typename VTypeIn, typename VTypeOut>
inline void PopulateSubKernelMap(vector<int> &in_map, const VTypeIn &in_index,
                                 vector<int> &out_map,
                                 const VTypeOut &out_index, int curr);
template <>
inline void PopulateSubKernelMap<int, int>(vector<int> &in_map,
                                           const int &in_index,
                                           vector<int> &out_map,
                                           const int &out_index, int curr) {
#ifdef DEBUG
  ASSERT(in_map.size() > curr, "Invalid map size. Current size", in_map.size(),
         "smaller than the pointed index", curr);
#endif
  in_map[curr] = in_index;
  out_map[curr] = out_index;
}

template <>
inline void PopulateSubKernelMap<CoordsMapVectorVType, int>(
    vector<int> &in_map, const CoordsMapVectorVType &in_indices,
    vector<int> &out_map, const int &out_index, int curr) {
  for (auto &in_index : in_indices) {
#ifdef DEBUG
    ASSERT(in_map.size() > curr, "Invalid map size. Current size",
           in_map.size(), "smaller than the pointed index", curr);
#endif
    in_map[curr] = in_index;
    out_map[curr] = out_index;
    curr++;
  }
}

template <>
inline void PopulateSubKernelMap<int, CoordsMapVectorVType>(
    vector<int> &in_map, const int &in_index, vector<int> &out_map,
    const CoordsMapVectorVType &out_indices, int curr) {
  for (auto &out_index : out_indices) {
#ifdef DEBUG
    ASSERT(in_map.size() > curr, "Invalid map size. Current size",
           in_map.size(), "smaller than the pointed index", curr);
#endif
    in_map[curr] = in_index;
    out_map[curr] = out_index;
    curr++;
  }
}

template <>
inline void PopulateSubKernelMap<CoordsMapVectorVType, CoordsMapVectorVType>(
    vector<int> &in_map, const CoordsMapVectorVType &in_indices,
    vector<int> &out_map, const CoordsMapVectorVType &out_indices, int curr) {
  for (auto &in_index : in_indices) {
    for (auto &out_index : out_indices) {
#ifdef DEBUG
      ASSERT(in_map.size() > curr, "Invalid map size. Current size",
             in_map.size(), "smaller than the pointed index", curr);
#endif
      in_map[curr] = in_index;
      out_map[curr] = out_index;
      curr++;
    }
  }
}

template <typename VType>
inline void kernel_push_back(vector<int> &map, const VType &vals);
template <> void kernel_push_back(vector<int> &map, const int &val) {
  map.push_back(val);
}
template <>
inline void kernel_push_back(vector<int> &map,
                             const CoordsMapVectorVType &vals) {
  for (auto val : vals) {
    map.push_back(val);
  }
}

} // namespace detail

template <typename MapType>
CoordsMap<MapType>::CoordsMap(int ncols_, const set<int> &batch_indices)
    : nrows(batch_indices.size()), ncols(ncols_) {
  int c = 0;
  map.reserve(nrows);
  vector<int> coord(ncols, 0);

  for (int b : batch_indices) {
    // Create a key
#ifdef BATCH_FIRST
    coord[0] = b;
#else
    coord[ncols - 1] = b;
#endif
    // Add to the map
    detail::Assign(map, coord, c++);
  }
}

template <typename MapType>
vector<int> CoordsMap<MapType>::initialize(const int *p_coords,
                                           const int nrows_, const int ncols_,
                                           const bool force_remap) {
  nrows = nrows_;
  ncols = ncols_;

  vector<int> mapping;

  mapping.reserve(nrows);
  map.reserve(nrows);

  int c = 0;
  for (int i = 0; i < nrows; i++) {
    vector<int> coord(ncols);
    std::copy_n(p_coords + i * ncols, ncols, coord.data());

    if (map.find(coord) == map.end()) {
      mapping.push_back(i);
      detail::MoveAssign(map, coord, force_remap ? c++ : i);
    }
  }

  nrows = map.size();
  return mapping;
}

template <typename MapType>
tuple<vector<int>, vector<int>, set<int>>
CoordsMap<MapType>::initialize_batch(const int *p_coords, const int nrows_,
                                     const int ncols_, const bool force_remap,
                                     const bool return_inverse) {
  nrows = nrows_;
  ncols = ncols_;

  vector<int> mapping, inverse_mapping;
  set<int> batch_indices;

  mapping.reserve(nrows);
  map.reserve(nrows);

  // index, inverse_index = initialize_with_inverse(coords)
  // unique_coords = coords[index]
  // coords == unique_coords[inverse_index]
  // coords == coords[index[inverse_index]]
  if (return_inverse) {
    inverse_mapping.reserve(nrows);
    int c = 0;
    for (int i = 0; i < nrows; i++) {
      vector<int> coord(ncols);
      std::copy_n(p_coords + i * ncols, ncols, coord.data());

      auto iter = map.find(coord);
      if (iter == map.end()) {
        mapping.push_back(i);
        inverse_mapping.push_back(c);
#ifdef BATCH_FIRST
        batch_indices.insert(coord[0]);
#else
        batch_indices.insert(coord[ncols - 1]);
#endif
        detail::MoveAssign(map, coord, c++);
      } else {
        inverse_mapping.push_back(detail::GetFirst(iter->second));
      }
    }
  } else {
    int c = 0;
    for (int i = 0; i < nrows; i++) {
      vector<int> coord(ncols);
      std::copy_n(p_coords + i * ncols, ncols, coord.data());

      if (map.find(coord) == map.end()) {
#ifdef BATCH_FIRST
        batch_indices.insert(coord[0]);
#else
        batch_indices.insert(coord[ncols - 1]);
#endif
        mapping.push_back(i);

        detail::MoveAssign(map, coord, force_remap ? c++ : i);
      }
    }
  }

  nrows = map.size();
  return make_tuple(mapping, inverse_mapping, batch_indices);
}

template <typename MapType>
CoordsMap<MapType>
CoordsMap<MapType>::stride(const vector<int> &tensor_strides) const {
  ASSERT(tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  CoordsMap<MapType> stride_map;
  stride_map.reserve(nrows);

  int c = 0;
  for (const auto &kv : map) {
    vector<int> strided_coord = stride_copy<int>(kv.first, tensor_strides);
    if (stride_map.find(strided_coord) == stride_map.end()) {
      detail::MoveAssign(stride_map, strided_coord, c++);
    }
  }
  stride_map.nrows = stride_map.size();
  stride_map.ncols = ncols;

  return stride_map;
}

template <typename MapType>
CoordsMap<MapType>
CoordsMap<MapType>::stride_region(const Region &region) const {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  CoordsMap<MapType> stride_map;
  const int K = region.size();
  stride_map.reserve(nrows * K);

  Region cregion(region);
  int c = 0;
  for (const auto &kv : map) {
    cregion.set_bounds(kv.first);
    for (const auto &point : cregion) {
      if (stride_map.find(point) == stride_map.end()) {
        detail::Assign(stride_map, point, c++);
      }
    }
  }
  stride_map.ncols = ncols;
  stride_map.nrows = stride_map.size();

  return stride_map;
};

template <typename MapType>
CoordsMap<MapType> CoordsMap<MapType>::prune(const bool *p_keep, int n) const {
  int c = 0;
  CoordsMap<MapType> pruned_map;
  pruned_map.reserve(nrows);
  for (const auto &kv : map) {
    detail::PrunedAssign(pruned_map, kv.first, p_keep, kv.second, c);
  }
  pruned_map.ncols = ncols;
  pruned_map.nrows = pruned_map.size();

  return pruned_map;
}

template <typename MapType>
CoordsMap<MapType> CoordsMap<MapType>::union_coords(
    const vector<reference_wrapper<CoordsMap<MapType>>> &maps) {
  const size_t num_tot =
      std::accumulate(maps.begin(), maps.end(), 0,
                      [](size_t count, const CoordsMap<MapType> &it) {
                        return count + it.size();
                      });

  const auto max_iter = std::max_element(
      maps.begin(), maps.end(),
      [](const CoordsMap<MapType> &a, const CoordsMap<MapType> &b) {
        return a.size() < b.size();
      });
  const auto max_index = std::distance(maps.begin(), max_iter);

  // Initialize with the largest coords map.
  const CoordsMap<MapType> &max_map = maps[max_index];
  CoordsMap<MapType> out_map(max_map);
  out_map.reserve(num_tot);
  size_t c = max_map.size();
  for (size_t i = 0; i < maps.size(); ++i) {
    // Skip the map it was initialized with
    if (i == max_index)
      continue;

    const CoordsMap<MapType> &map = maps[i];
    for (const auto &kv : map) {
      if (out_map.find(kv.first) == out_map.end()) {
        detail::Assign(out_map, kv.first, c++);
      }
    }
  }

  out_map.ncols = maps[0].get().ncols;
  out_map.nrows = out_map.size();

  return out_map;
}

template <typename MapType> // Current class map type
InOutMapsPair<int> CoordsMap<MapType>::kernel_map(
    const CoordsMap<CoordsToIndexMap> &out_coords_map,
    const Region &region) const {
  const int K = region.size();
  const int num_out = out_coords_map.size();

  InOutMaps<int> in_maps(K);
  InOutMaps<int> out_maps(K);
  for (int k = 0; k < K; k++) {
    // TODO update this for all functions with MapType CoordsToVectorMap
    in_maps[k].resize(num_out);
    out_maps[k].resize(num_out);
  }
  vector<int> num_used(K, 0);

  // OMP
  const auto &out_inner_map = out_coords_map.map;
  const size_t numElements =
      out_inner_map.calcNumElementsWithBuffer(out_inner_map.mask() + 1);

  // size_t stride = max((size_t)100, numElements / (2 *
  // omp_get_max_threads())); size_t N = (numElements + stride - 1) /
  // stride;

  // compute the chunk size per thread.
  // There's a trade-off between the thread initialization overhead and the
  // job sizes. If some jobs finish earlier than others due to imbalance in
  // hash distribution, these threads will be idle.
  size_t N = 2 * omp_get_max_threads();
  const size_t stride = (numElements + N - 1) / N;
  N = (numElements + stride - 1) / stride;

  // When no need to iterate through the region
  // Put if outside the loop for speed
  if (region.region_type != 2 && K == 1) {
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      int curr_index_begin;
      for (auto iter_out = out_inner_map.begin(stride * n);
           iter_out.num_steps() < std::min(stride, numElements - n * stride);
           ++iter_out) {

        const auto iter_map = map.find(iter_out->first);
        if (iter_map != map.end()) {
          const int n_submaps =
              detail::sub_kernel_size(iter_map->second, iter_out->second);
#pragma omp atomic capture
          {
            curr_index_begin = num_used[0];
            num_used[0] += n_submaps;
          }

          // Ensure that in_maps and out_maps are resized accordingly
          detail::PopulateSubKernelMap(in_maps[0], iter_map->second,
                                       out_maps[0], iter_out->second,
                                       curr_index_begin);
        }
      }
    }
  } else {
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      Region cregion(region);
      int kernel_ind, curr_index_begin;
      for (auto iter_out = out_inner_map.begin(stride * n);
           iter_out.num_steps() < std::min(stride, numElements - n * stride);
           ++iter_out) {

        // set the bounds for the current region
        cregion.set_bounds(iter_out->first);

        // For elements in the current region
        kernel_ind = 0;
        for (const auto &point : cregion) {
          // If the input coord exists
          const auto iter_map = map.find(point);
          if (iter_map != map.end()) {
            const int n_submaps =
                detail::sub_kernel_size(iter_map->second, iter_out->second);
#pragma omp atomic capture
            {
              curr_index_begin = num_used[kernel_ind];
              num_used[kernel_ind] += n_submaps;
            }
            // Ensure that in_maps and out_maps are resized accordingly
            detail::PopulateSubKernelMap(in_maps[kernel_ind], iter_map->second,
                                         out_maps[kernel_ind], iter_out->second,
                                         curr_index_begin);
          }
          // Post processings
          kernel_ind++;
        }
      }
    }
  }

  for (int i = 0; i < K; i++) {
    int max_num = num_used[i];
    in_maps[i].resize(max_num);
    out_maps[i].resize(max_num);
  }

  return make_pair(move(in_maps), move(out_maps));
}

template <typename MapType>
InOutMapsPair<int> CoordsMap<MapType>::pruned_kernel_map(
    const CoordsMap<MapType> &out_coords_map) const {
  InOutMaps<int> in_maps(1);
  InOutMaps<int> out_maps(1);

  in_maps.reserve(out_coords_map.size());
  out_maps.reserve(out_coords_map.size());

  for (const auto &out_kv : out_coords_map) {
    const auto &iter_map = map.find(out_kv.first);
    ASSERT(iter_map != map.end(), "Key not found: ", ArrToString(out_kv.first));
    detail::kernel_push_back(in_maps[0], iter_map->second);
    detail::kernel_push_back(out_maps[0], out_kv.second);
  }

  return make_pair(move(in_maps), move(out_maps));
}

template <typename MapType>
InOutMapsPair<int> CoordsMap<MapType>::global_reduction_map(
    const CoordsMap<MapType> &gout_coords_map, bool return_per_batch) const {
#ifdef BATCH_FIRST
  constexpr int batch_index = 0;
#else
  constexpr int batch_index = ncols - 1;
#endif

  if (!return_per_batch) {
    // Combine all maps into one
    InOutMaps<int> in_maps(1);
    InOutMaps<int> out_maps(1);

    // TODO: map size should be map.size() * CoordsMapVectorVType size.
    in_maps.reserve(map.size());
    out_maps.reserve(map.size());

    vector<int> coord(ncols, 0);
    for (const auto &kv : map) {
      coord[batch_index] = kv.first[batch_index];
      const auto &iter_out = gout_coords_map.find(coord);
      ASSERT(iter_out != gout_coords_map.end(),
             "Key not found: ", ArrToString(coord));
      detail::kernel_push_back(in_maps[0], kv.second);
      detail::kernel_push_back(out_maps[0], iter_out->second);
    }

    return make_pair(move(in_maps), move(out_maps));
  } else {
    // Return separate maps per batch
    const int batch_size = gout_coords_map.size();
    InOutMaps<int> in_maps(batch_size);
    InOutMaps<int> out_maps(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      // TODO: map size should be map.size() * CoordsMapVectorVType size.
      in_maps[b].reserve(nrows / batch_size);
      out_maps[b].reserve(nrows / batch_size);
    }

    vector<int> coord(ncols, 0);
    for (const auto &kv : map) {
      coord[batch_index] = kv.first[batch_index];
      const auto &iter_out = gout_coords_map.find(coord);
      const int b = detail::GetFirst(iter_out->second);
      ASSERT(iter_out != gout_coords_map.end(),
             "Key not found: ", ArrToString(coord));
      ASSERT(b < batch_size, "Invalid batch index: ", coord[batch_index],
             "max batch size: ", batch_size);
      detail::kernel_push_back(in_maps[b], kv.second);
      detail::kernel_push_back(out_maps[b], b);
    }

    return make_pair(move(in_maps), move(out_maps));
  }
}

template <typename MapType>
InOutMapsPair<int>
CoordsMap<MapType>::stride_map(const CoordsMap<MapType> &out_coords_map,
                               const vector<int> &tensor_strides) const {
  InOutMaps<int> in_maps(1);
  InOutMaps<int> out_maps(1);

  in_maps.reserve(map.size());
  out_maps.reserve(map.size());

  for (const auto &kv : map) {
    const auto strided_coord = stride_copy<int>(kv.first, tensor_strides);
    const auto &iter_out = out_coords_map.find(strided_coord);
    ASSERT(iter_out != out_coords_map.end(),
           "Key not found: ", ArrToString(strided_coord));

    detail::PopulateSubKernelMap(in_maps[0], kv.second, out_maps[0],
                                 iter_out->second);
  }

  return make_pair(move(in_maps), move(out_maps));
}

template <typename MapType>
InOutMapsPair<int> CoordsMap<MapType>::union_map(
    const vector<reference_wrapper<CoordsMap<MapType>>> &in_maps,
    const CoordsMap<MapType> &out_map) {
  const size_t num_in_maps = in_maps.size();
  InOutMaps<int> ins(num_in_maps);
  InOutMaps<int> outs(num_in_maps);
  for (size_t n = 0; n < num_in_maps; n++) {
    const size_t in_size = in_maps[n].get().size();
    // TODO: update the reserve size
    ins[n].reserve(in_size);
    outs[n].reserve(in_size);
  }

#pragma omp parallel for
  for (size_t n = 0; n < num_in_maps; n++) {
    for (const auto &kv : (const CoordsMap<MapType> &)in_maps[n]) {
      auto out_iter = out_map.find(kv.first);
      ASSERT(out_iter != out_map.end(), "Invalid out_map.");
      detail::PopulateSubKernelMap(ins[n], kv.second, outs[n],
                                   out_iter->second);
    }
  }

  return make_pair(move(ins), move(outs));
}

template <typename MapType> void CoordsMap<MapType>::print() const {
  for (const auto &kv : map) {
    std::cout << ArrToString(kv.first) << ":" << kv.second << "\n";
  }
  std::cout << std::flush;
}

template struct CoordsMap<CoordsToIndexMap>;
template struct CoordsMap<CoordsToVectorMap>;

} // end namespace minkowski
