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

CoordsMap::CoordsMap(int ncols_, const set<int> &batch_indices)
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
    map[coord] = c++;
  }
}

vector<int> CoordsMap::initialize(const int *p_coords, const int nrows_,
                                  const int ncols_, const bool force_remap) {
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
      map[move(coord)] = force_remap ? c++ : i;
    }
  }

  return mapping;
}

pair<vector<int>, set<int>>
CoordsMap::initialize_batch(const int *p_coords, const int nrows_,
                            const int ncols_, const bool force_remap) {
  nrows = nrows_;
  ncols = ncols_;

  vector<int> mapping;
  set<int> batch_indices;

  mapping.reserve(nrows);
  map.reserve(nrows);

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

      map[move(coord)] = force_remap ? c++ : i;
    }
  }

  return make_pair(mapping, batch_indices);
}

CoordsMap CoordsMap::stride(const vector<int> &tensor_strides) const {
  ASSERT(tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  CoordsMap stride_map;
  stride_map.reserve(nrows);

  int c = 0;
  for (const auto &kv : map) {
    const auto strided_coord = stride_copy<int>(kv.first, tensor_strides);
    if (stride_map.find(strided_coord) == stride_map.end()) {
      stride_map[move(strided_coord)] = c++;
    }
  }
  stride_map.nrows = stride_map.size();
  stride_map.ncols = ncols;

  return stride_map;
}

CoordsMap CoordsMap::stride_region(const Region &region) const {
  ASSERT(region.tensor_strides.size() == ncols - 1, "Invalid tensor strides");

  CoordsMap stride_map;
  const int K = region.size();
  stride_map.reserve(nrows * K);

  Region cregion(region);
  int c = 0;
  for (const auto &kv : map) {
    cregion.set_bounds(kv.first);
    for (const auto &point : cregion) {
      if (stride_map.find(point) == stride_map.end()) {
        stride_map[point] = c++;
      }
    }
  }
  stride_map.ncols = ncols;
  stride_map.nrows = stride_map.size();

  return stride_map;
};

CoordsMap CoordsMap::prune(const bool *p_keep, int n) const {
  int c = 0;
  CoordsMap pruned_map;
  pruned_map.reserve(nrows);

  for (const auto &kv : map) {
    if (p_keep[kv.second]) {
      pruned_map[kv.first] = c++;
    }
  }
  pruned_map.ncols = ncols;
  pruned_map.nrows = pruned_map.size();

  return pruned_map;
}

CoordsMap
CoordsMap::union_coords(const vector<reference_wrapper<CoordsMap>> &maps) {
  const size_t num_tot = std::accumulate(
      maps.begin(), maps.end(), 0,
      [](size_t count, const CoordsMap &it) { return count + it.size(); });

  const auto max_iter = std::max_element(
      maps.begin(), maps.end(), [](const CoordsMap &a, const CoordsMap &b) {
        return a.size() < b.size();
      });
  const auto max_index = std::distance(maps.begin(), max_iter);

  // Initialize with the largest coords map.
  const CoordsMap& max_map = maps[max_index];
  CoordsMap out_map(max_map);
  out_map.reserve(num_tot);
  size_t c = max_map.size();
  for (size_t i = 0; i < maps.size(); ++i) {
    // Skip the map it was initialized with
    if (i == max_index)
      continue;

    const CoordsMap &map = maps[i];
    for (const auto &kv : map) {
      if (out_map.find(kv.first) == out_map.end()) {
        out_map[kv.first] = c++;
      }
    }
  }

  out_map.ncols = maps[0].get().ncols;
  out_map.nrows = out_map.size();

  return out_map;
}

InOutMapsPair<int> CoordsMap::kernel_map(const CoordsMap &out_coords_map,
                                         const Region &region) const {
  const int K = region.size();
  const int num_out = out_coords_map.size();

  InOutMaps<int> in_maps(K);
  InOutMaps<int> out_maps(K);
  for (int k = 0; k < K; k++) {
    in_maps[k].resize(num_out);
    out_maps[k].resize(num_out);
  }
  vector<int> num_used(K, 0);

  // OMP
  const auto &out_inner_map = out_coords_map.map;
  const size_t numElements =
      out_inner_map.calcNumElementsWithBuffer(out_inner_map.mask() + 1);

  // size_t stride = max((size_t)100, numElements / (2 *
  // omp_get_max_threads())); size_t N = (numElements + stride - 1) / stride;

  // compute the chunk size per thread.
  // There's a trade-off between the thread initialization overhead and the job
  // sizes. If some jobs finish earlier than others due to imbalance in hash
  // distribution, these threads will be idle.
  size_t N = 2 * omp_get_max_threads();
  const size_t stride = (numElements + N - 1) / N;
  N = (numElements + stride - 1) / stride;

  // When no need to iterate through the region
  // Put if outside the loop for speed
  if (region.region_type != 2 && K == 1) {
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      int curr_index;
      for (auto iter_out = out_inner_map.begin(stride * n);
           iter_out.num_steps() < std::min(stride, numElements - n * stride);
           ++iter_out) {

        const auto iter_map = map.find(iter_out->first);
        if (iter_map != map.end()) {
#pragma omp atomic capture
          curr_index = num_used[0]++;
          // In index
          in_maps[0][curr_index] = iter_map->second;
          // Out index
          out_maps[0][curr_index] = iter_out->second;
        }
      }
    }
  } else {
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      Region cregion(region);
      int kernel_ind, curr_index;
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
#pragma omp atomic capture
            curr_index = num_used[kernel_ind]++;
            // In index
            in_maps[kernel_ind][curr_index] = iter_map->second;
            // Out index
            out_maps[kernel_ind][curr_index] = iter_out->second;
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

InOutMapsPair<int>
CoordsMap::pruned_kernel_map(const CoordsMap &out_coords_map) const {
  InOutMaps<int> in_maps(1);
  InOutMaps<int> out_maps(1);

  in_maps.reserve(out_coords_map.size());
  out_maps.reserve(out_coords_map.size());

  for (const auto &out_kv : out_coords_map) {
    const auto &iter_map = map.find(out_kv.first);
    ASSERT(iter_map != map.end(), "Key not found: ", ArrToString(out_kv.first));
    in_maps[0].push_back(iter_map->second);
    out_maps[0].push_back(out_kv.second);
  }

  return make_pair(move(in_maps), move(out_maps));
}

InOutMapsPair<int>
CoordsMap::global_reduction_map(const CoordsMap &gout_coords_map,
                                bool return_per_batch) const {
#ifdef BATCH_FIRST
  constexpr int batch_index = 0;
#else
  constexpr int batch_index = ncols - 1;
#endif

  if (!return_per_batch) {
    // Combine all maps into one
    InOutMaps<int> in_maps(1);
    InOutMaps<int> out_maps(1);

    in_maps.reserve(map.size());
    out_maps.reserve(map.size());

    vector<int> coord(ncols, 0);
    for (const auto &kv : map) {
      coord[batch_index] = kv.first[batch_index];
      const auto &iter_out = gout_coords_map.find(coord);
      ASSERT(iter_out != gout_coords_map.end(),
             "Key not found: ", ArrToString(coord));
      in_maps[0].push_back(kv.second);
      out_maps[0].push_back(iter_out->second);
    }

    return make_pair(move(in_maps), move(out_maps));
  } else {
    // Return separate maps per batch
    const int batch_size = gout_coords_map.size();
    InOutMaps<int> in_maps(batch_size);
    InOutMaps<int> out_maps(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      in_maps[b].reserve(nrows / batch_size);
      out_maps[b].reserve(nrows / batch_size);
    }

    vector<int> coord(ncols, 0);
    for (const auto &kv : map) {
      coord[batch_index] = kv.first[batch_index];
      const auto &iter_out = gout_coords_map.find(coord);
      const int b = iter_out->second;
      ASSERT(iter_out != gout_coords_map.end(),
             "Key not found: ", ArrToString(coord));
      ASSERT(b < batch_size, "Invalid batch index: ", coord[batch_index],
             "max batch size: ", batch_size);
      in_maps[b].push_back(kv.second);
      out_maps[b].push_back(b);
    }

    return make_pair(move(in_maps), move(out_maps));
  }
}

InOutMapsPair<int>
CoordsMap::stride_map(const CoordsMap &out_coords_map,
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

    in_maps[0].push_back(kv.second);
    out_maps[0].push_back(iter_out->second);
  }

  return make_pair(move(in_maps), move(out_maps));
}

InOutMapsPair<int>
CoordsMap::union_map(const vector<reference_wrapper<CoordsMap>> &in_maps,
                     const CoordsMap &out_map) {
  const size_t num_in_maps = in_maps.size();
  InOutMaps<int> ins(num_in_maps);
  InOutMaps<int> outs(num_in_maps);
  for (size_t n = 0; n < num_in_maps; n++) {
    const size_t in_size = in_maps[n].get().size();
    ins[n].reserve(in_size);
    outs[n].reserve(in_size);
  }

#pragma omp parallel for
  for (size_t n = 0; n < num_in_maps; n++) {
    for (const auto &kv : (const CoordsMap &)in_maps[n]) {
      auto out_iter = out_map.find(kv.first);
      ASSERT(out_iter != out_map.end(), "Invalid out_map.");
      ins[n].push_back(kv.second);
      outs[n].push_back(out_iter->second);
    }
  }

  return make_pair(move(ins), move(outs));
}

void CoordsMap::print() const {
  for (const auto &kv : map) {
    std::cout << ArrToString(kv.first) << ":" << kv.second << "\n";
  }
  std::cout << std::flush;
}

} // end namespace minkowski
