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
#ifndef COORDS_KERNELMAPS
#define COORDS_KERNELMAPS

#include "common.hpp"
#include "region_iter.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

long ComputeKernelVolume(int region_type, const std::vector<int> &kernel_size,
                         int n_offset) {
  int kernel_volume;
  if (region_type == 0) { // Hypercube
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume *= k;
  } else if (region_type == 1) { // Hypercross
    kernel_volume = 1;
    for (auto k : kernel_size)
      kernel_volume += k - 1;
  } else if (region_type == 2) {
    kernel_volume = n_offset;
  } else {
    throw std::invalid_argument("Invalid region type");
  }
  return kernel_volume;
}

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map}
*/
template <typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<Itype>::createInOutPerKernel(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const std::vector<int> &in_tensor_strides,
    const std::vector<int> &kernel_size, const std::vector<int> &dilations,
    int region_type, at::Tensor offsets) {
  ASSERT(existsCoordsKey(in_coords_key) and existsCoordsKey(out_coords_key),
         "The coords map doesn't exist for the given coords_key. ",
         "in_coords_key: ", std::to_string(in_coords_key),
         ", out_coords_key: ", std::to_string(out_coords_key));

  int D = _coords_pairs[in_coords_key].first - 1;
  ASSERT(D == in_tensor_strides.size() and D == kernel_size.size() and
             D == dilations.size(),
         "The dimension mismatch. tensor_strides: ",
         ArrToString(in_tensor_strides),
         ", kernel_size: ", ArrToString(kernel_size),
         ", dilations: ", ArrToString(dilations));

  int kernel_volume = 0;
  _CoordsHashMap<Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume(region_type, kernel_size, offsets.size(0));

  // Assign array with out_coords_num elements. Then remove the unused parts
  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  std::vector<int> num_used(kernel_volume);
  num_used.resize(kernel_volume);
  std::fill(num_used.begin(), num_used.end(), 0);

  int n_out = out_coords_hashmap.size();
  for (int i = 0; i < kernel_volume; i++) {
    in_map[i].resize(n_out);
    out_map[i].resize(n_out);
  }

  // Get the output coordinates
  Itype *p_out_coords = _coords_pairs[out_coords_key].second.data();

  // Parallel kernel map
#pragma omp parallel for
  for (int i = 0; i < n_out; i++) {
    // For each i'th output coordinate, find the neighbors
    int index;
    Coord<Itype> out_coord(D + 1);
    std::copy(p_out_coords + (D + 1) * i, p_out_coords + (D + 1) * (i + 1),
              out_coord.begin());
    auto pairs = region_neighbors<Itype>(
        in_coords_hashmap, out_coord, in_tensor_strides, kernel_size, dilations,
        region_type, offsets.data<Itype>(), offsets.size(0));

    for (int j = 0; j < pairs.size() / 2; j++) {
      int kernel_index = pairs[j * 2 + 0];

#pragma omp atomic capture
      index = num_used[kernel_index]++;

      in_map[kernel_index][index] = pairs[j * 2 + 1];
      out_map[kernel_index][index] = i;
    }
  }

  for (int i = 0; i < kernel_volume; i++) {
    int max_num = num_used[i];
    in_map[i].resize(max_num);
    out_map[i].resize(max_num);
  }

  return std::make_pair(std::move(in_map), std::move(out_map));
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<Itype>::createInOutPerKernelTranspose(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const std::vector<int> &out_tensor_strides,
    const std::vector<int> &kernel_size, const std::vector<int> &dilations,
    int region_type, at::Tensor offsets) {
  ASSERT(existsCoordsKey(in_coords_key) and existsCoordsKey(out_coords_key),
         "The coords map doesn't exist for the given coords_key. ",
         "in_coords_key: ", std::to_string(in_coords_key),
         ", out_coords_key: ", std::to_string(out_coords_key));

  int D = _coords_pairs[in_coords_key].first - 1;
  ASSERT(D == out_tensor_strides.size() and D == kernel_size.size() and
             D == dilations.size(),
         "The dimension mismatch. tensor_strides: ",
         ArrToString(out_tensor_strides),
         ", kernel_size: ", ArrToString(kernel_size),
         ", dilations: ", ArrToString(dilations));

  int kernel_volume;
  _CoordsHashMap<Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume(region_type, kernel_size, offsets.size(0));
  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  std::vector<int> num_used(kernel_volume);
  num_used.resize(kernel_volume);
  std::fill(num_used.begin(), num_used.end(), 0);

  int n_in = in_coords_hashmap.size();
  for (int i = 0; i < kernel_volume; i++) {
    in_map[i].resize(n_in);
    out_map[i].resize(n_in);
  }

  // Get the input coordinates
  Itype *p_in_coords = _coords_pairs[in_coords_key].second.data();

  // Parallel kernel map
#pragma omp parallel for
  for (int i = 0; i < n_in; i++) {
    // For each i'th output coordinate, find the neighbors
    int index;
    Coord<Itype> in_coord(D + 1);
    std::copy(p_in_coords + (D + 1) * i, p_in_coords + (D + 1) * (i + 1),
              in_coord.begin());
    auto pairs = region_neighbors<Itype>(
        out_coords_hashmap, in_coord, out_tensor_strides, kernel_size,
        dilations, region_type, offsets.data<Itype>(), offsets.size(0));

    for (int j = 0; j < pairs.size() / 2; j++) {
      int kernel_index = pairs[j * 2 + 0];

#pragma omp atomic capture
      index = num_used[kernel_index]++;

      in_map[kernel_index][index] = i;
      out_map[kernel_index][index] = pairs[j * 2 + 1];
    }
  }

  for (int i = 0; i < kernel_volume; i++) {
    int max_num = num_used[i];
    in_map[i].resize(max_num);
    out_map[i].resize(max_num);
  }

  return std::make_pair(std::move(in_map), std::move(out_map));
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<Itype>::createGlobalReductionInOutMap(
    const uint64_t in_coords_key, const uint64_t out_coords_key) {
  ASSERT(existsCoordsKey(in_coords_key) and existsCoordsKey(out_coords_key),
         "The coords map doesn't exist for the given coords_key. ",
         "in_coords_key: ", std::to_string(in_coords_key),
         ", out_coords_key: ", std::to_string(out_coords_key));

  int D = _coords_pairs[in_coords_key].first - 1;

  _CoordsHashMap<Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;

  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  std::map<Itype, Itype> in_out_map;
  Coord<Itype> coord(D + 1);
  std::fill(coord.begin(), coord.end(), 0);

  for (auto const &in_coord_iter : in_coords_hashmap) {
#ifdef BATCH_FIRST
    coord[0] = in_coord_iter.first[0];
#else
    coord[D] = in_coord_iter.first[D];
#endif
    auto out_coord_iter = out_coords_hashmap.find(coord);
    if (out_coord_iter != out_coords_hashmap.end()) {
      // Order by the input coord row index
      in_out_map[in_coord_iter.second] = out_coord_iter->second;
    } else {
      throw std::invalid_argument(Formatter()
                                  << "A key not found in the out coord map"
                                  << ArrToString(coord));
    }
  }

  // Extract key value as in_out (ascending) ordered by the in map
  for (auto const &pair : in_out_map) {
    in_map[0].push_back(pair.first);
    out_map[0].push_back(pair.second);
  }

  return std::make_pair(std::move(in_map), std::move(out_map));
}

template <typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<Itype>::createPruningInOutMap(const uint64_t in_coords_key,
                                            const uint64_t out_coords_key) {
  ASSERT(existsCoordsKey(in_coords_key) and existsCoordsKey(out_coords_key),
         "The coords map doesn't exist for the given coords_key. ",
         "in_coords_key: ", std::to_string(in_coords_key),
         ", out_coords_key: ", std::to_string(out_coords_key));

  int n_coords = getCoordsSize(out_coords_key);

  _CoordsHashMap<Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;

  int i = 0;
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  in_map[0].resize(n_coords);
  out_map[0].resize(n_coords);
  for (const auto &out_coord_iter : out_coords_hashmap) {
    auto out_coord = out_coord_iter.first;
    auto in_coord_iter = in_coords_hashmap.find(out_coord);
    if (in_coord_iter != in_coords_hashmap.end()) {
      in_map[0][i] = in_coord_iter->second;
      out_map[0][i] = out_coord_iter.second;
      i++;
    }
  }
  return std::make_pair(std::move(in_map), std::move(out_map));
}
#endif
