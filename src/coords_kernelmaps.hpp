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

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map}
*/
template <uint8_t D, typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createInOutPerKernel(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const Arr<D, int> &in_tensor_strides, const Arr<D, int> &kernel_size,
    const Arr<D, int> &dilations, int region_type, at::Tensor offsets) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  int kernel_volume = 0;
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume<D>(region_type, kernel_size, offsets.size(0));

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
    Coord<D, Itype> out_coord;
    std::copy(p_out_coords + (D + 1) * i, p_out_coords + (D + 1) * (i + 1),
              out_coord.begin());
    auto pairs = region_neighbors<D, Itype>(
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

template <uint8_t D, typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createInOutPerKernelTranspose(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const Arr<D, int> &out_tensor_strides, const Arr<D, int> &kernel_size,
    const Arr<D, int> &dilations, int region_type, at::Tensor offsets) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  int kernel_volume;
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume<D>(region_type, kernel_size, offsets.size(0));
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
    Coord<D, Itype> in_coord;
    std::copy(p_in_coords + (D + 1) * i, p_in_coords + (D + 1) * (i + 1),
              in_coord.begin());
    auto pairs = region_neighbors<D, Itype>(
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

template <uint8_t D, typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createGlobalReductionInOutMap(
    const uint64_t in_coords_key, const uint64_t out_coords_key) {
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  std::map<Itype, Itype> in_out_map;
  Coord<D, Itype> coord;
  coord.fill(0);
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
                                  << ArrToString<Coord<D, Itype>>(coord));
    }
  }

  // Extract key value as in_out (ascending) ordered by the in map
  for (auto const &pair : in_out_map) {
    in_map[0].push_back(pair.first);
    out_map[0].push_back(pair.second);
  }

  return std::make_pair(std::move(in_map), std::move(out_map));
}

template <uint8_t D, typename Itype>
std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createPruningInOutMap(const uint64_t in_coords_key,
                                               const uint64_t out_coords_key) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      _coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      _coords_hashmaps[out_coords_key].map;
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  for (const auto &out_coord_iter : out_coords_hashmap) {
    auto out_coord = out_coord_iter.first;
    auto in_coord_iter = in_coords_hashmap.find(out_coord);
    if (in_coord_iter != in_coords_hashmap.end()) {
      in_map[0].push_back(in_coord_iter->second);
      out_map[0].push_back(out_coord_iter.second);
    }
  }
  return std::make_pair(std::move(in_map), std::move(out_map));
}
#endif
