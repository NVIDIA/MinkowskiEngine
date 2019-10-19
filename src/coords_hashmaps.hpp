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
#ifndef COORDS_HASHMAPS
#define COORDS_HASHMAPS

#include "common.hpp"
#include "region_iter.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

long computeKernelVolume(int region_type, const std::vector<int> &kernel_size,
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

std::vector<int> computeOutTensorStride(const std::vector<int> &tensor_strides,
                                        const std::vector<int> &strides,
                                        bool is_transpose) {
  std::vector<int> out_tensor_strides;
  ASSERT(tensor_strides.size() == strides.size(),
         "The dimension of tensor_stride: ", ArrToString(tensor_strides),
         " does not match the dimension of strides: ", ArrToString(strides));
  for (int i = 0; i < strides.size(); i++) {
    if (is_transpose) {
      ASSERT(tensor_strides[i] % strides[i] == 0,
             "The output tensor stride is not divisible by ",
             "up_strides. tensor stride: ", ArrToString(tensor_strides),
             ", up_strides: ", ArrToString(strides));
      out_tensor_strides.push_back(tensor_strides[i] / strides[i]);
    } else
      out_tensor_strides.push_back(tensor_strides[i] * strides[i]);
  }
  return out_tensor_strides;
}

template <typename Itype>
std::tuple<CoordsHashMap<Itype>, std::set<Itype>, std::vector<Itype>>
CoordsManager<Itype>::createCoordsHashMap(at::Tensor coords) {
  // Find all unique batch indices
  std::set<Itype> batch_indices;
  int nrows = coords.size(0), ncols = coords.size(1);
  // int D = ncols - 1;

  CoordsHashMap<Itype> coords_hashmap;
  coords_hashmap.map.resize(nrows);

  Coord<Itype> coord(ncols);
  std::fill(coord.begin(), coord.end(), 0);

  Itype *p_coords = coords.data<Itype>();
  for (int i = 0; i < nrows; i++) {
    // TODO: BATCH_FIRST, ncols == D
    std::copy(p_coords, p_coords + ncols, coord.begin());
    auto exists_iter = coords_hashmap.map.find(coord);

    // Track all batch indices for faster pooling
#ifdef BATCH_FIRST
    batch_indices.insert(p_coords[0]);
#else
    batch_indices.insert(p_coords[ncols - 1]);
#endif

    ASSERT(exists_iter == coords_hashmap.map.end(),
           "A duplicate key found. Existing coord: ",
           ArrToString(exists_iter->first),
           ", new coord: : ", ArrToString(coord),
           ". If the duplication was intentional, use ",
           "initialize_coords_with_duplicates.");
    coords_hashmap.map[coord] = i;
    p_coords += ncols;
  }

  // Copy all coordinates to a vector
  std::vector<Itype> vec_coords(nrows * ncols);
  vec_coords.resize(nrows * ncols);

  p_coords = coords.data<Itype>();
  std::copy(p_coords, p_coords + nrows * ncols, vec_coords.data());

  // make the values into rvalues
  return std::make_tuple(std::move(coords_hashmap), std::move(batch_indices),
                         std::move(vec_coords));
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <typename Itype>
std::pair<CoordsHashMap<Itype>, std::vector<Itype>>
CoordsManager<Itype>::createOutCoordsHashCoordsPair(
    uint64_t coords_key, const std::vector<int> &tensor_strides,
    const std::vector<int> &strides) {
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         std::to_string(coords_key));

  int D = _coords_pairs[coords_key].first - 1;

  ASSERT(D == tensor_strides.size(), "The tensor_strides, ",
         ArrToString(tensor_strides),
         ", does not match the coordinate dimension, ", std::to_string(D), ".");

  ASSERT(D == strides.size(), "The strides, ", ArrToString(tensor_strides),
         ", does not match the coordinate dimension, ", std::to_string(D), ".");

  // Assert that all strides are non-identity.
  bool is_identity = true;
  for (auto s : strides)
    if (s != 1)
      is_identity = false;

  ASSERT(!is_identity, "Creating new hash-coords pairs must be ",
         "called for non-identity strides: ", ArrToString(strides));

  // Define the new tensor strides
  std::vector<Itype> new_tensor_strides(D);
  for (int i = 0; i < D; i++)
    new_tensor_strides[i] = tensor_strides[i] * strides[i];

  // Define output coordinates and hashmap
  CoordsHashMap<Itype> out_coords_hashmap;
  auto &in_coords = _coords_pairs[coords_key].second;
  // over estimate the size
  std::vector<Itype> out_coords(in_coords.size());

  Itype *p_out_coord = out_coords.data();
  int n_out = 0, n_in = in_coords.size() / (D + 1);

  for (int i = 0; i < n_in; i++) {
    Coord<Itype> coord(D + 1);
    Itype *p_in_coord = &in_coords[(D + 1) * i];
#ifdef BATCH_FIRST
    coord[0] = p_in_coord[0];
    for (int j = 1; j < D + 1; j++)
      coord[j] = int(floor(((float)p_in_coord[j]) / new_tensor_strides[j])) *
                 new_tensor_strides[j];
#else
    for (int j = 0; j < D; j++)
      coord[j] = int(floor(((float)p_in_coord[j]) / new_tensor_strides[j])) *
                 new_tensor_strides[j];
    coord[D] = p_in_coord[D];
#endif
    if (out_coords_hashmap.map.find(coord) == out_coords_hashmap.map.end()) {
      out_coords_hashmap.map[coord] = n_out++;
      std::copy(coord.begin(), coord.end(), p_out_coord);
      p_out_coord += (D + 1);
    }
  }
  out_coords.resize(n_out * (D + 1));
  return std::make_pair(std::move(out_coords_hashmap), std::move(out_coords));
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <typename Itype>
std::pair<CoordsHashMap<Itype>, std::vector<Itype>>
CoordsManager<Itype>::createTransposedOutCoordsHashCoordsPair(
    uint64_t coords_key, const std::vector<int> &tensor_strides,
    const std::vector<int> &strides, std::vector<int> kernel_sizes,
    std::vector<int> dilations, int region_type, at::Tensor offsets) {
  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         std::to_string(coords_key));

  int D = _coords_pairs[coords_key].first - 1;

  ASSERT(D == tensor_strides.size(), "The tensor_strides, ",
         ArrToString(tensor_strides),
         ", does not match the coordinate dimension, ", std::to_string(D), ".");

  ASSERT(D == strides.size(), "The strides, ", ArrToString(tensor_strides),
         ", does not match the coordinate dimension, ", std::to_string(D), ".");

  // Define the new tensor strides
  auto new_tensor_strides =
      computeOutTensorStride(tensor_strides, strides, true);

  // Define output coordinates and hashmap
  CoordsHashMap<Itype> out_coords_hashmap;
  auto &in_coords = _coords_pairs[coords_key].second;

  long kernel_volume =
      computeKernelVolume(region_type, kernel_sizes, offsets.size(0));

  // over estimate the size
  std::vector<Itype> out_coords(in_coords.size() * kernel_volume);

  Itype *p_out_coord = out_coords.data();
  int n_out = 0, n_in = in_coords.size() / (D + 1);

  // Get the input coordinates
  Itype *p_in_coords = _coords_pairs[coords_key].second.data();

  for (int i = 0; i < n_in; i++) {
    // For each i'th output coordinate, find the neighbors
    Coord<Itype> in_coord(D + 1);
    std::copy(p_in_coords + (D + 1) * i, p_in_coords + (D + 1) * (i + 1),
              in_coord.begin());

    auto region =
        Region<Itype>(in_coord, new_tensor_strides, kernel_sizes, dilations,
                      region_type, offsets.data<Itype>(), offsets.size(0));
    for (auto &point : region) {
      auto out_coord_iter = out_coords_hashmap.map.find(point);
      if (out_coord_iter == out_coords_hashmap.map.end()) {
        out_coords_hashmap.map[point] = n_out++;
        std::copy(point.begin(), point.end(), p_out_coord);
        p_out_coord += (D + 1);
      }
    }
  }
  out_coords.resize(n_out * (D + 1));
  return std::make_pair(std::move(out_coords_hashmap), std::move(out_coords));
}

/*
 * Coord map with the origin only
 */
template <typename Itype>
std::pair<CoordsHashMap<Itype>, std::vector<Itype>>
CoordsManager<Itype>::createOriginCoordsHashMap(int D) {
  CoordsHashMap<Itype> out_coord_map;

  std::vector<Itype> out_coords((D + 1) * _batch_indices.size());
  std::fill(out_coords.begin(), out_coords.end(), 0);

  Coord<Itype> coord(D + 1);
  std::fill(coord.begin(), coord.end(), 0);

  for (std::size_t i = 0; i < _batch_indices.size(); ++i) {
#ifdef BATCH_FIRST
    coord[0] = _batch_indices[i];
    out_coords[(D + 1) * i] = _batch_indices[i];
#else
    coord[D] = _batch_indices[i];
    out_coords[(D + 1) * i + D] = _batch_indices[i];
#endif
    // Iterating over a set. All guaranteed to be unique.
    out_coord_map.map[coord] = i;
  }
  // No need to resize since batch_indices are all unique.
  return std::make_pair(std::move(out_coord_map), std::move(out_coords));
}

/*
 * prune coords
 */
template <typename Itype>
std::pair<CoordsHashMap<Itype>, std::vector<Itype>>
CoordsManager<Itype>::createPrunedCoordsHashMap(uint64_t coords_key,
                                                at::Tensor use_feat) {
  int n_coords = getCoordsSize(coords_key);
  ASSERT(n_coords == use_feat.numel(),
         "Number of coords and use_feat.numel() mismatch. coords_size: ",
         std::to_string(n_coords),
         ", use_feat.numel(): ", std::to_string(use_feat.numel()));

  ASSERT(existsCoordsKey(coords_key),
         "The coord map doesn't exist for the given coords_key: ",
         std::to_string(coords_key), ".");

  int D = _coords_pairs[coords_key].first - 1;

  CoordsHashMap<Itype> *p_in_coords = &_coords_hashmaps[coords_key];
  auto &in_coords = _coords_pairs[coords_key].second;
  CoordsHashMap<Itype> out_coords_hashmap;
  // Overestimate the size
  std::vector<Itype> out_coords(in_coords.size());
  Itype *p_out_coords = out_coords.data();

  // int n_in = in_coords.size() / (D + 1);
  int n_out = 0;
  uint8_t *p_use_feat = use_feat.data<uint8_t>();

  for (const auto &in_pair : p_in_coords->map) {
    int n = in_pair.second;
    // We use it for the out coords
    if (p_use_feat[n] > 0) {
      Coord<Itype> coord(in_pair.first);
      if (out_coords_hashmap.map.find(coord) == out_coords_hashmap.map.end()) {
        out_coords_hashmap.map[coord] = n_out++;
        std::copy(coord.begin(), coord.end(), p_out_coords);
        p_out_coords += (D + 1);
      }
    }
  }
  out_coords.resize((D + 1) * n_out);
  return std::make_pair(std::move(out_coords_hashmap), std::move(out_coords));
}

#endif
