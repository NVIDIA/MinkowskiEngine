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

template <uint8_t D, typename Itype>
std::tuple<CoordsHashMap<D, Itype>, std::set<Itype>, std::vector<Itype>>
CoordsManager<D, Itype>::createCoordsHashMap(at::Tensor coords) {
  // Find all unique batch indices
  std::set<Itype> set_batch_indices;
  int nrows = coords.size(0), ncols = coords.size(1);
  if (ncols != D + 1 && ncols != D)
    throw std::invalid_argument(
        Formatter() << "Dimension mismatch. The CoordsManager<" << D << "> "
                    << "cannot take size (" << nrows << ", " << ncols << ") "
                    << "tensor for coordinates.");

  CoordsHashMap<D, Itype> coords_hashmap;
  coords_hashmap.map.resize(nrows);
  Coord<D, Itype> coord;
  coord.fill(0); // if NxD array is given, fill the batch index to 0.
  Itype *p_coords = coords.data<Itype>();
  for (int i = 0; i < nrows; i++) {
    // TODO: BATCH_FIRST, ncols == D
    std::copy(p_coords, p_coords + ncols, coord.data());
    auto exists_iter = coords_hashmap.map.find(coord);

    // Track all batch indices for faster pooling
#ifdef BATCH_FIRST
    set_batch_indices.insert(p_coords[0]);
#else
    set_batch_indices.insert(p_coords[ncols - 1]);
#endif

    if (exists_iter == coords_hashmap.map.end()) {
      coords_hashmap.map[coord] = i;
    } else {
      throw std::invalid_argument(
          Formatter() << "A duplicate key found. Existing coord: "
                      << ArrToString<Coord<D, Itype>>(exists_iter->first)
                      << ", new coord: : "
                      << ArrToString<Coord<D, Itype>>(coord)
                      << ". If the duplication was intentional, use "
                         "initialize_coords_with_duplicates.");
    }
    p_coords += ncols;
  }

  // Copy all coordinates to a vector
  std::vector<Itype> vec_coords(nrows * ncols);
  vec_coords.resize(nrows * ncols);

  p_coords = coords.data<Itype>();
  std::copy(p_coords, p_coords + nrows * ncols, vec_coords.data());

  // make the values into rvalues
  return std::make_tuple(std::move(coords_hashmap),
                         std::move(set_batch_indices), std::move(vec_coords));
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <uint8_t D, typename Itype>
std::pair<CoordsHashMap<D, Itype>, std::vector<Itype>>
CoordsManager<D, Itype>::createOutCoordsHashCoordsPair(
    uint64_t coords_key, const Arr<D, int> &tensor_strides,
    const Arr<D, int> &strides) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key. "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);

  // Assert that all strides are non-identity.
  bool is_identity = true;
  for (auto s : strides)
    if (s != 1)
      is_identity = false;

  if (is_identity)
    throw std::invalid_argument(Formatter()
                                << "Creating new hash-coords pairs must be "
                                   "called when strides is non-identity. "
                                << "strides: " << ArrToString(strides) << " at "
                                << __FILE__ << ":" << __LINE__);

  // Define the new tensor strides
  Arr<D, Itype> new_tensor_strides;
  for (int i = 0; i < D; i++)
    new_tensor_strides[i] = tensor_strides[i] * strides[i];

  // Define output coordinates and hashmap
  CoordsHashMap<D, Itype> out_coords_hashmap;
  auto &in_coords = _coords_pairs[coords_key].second;
  // over estimate the size
  std::vector<Itype> out_coords(in_coords.size());
  out_coords.resize(in_coords.size());

  Itype *p_out_coord = out_coords.data();
  int n_out = 0, n_in = in_coords.size() / (D + 1);

  for (int i = 0; i < n_in; i++) {
    Coord<D, Itype> coord;
    Itype *p_in_coord = &in_coords[(D + 1) * i];
    for (int j = 0; j < D; j++)
      coord[j] = int(floor(((float)p_in_coord[j]) / new_tensor_strides[j])) *
                 new_tensor_strides[j];
    coord[D] = p_in_coord[D];
    if (out_coords_hashmap.map.find(coord) == out_coords_hashmap.map.end()) {
      out_coords_hashmap.map[coord] = n_out++;
      std::copy(coord.begin(), coord.end(), p_out_coord);
      p_out_coord += (D + 1);
    }
  }
  out_coords.resize(n_out * (D + 1));
  return std::make_pair(std::move(out_coords_hashmap), std::move(out_coords));
}

/*
 * Coord map with the origin only
 */
template <uint8_t D, typename Itype>
std::pair<CoordsHashMap<D, Itype>, std::vector<Itype>>
CoordsManager<D, Itype>::createOriginCoordsHashMap() {
  CoordsHashMap<D, Itype> out_coord_map;
  std::vector<Itype> out_coords((D + 1) * _batch_indices.size());
  out_coords.resize((D + 1) * _batch_indices.size());

  std::fill(out_coords.begin(), out_coords.end(), 0);

  Coord<D, Itype> coord;
  coord.fill(0);
  for (std::size_t i = 0; i < _batch_indices.size(); ++i) {
#ifdef BATCH_FIRST
    coord[0] = _batch_indices[i];
    out_coords[(D + 1) * i] = _batch_indices[i];
#else
    coord[D] = _batch_indices[i];
    out_coords[(D + 1) * i + D] = _batch_indices[i];
#endif
    out_coord_map.map[coord] = i;
  }
  return std::make_pair(std::move(out_coord_map), std::move(out_coords));
}

/*
 * prune coords
 */
template <uint8_t D, typename Itype>
std::pair<CoordsHashMap<D, Itype>, std::vector<Itype>>
CoordsManager<D, Itype>::createPrunedCoordsHashMap(uint64_t coords_key,
                                                   at::Tensor use_feat) {
  int n_coords = getCoordsSize(coords_key);
  if (n_coords != use_feat.numel())
    throw std::invalid_argument(
        Formatter()
        << "Number of coords and use_feat.numel() mismatch. coords_size: "
        << std::to_string(n_coords)
        << ", use_feat.numel(): " << std::to_string(use_feat.numel()) << " at "
        << __FILE__ << ":" << __LINE__);

  // use_feat.data<uint8_t>();
  CoordsHashMap<D, Itype> *p_in_coords = &_coords_hashmaps[coords_key];
  auto &in_coords = _coords_pairs[coords_key].second;
  CoordsHashMap<D, Itype> out_coords_hashmap;
  std::vector<Itype> out_coords(in_coords.size());
  out_coords.resize(in_coords.size());
  Itype *p_out_coords = out_coords.data();

  // int n_in = in_coords.size() / (D + 1);
  int n_out = 0;
  uint8_t *p_use_feat = use_feat.data<uint8_t>();

  for (const auto &in_pair : p_in_coords->map) {
    int n = in_pair.second;
    // We use it for the out coords
    if (p_use_feat[n] > 0) {
      Coord<D, Itype> coord(in_pair.first);
      if (out_coords_hashmap.map.find(coord) == out_coords_hashmap.map.end()) {
        out_coords_hashmap.map[coord] = n_out++;
        std::copy(coord.begin(), coord.end(), p_out_coords);
        p_out_coords += (D + 1);
      }
    }
  }
  return std::make_pair(std::move(out_coords_hashmap), std::move(out_coords));
}

#endif
