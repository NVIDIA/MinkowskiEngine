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
std::pair<CoordsHashMap<D, Itype>, std::set<Itype>>
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
  coord.fill(0);  // if NxD array is given, fill the batch index to 0.
  Itype *p_coords = coords.data<Itype>();
  for (int i = 0; i < nrows; i++) {
      // TODO: BATCH_FIRST, ncols == D
      std::copy(p_coords, p_coords + ncols, coord.data());
      auto exists_iter = coords_hashmap.map.find(coord);

      // Track all batch indices
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

  // make the values into rvalues
  return std::make_pair(std::move(coords_hashmap), std::move(set_batch_indices));
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype> CoordsManager<D, Itype>::createOutCoordsHashMap(
    uint64_t coords_key, const Arr<D, int> &tensor_strides,
    const Arr<D, int> &strides) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key. "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);

  CoordsHashMap<D, Itype> *p_in_coords = &coords_hashmaps[coords_key];
  bool gt_one = true;
  for (auto s : strides)
    if (s != 1)
      gt_one = false;
  if (gt_one)
    return *p_in_coords;

  CoordsHashMap<D, Itype> out_coords;
  Arr<D, Itype> new_tensor_strides;
  int n_out = 0;
  for (int i = 0; i < D; i++)
    new_tensor_strides[i] = tensor_strides[i] * strides[i];
  for (const auto &in_pair : p_in_coords->map) {
    Coord<D, Itype> coord(in_pair.first);
    for (int j = 0; j < D; j++)
      coord[j] = int(floor(((float)coord[j]) / new_tensor_strides[j])) *
                 new_tensor_strides[j];
    if (out_coords.map.find(coord) == out_coords.map.end())
      out_coords.map[coord] = n_out++;
  }
  return std::move(out_coords);
}

/*
 * Coord map with the origin only
 */
template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype> CoordsManager<D, Itype>::createOriginCoordsHashMap() {
  CoordsHashMap<D, Itype> out_coord_map;
  Coord<D, Itype> coord;
  coord.fill(0);
  for (std::size_t i = 0; i < batch_indices.size(); ++i) {
#ifdef BATCH_FIRST
    coord[0] = batch_indices[i];
#else
    coord[D] = batch_indices[i];
#endif
    out_coord_map.map[coord] = i;
  }
  return std::move(out_coord_map);
}

/*
 * prune coords
 */
template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype>
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
  CoordsHashMap<D, Itype> *p_in_coords = &coords_hashmaps[coords_key];
  CoordsHashMap<D, Itype> out_coords;
  uint8_t *p_use_feat = use_feat.data<uint8_t>();
  int n_out = 0;
  for (const auto &in_pair : p_in_coords->map) {
    int n = in_pair.second;
    // We use it for the out coords
    if (p_use_feat[n] > 0) {
      if (p_use_feat[n] != 1)
        throw std::invalid_argument(
            Formatter() << "use_feat should be boolean. use_feat[n]: "
                        << std::to_string(p_use_feat[n]));
      Coord<D, Itype> coord(in_pair.first);
      if (out_coords.map.find(coord) == out_coords.map.end())
        out_coords.map[coord] = n_out++;
    }
  }
  return std::move(out_coords);
}

#endif
