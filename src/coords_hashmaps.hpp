#ifndef COORDS_HASHMAPS
#define COORDS_HASHMAPS

#include "common.hpp"
#include "region_iter.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype>
CoordsManager<D, Itype>::createCoordsHashMap(at::Tensor coords) {
  int nrows = coords.size(0), ncols = coords.size(1);
  CoordsHashMap<D, Itype> coords_hashmap;
  coords_hashmap.map.resize(nrows);
  Coord<D, Itype> coord;
  Itype *p_coords = coords.data<Itype>();
  for (int i = 0; i < nrows; i++) {
    std::copy(&p_coords[i * ncols], &p_coords[(i + 1) * ncols], coord.data());
    auto exists_iter = coords_hashmap.map.find(coord);
    if (exists_iter == coords_hashmap.map.end()) {
      coords_hashmap.map[std::move(coord)] = i;
    } else {
      throw std::invalid_argument(
          Formatter() << "A duplicate key found. Existing coord: "
                      << ArrToString<Coord<D, Itype>>(exists_iter->first)
                      << ", new coord: : "
                      << ArrToString<Coord<D, Itype>>(coord)
                      << ". If the duplication was intentional, use "
                         "initialize_coords_with_duplicates.");
    }
  }
  return coords_hashmap;
}

/**
  Given the input coordinate to index map, kernel size, stride, and dilation,
  compute the output coordinates and corresponding index.

  is_transpose is not used as we assume that the unpooled coords should
  correspond to one of the existing coord maps.
*/
template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype>
CoordsManager<D, Itype>::createOutCoordsHashMap(uint64_t coords_key,
                                                const Arr<D, int> &pixel_dists,
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
  Arr<D, Itype> new_pixel_dists;
  int n_out = 0;
  for (int i = 0; i < D; i++)
    new_pixel_dists[i] = pixel_dists[i] * strides[i];
  for (const auto &in_pair : p_in_coords->map) {
    Coord<D, Itype> coord(in_pair.first);
    for (int j = 0; j < D; j++)
      coord[j] = int(floor(((float)coord[j]) / new_pixel_dists[j])) *
                 new_pixel_dists[j];
    if (out_coords.map.find(coord) == out_coords.map.end())
      out_coords.map[coord] = n_out++;
  }
  return out_coords;
}

/*
 * Coord map with the origin only
 */
template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype>
CoordsManager<D, Itype>::createOriginCoordsHashMap(uint64_t coords_key,
                                                   int batch_size) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key. "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);
  CoordsHashMap<D, Itype> &in_coords = coords_hashmaps[coords_key];
  CoordsHashMap<D, Itype> out_coord_map;
  int n_out = 0;
  // When batch size is not given (0, negative), go over all values
  if (batch_size < 1) {
    // Order all batch indices first
    std::map<Itype, Itype> batch_indices;
    for (auto in_pair : in_coords.map) {
      Coord<D, Itype> coord(in_pair.first);
      batch_indices[coord[D]] = 0; // Insert a new batch index
    }
    // Once we collected all batch indices, insert it into the map
    Coord<D, Itype> coord;
    for (int j = 0; j < D; j++)
      coord[j] = 0;
    for (const auto &i : batch_indices) {
      coord[D] = i.first;
      out_coord_map.map[coord] = n_out++;
    }
  } else {
    for (int b = 0; b < batch_size; b++) {
      Coord<D, Itype> coord;
      for (int j = 0; j < D; j++)
        coord[j] = 0;
      coord[D] = b;
      out_coord_map.map[coord] = b;
    }
  }
  return out_coord_map;
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
  return out_coords;
}

#endif
