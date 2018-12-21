#include "common.hpp"
#include "kernel_region.hpp"
#include "utils.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <uint8_t D>
std::vector<int> computeOutPixelDist(const Arr<D, int> &pixel_dists,
                                     const Arr<D, int> &strides,
                                     bool is_transpose) {
  std::vector<int> out_pixel_dists;
  for (int i = 0; i < D; i++) {
    if (is_transpose) {
      if (pixel_dists[i] % strides[i] > 0)
        throw std::invalid_argument(
            Formatter() << "The output pixel dist is not divisible by "
                           "up_strides. pixel dists: "
                        << ArrToString(pixel_dists)
                        << ", up_strides: " << ArrToString(strides));
      out_pixel_dists.push_back(pixel_dists[i] / strides[i]);
    } else
      out_pixel_dists.push_back(pixel_dists[i] * strides[i]);
  }
  return out_pixel_dists;
}

template <uint8_t D>
long ComputeKernelVolume(int region_type, const Arr<D, int> &kernel_size,
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

template <uint8_t D, typename Itype>
uint64_t CoordsManager<D, Itype>::getCoordsKey(const Arr<D, int> &pixel_dists) {
  auto pixel_dist_hash = hash_vec<Arr<D, int>>(pixel_dists);
  uint64_t r_coords_key = 0;

  // Following lines are from INITIALIZE_IN_COORDS
  /* Prioritize the p_coords_key */
  if (coords_hashmaps.find(pixel_dist_hash) != coords_hashmaps.end()) {
    r_coords_key = pixel_dist_hash;
  } else {
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given pixel dists "
                    << "pixel_dist: " << ArrToString(pixel_dists) << " at "
                    << __FILE__ << ":" << __LINE__);
  }

  return r_coords_key;
}

template <uint8_t D, typename Itype>
bool CoordsManager<D, Itype>::existsCoordsKey(uint64_t coords_key) {
  bool exist = false;
  // Following lines are from INITIALIZE_IN_COORDS
  /* Prioritize the p_coords_key */
  if (coords_hashmaps.find(coords_key) != coords_hashmaps.end())
    exist = true;
  return exist;
}

template <uint8_t D, typename Itype>
bool CoordsManager<D, Itype>::existsCoordsKey(py::object py_coords_key) {
  PyCoordsKey<D> *p_coords_key = py_coords_key.cast<PyCoordsKey<D> *>();
  return existsCoordsKey(p_coords_key->getKey());
}

template <uint8_t D, typename Itype>
int CoordsManager<D, Itype>::getCoordsSize(uint64_t coords_key) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);
  return coords_hashmaps[coords_key].size();
}

template <uint8_t D, typename Itype>
int CoordsManager<D, Itype>::getCoordsSize(py::object py_coords_key) {
  PyCoordsKey<D> *p_coords_key = py_coords_key.cast<PyCoordsKey<D> *>();
  return getCoordsSize(p_coords_key->getKey());
}

template <uint8_t D, typename Itype>
CoordsHashMap<D, Itype>
CoordsManager<D, Itype>::createCoordsHashMap(at::Tensor coords) {
  int nrows = coords.size(0), ncols = coords.size(1);
  CoordsHashMap<D, Itype> coords_hashmap;
  coords_hashmap.map.resize(nrows);
  Coord<D, Itype> coord;
  Itype *loc = coords.data<Itype>();
  for (int i = 0; i < nrows; i++) {
    std::copy(&loc[i * ncols], &loc[(i + 1) * ncols], coord.data());
    if (coords_hashmap.map.find(coord) == coords_hashmap.map.end()) {
      coords_hashmap.map[std::move(coord)] = i;
    } else {
      std::cerr << "Duplicate key found. Use initialize_coords_with_duplicates "
                << "or remove duplicates";
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
  for (auto in_pair : p_in_coords->map) {
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
  Coord<D, Itype> coord;
  int n_out = 0;
  // When batch size is not given (0, negative), go over all values
  if (batch_size < 1) {
    for (auto in_pair : in_coords.map) {
      Coord<D, Itype> coord(in_pair.first);
      for (int j = 0; j < D; j++)
        coord[j] = 0;
      if (out_coord_map.map.find(coord) == out_coord_map.map.end())
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

template <uint8_t D, typename Itype>
uint64_t CoordsManager<D, Itype>::initializeCoords(at::Tensor coords,
                                                   py::object py_coords_key) {
  PyCoordsKey<D> *p_coords_key = py_coords_key.cast<PyCoordsKey<D> *>();
  uint64_t in_coords_key = initializeCoords(coords, p_coords_key->pixel_dists_);
  p_coords_key->setKey(in_coords_key);
  return in_coords_key;
}

template <uint8_t D, typename Itype>
uint64_t
CoordsManager<D, Itype>::initializeCoords(at::Tensor coords,
                                          const Arr<D, int> &pixel_dists) {
  uint64_t pixel_dist_hash = hash_vec<Arr<D, int>>(pixel_dists);
  if (coords_hashmaps.find(pixel_dist_hash) != coords_hashmaps.end())
    throw std::invalid_argument(
        Formatter() << "The coord map already exists for the given pixel dist "
                    << "pixel_dist: " << ArrToString(pixel_dists) << " at "
                    << __FILE__ << ":" << __LINE__);
  coords_hashmaps[pixel_dist_hash] = createCoordsHashMap(coords);
  return pixel_dist_hash;
}

template <uint8_t D, typename Itype>
uint64_t CoordsManager<D, Itype>::createOutCoords(
    uint64_t coords_key, const Arr<D, int> &pixel_dists,
    const Arr<D, int> &strides, bool is_transpose) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key. "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);

  auto out_pixel_dists =
      computeOutPixelDist<D>(pixel_dists, strides, is_transpose);
  uint64_t out_coords_key = hash_vec<std::vector<int>>(out_pixel_dists);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  coords_hashmaps[out_coords_key] =
      createOutCoordsHashMap(coords_key, pixel_dists, strides);
  return out_coords_key;
}

template <uint8_t D, typename Itype>
uint64_t CoordsManager<D, Itype>::createOriginCoords(uint64_t coords_key,
                                                     int batch_size) {
  if (!existsCoordsKey(coords_key))
    throw std::invalid_argument(
        Formatter() << "The coord map doesn't exist for the given coords_key. "
                    << "coords_key: " << coords_key << " at " << __FILE__ << ":"
                    << __LINE__);
  Arr<D, int> zero_pixel_dists;
  zero_pixel_dists.fill(0);
  uint64_t out_coords_key = hash_vec<Arr<D, int>>(zero_pixel_dists);
  // If the coordinates already exists, return the key.
  if (existsCoordsKey(out_coords_key))
    return out_coords_key;

  coords_hashmaps[out_coords_key] =
      createOriginCoordsHashMap(coords_key, batch_size);
  return out_coords_key;
}

/**
  Given the index map, kernel size, stride, and dilation, compute the input
  index to output index. Returns {in_map, out_map}
*/
template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createInOutPerKernel(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const Arr<D, int> &in_pixel_dists, const Arr<D, int> &kernel_size,
    const Arr<D, int> &dilations, int region_type, at::Tensor offsets) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  int kernel_volume, kernel_ind = 0;
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume<D>(region_type, kernel_size, offsets.size(0));
  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  for (auto const out_coord_iter : out_coords_hashmap) {
    auto out_coord = out_coord_iter.first;
    auto kernel_region = KernelRegion<D, Itype>(
        out_coord, in_pixel_dists, kernel_size, dilations, region_type,
        offsets.data<Itype>(), offsets.size(0));
    kernel_ind = 0;
    for (auto point : kernel_region) {
      auto in_coord_iter = in_coords_hashmap.find(point);
      if (in_coord_iter != in_coords_hashmap.end()) {
        in_map[kernel_ind].push_back(in_coord_iter->second);
        out_map[kernel_ind].push_back(out_coord_iter.second);
      }
      kernel_ind++;
    }
  }
  return std::make_tuple(in_map, out_map);
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createInOutPerKernelTranspose(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const Arr<D, int> &out_pixel_dists, const Arr<D, int> &kernel_size,
    const Arr<D, int> &dilations, int region_type, at::Tensor offsets) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  int kernel_volume, kernel_ind = 0;
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      coords_hashmaps[out_coords_key].map;
  kernel_volume =
      ComputeKernelVolume<D>(region_type, kernel_size, offsets.size(0));
  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);
  for (auto const in_coord_iter : in_coords_hashmap) {
    auto in_coord = in_coord_iter.first;
    auto kernel_region = KernelRegion<D, Itype>(
        in_coord, out_pixel_dists, kernel_size, dilations, region_type,
        offsets.data<Itype>(), offsets.size(0));
    kernel_ind = 0;
    for (auto point : kernel_region) {
      auto out_coord_iter = out_coords_hashmap.find(point);
      if (out_coord_iter != out_coords_hashmap.end()) {
        in_map[kernel_ind].push_back(in_coord_iter.second);
        out_map[kernel_ind].push_back(out_coord_iter->second);
      }
      kernel_ind++;
    }
  }
  return std::make_tuple(in_map, out_map);
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createGlobalReductionInOutMap(
    const uint64_t in_coords_key, const uint64_t out_coords_key) {
  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      coords_hashmaps[out_coords_key].map;
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  std::map<Itype, Itype> in_out_map;
  // The out_coord_map.size() == 1
  for (auto const in_coord_iter : in_coords_hashmap) {
    Coord<D, Itype> coord(in_coord_iter.first);
    for (int j = 0; j < D; j++)
      coord[j] = 0;
    auto out_coord_iter = out_coords_hashmap.find(coord);
    if (out_coord_iter != out_coords_hashmap.end()) {
      in_out_map[in_coord_iter.second] = out_coord_iter->second;
    } else {
      throw std::invalid_argument(Formatter()
                                  << "Coord not found in out coord map"
                                  << ArrToString<Coord<D, Itype>>(coord));
    }
  }

  // Extract key value as in out (ascending) ordered by the in map
  for (auto const &pair : in_out_map) {
    in_map[0].push_back(pair.first);
    out_map[0].push_back(pair.second);
  }

  return std::make_tuple(in_map, out_map);
}

template <uint8_t D, typename Itype>
InOutMapKey CoordsManager<D, Itype>::getMapHashKey(
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose) {
  if (pixel_dists.size() != D || strides.size() != D ||
      kernel_sizes.size() != D || dilations.size() != D) {
    throw std::invalid_argument(
        Formatter() << "Size mismatch. pixel_dists: "
                    << ArrToString(pixel_dists)
                    << ", strides: " << ArrToString(strides)
                    << ", kernel_sizes: " << ArrToString(kernel_sizes)
                    << ", dilations: " << ArrToString(dilations));
  }
  Arr<D, int> arr_pixel_dists;
  Arr<D, int> arr_strides;
  Arr<D, int> arr_kernel_sizes;
  Arr<D, int> arr_dilations;
  std::copy_n(pixel_dists.begin(), D, arr_pixel_dists.begin());
  std::copy_n(strides.begin(), D, arr_strides.begin());
  std::copy_n(kernel_sizes.begin(), D, arr_kernel_sizes.begin());
  std::copy_n(dilations.begin(), D, arr_dilations.begin());
  return getMapHashKey(arr_pixel_dists, arr_strides, arr_kernel_sizes,
                       arr_dilations, region_type, py_in_coords_key,
                       py_out_coords_key, is_transpose);
}

template <uint8_t D, typename Itype>
InOutMapKey CoordsManager<D, Itype>::getMapHashKey(
    Arr<D, int> pixel_dists, Arr<D, int> strides, Arr<D, int> kernel_sizes,
    Arr<D, int> dilations, int region_type, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose) {
  PyCoordsKey<D> *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey<D> *>();
  PyCoordsKey<D> *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey<D> *>();
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t stride_hash = hash_vec<Arr<D, int>>(strides);
  uint64_t kernel_size_hash = hash_vec<Arr<D, int>>(kernel_sizes);
  uint64_t dilation_hash = hash_vec<Arr<D, int>>(dilations);
  InOutMapKey map_key = {
      in_coords_key, out_coords_key,        stride_hash, kernel_size_hash,
      dilation_hash, (uint64_t)region_type, is_transpose};
  return map_key;
}

template <uint8_t D, typename Itype>
InOutMapKey
CoordsManager<D, Itype>::getOriginMapHashKey(py::object py_in_coords_key,
                                             py::object py_out_coords_key) {
  PyCoordsKey<D> *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey<D> *>();
  PyCoordsKey<D> *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey<D> *>();
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t zero_hash = hash_vec<Arr<D, int>>(Arr<D, int>());
  InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false};
  return map_key;
}

template <uint8_t D, typename Itype>
InOutMapKey CoordsManager<D, Itype>::getOriginMapHashKeyCheck(
    py::object py_in_coords_key, py::object py_out_coords_key) {
  PyCoordsKey<D> *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey<D> *>();
  PyCoordsKey<D> *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey<D> *>();
  if (!p_in_coords_key->key_set or !p_out_coords_key->key_set)
    throw std::invalid_argument(Formatter()
                                << "Key is not set. in_coords_key: "
                                << std::to_string(p_in_coords_key->getKey())
                                << ", out_coords_key: "
                                << std::to_string(p_out_coords_key->getKey()));
  // Use the global pooling mapping
  uint64_t out_coords_key = p_out_coords_key->getKey();
  uint64_t in_coords_key = p_in_coords_key->getKey();
  uint64_t zero_hash = hash_vec<Arr<D, int>>(Arr<D, int>());
  InOutMapKey map_key = {
      in_coords_key, out_coords_key, zero_hash, zero_hash, zero_hash, 0, false};

  return map_key;
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<D, Itype>::setupAndReturnInOutPerKernel(
    std::vector<int> pixel_dists, std::vector<int> strides,
    std::vector<int> kernel_sizes, std::vector<int> dilations, int region_type,
    at::Tensor offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose) {
  if (pixel_dists.size() != D || strides.size() != D ||
      kernel_sizes.size() != D || dilations.size() != D) {
    throw std::invalid_argument(
        Formatter() << "Size mismatch. pixel_dists: " << pixel_dists.size()
                    << ", strides: " << strides.size()
                    << ", kernel_sizes: " << kernel_sizes.size()
                    << ", dilations: " << dilations.size());
  }
  Arr<D, int> arr_pixel_dists;
  Arr<D, int> arr_strides;
  Arr<D, int> arr_kernel_sizes;
  Arr<D, int> arr_dilations;
  std::copy_n(pixel_dists.begin(), D, arr_pixel_dists.begin());
  std::copy_n(strides.begin(), D, arr_strides.begin());
  std::copy_n(kernel_sizes.begin(), D, arr_kernel_sizes.begin());
  std::copy_n(dilations.begin(), D, arr_dilations.begin());
  return setupAndReturnInOutPerKernel(
      arr_pixel_dists, arr_strides, arr_kernel_sizes, arr_dilations,
      region_type, offsets, py_in_coords_key, py_out_coords_key, is_transpose);
}

/**
 * Setup output coords, set the py_out_coords_key, create in_out_kernel_map
 */
template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<D, Itype>::setupAndReturnInOutPerKernel(
    Arr<D, int> pixel_dists, Arr<D, int> strides, Arr<D, int> kernel_sizes,
    Arr<D, int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose) {
  PyCoordsKey<D> *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey<D> *>();
  PyCoordsKey<D> *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey<D> *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    out_coords_key = createOutCoords(p_in_coords_key->getKey(), pixel_dists,
                                     strides, is_transpose);
    p_out_coords_key->setKey(out_coords_key);
  } else
    out_coords_key = p_out_coords_key->getKey();

  InOutMapKey map_key =
      getMapHashKey(pixel_dists, strides, kernel_sizes, dilations, region_type,
                    py_in_coords_key, py_out_coords_key, is_transpose);

  if (!is_transpose) { // NON TRANSPOSE
    p_out_coords_key->setPixelDist(pixel_dists);
    p_out_coords_key->stride(strides);
    // For non transpose case
    // make a kernel mapping. The kernel will be saved with the map_key.
    if (in_maps.find(map_key) == in_maps.end()) {
      auto in_out =
          createInOutPerKernel(in_coords_key, out_coords_key, pixel_dists,
                               kernel_sizes, dilations, region_type, offsets);
      in_maps[map_key] = std::get<0>(in_out);
      out_maps[map_key] = std::get<1>(in_out);
    }
    return std::make_tuple(std::ref(in_maps[map_key]),
                           std::ref(out_maps[map_key]));

  } else { // TRANSPOSE
    p_out_coords_key->setPixelDist(pixel_dists);
    p_out_coords_key->up_stride(strides);
    // Create temporary key for the flipped in/out
    InOutMapKey tmp_map_key =
        getMapHashKey(pixel_dists, strides, kernel_sizes, dilations,
                      region_type, py_out_coords_key, py_in_coords_key, false);
    // Check if the temporary key exists and return swapped in/out
    if (in_maps.find(tmp_map_key) != in_maps.end()) {
      return std::make_tuple(std::ref(out_maps[tmp_map_key]),
                             std::ref(in_maps[tmp_map_key]));
    } else {
      // create in out kernel if it doesn't exist
      auto out_pixel_dists = p_out_coords_key->getPixelDist();
      auto in_out = createInOutPerKernelTranspose(
          in_coords_key, out_coords_key, out_pixel_dists, kernel_sizes,
          dilations, region_type, offsets);
      in_maps[map_key] = std::get<0>(in_out);
      out_maps[map_key] = std::get<1>(in_out);

      return std::make_tuple(std::ref(in_maps[map_key]),
                             std::ref(out_maps[map_key]));
    }
  }
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
CoordsManager<D, Itype>::setupAndReturnOriginInOutPerKernel(
    int batch_size, py::object py_in_coords_key, py::object py_out_coords_key) {
  PyCoordsKey<D> *p_in_coords_key = py_in_coords_key.cast<PyCoordsKey<D> *>();
  PyCoordsKey<D> *p_out_coords_key = py_out_coords_key.cast<PyCoordsKey<D> *>();
  uint64_t out_coords_key, in_coords_key = p_in_coords_key->getKey();

  // Create output coordinates if it doesn't exist
  if (!p_out_coords_key->key_set) {
    out_coords_key = createOriginCoords(p_in_coords_key->getKey(), batch_size);
    p_out_coords_key->setKey(out_coords_key);
    p_out_coords_key->setPixelDist(Arr<D, int>());
  } else
    out_coords_key = p_out_coords_key->getKey();

  // Map key for origin hash map
  InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);
  // For non transpose case
  // make a kernel mapping. The kernel will be saved with the map_key.
  if (in_maps.find(map_key) == in_maps.end()) {
    auto in_out = createGlobalReductionInOutMap(in_coords_key, out_coords_key);
    in_maps[map_key] = std::get<0>(in_out);
    out_maps[map_key] = std::get<1>(in_out);
  }
  return std::make_tuple(std::ref(in_maps[map_key]),
                         std::ref(out_maps[map_key]));
}

template <uint8_t D, typename Itype>
std::string CoordsManager<D, Itype>::toString() const {
  std::string tmp;
  tmp += "< CoordsManager, Number of Coords: ";
  tmp += std::to_string(coords_hashmaps.size());
  for (auto kv : coords_hashmaps) {
    tmp += " \n\tCoords Key: ";
    tmp += std::to_string(kv.first);
    tmp += ", Size: ";
    tmp += std::to_string((kv.second).size());
  }
  tmp += "\n  Number of Kernel Maps: ";
  tmp += std::to_string(in_maps.size());
  for (auto kv : in_maps) {
    tmp += " \n\tKernel Map Key: ";
    tmp += std::to_string(hash_vec<InOutMapKey>(kv.first));
    int size = 0;
    for (auto map : kv.second) {
      size += map.size();
    }
    tmp += ", Size: ";
    tmp += std::to_string(size);
  }
  tmp += " >";
  return tmp;
}

INSTANTIATE_CLASS_DIM_ITYPE(CoordsManager, int32_t);
