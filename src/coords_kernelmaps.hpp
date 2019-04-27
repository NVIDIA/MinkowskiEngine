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
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
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
    auto kernel_region =
        Region<D, Itype>(out_coord, in_tensor_strides, kernel_size, dilations,
                         region_type, offsets.data<Itype>(), offsets.size(0));
    kernel_ind = 0;
    for (auto &point : kernel_region) {
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

/**
 * Multithreaded in out kernel generator
 */
template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createInOutPerKernelInThreads(
    const uint64_t in_coords_key, const uint64_t out_coords_key,
    const Arr<D, int> &in_tensor_strides, const Arr<D, int> &kernel_size,
    const Arr<D, int> &dilations, int region_type, at::Tensor offsets) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      coords_hashmaps[out_coords_key].map;
  int kernel_volume =
      ComputeKernelVolume<D>(region_type, kernel_size, offsets.size(0));
  InOutMapPerKernel<Itype> in_map(kernel_volume), out_map(kernel_volume);

  std::vector<std::future<Triplets>> results;
  KernelMapFunctor<D, Itype> f;
  for (auto const out_coord_iter : out_coords_hashmap) {
    auto out_coord = out_coord_iter.first;
    int out_coord_index = out_coord_iter.second;
    results.emplace_back(CoordsManager<D, Itype>::pool->enqueue(
        f, out_coord, std::ref(in_tensor_strides), std::ref(kernel_size),
        std::ref(dilations), region_type, offsets.data<Itype>(),
        offsets.size(0), out_coord_index, std::ref(in_coords_hashmap)));
  }

  for (auto &result : results) {
    Triplets triplets = result.get();
    for (auto &triplet : triplets) {
      int kernel_id = triplet[0];
      in_map[kernel_id].push_back(triplet[1]);
      out_map[kernel_id].push_back(triplet[2]);
      // std::cout << kernel_id << ", " << triplet[1] << ", " << triplet[2] <<
      // std::endl;
    }
  }

  return std::make_tuple(in_map, out_map);
}

template <uint8_t D, typename Itype>
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
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
    auto kernel_region =
        Region<D, Itype>(in_coord, out_tensor_strides, kernel_size, dilations,
                         region_type, offsets.data<Itype>(), offsets.size(0));
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
std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
CoordsManager<D, Itype>::createPruningInOutMap(const uint64_t in_coords_key,
                                               const uint64_t out_coords_key) {
  if (!existsCoordsKey(in_coords_key) || !existsCoordsKey(out_coords_key))
    throw std::invalid_argument(
        Formatter() << "The coords map doesn't exist for the given coords_key. "
                    << "in_coords_key: " << in_coords_key
                    << ", out_coords_key: " << out_coords_key << " at "
                    << __FILE__ << ":" << __LINE__);

  _CoordsHashMap<D, Itype> &in_coords_hashmap =
      coords_hashmaps[in_coords_key].map;
  _CoordsHashMap<D, Itype> &out_coords_hashmap =
      coords_hashmaps[out_coords_key].map;
  InOutMapPerKernel<Itype> in_map(1), out_map(1);
  for (const auto &out_coord_iter : out_coords_hashmap) {
    auto out_coord = out_coord_iter.first;
    auto in_coord_iter = in_coords_hashmap.find(out_coord);
    if (in_coord_iter != in_coords_hashmap.end()) {
      in_map[0].push_back(in_coord_iter->second);
      out_map[0].push_back(out_coord_iter.second);
    }
  }
  return std::make_tuple(in_map, out_map);
}
#endif
