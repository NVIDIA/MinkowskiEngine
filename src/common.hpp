#ifndef COMMON
#define COMMON
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <google/dense_hash_map>

#include <torch/extension.h>

#include "utils.hpp"

#ifndef CPU_ONLY
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <driver_types.h> // cuda driver types

#include <THC/THCBlas.h>
#include <thrust/device_vector.h>

#include "gpu.cuh"
#endif

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
  char gInstantiationGuard##classname;                                         \
  template class classname<float>;                                             \
  // template class classname<double>
  //
#define INSTANTIATE_CLASS_DIM(CLASSNAME)                                       \
  template class CLASSNAME<1>;                                                 \
  template class CLASSNAME<2>;                                                 \
  template class CLASSNAME<3>;                                                 \
  template class CLASSNAME<4>;                                                 \
  template class CLASSNAME<5>;                                                 \
  template class CLASSNAME<6>;                                                 \
  template class CLASSNAME<7>;

#define INSTANTIATE_CLASS_DIM_ITYPE(CLASSNAME, ITYPE)                          \
  template class CLASSNAME<1, ITYPE>;                                          \
  template class CLASSNAME<2, ITYPE>;                                          \
  template class CLASSNAME<3, ITYPE>;                                          \
  template class CLASSNAME<4, ITYPE>;                                          \
  template class CLASSNAME<5, ITYPE>;                                          \
  template class CLASSNAME<6, ITYPE>;                                          \
  template class CLASSNAME<7, ITYPE>;

#define SWITCH_DIM_TYPES(func, Dtype, Itype, ...)                              \
  switch (D) {                                                                 \
  case 1:                                                                      \
    func<1, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 2:                                                                      \
    func<2, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 3:                                                                      \
    func<3, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 4:                                                                      \
    func<4, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 5:                                                                      \
    func<5, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 6:                                                                      \
    func<6, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  case 7:                                                                      \
    func<7, Dtype, Itype>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    throw std::invalid_argument(Formatter() << "Not supported D " << D);       \
  }

// N-Dimensional coordinate + batch index = N + 1
template <uint8_t D, typename Itype> using Coord = std::array<Itype, D + 1>;

// For hashing kernel sizes, strides, and dilations.
template <uint8_t D, typename Itype> using Arr = std::array<Itype, D>;

template <typename T> std::string ArrToString(T arr) {
  std::string buf = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    buf += (i ? ", " : "") + std::to_string(arr[i]);
  }
  buf += "]";
  return buf;
}

template <typename T> void PyPrintArr(T arr) { py::print(ArrToString(arr)); }

// Key for InOutMap
// (in_coords_key, out_coords_key, stride hash, kernel size, dilation,
// is_transpose)
using InOutMapKey = std::array<uint64_t, 7>;

// FNV64-1a
// uint64_t for unsigned long, must use CXX -m64
// WARNING: IType for T must be int32
template <typename T> uint64_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint32_t x : p) {
    hash ^= x;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

struct InOutMapKeyHash {
  uint64_t operator()(InOutMapKey const &p) const {
    return hash_vec<InOutMapKey>(p);
  }
};

// Input index to output index mapping for each spatial kernel
template <typename Itype>
using InOutMapPerKernel = std::vector<std::vector<Itype>>;

template <typename Itype>
using InOutKernelMapPair =
    std::pair<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>;

// For Used for fast index of coordinate retrieval
template <uint8_t D, typename Itype> struct CoordHash {
  uint64_t operator()(Coord<D, Itype> const &p) const {
    return hash_vec<Coord<D, Itype>>(p);
  }
};

template <uint8_t D, typename Itype> struct ArrHash {
  uint64_t operator()(Arr<D, Itype> const &p) const {
    return hash_vec<Arr<D, Itype>>(p);
  }
};

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

// Location to index of the feature
template <uint8_t D, typename Itype>
using _CoordsHashMap =
    google::dense_hash_map<Coord<D, Itype>, uint64_t, CoordHash<D, Itype>,
                           std::equal_to<Coord<D, Itype>>>;

template <uint8_t D, typename Itype> class CoordsHashMap {
public:
  _CoordsHashMap<D, Itype> map;
  CoordsHashMap() {
    Coord<D, Itype> empty_key;
    for (int i = 0; i < D + 1; ++i)
      empty_key[i] = -std::numeric_limits<Itype>::max();
    map.set_empty_key(empty_key);
  }
  size_t size() { return map.size(); }
};

// Will be exported to python for lazy key initialization.
// For instance, ConvModule.out_coords_key can be used for other layers before
// feedforward
template <uint8_t D> class PyCoordsKey {
private:
  uint64_t key_; // Use the key_ for all coordshashmap query. Lazily set

public:
  bool key_set = false;
  Arr<D, int> pixel_dists_;
  PyCoordsKey() { reset(); }
  void reset();
  void copy(py::object ohter);
  void setKey(uint64_t key);
  uint64_t getKey();
  void setPixelDist(const Arr<D, int> &pixel_dists);
  void stride(const Arr<D, int> &strides);
  void up_stride(const Arr<D, int> &strides);
  Arr<D, int> getPixelDist() { return pixel_dists_; };
  std::string toString() const;
};

template <uint8_t D, typename Itype> class CoordsManager {
public:
  ~CoordsManager() { clear(); }

  // Coordinate hash key to coordinate hash map
  std::unordered_map<uint64_t, CoordsHashMap<D, Itype>> coords_hashmaps;
  // In to out index mapping for each kernel, pooling
  std::unordered_map<InOutMapKey, InOutMapPerKernel<Itype>, InOutMapKeyHash>
      in_maps;
  std::unordered_map<InOutMapKey, InOutMapPerKernel<Itype>, InOutMapKeyHash>
      out_maps;

  bool existsCoordsKey(uint64_t coords_key);
  bool existsCoordsKey(py::object py_coords_key);
  int getCoordsSize(uint64_t coords_key);
  int getCoordsSize(py::object py_coords_key);
  uint64_t getCoordsKey(const Arr<D, int> &pixel_dists);

  void getCoordsMapping(at::Tensor mapping, py::object py_in_coords_key,
                        py::object py_out_coords_key);
  void getCoords(at::Tensor coords, py::object py_coords_key);

  // New coords map initialzation entry
  uint64_t initializeCoords(at::Tensor coords, const Arr<D, int> &pixel_dists);
  uint64_t initializeCoords(at::Tensor coords, py::object py_coords_key);
  // New coords map given an input
  uint64_t createOutCoords(uint64_t coords_key, const Arr<D, int> &pixel_dists,
                           const Arr<D, int> &strides, bool is_transpose);
  uint64_t createOriginCoords(uint64_t coords_key, int batch_size);
  uint64_t createPruneCoords(at::Tensor use_feat, py::object py_in_coords_key,
                             py::object py_out_coords_key);

  // Helper functions for hashmap creation
  CoordsHashMap<D, Itype> createCoordsHashMap(at::Tensor coords);
  CoordsHashMap<D, Itype> createOutCoordsHashMap(uint64_t coords_key,
                                                 const Arr<D, int> &pixel_dists,
                                                 const Arr<D, int> &strides);
  CoordsHashMap<D, Itype> createOriginCoordsHashMap(uint64_t coords_key,
                                                    int batch_size);
  CoordsHashMap<D, Itype> createPrunedCoordsHashMap(uint64_t coords_key,
                                                    at::Tensor use_feat);

  // Mappings
  InOutMapKey getMapHashKey(Arr<D, int> pixel_dists, Arr<D, int> strides,
                            Arr<D, int> kernel_sizes, Arr<D, int> dilations,
                            int region_type, py::object py_in_coords_key,
                            py::object py_out_coords_key, bool is_transpose);
  InOutMapKey getMapHashKey(std::vector<int> pixel_dists,
                            std::vector<int> strides,
                            std::vector<int> kernel_sizes,
                            std::vector<int> dilations, int region_type,
                            py::object py_in_coords_key,
                            py::object py_out_coords_key, bool is_transpose);
  InOutMapKey getOriginMapHashKey(py::object py_in_coords_key,
                                  py::object py_out_coords_key);
  InOutMapKey getOriginMapHashKeyCheck(py::object py_in_coords_key,
                                       py::object py_out_coords_key);

  // Kernel Maps
  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createInOutPerKernel(const uint64_t in_coords_key,
                       const uint64_t out_coords_key,
                       const Arr<D, int> &in_pixel_dists,
                       const Arr<D, int> &kernel_size,
                       const Arr<D, int> &dilations, int region_type,
                       at::Tensor offsets);
  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createInOutPerKernelTranspose(const uint64_t in_coords_key,
                                const uint64_t out_coords_key,
                                const Arr<D, int> &out_pixel_dists,
                                const Arr<D, int> &kernel_size,
                                const Arr<D, int> &dilations, int region_type,
                                at::Tensor offsets);
  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createGlobalReductionInOutMap(const uint64_t in_coords_key,
                                const uint64_t out_coords_key);
  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createPruningInOutMap(const uint64_t in_coords_key,
                        const uint64_t out_coords_key);

  // Wrapper functions for setting up coords and returning maps
  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnInOutPerKernel(std::vector<int> pixel_dists,
                               std::vector<int> strides,
                               std::vector<int> kernel_sizes,
                               std::vector<int> dilations, int region_type,
                               at::Tensor offsets, py::object py_in_coords_key,
                               py::object py_out_coords_key, bool is_transpose);
  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnInOutPerKernel(Arr<D, int> pixel_dists, Arr<D, int> strides,
                               Arr<D, int> kernel_sizes, Arr<D, int> dilations,
                               int region_type, at::Tensor offsets,
                               py::object py_in_coords_key,
                               py::object py_out_coords_key, bool is_transpose);
  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnOriginInOutPerKernel(int batch_size,
                                     py::object py_in_coords_key,
                                     py::object py_out_coords_key);
  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnPruningInOutPerKernel(at::Tensor use_feat,
                                      py::object py_in_coords_key,
                                      py::object py_out_coords_key);

  std::string toString() const;
  void clear() {
    coords_hashmaps.clear();
    in_maps.clear();
    out_maps.clear();
  }
};

#endif
