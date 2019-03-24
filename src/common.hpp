#ifndef COMMON
#define COMMON
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>

#include "instantiation.hpp"
#include "thread_pool.hpp"
#include "types.hpp"
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

template <typename T> std::string ArrToString(T arr) {
  std::string buf = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    buf += (i ? ", " : "") + std::to_string(arr[i]);
  }
  buf += "]";
  return buf;
}

template <typename T> void PyPrintArr(T arr) { py::print(ArrToString(arr)); }

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
  // Static multi threaded pool
  static int nthreads;
  static std::unique_ptr<CoordsThreadPool<D, Itype>> pool;

  CoordsManager();
  CoordsManager(int nthreads_);
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
  void getKernelMap(at::Tensor kernel_map, std::vector<int> pixel_dists,
                    std::vector<int> strides, std::vector<int> kernel_sizes,
                    std::vector<int> dilations, int region_type,
                    py::object py_in_coords_key, py::object py_out_coords_key,
                    bool is_transpose);

  // New coords map initialzation entry
  uint64_t initializeCoords(at::Tensor coords, const Arr<D, int> &pixel_dists,
                            bool enforce_creation);
  uint64_t initializeCoords(at::Tensor coords, py::object py_coords_key,
                            bool enforce_creation);
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
  createInOutPerKernelInThreads(const uint64_t in_coords_key,
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

template <uint8_t D, typename Itype> int CoordsManager<D, Itype>::nthreads;

template <uint8_t D, typename Itype>
std::unique_ptr<CoordsThreadPool<D, Itype>> CoordsManager<D, Itype>::pool;

#endif
