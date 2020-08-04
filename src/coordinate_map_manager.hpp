/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu)
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef COORDINATE_MAP_MANAGER
#define COORDINATE_MAP_MANAGER

#include "coordinate_map.hpp"
#include "coordinate_map_cpu.hpp"
#include "coordinate_map_key.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "coordinate_map_gpu.cuh"
#include "kernel_map.cuh"
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <robin_hood.h>

#include <torch/extension.h>

namespace minkowski {

namespace detail {

template <template <typename T, template <typename Q> class A>
          class CoordinateMapType>
struct is_cpu_coordinate_map : std::false_type {};

template <> struct is_cpu_coordinate_map<CoordinateMapCPU> : std::true_type {};

template <typename T1, typename T2> void copy_types(const T1 &src, T2 &dst) {
  size_t curr_it = 0;
  for (const auto s : src)
    dst[curr_it++] = s;
}

/*
template <typename index_type, typename stride_type>
struct coordinate_map_key_hasher {
  uint64_t operator()(coordinate_map_key_type const &key) {
    auto hash_vec = robin_hood::hash_bytes(
        key.first.data(), sizeof(index_type) * key.first.size());
    hash_vec ^= robin_hood::hash_bytes(key.second.data(), key.second.length());
    return hash_vec;
  }
};
*/

} // namespace detail

/*
template <typename VType> int getInOutMapsSize(const VType &map) {
  // can't use ::accumulate as pVector template instantiation requires a bit
  // dirty syntax
  int n = 0;
  for (auto cmap = begin(map); cmap != end(map); ++cmap)
    n += cmap->size();
  return n;
}

inline std::vector<int>
computeOutTensorStride(const vector<int> &tensor_strides,
                       const vector<int> &strides, bool is_transpose) {
  vector<int> out_tensor_strides;
  ASSERT(tensor_strides.size() == strides.size(),
         "The dimension of tensor_stride: ", ArrToString(tensor_strides),
         " does not match the dimension of strides: ", ArrToString(strides));
  for (size_t i = 0; i < strides.size(); i++) {
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
*/

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
class CoordinateMapManager {
public:
  using size_type = default_types::size_type;
  using index_type = default_types::index_type;
  using stride_type = default_types::stride_type;
  using map_type = CoordinateMapType<coordinate_type, TemplatedAllocator>;
  using self_type = CoordinateMapManager<coordinate_type, TemplatedAllocator,
                                         CoordinateMapType>;
  using map_collection_type = std::map<coordinate_map_key_type, map_type,
                                       coordinate_map_key_comparator>;
  using kernel_map_type =
#ifndef CPU_ONLY
      typename std::conditional<
          detail::is_cpu_coordinate_map<CoordinateMapType>::value,
          cpu_kernel_map,
          gpu_kernel_map<index_type, TemplatedAllocator<char>>>::type;
#else
      cpu_kernel_map;
#endif
  using kernel_map_reference_type =
#ifndef CPU_ONLY
      typename std::conditional<
          detail::is_cpu_coordinate_map<CoordinateMapType>::value,
          cpu_kernel_map,
          gpu_kernel_map<index_type, TemplatedAllocator<char>>>::type;
#else
      cpu_kernel_map_reference;
#endif

public:
  // allocator backend will be ignored when coordinate map backend is CPU
  CoordinateMapManager(CUDAKernelMapMode::Mode kernel_map_mode =
                           CUDAKernelMapMode::SPEED_OPTIMIZED,
                       size_type num_threads = 0)
      : m_kernel_map_mode(kernel_map_mode) {
    if (num_threads > 0) {
      // Doesn't seem to work. use `export OMP_NUM_THREADS=N;` in bash.
      omp_set_dynamic(0);
      omp_set_num_threads(num_threads);
    }
    if (kernel_map_mode == CUDAKernelMapMode::SPEED_OPTIMIZED) {
      m_gpu_default_occupancy = 25;
    } else {
      m_gpu_default_occupancy = 50;
    }
  }
  ~CoordinateMapManager() {}

  /****************************************************************************
   * Coordinate generation, modification, and initialization entry functions
   ****************************************************************************/
  // TODO
  // py::object insert(at::Tensor const &th_coordinate,
  //                  stride_type const tensor_stride,
  //                  std::string const string_id = "");

  /*
   * New coordinate map initialzation function.
   *
   * returns key and map, inverse map
   */
  std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
  insert_and_map(at::Tensor const &th_coordinate,
                 stride_type const tensor_stride,
                 std::string const string_id = "");

  /*
   * Generate a new coordinate_map if it doesn't exists
   */
  // returns out_map_key and flag which is true if a new map is created
  std::pair<coordinate_map_key_type, bool>
  stride(coordinate_map_key_type const &in_map_key,
         stride_type const &kernel_stride);

  // python-side stride function
  py::object py_stride(CoordinateMapKey const *in_map_key,
                       stride_type const &kernel_stride) {
    auto key = std::get<0>(stride(in_map_key->get_key(), kernel_stride));
    return py::cast(new CoordinateMapKey(key.first.size() + 1, key));
  }

  // stride region: new coordinate generation
  std::pair<coordinate_map_key_type, bool>
  stride_region(coordinate_map_key_type const &in_map_key,
                cpu_kernel_region<coordinate_type> &kernel, bool is_transpose);

  // origin coordinate map creation
  std::pair<coordinate_map_key_type, bool> origin();

  // python-side stride function
  py::object py_origin() {
    auto map_key_bool = origin();
    LOG_DEBUG("Return origin map key");
    return py::cast(new CoordinateMapKey(map_key_bool.first.first.size() + 1,
                                         map_key_bool.first));
  }

  /****************************************************************************
   * Coordinate management helper functions
   ****************************************************************************/
  bool insert(coordinate_map_key_type map_key, map_type &map) {
    LOG_DEBUG("insert map with tensor_stride", map_key.first);
    auto result = m_coordinate_maps.insert(
        std::make_pair<coordinate_map_key_type, map_type>(std::move(map_key),
                                                          std::move(map)));
    LOG_DEBUG("map insertion", result.second);
    return result.second;
  }

  typename map_collection_type::iterator
  find(coordinate_map_key_type const &map_key) {
    return m_coordinate_maps.find(map_key);
  }

  typename map_collection_type::const_iterator map_end() const {
    return m_coordinate_maps.cend();
  }

  inline bool exists(coordinate_map_key_type const &key) const noexcept {
    return m_coordinate_maps.find(key) != m_coordinate_maps.end();
  }

  // when the key is the python coordinate map key
  inline bool exists(CoordinateMapKey const *p_key) const {
    // key set exception
    return exists(p_key->get_key());
  }

  inline size_type size(coordinate_map_key_type const &key) const {
    auto it = m_coordinate_maps.find(key);
    ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
    return it->second.size();
  }

  inline size_type size(CoordinateMapKey const *p_key) const {
    return size(p_key->get_key());
  }

  inline size_type capacity(coordinate_map_key_type const &key) const {
    auto it = m_coordinate_maps.find(key);
    ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
    return it->second.capacity();
  }

  at::Tensor get_coordinates(CoordinateMapKey const *p_key) const;

  std::vector<py::object>
  get_coordinate_map_keys(stride_type const tensor_stride) const {
    std::vector<py::object> keys;
    for (auto it = m_coordinate_maps.begin(); it != m_coordinate_maps.end();
         ++it) {
      coordinate_map_key_type const &key = it->first;
      if (key.first == tensor_stride) {
        keys.push_back(py::cast(new CoordinateMapKey(key.first.size(), key)));
      }
    }
    return keys;
  }

  std::string print_key(coordinate_map_key_type const &key) const {
    Formatter out;
    out << ArrToString(key.first);
    if (key.second.length() > 0)
      out << "-" << key.second;
    return out.str();
  }

  std::string to_string(CoordinateMapKey const *p_key) const {
    auto it = m_coordinate_maps.find(p_key->get_key());
    ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
    return print_key(it->first) + " : " + it->second.to_string();
  }

  std::string to_string() const {
    Formatter o;
    for (auto const &kv : m_coordinate_maps) {
      o << "\t" << print_key(kv.first) << ":\t" << kv.second.to_string()
        << "\n";
    }
    for (auto const &kv : m_kernel_maps) {
      o << "\t" << print_key(std::get<0>(kv.first)) << "->"
        << print_key(std::get<1>(kv.first)) << ":\t" << kv.second << "\n";
    }
    return o.str();
  }

  /****************************************************************************
   * Kernel map related functions
   ****************************************************************************/

  // return kernel map. for cpu it is {in maps, out maps}.
  // For gpu it could be {in maps, out maps}, or {kernel index, in map, out map}
  kernel_map_type const &
  kernel_map(CoordinateMapKey const *py_in_coords_key,  //
             CoordinateMapKey const *py_out_coords_key, //
             stride_type const &kernel_size,            //
             stride_type const &kernel_stride,          //
             stride_type const &kernel_dilation,        //
             RegionType::Type const region_type,        //
             at::Tensor const &offsets, bool is_transpose, bool is_pool);

  std::pair<at::Tensor, std::vector<at::Tensor>>
  origin_map(CoordinateMapKey const *py_out_coords_key);

  /*
  bool existsInOutMapKey(const InOutMapKey &map_key) const {
    return in_maps.find(map_key) != in_maps.end();
  }

  // Mappings
  const InOutMapsRefPair<int> getPruningInOutMaps(at::Tensor use_feat,
                                                  py::object
py_in_coords_key, py::object py_out_coords_key);

  const InOutMapsRefPair<int>
  getUnionInOutMaps(vector<py::object> py_in_coords_keys,
                    py::object py_out_coords_key);

  int getMapSize(const InOutMaps<int> &in_maps) {
    int n = 0;
    for (auto &map : in_maps)
      n += (int)(map.size());
    return n;
  }

  int getMaxMapSize(const InOutMaps<int> &in_maps) {
    int max_n_active = -1;
    for (auto &map : in_maps)
      if (max_n_active < (int)(map.size()))
        max_n_active = (int)(map.size());
    return max_n_active;
  }

  int getMaxMapSize(const pair<InOutMaps<int> &, InOutMaps<int> &> &in_out)
{ return getMaxMapSize(in_out.first);
  }

  uint64_t getRandomCoordsKey();

  string toString() const;
  void clear() {
    coords_maps.clear();
    in_maps.clear();
    out_maps.clear();
  }

  at::Tensor getRowIndicesAtBatchIndex(py::object py_in_coords_key,
                                       py::object py_out_coords_key,
                                       const int batch_index);
  vector<at::Tensor> getRowIndicesPerBatch(py::object py_in_coords_key,
                                           py::object py_out_coords_key);
#ifndef CPU_ONLY
  // Keep all in out maps throughout the lifecycle of the coords manager
  unordered_map<InOutMapKey, pInOutMaps<int>, InOutMapKeyHash> d_in_maps;
  unordered_map<InOutMapKey, pInOutMaps<int>, InOutMapKeyHash> d_out_maps;

  const pInOutMaps<int> copyInOutMapToGPU(const InOutMaps<int> &map);
  void copyInOutMapsToGPU(const InOutMapKey &map_key);

  const pInOutMapsRefPair<int>
  getInOutMapsGPU(const vector<int> &tensor_strides, const vector<int>
&strides, const vector<int> &kernel_sizes, const vector<int> &dilations, int
region_type, const at::Tensor &offsets, py::object py_in_coords_key,
py::object py_out_coords_key, bool is_transpose, bool is_pool = false, bool
force_creation = false);

  const pInOutMapsRefPair<int>
  getOriginInOutMapsGPU(py::object py_in_coords_key,
                        py::object py_out_coords_key);

  const pInOutMapsRefPair<int>
  getPruningInOutMapsGPU(at::Tensor use_feat, py::object py_in_coords_key,
                         py::object py_out_coords_key);

  const pInOutMapsRefPair<int>
  getUnionInOutMapsGPU(vector<py::object> py_in_coords_keys,
                       py::object py_out_coords_key);

  void *allocate(size_type size) {

    return gpu_memory_manager.get()->tmp_data(size);
  }

  void clearScratchGPUMemory() { gpu_memory_manager.get()->clear_tmp(); }

#endif // CPU_ONLY
  */

private:
  void coordinate_map_key_check(CoordinateMapKey const *p_map_key) const {
    ASSERT(p_map_key != nullptr, "Input coordinate map key not defined.");
    ASSERT(p_map_key->is_key_set(), "Key not defined.");
    ASSERT(exists(p_map_key->get_key()), "Key does not exist.");
  }

  // random string generator
  std::string random_string(size_t length) {
    auto randchar = []() -> char {
      const char charset[] = "0123456789"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
  }

  coordinate_map_key_type get_random_string_id(stride_type const &tensor_stride,
                                               std::string string_id) {
    coordinate_map_key_type key =
        std::make_pair(tensor_stride, string_id + '-' + random_string(5));
    while (m_coordinate_maps.find(key) != m_coordinate_maps.end()) {
      key = std::make_pair(tensor_stride, string_id + '-' + random_string(5));
    }
    return key;
  }

  kernel_map_key_type
  origin_map_key(coordinate_map_key_type const &in_key) const {
    map_type const &random_map = m_coordinate_maps.begin()->second;
    stride_type zero_vec(random_map.coordinate_size() - 1);
    std::for_each(zero_vec.begin(), zero_vec.end(), [](auto &i) { i = 0; });
    coordinate_map_key_type origin_key = std::make_pair(zero_vec, "");

    return std::make_tuple(in_key, origin_key,           // maps
                           zero_vec, zero_vec, zero_vec, // kernels
                           RegionType::HYPER_CUBE, false, false);
  }

public:
  size_t m_gpu_default_occupancy;
#ifndef CPU_ONLY
  void *allocate(size_type n) { return m_allocator.allocate(n); }

  void deallocate(void *p, size_type n) { m_allocator.deallocate((char *)p, n); }
#endif
private:
  // NOTE: operator[] required mapped_type(), which is not defined.
  //
  // CoordinateMapManager owns the coordinate maps
  std::map<coordinate_map_key_type, map_type, coordinate_map_key_comparator>
      m_coordinate_maps;

  // CoordinateMapManager owns the tensors
  std::unordered_map<kernel_map_key_type, kernel_map_type,
                     kernel_map_key_hasher<coordinate_map_key_hasher>>
      m_kernel_maps;

#ifndef CPU_ONLY
  TemplatedAllocator<char> m_allocator;
#endif
  // kernel map mode
  CUDAKernelMapMode::Mode m_kernel_map_mode;

  // In to out index mapping for each kernel, pooling
  // unordered_map<InOutMapKey, InOutMaps<int>, InOutMapKeyHash> in_maps;
  // unordered_map<InOutMapKey, InOutMaps<int>, InOutMapKeyHash> out_maps;

}; // coordsmanager

namespace detail {

// a partial specialization functor for insertion
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
struct insert_and_map_functor {
  std::pair<at::Tensor, at::Tensor>
  operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
             CoordinateMapManager<coordinate_type, TemplatedAllocator,
                                  CoordinateMapType> &manager);
};

// a partial specialization functor for kernel map generation
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct kernel_map_functor {
  kernel_map_type operator()(
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &out_map,
      CUDAKernelMapMode::Mode kernel_map_mode,
      cpu_kernel_region<coordinate_type> &kernel);
};

// a partial specialization functor for stride map generation
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct stride_map_functor {
  using stride_type = default_types::stride_type;

  kernel_map_type operator()(
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &out_map,
      stride_type const &kernel);
};

// a partial specialization functor for kernel map in/out swap
template <typename kernel_map_type> struct swap_in_out_map_functor {

  kernel_map_type operator()(kernel_map_type const &kernel_map);
};

// a partial specialization functor for origin_map
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct origin_map_functor {
  std::pair<at::Tensor, std::vector<at::Tensor>>
  operator()(CoordinateMapType<coordinate_type, TemplatedAllocator> const
                 &origin_coordinate_map,
             kernel_map_type const &origin_map);
};

} // namespace detail

// type defs
template <typename coordinate_type>
using cpu_manager_type =
    CoordinateMapManager<coordinate_type, std::allocator, CoordinateMapCPU>;

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
using gpu_manager_type =
    CoordinateMapManager<coordinate_type, TemplatedAllocator, CoordinateMapGPU>;

template <typename coordinate_type>
using gpu_default_manager_type =
    gpu_manager_type<coordinate_type, detail::default_allocator>;

template <typename coordinate_type>
using gpu_c10_manager_type =
    gpu_manager_type<coordinate_type, detail::c10_allocator>;

#endif

} // namespace minkowski

#endif // COORDINATE_MAP_MANAGER
