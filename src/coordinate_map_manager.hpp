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

template <default_types::index_type V>
default_types::stride_type _fill_vec(size_t const len) {
  default_types::stride_type vec(len);
  std::for_each(vec.begin(), vec.end(), [](auto &i) { i = V; });
  return vec;
}

} // namespace detail

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
class CoordinateMapManager {
public:
  using size_type = default_types::size_type;
  using index_type = default_types::index_type;
  using stride_type = default_types::stride_type;
  using map_type = CoordinateMapType<coordinate_type, TemplatedAllocator>;
#ifndef CPU_ONLY
  using field_map_type = typename std::conditional<
      detail::is_cpu_coordinate_map<CoordinateMapType>::value,
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>,
      CoordinateFieldMapGPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>>::type;
#else
  using field_map_type =
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>;
#endif
  using self_type = CoordinateMapManager<coordinate_type, coordinate_field_type,
                                         TemplatedAllocator, CoordinateMapType>;
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
  CoordinateMapManager(
      MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT,
      size_type num_threads = 0)
      : m_algorithm(algo) {
    if (num_threads > 0) {
      // Doesn't seem to work. use `export OMP_NUM_THREADS=N;` in bash.
      omp_set_dynamic(0);
      omp_set_num_threads(num_threads);
    }
    switch (m_algorithm) {
    case MinkowskiAlgorithm::DEFAULT: {
      m_kernel_map_mode = CUDAKernelMapMode::SPEED_OPTIMIZED;
      m_gpu_default_occupancy = 25;
      break;
    }
    case MinkowskiAlgorithm::MEMORY_EFFICIENT: {
      m_kernel_map_mode = CUDAKernelMapMode::MEMORY_EFFICIENT;
      m_gpu_default_occupancy = 50;
      break;
    }
    case MinkowskiAlgorithm::SPEED_OPTIMIZED: {
      m_kernel_map_mode = CUDAKernelMapMode::SPEED_OPTIMIZED;
      m_gpu_default_occupancy = 25;
      break;
    }
    }
  }
  ~CoordinateMapManager() {}

  /****************************************************************************
   * Coordinate generation, modification, and initialization entry functions
   ****************************************************************************/
  py::object insert_field(at::Tensor const &th_coordinate,
                          stride_type const tensor_stride,
                          std::string const string_id = "");

  /*
   * New coordinate map initialzation function.
   *
   * returns key and map, inverse map
   */
  std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map(CoordinateMapKey const *p_in_field_map_key,
                                 stride_type const sparse_tensor_stride,
                                 std::string const sparse_string_id = "");

  std::pair<at::Tensor, at::Tensor>
  field_to_sparse_map(CoordinateMapKey const *p_in_field_map_key,
                      CoordinateMapKey const *p_out_sparse_map_key);

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
         stride_type const &kernel_stride, std::string const string_id = "");

  // python-side stride function
  py::object py_stride(CoordinateMapKey const *in_map_key,
                       stride_type const &kernel_stride,
                       std::string const string_id = "") {
    auto key =
        std::get<0>(stride(in_map_key->get_key(), kernel_stride, string_id));
    return py::cast(new CoordinateMapKey(key.first.size() + 1, key));
  }

  // stride region: new coordinate generation
  std::pair<coordinate_map_key_type, bool>
  stride_region(coordinate_map_key_type const &in_map_key,
                cpu_kernel_region<coordinate_type> &kernel,
                stride_type const &out_tensor_stride,
                bool const expand_coordinates);

  // origin coordinate map creation
  std::pair<coordinate_map_key_type, bool> origin();
  std::pair<coordinate_map_key_type, bool> origin_field();

  // pruning
  coordinate_map_key_type prune(coordinate_map_key_type const &in_key,
                                bool const *keep_begin, bool const *keep_end);

  // python-side stride function
  py::object py_origin() {
    auto map_key_bool = origin();
    LOG_DEBUG("Return origin map key");
    return py::cast(new CoordinateMapKey(map_key_bool.first.first.size() + 1,
                                         map_key_bool.first));
  }

  // python-side stride function
  py::object py_origin_field() {
    auto map_key_bool = origin_field();
    LOG_DEBUG("Return origin map key");
    return py::cast(new CoordinateMapKey(map_key_bool.first.first.size() + 1,
                                         map_key_bool.first));
  }

  // Merge
  coordinate_map_key_type
  merge(std::vector<coordinate_map_key_type> const &map_keys);
  std::pair<coordinate_map_key_type, std::vector<at::Tensor>>
  union_map(std::vector<coordinate_map_key_type> const &map_keys);
  std::vector<at::Tensor>
  union_map_th(std::vector<CoordinateMapKey *> const &map_keys,
               CoordinateMapKey *p_out_key);

  /****************************************************************************
   * Tensor field related operations
   ****************************************************************************/
  py::object insert(at::Tensor const &coordinates);

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

  bool insert_field_map(coordinate_map_key_type map_key, field_map_type &map) {
    LOG_DEBUG("insert map with tensor_stride", map_key.first);
    auto result = m_field_coordinates.insert(
        std::make_pair<coordinate_map_key_type, field_map_type>(
            std::move(map_key), std::move(map)));
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

  inline bool exists_field(coordinate_map_key_type const &key) const noexcept {
    return m_field_coordinates.find(key) != m_field_coordinates.end();
  }

  inline bool exists_field_to_sparse(
      coordinate_map_key_type const &field_key,
      coordinate_map_key_type const &sparse_key) const noexcept {
    auto key = std::pair<coordinate_map_key_type, coordinate_map_key_type>{
        field_key, sparse_key};
    return m_field_to_sparse_maps.find(key) != m_field_to_sparse_maps.end();
  }

  std::vector<py::object>
  field_to_sparse_keys(coordinate_map_key_type const &field_key) const {
    std::vector<py::object> return_keys;
    for (auto const &elem : m_field_to_sparse_maps) {
      if (elem.first.first == field_key) {
        auto const &tensor_key = elem.first.second;
        return_keys.push_back(py::cast(
            new CoordinateMapKey(tensor_key.first.size() + 1, tensor_key)));
      }
    }
    return return_keys;
  }

  // when the key is the python coordinate map key
  inline bool exists(CoordinateMapKey const *p_key) const {
    // key set exception
    return exists(p_key->get_key());
  }

  // when the key is the python coordinate map key
  inline bool exists_field(CoordinateMapKey const *p_key) const {
    // key set exception
    return exists_field(p_key->get_key());
  }

  inline bool
  exists_field_to_sparse(CoordinateMapKey const *p_field_key,
                         CoordinateMapKey const *p_sparse_key) const {
    // key set exception
    return exists_field_to_sparse(p_field_key->get_key(),
                                  p_sparse_key->get_key());
  }

  inline size_type size(coordinate_map_key_type const &key) const {
    auto const it = m_coordinate_maps.find(key);
    auto const field_it = m_field_coordinates.find(key);
    ASSERT(it != m_coordinate_maps.end() ||
               field_it != m_field_coordinates.end(),
           ERROR_MAP_NOT_FOUND);
    if (it != m_coordinate_maps.end())
      return it->second.size();
    else
      return field_it->second.size();
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

  at::Tensor get_coordinate_field(CoordinateMapKey const *p_key) const;

  std::pair<at::Tensor, at::Tensor>
  get_field_to_sparse_map(CoordinateMapKey const *p_field_key,
                          CoordinateMapKey const *p_sparse_key) const;

  std::vector<py::object>
  get_coordinate_map_keys(stride_type const tensor_stride) const {
    std::vector<py::object> keys;
    for (auto it = m_coordinate_maps.begin(); it != m_coordinate_maps.end();
         ++it) {
      coordinate_map_key_type const &key = it->first;
      if (key.first == tensor_stride) {
        keys.push_back(
            py::cast(new CoordinateMapKey(key.first.size() + 1, key)));
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

    if (m_field_coordinates.size() > 0) {
      for (auto const &kv : m_field_coordinates) {
        o << "\tTensorField " << print_key(kv.first) << ":\t"
          << kv.second.to_string() << "\n";
      }
    }

    for (auto const &kv : m_kernel_maps) {
      o << "\t" << print_key(std::get<0>(kv.first)) << "->"
        << print_key(std::get<1>(kv.first)) << ":\t" << kv.second << "\n";
    }
    return o.str();
  }

  MinkowskiAlgorithm::Mode algorithm() const { return m_algorithm; }

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

  // for kernel size 0
  kernel_map_type const &kernel_map(CoordinateMapKey const *py_in_coords_key,
                                    CoordinateMapKey const *py_out_coords_key);

  kernel_map_type const &origin_map(CoordinateMapKey const *py_out_coords_key);
  kernel_map_type const &
  origin_field_map(CoordinateMapKey const *py_out_coords_key);

  // return kernel map. for cpu it is {in maps, out maps}.
  // For gpu it could be {in maps, out maps}, or {kernel index, in map, out map}
  std::unordered_map<int64_t, at::Tensor>
  kernel_map_th(CoordinateMapKey const *py_in_coords_key,  //
                CoordinateMapKey const *py_out_coords_key, //
                stride_type const &kernel_size,            //
                stride_type const &kernel_stride,          //
                stride_type const &kernel_dilation,        //
                RegionType::Type const region_type,        //
                at::Tensor const &offsets, bool is_transpose, bool is_pool);

  // interpolation map
  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield,
                           CoordinateMapKey const *py_in_coords_key);

  std::pair<at::Tensor, std::vector<at::Tensor>>
  origin_map_th(CoordinateMapKey const *py_out_coords_key);

  std::pair<at::Tensor, std::vector<at::Tensor>>
  origin_field_map_th(CoordinateMapKey const *py_out_coords_key);

  std::pair<at::Tensor, at::Tensor>
  stride_map_th(CoordinateMapKey const *p_in_map_key,
                CoordinateMapKey const *p_strided_map_key);

  size_t origin_map_size() {
    ASSERT(m_coordinate_maps.size() > 0 or m_field_coordinates.size() > 0,
           "No coordinate map found.");
    if (m_coordinate_maps.size() > 0) {
      auto const key = origin().first;
      return m_coordinate_maps.find(key)->second.size();
    } else {
      auto const key = origin_field().first;
      return m_coordinate_maps.find(key)->second.size();
    }
  }

  coordinate_map_key_type get_random_string_id(stride_type const &tensor_stride,
                                               std::string string_id) {
    coordinate_map_key_type key = std::make_pair(
        tensor_stride, string_id.size() > 0 ? string_id + '-' + random_string(5)
                                            : random_string(5));
    while (m_coordinate_maps.find(key) != m_coordinate_maps.end()) {
      key =
          std::make_pair(tensor_stride, string_id.size() > 0
                                            ? string_id + '-' + random_string(5)
                                            : random_string(5));
    }
    return key;
  }

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

  void deallocate(void *p, size_type n) {
    m_allocator.deallocate((char *)p, n);
  }
#endif
private:
  // NOTE: operator[] required mapped_type(), which is not defined.
  //
  // CoordinateMapManager owns the coordinate maps
  std::map<coordinate_map_key_type, map_type, coordinate_map_key_comparator>
      m_coordinate_maps;

  // CoordinateMapManager managed coordinates
  std::map<coordinate_map_key_type, field_map_type,
           coordinate_map_key_comparator>
      m_field_coordinates;

  // CoordinateMapManager owns the kernel maps
  std::unordered_map<kernel_map_key_type, kernel_map_type,
                     kernel_map_key_hasher<coordinate_map_key_hasher>>
      m_kernel_maps;

  std::unordered_map<kernel_map_key_type, kernel_map_type,
                     kernel_map_key_hasher<coordinate_map_key_hasher>>
      m_field_kernel_maps;

  std::unordered_map<
      const std::pair<coordinate_map_key_type, coordinate_map_key_type>,
      const std::pair<at::Tensor, at::Tensor>,
      field_to_sparse_map_key_hasher<coordinate_map_key_hasher>>
      m_field_to_sparse_maps;

#ifndef CPU_ONLY
  TemplatedAllocator<char> m_allocator;
#endif
  // kernel map mode
  CUDAKernelMapMode::Mode m_kernel_map_mode;

  // Algorithm index
  MinkowskiAlgorithm::Mode m_algorithm;

}; // coordsmanager

namespace detail {

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename field_map_type>
struct insert_field_functor {

  void operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapType> &manager);
};

// a partial specialization functor for insertion
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
struct insert_and_map_functor {
  std::pair<at::Tensor, at::Tensor> operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapType> &manager);
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

// a partial specialization functor for stride map generation
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct empty_map_functor {
  using stride_type = default_types::stride_type;

  kernel_map_type operator()();
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

// a partial specialization functor for stride map
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct stride_map2tensor_functor {
  std::pair<at::Tensor, at::Tensor>
  operator()(kernel_map_type const &origin_map);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct kernel_map_to_tensors {
  using index_type = default_types::index_type;

  std::unordered_map<int64_t, at::Tensor>
  operator()(kernel_map_type const &kernel_map);
};

} // namespace detail

// type defs
template <typename coordinate_type>
using cpu_manager_type =
    CoordinateMapManager<coordinate_type, default_types::ccoordinate_type,
                         std::allocator, CoordinateMapCPU>;

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
using gpu_manager_type =
    CoordinateMapManager<coordinate_type, default_types::ccoordinate_type,
                         TemplatedAllocator, CoordinateMapGPU>;

template <typename coordinate_type>
using gpu_default_manager_type =
    gpu_manager_type<coordinate_type, detail::default_allocator>;

template <typename coordinate_type>
using gpu_c10_manager_type =
    gpu_manager_type<coordinate_type, detail::c10_allocator>;

#endif

} // namespace minkowski

#endif // COORDINATE_MAP_MANAGER
