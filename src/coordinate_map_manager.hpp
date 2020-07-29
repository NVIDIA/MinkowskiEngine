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
#include "types.hpp"
#include "utils.hpp"

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

template <>
struct is_cpu_coordinate_map<CoordinateMapCPU> : std::true_type {};

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

template <typename index_type, typename stride_type>
struct coordinate_map_key_comparator {
  bool operator()(coordinate_map_key_type const &lhs,
                  coordinate_map_key_type const &rhs) {
    auto vec_less = std::lexicographical_compare(
        lhs.first.begin(), lhs.first.end(), rhs.first.begin(), rhs.first.end());
    if (!vec_less & std::equal(lhs.first.begin(), lhs.first.end(),
                               rhs.first.begin(), rhs.first.end())) {
      return std::lexicographical_compare(lhs.second.begin(), lhs.second.end(),
                                          rhs.second.begin(), rhs.second.end());
    }
    return vec_less;
  }
};

} // namespace detail

using std::vector;

template <typename VType> int getInOutMapsSize(const VType &map) {
  // can't use ::accumulate as pVector template instantiation requires a bit
  // dirty syntax
  int n = 0;
  for (auto cmap = begin(map); cmap != end(map); ++cmap)
    n += cmap->size();
  return n;
}

inline vector<int> computeOutTensorStride(const vector<int> &tensor_strides,
                                          const vector<int> &strides,
                                          bool is_transpose) {
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
  using map_collection_type =
      std::map<coordinate_map_key_type, map_type,
               detail::coordinate_map_key_comparator<index_type, stride_type>>;

public:
  // allocator backend will be ignored when coordinate map backend is CPU
  CoordinateMapManager(size_type num_threads = 0) {
    if (num_threads > 0) {
      // Doesn't seem to work. use `export OMP_NUM_THREADS=N;` in bash.
      omp_set_dynamic(0);
      omp_set_num_threads(num_threads);
    }
  }
  ~CoordinateMapManager() { // clear();
  }

  /*
   * New coordinate map initialzation function.
   *
   * returns key and map, inverse map
   */
  std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
  insert_and_map(at::Tensor const &th_coordinate,
                 stride_type const tensor_stride,
                 std::string const string_id = "");

  // return kernel map
  // std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  // kernel_map(py::object py_in_coords_key, py::object py_out_coords_key,
  //            stride_type strides, stride_type kernel_sizes,
  //            stride_type dilations, REGION_TYPE:: region_type, at::Tensor
  //            offsets);

  /*
  void printDiagnostics(py::object py_coords_key) const;

  bool existsCoordsKey(uint64_t coords_key) const;
  bool existsCoordsKey(py::object py_coords_key) const;
  bool existsInOutMapKey(const InOutMapKey &map_key) const {
    return in_maps.find(map_key) != in_maps.end();
  }
  int getCoordsSize(uint64_t coords_key) const;
  int getCoordsSize(py::object py_coords_key) const;
  uint64_t getCoordsKey(const vector<int> &tensor_strides) const;
  long int getBatchSize() const { return batch_indices.size(); }
  set<int> getBatchIndices() const { return batch_indices; }
  void getCoords(at::Tensor coords, py::object py_coords_key) const;
  vector<vector<at::Tensor>>
  getKernelMap(vector<int> tensor_strides, vector<int> strides,
               vector<int> kernel_sizes, vector<int> dilations, int region_type,
               at::Tensor offsets, py::object py_in_coords_key,
               py::object py_out_coords_key, bool is_transpose, bool is_pool);

#ifndef CPU_ONLY
  vector<vector<at::Tensor>>
  getKernelMapGPU(vector<int> tensor_strides, vector<int> strides,
                  vector<int> kernel_sizes, vector<int> dilations,
                  int region_type, at::Tensor offsets,
                  py::object py_in_coords_key, py::object py_out_coords_key,
                  bool is_transpose, bool is_pool);
#endif

  // TODO make this function non-const with ability to generate a new map
  vector<at::Tensor> getCoordsMap(py::object py_in_coords_key,
                                  py::object py_out_coords_key) const;
  pair<vector<at::Tensor>, vector<at::Tensor>>
  getUnionMap(vector<py::object> py_in_coords_keys,
              py::object py_out_coords_key);

  // Set the py_coords_key to the origin coords map key
  void setOriginCoordsKey(py::object py_coords_key);

  // New coords map given an input
  uint64_t createStridedCoords(uint64_t coords_key,
                               const vector<int> &tensor_strides,
                               const vector<int> &strides, bool force_creation);
  uint64_t createTransposedStridedRegionCoords(
      uint64_t coords_key, const vector<int> &tensor_strides,
      const vector<int> &strides, vector<int> kernel_sizes,
      vector<int> dilations, int region_type, at::Tensor offsets,
      bool force_creation);
  uint64_t createPrunedCoords(at::Tensor use_feat, py::object py_in_coords_key,
                              py::object py_out_coords_key);
  uint64_t createOriginCoords(const int D);
  uint64_t createUnionCoords(vector<py::object> py_in_coords_keys,
                             py::object py_out_coords_key);

  // Mappings
  const InOutMapKey getMapHashKey(vector<int> tensor_strides,
                                  vector<int> strides, vector<int> kernel_sizes,
                                  vector<int> dilations, int region_type,
                                  py::object py_in_coords_key,
                                  py::object py_out_coords_key,
                                  bool is_transpose, bool is_pool) const;
  const InOutMapKey getOriginMapHashKey(py::object py_in_coords_key,
                                        py::object py_out_coords_key) const;
  const InOutMapKey getUnionMapHashKey(vector<py::object> py_in_coords_keys,
                                       py::object py_out_coords_key) const;

  // Wrapper functions for setting up coords and returning maps
  const InOutMapsRefPair<int>
  getInOutMaps(const vector<int> &tensor_strides, const vector<int> &strides,
               const vector<int> &kernel_sizes, const vector<int> &dilations,
               int region_type, const at::Tensor &offsets,
               py::object py_in_coords_key, py::object py_out_coords_key,
               bool is_transpose, bool is_pool = false,
               bool generate_new_coords = false);

  const InOutMapsRefPair<int> getOriginInOutMaps(py::object py_in_coords_key,
                                                 py::object py_out_coords_key);

  const InOutMapsRefPair<int> getPruningInOutMaps(at::Tensor use_feat,
                                                  py::object py_in_coords_key,
                                                  py::object py_out_coords_key);

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

  int getMaxMapSize(const pair<InOutMaps<int> &, InOutMaps<int> &> &in_out) {
    return getMaxMapSize(in_out.first);
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
  getInOutMapsGPU(const vector<int> &tensor_strides, const vector<int> &strides,
                  const vector<int> &kernel_sizes, const vector<int> &dilations,
                  int region_type, const at::Tensor &offsets,
                  py::object py_in_coords_key, py::object py_out_coords_key,
                  bool is_transpose, bool is_pool = false,
                  bool force_creation = false);

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

  coordinate_map_key_type get_random_string_id(stride_type const &tensor_stride,
                                               std::string string_id) {
    coordinate_map_key_type key =
        std::make_pair(tensor_stride, string_id + '-' + random_string(5));
    while (m_coordinate_maps.find(key) != m_coordinate_maps.end()) {
      key = std::make_pair(tensor_stride, string_id + '-' + random_string(5));
    }
    return key;
  }

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

private:
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

private:
  std::map<coordinate_map_key_type, map_type,
           detail::coordinate_map_key_comparator<index_type, stride_type>>
      m_coordinate_maps;
#ifndef CPU_ONLY
  TemplatedAllocator<char> m_allocator;
#endif
  // Track whether the batch indices are set
  bool is_batch_indices_set = false;

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

} // namespace detail

} // namespace minkowski

#endif // COORDINATE_MAP_MANAGER
