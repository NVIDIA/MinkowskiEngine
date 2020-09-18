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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef GPU_COORDS_MAN
#define GPU_COORDS_MAN

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "gpu_coordsmap.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "gpu_memory_manager.hpp"
#include <ATen/cuda/CUDAContext.h>
#endif // CPU_ONLY

namespace minkowski {

using std::begin;
using std::end;
using std::get;
using std::move;
using std::ref;
using std::string;
using std::to_string;
using std::unordered_map;

/*
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
*/

#ifndef CPU_ONLY

template <typename VType> int getInOutMapsSizeGPU(const VType &map) {
  int n = 0;
  for (auto cmap = begin(map); cmap != end(map); ++cmap)
    n += cmap->size(0);
  return n;
}

template <typename MapType = CoordsToIndexMapGPU> class GPUCoordsManager {
public:
  // Variables
  //
  // Coordinate hash key to coordinate hash map
  unordered_map<uint64_t, std::shared_ptr<GPUCoordsMap<MapType>>> coords_maps;

  set<int> batch_indices;
  int batch_size;
  int D;
  int device_id;
  c10::Device device;
  int min_nrows;
  uint64_t min_coords_key;

  std::shared_ptr<GPUMemoryManager> gpu_memory_manager;

  // In to out index mapping for each kernel, pooling
  unordered_map<InOutMapKey, vector<at::Tensor>, InOutMapKeyHash> in_maps;
  unordered_map<InOutMapKey, vector<at::Tensor>, InOutMapKeyHash> out_maps;

  GPUCoordsManager(int D,
                   int device_id,
                   MemoryManagerBackend backend) : batch_size(-1), device(c10::DeviceType::CUDA, 0) {
    gpu_memory_manager = std::make_shared<GPUMemoryManager>(backend);
    this->device_id = device_id;
    this->D = D;
    min_nrows = INT_MAX;
  }
  ~GPUCoordsManager() { clear(); }

// TODO(ljm): implement GPUCoordsMap<MapType>::print
//  void printDiagnostics(py::object py_coords_key) const;

  uint64_t getCoordsKey(const vector<int> &tensor_strides) const;
  bool existsCoordsKey(uint64_t coords_key) const;
  bool existsCoordsKey(py::object py_coords_key) const;
  bool existsInOutMapKey(const InOutMapKey &map_key) const {
    return in_maps.find(map_key) != in_maps.end();
  }
  int getCoordsSize(uint64_t coords_key) const;
  int getCoordsSize(py::object py_coords_key) const;
  uint64_t getRandomCoordsKey();
  long int getBatchSize();
  set<int> getBatchIndices() {
    if (batch_indices.empty()) {
      for (int b = 0; b != getBatchSize(); ++b) batch_indices.insert(b);
    }
    ASSERT((int)batch_indices.size() == getBatchSize(),
           "batch_indices.size() must be equal to getBatchSize()");
    return batch_indices;
  }
  void getCoords(at::Tensor coords, py::object py_coords_key) const;
  vector<vector<at::Tensor>>
  getKernelMap(const vector<int>& tensor_strides, const vector<int>& strides,
               const vector<int>& kernel_sizes, const vector<int>& dilations, int region_type,
               at::Tensor offsets, py::object py_in_coords_key,
               py::object py_out_coords_key, bool is_transpose, bool is_pool);
  // TODO make this function non-const with ability to generate a new map
  vector<at::Tensor> getCoordsMap(py::object py_in_coords_key,
                                  py::object py_out_coords_key) const;
  pair<vector<at::Tensor>, vector<at::Tensor>>
  getUnionMap(vector<py::object> py_in_coords_keys,
              py::object py_out_coords_key);

  // Set the py_coords_key to the origin coords map key
  void setOriginCoordsKey(py::object py_coords_key);

  // New coords map initialzation entry
  uint64_t initializeCoords(at::Tensor coords, at::Tensor mapping,
                            at::Tensor inverse_mapping,
                            const vector<int> &tensor_strides,
                            const bool force_creation, const bool force_remap,
                            const bool allow_duplicate_coords,
                            const bool return_inverse);

  uint64_t initializeCoords(at::Tensor coords, at::Tensor mapping,
                            at::Tensor inverse_mapping,
                            py::object py_coords_key, const bool force_creation,
                            const bool force_remap,
                            const bool allow_duplicate_coords,
                            const bool return_inverse);

  // New coords map given an input
  uint64_t createStridedCoords(uint64_t coords_key,
                               const vector<int> &tensor_strides,
                               const vector<int> &strides, bool force_creation);
  uint64_t createTransposedStridedRegionCoords(
      uint64_t coords_key, const vector<int> &tensor_strides,
      const vector<int> &strides, vector<int> kernel_sizes,
      vector<int> dilations, int region_type, at::Tensor offsets,
      bool force_creation);
  uint64_t createPruningCoords(at::Tensor use_feat, py::object py_in_coords_key,
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
  const InOutMapKey
  getInOutMaps(const vector<int> &tensor_strides, const vector<int> &strides,
               const vector<int> &kernel_sizes, const vector<int> &dilations,
               int region_type, const at::Tensor &offsets,
               py::object py_in_coords_key, py::object py_out_coords_key,
               bool is_transpose, bool is_pool = false,
               bool generate_new_coords = false);

  const InOutMapKey getOriginInOutMaps(py::object py_in_coords_key,
                                    py::object py_out_coords_key);

  const InOutMapKey getPruningInOutMaps(at::Tensor use_feat,
                                                  py::object py_in_coords_key,
                                                  py::object py_out_coords_key);

  const InOutMapKey
  getUnionInOutMaps(vector<py::object> py_in_coords_keys,
                    py::object py_out_coords_key);

  const InOutMapKey
  getStridedInOutMaps(
      py::object py_in_coords_key, py::object py_out_coords_key,
      const vector<int>& tensor_strides, const vector<int>& strides,
      const vector<int>& kernel_sizes, const vector<int>& dilations, int region_type,
      bool is_transpose, bool is_pool,
      bool force_creation);

  const InOutMapKey
  createStridedInOutMaps(
      py::object py_in_coords_key, py::object py_out_coords_key,
      const vector<int> &tensor_strides,
      const vector<int> &strides,
      vector<int> kernel_sizes, vector<int> dilations, int region_type,
      bool is_transpose, bool is_pool,
      bool force_creation);

  const InOutMapKey
  getTransposedStridedRegionInOutMaps(
      py::object py_in_coords_key, py::object py_out_coords_key,
      const vector<int>& tensor_strides,
      const vector<int>& strides, const vector<int>& kernel_sizes, const vector<int>& dilations,
      int region_type,
      bool is_transpose, bool is_pool,
      at::Tensor offsets,
      bool force_creation);

  const InOutMapKey
  createTransposedStridedRegionInOutMaps(
      py::object py_in_coords_key, py::object py_out_coords_key,
      const vector<int>& tensor_strides,
      const vector<int>& strides, const vector<int>& kernel_sizes, const vector<int>& dilations,
      int region_type,
      bool is_transpose, bool is_pool,
      at::Tensor offsets, bool force_creation);

  const InOutMapKey
  createUnionInOutMaps(const vector<py::object>& py_in_coords_keys,
                       py::object py_out_coords_key);

  const InOutMapKey
  createPruningInOutMaps(at::Tensor use_feat,
                         py::object py_in_coords_key,
                         py::object py_out_coords_key);

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

  void *getScratchGPUMemory(size_t size) {
    return gpu_memory_manager.get()->tmp_data(size);
  }

  void clearScratchGPUMemory() { gpu_memory_manager.get()->clear_tmp(); }

};     // gpucoordsmanager
#endif

} // namespace minkowski

#endif // GPU_COORDS_MAN
