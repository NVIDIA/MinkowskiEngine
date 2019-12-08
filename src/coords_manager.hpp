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
#ifndef COORDS_MAN
#define COORDS_MAN

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>

#include "concurrent_coordsmap.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "gpu_memory_manager.hpp"
#endif // CPU_ONLY

inline long computeKernelVolume(int region_type, const vector<int> &kernel_size,
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
    throw invalid_argument("Invalid region type");
  }
  return kernel_volume;
}

inline vector<int> computeOutTensorStride(const vector<int> &tensor_strides,
                                          const vector<int> &strides,
                                          bool is_transpose) {
  vector<int> out_tensor_strides;
  ASSERT(tensor_strides.size() == strides.size(),
         "The dimension of tensor_stride: ", ArrToString(tensor_strides),
         " does not match the dimension of strides: ", ArrToString(strides));
  for (int i = 0; i < strides.size(); i++) {
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

class CoordsManager {
public:
  // Variables
  //
  // Coordinate hash key to coordinate hash map
  unordered_map<uint64_t, ConcurrentCoordsMap> coords_maps;
  set<int> batch_indices;

  // In to out index mapping for each kernel, pooling
  unordered_map<InOutMapKey, InOutMaps<int>, InOutMapKeyHash> in_maps;
  unordered_map<InOutMapKey, InOutMaps<int>, InOutMapKeyHash> out_maps;

  CoordsManager(){};
  ~CoordsManager() { clear(); }

  void printDiagnostics(py::object py_coords_key);

  bool existsCoordsKey(uint64_t coords_key);
  bool existsCoordsKey(py::object py_coords_key);
  int getCoordsSize(uint64_t coords_key);
  int getCoordsSize(py::object py_coords_key);
  uint64_t getCoordsKey(const vector<int> &tensor_strides);

  void getCoords(at::Tensor coords, py::object py_coords_key);
  void getKernelMap(at::Tensor kernel_map, vector<int> tensor_strides,
                    vector<int> strides, vector<int> kernel_sizes,
                    vector<int> dilations, int region_type,
                    py::object py_in_coords_key, py::object py_out_coords_key,
                    bool is_transpose, bool is_pool);

  // New coords map initialzation entry
  uint64_t initializeCoords(at::Tensor coords, at::Tensor mapping,
                            const vector<int> &tensor_strides,
                            bool force_creation, bool force_remap,
                            bool allow_duplicate);

  uint64_t initializeCoords(at::Tensor coords, at::Tensor mapping,
                            py::object py_coords_key, bool force_creation,
                            bool force_remap, bool allow_duplicate);

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
  uint64_t createOriginCoords(int D);

  // Mappings
  InOutMapKey getMapHashKey(vector<int> tensor_strides, vector<int> strides,
                            vector<int> kernel_sizes, vector<int> dilations,
                            int region_type, py::object py_in_coords_key,
                            py::object py_out_coords_key, bool is_transpose,
                            bool is_pool);
  InOutMapKey getOriginMapHashKey(py::object py_in_coords_key,
                                  py::object py_out_coords_key);

  // Wrapper functions for setting up coords and returning maps
  InOutMapsRefPair<int>
  getInOutMaps(const vector<int> &tensor_strides, const vector<int> &strides,
               const vector<int> &kernel_sizes, const vector<int> &dilations,
               int region_type, const at::Tensor &offsets,
               py::object py_in_coords_key, py::object py_out_coords_key,
               bool is_transpose, bool is_pool = false,
               bool generate_new_coords = false);

  InOutMapsRefPair<int> getOriginInOutMaps(py::object py_in_coords_key,
                                           py::object py_out_coords_key);

  InOutMapsRefPair<int> getPruningInOutMaps(at::Tensor use_feat,
                                            py::object py_in_coords_key,
                                            py::object py_out_coords_key);

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

  pair<vector<int>, vector<vector<int>>>
  getRowIndicesPerBatch(py::object py_in_coords_key,
                        py::object py_out_coords_key);

#ifndef CPU_ONLY
  // GPU memory manager
  GPUMemoryManager<int> _gpu_memory_manager;
  GPUMemoryManager<int8_t> _dgpu_memory_manager;

  // Coordinates on gpu memory (ncols, pointer) pair
  unordered_map<uint64_t, pair<int, int *>> _gpu_coords;

  // resize and return data_pointer
  int *getScratchGPUMemory(int size) {
    return static_cast<int *>(_gpu_memory_manager.data(size));
  }
  int8_t *getDScratchGPUMemory(int size) {
    return _dgpu_memory_manager.data(size);
  }
#endif // CPU_ONLY
};

#endif // COORDS_MAN
