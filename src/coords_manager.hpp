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

#include "instantiation.hpp"
#include "thread_pool.hpp"
#include "types.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "gpu_memory_manager.hpp"
#endif

template <uint8_t D, typename Itype> class CoordsManager {
public:
  // Static multi threaded pool
  static int nthreads;
  static std::unique_ptr<CoordsThreadPool<D, Itype>> pool;

  CoordsManager();
  CoordsManager(int nthreads_);
  ~CoordsManager() { clear(); }

  bool existsCoordsKey(uint64_t coords_key);
  bool existsCoordsKey(py::object py_coords_key);
  int getCoordsSize(uint64_t coords_key);
  int getCoordsSize(py::object py_coords_key);
  uint64_t getCoordsKey(const Arr<D, int> &tensor_strides);

  void getCoordsMapping(at::Tensor mapping, py::object py_in_coords_key,
                        py::object py_out_coords_key);
  void getCoords(at::Tensor coords, py::object py_coords_key);
  void getKernelMap(at::Tensor kernel_map, std::vector<int> tensor_strides,
                    std::vector<int> strides, std::vector<int> kernel_sizes,
                    std::vector<int> dilations, int region_type,
                    py::object py_in_coords_key, py::object py_out_coords_key,
                    bool is_transpose);

  // New coords map initialzation entry
  uint64_t initializeCoords(at::Tensor coords,
                            const Arr<D, int> &tensor_strides,
                            bool enforce_creation);
  uint64_t initializeCoords(at::Tensor coords, py::object py_coords_key,
                            bool enforce_creation);
  // New coords map given an input
  uint64_t createOutCoords(uint64_t coords_key,
                           const Arr<D, int> &tensor_strides,
                           const Arr<D, int> &strides, bool is_transpose);
  uint64_t createOriginCoords();
  uint64_t createPruneCoords(at::Tensor use_feat, py::object py_in_coords_key,
                             py::object py_out_coords_key);

  // Helper functions for hashmap creation that returns the hashmap and the
  // batch indieces
  std::pair<CoordsHashMap<D, Itype>, std::set<Itype>>
  createCoordsHashMap(at::Tensor coords);

  CoordsHashMap<D, Itype>
  createOutCoordsHashMap(uint64_t coords_key, const Arr<D, int> &tensor_strides,
                         const Arr<D, int> &strides);
  CoordsHashMap<D, Itype> createOriginCoordsHashMap();
  CoordsHashMap<D, Itype> createPrunedCoordsHashMap(uint64_t coords_key,
                                                    at::Tensor use_feat);

  // Mappings
  InOutMapKey getMapHashKey(Arr<D, int> tensor_strides, Arr<D, int> strides,
                            Arr<D, int> kernel_sizes, Arr<D, int> dilations,
                            int region_type, py::object py_in_coords_key,
                            py::object py_out_coords_key, bool is_transpose);
  InOutMapKey getMapHashKey(std::vector<int> tensor_strides,
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
                       const Arr<D, int> &in_tensor_strides,
                       const Arr<D, int> &kernel_size,
                       const Arr<D, int> &dilations, int region_type,
                       at::Tensor offsets);

  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createInOutPerKernelInThreads(const uint64_t in_coords_key,
                                const uint64_t out_coords_key,
                                const Arr<D, int> &in_tensor_strides,
                                const Arr<D, int> &kernel_size,
                                const Arr<D, int> &dilations, int region_type,
                                at::Tensor offsets);

  std::tuple<InOutMapPerKernel<Itype>, InOutMapPerKernel<Itype>>
  createInOutPerKernelTranspose(const uint64_t in_coords_key,
                                const uint64_t out_coords_key,
                                const Arr<D, int> &out_tensor_strides,
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
  setupAndReturnInOutPerKernel(const std::vector<int> &tensor_strides,
                               const std::vector<int> &strides,
                               const std::vector<int> &kernel_sizes,
                               const std::vector<int> &dilations,
                               int region_type, const at::Tensor &offsets,
                               py::object py_in_coords_key,
                               py::object py_out_coords_key, bool is_transpose);

  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnInOutPerKernel(const Arr<D, int> &tensor_strides,
                               const Arr<D, int> &strides,
                               const Arr<D, int> &kernel_sizes,
                               const Arr<D, int> &dilations, int region_type,
                               const at::Tensor &offsets,
                               py::object py_in_coords_key,
                               py::object py_out_coords_key, bool is_transpose);

  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnOriginInOutPerKernel(py::object py_in_coords_key,
                                     py::object py_out_coords_key);

  std::tuple<InOutMapPerKernel<Itype> &, InOutMapPerKernel<Itype> &>
  setupAndReturnPruningInOutPerKernel(at::Tensor use_feat,
                                      py::object py_in_coords_key,
                                      py::object py_out_coords_key);

  int getMaxMapSize(const InOutMapPerKernel<Itype> &in_maps) {
    int max_n_active = -1;
    for (auto &map : in_maps)
      if (max_n_active < (int)(map.size()))
        max_n_active = (int)(map.size());
    return max_n_active;
  }

  int getMaxMapSize(const std::tuple<InOutMapPerKernel<Itype> &,
                                     InOutMapPerKernel<Itype> &> &in_out) {
    return getMaxMapSize(std::get<0>(in_out));
  }

  std::string toString() const;
  void clear() {
    coords_hashmaps.clear();
    in_maps.clear();
    out_maps.clear();
  }

  std::pair<std::vector<Itype>, std::vector<std::vector<Itype>>>
  getRowIndicesPerBatch(py::object py_in_coords_key,
                        py::object py_out_coords_key);

  // Coordinate hash key to coordinate hash map
  std::unordered_map<uint64_t, CoordsHashMap<D, Itype>> coords_hashmaps;
  // In to out index mapping for each kernel, pooling
  std::unordered_map<InOutMapKey, InOutMapPerKernel<Itype>, InOutMapKeyHash>
      in_maps;
  std::unordered_map<InOutMapKey, InOutMapPerKernel<Itype>, InOutMapKeyHash>
      out_maps;

  // Batch indices must be consistent throughout the lifetime of the coordsman
  std::vector<Itype> batch_indices;

#ifndef CPU_ONLY
  // GPU memory manager
  GPUMemoryManager<Itype> gpu_memory_manager;
  GPUMemoryManager<int8_t> dgpu_memory_manager;

  // resize and return data_pointer
  Itype *getScratchGPUMemory(int size) {
    return static_cast<Itype *>(gpu_memory_manager.data(size));
  }
  int8_t *getDScratchGPUMemory(int size) {
    return dgpu_memory_manager.data(size);
  }
#endif
};

template <uint8_t D, typename Itype> int CoordsManager<D, Itype>::nthreads;

template <uint8_t D, typename Itype>
std::unique_ptr<CoordsThreadPool<D, Itype>> CoordsManager<D, Itype>::pool;

#endif
