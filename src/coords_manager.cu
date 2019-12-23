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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "coords_manager.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

const pInOutMaps<int>
CoordsManager::copyInOutMapToGPU(const InOutMaps<int> &map) {
  pInOutMaps<int> d_map;

  int n = getInOutMapsSize(map);
  int *d_scr = (int *)gpu_memory_manager.gpuMalloc(n * sizeof(int));

  for (const auto &cmap : map) {
    // Copy (*p_in_maps)[k] to GPU
    CUDA_CHECK(cudaMemcpy(d_scr, cmap.data(), cmap.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    d_map.push_back(pVector<int>(d_scr, cmap.size()));
    d_scr += cmap.size();
  }

  return d_map;
}

void CoordsManager::copyInOutMapsToGPU(const InOutMapKey &map_key) {
  if (d_in_maps.find(map_key) == d_in_maps.end()) {
    ASSERT(in_maps.find(map_key) != in_maps.end(),
           "The InOutMap doesn't exists.");
    d_in_maps[map_key] = copyInOutMapToGPU(in_maps[map_key]);
    d_out_maps[map_key] = copyInOutMapToGPU(out_maps[map_key]);
  }
}

const pInOutMapsRefPair<int> CoordsManager::getInOutMapsGPU(
    const vector<int> &tensor_strides, const vector<int> &strides,
    const vector<int> &kernel_sizes, const vector<int> &dilations,
    int region_type, const at::Tensor &offsets, py::object py_in_coords_key,
    py::object py_out_coords_key, bool is_transpose, bool is_pool,
    bool force_creation) {

  const auto &in_out =
      getInOutMaps(tensor_strides, strides, kernel_sizes, dilations,
                   region_type, offsets, py_in_coords_key, py_out_coords_key,
                   is_transpose, is_pool, force_creation);

  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

const pInOutMapsRefPair<int>
CoordsManager::getOriginInOutMapsGPU(py::object py_in_coords_key,
                                     py::object py_glob_coords_key) {
  const auto &in_out = getOriginInOutMaps(py_in_coords_key, py_glob_coords_key);

  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_glob_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

const pInOutMapsRefPair<int>
CoordsManager::getPruningInOutMapsGPU(at::Tensor use_feat,
                                      py::object py_in_coords_key,
                                      py::object py_out_coords_key) {
  const auto &in_out =
      getPruningInOutMaps(use_feat, py_in_coords_key, py_out_coords_key);

  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

const pInOutMapsRefPair<int>
CoordsManager::getUnionInOutMapsGPU(vector<py::object> py_in_coords_keys,
                                    py::object py_out_coords_key) {
  const auto &in_out = getUnionInOutMaps(py_in_coords_keys, py_out_coords_key);

  const InOutMapKey map_key =
      getUnionMapHashKey(py_in_coords_keys, py_out_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}
