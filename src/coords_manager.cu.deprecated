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

namespace minkowski {

namespace detail {

template <typename SrcType, typename DstType>
__global__ void dtypeCopy(SrcType const *src, DstType *dst, size_t n) {
  CUDA_KERNEL_LOOP(index, n) { dst[index] = src[index]; }
}

} // namespace detail

template <typename MapType>
const pInOutMaps<int>
CoordsManager<MapType>::copyInOutMapToGPU(const InOutMaps<int> &map) {
  pInOutMaps<int> d_map;

  const int n = getInOutMapsSize(map);
  int *d_scr = (int *)gpu_memory_manager.get()->gpuMalloc(n * sizeof(int));

  for (const auto &cmap : map) {
    // Copy (*p_in_maps)[k] to GPU
    CUDA_CHECK(cudaMemcpy(d_scr, cmap.data(), cmap.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    d_map.push_back(pVector<int>(d_scr, cmap.size()));
    d_scr += cmap.size();
  }

  return d_map;
}

template <typename MapType>
void CoordsManager<MapType>::copyInOutMapsToGPU(const InOutMapKey &map_key) {
  if (d_in_maps.find(map_key) == d_in_maps.end()) {
    ASSERT(in_maps.find(map_key) != in_maps.end(),
           "The InOutMap doesn't exists.");
    d_in_maps[map_key] = copyInOutMapToGPU(in_maps[map_key]);
    d_out_maps[map_key] = copyInOutMapToGPU(out_maps[map_key]);
  }
}

template <typename MapType>
const pInOutMapsRefPair<int> CoordsManager<MapType>::getInOutMapsGPU(
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

template <typename MapType>
const pInOutMapsRefPair<int>
CoordsManager<MapType>::getOriginInOutMapsGPU(py::object py_in_coords_key,
                                              py::object py_glob_coords_key) {
  const auto &in_out = getOriginInOutMaps(py_in_coords_key, py_glob_coords_key);

  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_glob_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

template <typename MapType>
const pInOutMapsRefPair<int>
CoordsManager<MapType>::getPruningInOutMapsGPU(at::Tensor use_feat,
                                               py::object py_in_coords_key,
                                               py::object py_out_coords_key) {
  const auto &in_out =
      getPruningInOutMaps(use_feat, py_in_coords_key, py_out_coords_key);

  const InOutMapKey map_key =
      getOriginMapHashKey(py_in_coords_key, py_out_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

template <typename MapType>
const pInOutMapsRefPair<int> CoordsManager<MapType>::getUnionInOutMapsGPU(
    vector<py::object> py_in_coords_keys, py::object py_out_coords_key) {
  const auto &in_out = getUnionInOutMaps(py_in_coords_keys, py_out_coords_key);

  const InOutMapKey map_key =
      getUnionMapHashKey(py_in_coords_keys, py_out_coords_key);

  copyInOutMapsToGPU(map_key);

  return make_pair(ref(d_in_maps[map_key]), ref(d_out_maps[map_key]));
}

/*
 * Given tensor_stride_src and tensor_stride_dst, find the respective coord_maps
 * and return the indices of the coord_map_ind in coord_map_dst
 */
template <typename MapType>
vector<vector<at::Tensor>> CoordsManager<MapType>::getKernelMapGPU(
    vector<int> tensor_strides, vector<int> strides, vector<int> kernel_sizes,
    vector<int> dilations, int region_type, at::Tensor offsets,
    py::object py_in_coords_key, py::object py_out_coords_key,
    bool is_transpose, bool is_pool) {
  // WARNING: This function will not work properly with custon region types.
  ASSERT(region_type != 2,
         "Currently, it does not support the custom region type.");
  const InOutMapKey map_key = getMapHashKey(
      tensor_strides, strides, kernel_sizes, dilations, region_type,
      py_in_coords_key, py_out_coords_key, is_transpose, is_pool);

  const auto &in_out = getInOutMapsGPU(
      tensor_strides, strides, kernel_sizes, dilations, region_type, offsets,
      py_in_coords_key, py_out_coords_key, false);

  const pInOutMaps<int> &in_maps = in_out.first;
  const pInOutMaps<int> &out_maps = in_out.second;

  int all_volume = 0, kernel_volume = in_maps.size();
  for (int k = 0; k < kernel_volume; k++)
    all_volume += in_maps[k].size();

  // CUDA_CHECK(cudaGetDevice(&device_id));
  torch::TensorOptions options =
      torch::TensorOptions()
          .dtype(torch::kInt64)
          // .device(torch::kCUDA)
          .device(torch::kCUDA, gpu_memory_manager.get()->get_device_id())
          .requires_grad(false);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vector<at::Tensor> in_tensors, out_tensors;

  for (int k = 0; k < kernel_volume; k++) {
    auto curr_volume = in_maps[k].size();
    if (curr_volume <= 0)
      continue;

    at::Tensor in_kernel_map =
        torch::empty({(long)curr_volume}, options).contiguous();
    at::Tensor out_kernel_map =
        torch::empty({(long)curr_volume}, options).contiguous();

    // Wait until both memory chunks are allocated
    CUDA_CHECK(cudaStreamSynchronize(stream));

    detail::dtypeCopy<int, long>
        <<<GET_BLOCKS(curr_volume), CUDA_NUM_THREADS, 0, stream>>>(
            in_maps[k].data(), in_kernel_map.data<long>(), curr_volume);
    detail::dtypeCopy<int, long>
        <<<GET_BLOCKS(curr_volume), CUDA_NUM_THREADS, 0, stream>>>(
            out_maps[k].data(), out_kernel_map.data<long>(), curr_volume);

    in_tensors.push_back(move(in_kernel_map));
    out_tensors.push_back(move(out_kernel_map));
  }

  return {in_tensors, out_tensors};
}

template class CoordsManager<CoordsToIndexMap>;
// template class CoordsManager<CoordsToVectorMap>;

} // end namespace minkowski
