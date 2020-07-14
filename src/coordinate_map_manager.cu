/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#include "coordinate_map_gpu.cuh"
#include "coordinate_map_manager.cpp"
#include "coordinate_map_manager.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace minkowski {

namespace detail {

template <typename SrcType, typename DstType>
__global__ void dtypeCopy(SrcType const *src, DstType *dst, size_t n) {
  CUDA_KERNEL_LOOP(index, n) { dst[index] = src[index]; }
}

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct insert_and_map_functor<coordinate_type, TemplatedAllocator,
                              CoordinateMapGPU> {

  std::pair<at::Tensor, at::Tensor>
  operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
             CoordinateMapManager<coordinate_type, TemplatedAllocator,
                                  CoordinateMapGPU> &manager) {
    uint32_t const N = th_coordinate.size(0);
    uint32_t const coordinate_size = th_coordinate.size(1);
    coordinate_type *p_coordinate = th_coordinate.data_ptr<coordinate_type>();

    auto coordinate_map = CoordinateMapGPU<coordinate_type, TemplatedAllocator>(
        N, coordinate_size, DEFAULT_HASH_TABLE_OCCUPANCY, map_key.first);

    auto input_coordinate_range =
        coordinate_range<coordinate_type>(N, coordinate_size, p_coordinate);

    auto map_inverse_map = coordinate_map.insert_and_map(
        input_coordinate_range.begin(), input_coordinate_range.end());
    LOG_DEBUG("mapping size:", map_inverse_map.first.size());

    // insert moves map
    manager.insert(map_key, coordinate_map);

    auto const &mapping = map_inverse_map.first;
    auto const &inverse_mapping = map_inverse_map.second;

    // return tensors
    at::Tensor th_mapping =
        torch::empty({(int64_t)mapping.size()},
                     th_coordinate.options().requires_grad(false));
    at::Tensor th_inverse_mapping =
        torch::empty({(int64_t)inverse_mapping.size()},
                     th_coordinate.options().requires_grad(false));

    static_assert(sizeof(coordinate_type) == sizeof(default_types::index_type));
    CUDA_CHECK(cudaMemcpy(th_mapping.data_ptr<coordinate_type>(),
                          thrust::raw_pointer_cast(mapping.data()),
                          mapping.size() * sizeof(default_types::index_type),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(th_inverse_mapping.data_ptr<coordinate_type>(),
                   thrust::raw_pointer_cast(inverse_mapping.data()),
                   inverse_mapping.size() * sizeof(default_types::index_type),
                   cudaMemcpyDeviceToDevice));

    return std::make_pair(std::move(th_mapping), std::move(th_inverse_mapping));
  }
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct kernel_map_functor<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {

  gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>
  operator()(
      CoordinateMapGPU<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapGPU<coordinate_type, TemplatedAllocator> const &out_map,
      CUDAKernelMapMode::Mode kernel_map_mode,
      cpu_kernel_region<coordinate_type> &kernel) {
    LOG_DEBUG("cpu_kernel_region initialized with volume", kernel.volume());
    kernel.to_gpu();
    auto gpu_kernel = gpu_kernel_region<coordinate_type>(kernel);
    LOG_DEBUG("gpu_kernel_region initialization");

    return in_map.kernel_map(out_map, gpu_kernel, kernel_map_mode,
                             CUDA_NUM_THREADS);
  }
};

} // namespace detail

template class CoordinateMapManager<int32_t, detail::default_allocator,
                                    CoordinateMapGPU>;
template class CoordinateMapManager<int32_t, detail::c10_allocator,
                                    CoordinateMapGPU>;

} // end namespace minkowski
