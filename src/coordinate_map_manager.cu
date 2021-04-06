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
#include <torch/extension.h>
#include <unordered_map>

namespace py = pybind11;

namespace minkowski {

namespace detail {

template <typename src_type, typename dst_type>
__global__ void cuda_copy_n(src_type const *src, uint32_t N, dst_type *dst) {
  CUDA_KERNEL_LOOP(index, N) { dst[index] = src[index]; }
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator>
struct insert_and_map_functor<coordinate_type, coordinate_field_type,
                              TemplatedAllocator, CoordinateMapGPU> {

  std::pair<at::Tensor, at::Tensor> operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapGPU> &manager) {
    uint32_t const N = th_coordinate.size(0);
    uint32_t const coordinate_size = th_coordinate.size(1);
    coordinate_type *p_coordinate = th_coordinate.data_ptr<coordinate_type>();

    auto coordinate_map = CoordinateMapGPU<coordinate_type, TemplatedAllocator>(
        N, coordinate_size, manager.m_gpu_default_occupancy, map_key.first);

    LOG_DEBUG("inserting", N,
              "coordinates with coordinate_size:", coordinate_size);
    auto input_coordinate_range =
        coordinate_range<coordinate_type>(N, coordinate_size, p_coordinate);

    LOG_DEBUG("insert_and_map");
    auto map_inverse_map = coordinate_map.template insert_and_map<true>(
        input_coordinate_range.begin(), input_coordinate_range.end());
    LOG_DEBUG("mapping size:", map_inverse_map.first.size());

    // insert moves map
    manager.insert(map_key, coordinate_map);

    auto const &mapping = map_inverse_map.first;
    auto const &inverse_mapping = map_inverse_map.second;

    // return tensors
    // TODO int64_t
    LOG_DEBUG("Reserve mapping torch output tensors.");
    at::Tensor th_mapping = torch::empty(
        {(int64_t)mapping.size()},
        th_coordinate.options().requires_grad(false).dtype(torch::kInt64));
    at::Tensor th_inverse_mapping = torch::empty(
        {(int64_t)inverse_mapping.size()},
        th_coordinate.options().requires_grad(false).dtype(torch::kInt64));

    auto const num_blocks =
        (mapping.size() + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    LOG_DEBUG("cuda_copy_n with num_blocks:", num_blocks,
              "mapping.size():", mapping.size());
    detail::cuda_copy_n<default_types::index_type, int64_t>
        <<<num_blocks, CUDA_NUM_THREADS>>>(mapping.cbegin(), mapping.size(),
                                           th_mapping.data_ptr<int64_t>());

    auto const num_inv_blocks =
        (inverse_mapping.size() + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    LOG_DEBUG("cuda_copy_n with num_inv_blocks:", num_inv_blocks,
              "inverse_mapping.size():", inverse_mapping.size());
    if (inverse_mapping.size() > 0) {
      detail::cuda_copy_n<default_types::index_type, int64_t>
          <<<num_inv_blocks, CUDA_NUM_THREADS>>>(
              inverse_mapping.cbegin(), inverse_mapping.size(),
              th_inverse_mapping.data_ptr<int64_t>());
      CUDA_CHECK(cudaStreamSynchronize(0));
    }

    LOG_DEBUG("End of insert_map_functor");
    // return std::make_pair(std::move(th_mapping),
    // std::move(th_inverse_mapping));
    return std::make_pair(th_mapping, th_inverse_mapping);
  }
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct stride_map2tensor_functor<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {

  using gpu_kernel_map_type =
      gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>;

  std::pair<at::Tensor, at::Tensor>
  operator()(gpu_kernel_map_type const &stride_kernel_map) {

    ASSERT(stride_kernel_map.in_maps.size(0) ==
               stride_kernel_map.out_maps.size(0),
           "Invalid kernel map");

    auto curr_device = at::cuda::current_device();
    auto options = torch::TensorOptions({at::kCUDA, curr_device})
                       .dtype(torch::kLong)
                       .requires_grad(false);
    auto const out_size = stride_kernel_map.size();

    // return tensors
    LOG_DEBUG("Reserve mapping torch output tensors with size:", out_size);
    at::Tensor th_in_map = torch::empty({(int64_t)out_size}, options);
    at::Tensor th_out_map = torch::empty({(int64_t)out_size}, options);

    auto const num_blocks =
        (out_size + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    LOG_DEBUG("cuda_copy_n with num_blocks:", num_blocks,
              "mapping size:", out_size);
    detail::cuda_copy_n<default_types::index_type, int64_t>
        <<<num_blocks, CUDA_NUM_THREADS>>>(stride_kernel_map.in_maps.begin(),
                                           out_size,
                                           th_in_map.data_ptr<int64_t>());

    detail::cuda_copy_n<default_types::index_type, int64_t>
        <<<num_blocks, CUDA_NUM_THREADS>>>(stride_kernel_map.out_maps.begin(),
                                           out_size,
                                           th_out_map.data_ptr<int64_t>());

    return std::make_pair(std::move(th_in_map), std::move(th_out_map));
  }
};

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator>
struct insert_field_functor<
    coordinate_type, coordinate_field_type, TemplatedAllocator,
    CoordinateMapGPU,
    CoordinateFieldMapGPU<coordinate_field_type, coordinate_type,
                          TemplatedAllocator>> {

  void operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapGPU> &manager) {
    LOG_DEBUG("insert field");
    uint32_t const N = th_coordinate.size(0);
    uint32_t const coordinate_size = th_coordinate.size(1);
    coordinate_field_type *p_coordinate =
        th_coordinate.data_ptr<coordinate_field_type>();
    auto map = CoordinateFieldMapGPU<coordinate_field_type, coordinate_type,
                                     TemplatedAllocator>(N, coordinate_size,
                                                         map_key.first);
    map.insert(p_coordinate, p_coordinate + N * coordinate_size);

    LOG_DEBUG("insert map with tensor_stride", map_key.first);
    manager.insert_field_map(map_key, map);
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

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct stride_map_functor<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {

  gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>
  operator()(
      CoordinateMapGPU<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapGPU<coordinate_type, TemplatedAllocator> const &out_map,
      default_types::stride_type const &stride) {
    return in_map.stride_map(out_map, stride, CUDA_NUM_THREADS);
  }
};

// a partial specialization functor for kernel map in/out swap
template <>
struct swap_in_out_map_functor<gpu_kernel_map<
    default_types::index_type, detail::default_allocator<char>>> {
  using gpu_kernel_map_type = gpu_kernel_map<default_types::index_type,
                                             detail::default_allocator<char>>;

  gpu_kernel_map_type operator()(gpu_kernel_map_type const &kernel_map) {
    auto swapped_kernel_map = kernel_map.swap();
    LOG_DEBUG("Transposed kernel map in_maps:",
              swapped_kernel_map.out_maps.begin() -
                  swapped_kernel_map.in_maps.begin());
    return std::move(swapped_kernel_map);
  }
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct empty_map_functor<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {
  using gpu_kernel_map_type =
      gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>;

  gpu_kernel_map_type operator()() { return gpu_kernel_map_type{}; }
};

template <>
struct swap_in_out_map_functor<
    gpu_kernel_map<default_types::index_type, detail::c10_allocator<char>>> {
  using gpu_kernel_map_type =
      gpu_kernel_map<default_types::index_type, detail::c10_allocator<char>>;

  gpu_kernel_map_type operator()(gpu_kernel_map_type const &kernel_map) {
    auto swapped_kernel_map = kernel_map.swap();
    LOG_DEBUG("Transposed kernel map in_maps:",
              swapped_kernel_map.out_maps.begin() -
                  swapped_kernel_map.in_maps.begin());
    return std::move(swapped_kernel_map);
  }
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct kernel_map_to_tensors<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {
  using index_type = default_types::index_type;

  std::unordered_map<int64_t, at::Tensor> operator()(
      gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>> const
          &kernel_map) {
    auto curr_device = at::cuda::current_device();
    auto options = torch::TensorOptions({at::kCUDA, curr_device})
                       .dtype(torch::kInt)
                       .requires_grad(false);

    std::unordered_map<int64_t, at::Tensor> kernel_map_th;

    if (kernel_map.size() > 0)
      for (auto it = kernel_map.key_cbegin(); it != kernel_map.key_cend();
           ++it) {
        auto const &key = it->first;
        long const N = kernel_map.size(key);
        at::Tensor curr_map = torch::empty({2, N}, options);
        int32_t *p_map = curr_map.data_ptr<int32_t>();
        CUDA_CHECK(cudaMemcpy(p_map, kernel_map.in_maps.begin(key),
                              sizeof(int32_t) * N, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(p_map + N, kernel_map.out_maps.begin(key),
                              sizeof(int32_t) * N, cudaMemcpyDeviceToDevice));

        kernel_map_th[key] = std::move(curr_map);
      }

    return kernel_map_th;
  }
};

namespace detail {

template <typename dst_type, typename src_type, typename size_type>
__global__ void strided_copy(dst_type *__restrict__ dst,       //
                             size_type const num_threads,      //
                             src_type const *__restrict__ src, //
                             size_type const stride_size) {
  auto const tx = threadIdx.x;
  auto const bx = blockIdx.x;
  auto const x = blockDim.x * bx + tx;

  if (x < num_threads) {
    dst[x] = src[x * stride_size];
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
struct origin_map_functor<
    coordinate_type, TemplatedAllocator, CoordinateMapGPU,
    gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>>> {
  std::pair<at::Tensor, std::vector<at::Tensor>> operator()(
      CoordinateMapGPU<coordinate_type, TemplatedAllocator> const
          &origin_coordinate_map,
      gpu_kernel_map<default_types::index_type, TemplatedAllocator<char>> const
          &origin_map) {
    auto curr_device = at::cuda::current_device();

    auto options = torch::TensorOptions({at::kCUDA, curr_device})
                       .dtype(torch::kLong)
                       .requires_grad(false);
    auto const out_size = origin_coordinate_map.size();
    auto const coordinate_size = origin_coordinate_map.coordinate_size();

    at::Tensor batch_indices = torch::empty({out_size}, options);
    int64_t *d_batch_indices = batch_indices.data_ptr<int64_t>();

    LOG_DEBUG("manager origin map strided_copy");
    // GPU batch indices are sorted
    detail::strided_copy<int64_t, default_types::dcoordinate_type,
                         default_types::size_type>
        <<<GET_BLOCKS(out_size, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
            d_batch_indices, out_size,
            origin_coordinate_map.const_coordinate_data(), coordinate_size);
    CUDA_CHECK(cudaStreamSynchronize(0));

    LOG_DEBUG("manager batch copy");
    std::vector<int64_t> vec_batch_indices(out_size);
    CUDA_CHECK(cudaMemcpy(vec_batch_indices.data(), d_batch_indices,
                          out_size * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));
#ifdef DEBUG
    LOG_DEBUG("Batch indices:", vec_batch_indices);
#endif
    // gpu origin() sort batch indices
    auto const max_batch_index = vec_batch_indices[out_size - 1];

    std::vector<at::Tensor> in_maps;
    default_types::index_type current_batch_row_index = 0;
    for (default_types::index_type i = 0; i < (max_batch_index + 1);) {
      if (vec_batch_indices[current_batch_row_index] == i) {
        auto p_curr_map = origin_map.in_maps.begin(current_batch_row_index);
        auto const curr_size = origin_map.size(current_batch_row_index);
        at::Tensor row_indices = torch::empty({curr_size}, options);
        int64_t *d_row_indices = row_indices.data_ptr<int64_t>();

        LOG_DEBUG("manager batch copy", i);
        detail::strided_copy<int64_t, default_types::index_type,
                             default_types::size_type>
            <<<GET_BLOCKS(curr_size, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
                d_row_indices, curr_size, p_curr_map, 1);
        in_maps.push_back(std::move(row_indices));

        // if there is a match, move the index.
        ++current_batch_row_index;
        if (current_batch_row_index >= out_size) {
          // Should not happen, but for safety
          break;
        }
      } else {
        at::Tensor row_indices = torch::empty({0}, options);
        in_maps.push_back(std::move(row_indices));
      }
      ++i;
    }
    CUDA_CHECK(cudaStreamSynchronize(0));

    return std::make_pair(batch_indices, in_maps);
  }
};

} // namespace detail

template class CoordinateMapManager<
    default_types::dcoordinate_type, default_types::ccoordinate_type,
    detail::default_allocator, CoordinateMapGPU>;
template class CoordinateMapManager<default_types::dcoordinate_type,
                                    default_types::ccoordinate_type,
                                    detail::c10_allocator, CoordinateMapGPU>;

} // end namespace minkowski
